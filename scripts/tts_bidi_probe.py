#!/usr/bin/env python3
"""Probe the Volcengine 豆包 V3 **双向流式** (bidirectional streaming) TTS endpoint.

Why this exists
---------------
Today the engine synthesizes one *independent* unidirectional-SSE request per
sentence (``tts_volcengine.py`` → ``/api/v3/tts/unidirectional/sse``). That
resets prosody/语调 every sentence, which is the structural root cause of the
cross-sentence 音色起伏. The fix is to migrate to the bidirectional WebSocket
endpoint where ONE session spans the whole turn and the model plans prosody
across all sentences. Before committing to that migration we must confirm the
account is actually granted the bidi resource for our voices.

This script is that confirmation. It:
  1. opens ``wss://openspeech.bytedance.com/api/v3/tts/bidirection``
  2. StartConnection → ConnectionStarted
  3. StartSession → SessionStarted
  4. sends the demo paragraph as MULTIPLE TaskRequests (one per sentence) inside
     ONE session — exactly the migration usage — then FinishSession
  5. collects the returned PCM and writes a WAV so you can LISTEN for whether the
     cross-sentence delivery is continuous (the whole point)
  6. prints a clear PASS / FAIL grant verdict

Credentials are read from the environment ONLY — nothing is hardcoded or dumped.
Run it locally (export the keys) or on the ECS box (where the keys already live).

Auth (set whichever your console uses):
  New console:    ISALES_VOLC_API_KEY=<uuid>
  Legacy console: ISALES_VOLC_APP_KEY=<app id>  ISALES_VOLC_ACCESS_KEY=<token>
  Optional:       ISALES_VOLC_APP_ID=<app id>   (sent as X-Api-App-Id if set)

Tunables (env, all optional):
  ISALES_VOLC_TTS_RESOURCE_ID  default seed-tts-2.0  (try volc.service_type.10029 if grant error)
  ISALES_VOLC_VOICE            default zh_female_xiaohe_uranus_bigtts
  ISALES_VOLC_SAMPLE_RATE      default 16000
  ISALES_VOLC_EMOTION_SCALE    default 2
  ISALES_VOLC_TEXT             default a 4-sentence demo paragraph (use ; to separate sentences)
  ISALES_VOLC_OUT              default /tmp/tts_bidi_probe.wav

Usage:
  cd ../isales-engine
  ISALES_VOLC_API_KEY=... .venv/bin/python scripts/tts_bidi_probe.py
"""

from __future__ import annotations

import asyncio
import json
import os
import struct
import sys
import uuid
import wave

BIDI_URL = "wss://openspeech.bytedance.com/api/v3/tts/bidirection"

# --- binary protocol constants (V3, same framing family as ASR SAUC) ---------
PROTO_VERSION = 0b0001
HEADER_SIZE_UNITS = 0b0001  # 1 * 4 = 4-byte header

MSG_FULL_CLIENT = 0b0001
MSG_AUDIO_CLIENT = 0b0010
MSG_FULL_SERVER = 0b1001
MSG_AUDIO_SERVER = 0b1011
MSG_FRONTEND_SERVER = 0b1100
MSG_ERROR = 0b1111

FLAG_WITH_EVENT = 0b0100
SER_JSON = 0b0001
SER_RAW = 0b0000
COMP_NONE = 0b0000
COMP_GZIP = 0b0001

# events (client→server)
EV_START_CONNECTION = 1
EV_FINISH_CONNECTION = 2
EV_START_SESSION = 100
EV_FINISH_SESSION = 102
EV_TASK_REQUEST = 200
# events (server→client)
EV_CONNECTION_STARTED = 50
EV_CONNECTION_FAILED = 51
EV_CONNECTION_FINISHED = 52
EV_SESSION_STARTED = 150
EV_SESSION_FINISHED = 152
EV_SESSION_FAILED = 153
EV_TTS_SENTENCE_START = 350
EV_TTS_SENTENCE_END = 351
EV_TTS_RESPONSE = 352
EV_TTS_ENDED = 359

CONNECTION_EVENTS = {
    EV_START_CONNECTION,
    EV_FINISH_CONNECTION,
    EV_CONNECTION_STARTED,
    EV_CONNECTION_FAILED,
    EV_CONNECTION_FINISHED,
}

EVENT_NAMES = {
    1: "StartConnection", 2: "FinishConnection", 100: "StartSession",
    102: "FinishSession", 200: "TaskRequest", 50: "ConnectionStarted",
    51: "ConnectionFailed", 52: "ConnectionFinished", 150: "SessionStarted",
    152: "SessionFinished", 153: "SessionFailed", 350: "TTSSentenceStart",
    351: "TTSSentenceEnd", 352: "TTSResponse", 359: "TTSEnded",
}


def _encode(event: int, session_id: str, payload: bytes,
            *, msg_type: int = MSG_FULL_CLIENT, serialization: int = SER_JSON) -> bytes:
    header = bytes([
        (PROTO_VERSION << 4) | HEADER_SIZE_UNITS,
        (msg_type << 4) | FLAG_WITH_EVENT,
        (serialization << 4) | COMP_NONE,
        0,
    ])
    out = [header, struct.pack(">i", event)]
    if event not in CONNECTION_EVENTS:
        sid = session_id.encode("utf-8")
        out.append(struct.pack(">I", len(sid)))
        out.append(sid)
    out.append(struct.pack(">I", len(payload)))
    out.append(payload)
    return b"".join(out)


def _decode(raw: bytes) -> dict:
    b0, b1, b2 = raw[0], raw[1], raw[2]
    msg_type = b1 >> 4
    flags = b1 & 0x0F
    serialization = b2 >> 4
    compression = b2 & 0x0F
    cur = (b0 & 0x0F) * 4
    out: dict = {"msg_type": msg_type, "flags": flags, "serialization": serialization,
                 "compression": compression, "event": None,
                 "session_id": None, "error_code": None, "payload": b""}
    if flags & FLAG_WITH_EVENT:
        out["event"] = struct.unpack(">i", raw[cur:cur + 4])[0]
        cur += 4
        if out["event"] not in CONNECTION_EVENTS:
            slen = struct.unpack(">I", raw[cur:cur + 4])[0]
            cur += 4
            out["session_id"] = raw[cur:cur + slen].decode("utf-8", "replace")
            cur += slen
    if msg_type == MSG_ERROR:
        out["error_code"] = struct.unpack(">i", raw[cur:cur + 4])[0]
        cur += 4
    if cur + 4 <= len(raw):
        psize = struct.unpack(">I", raw[cur:cur + 4])[0]
        cur += 4
        payload = raw[cur:cur + psize]
        if compression == COMP_GZIP and payload:
            import gzip
            try:
                payload = gzip.decompress(payload)
            except Exception:  # noqa: BLE001
                pass
        out["payload"] = payload
    return out


async def _load_creds_from_db() -> None:
    """ECS-side: decrypt volcengine_speech creds via the engine CredentialStore
    and populate os.environ. Run with ISALES_VOLC_FROM_DB=1 after sourcing the
    engine env (needs ISALES_DATABASE_URL / DATABASE_URL + ISALES_FERNET_KEY).
    Secrets never leave the box and are never printed."""
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from isales_common.credentials import CredentialStore

    dsn = os.environ.get("ISALES_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not dsn:
        sys.exit("ISALES_VOLC_FROM_DB=1 but ISALES_DATABASE_URL/DATABASE_URL unset")
    if dsn.startswith("postgresql://"):
        dsn = dsn.replace("postgresql://", "postgresql+asyncpg://", 1)
    engine = create_async_engine(dsn, echo=False)
    try:
        async with async_sessionmaker(engine, expire_on_commit=False)() as session:
            store = await CredentialStore.from_db(session)
    finally:
        await engine.dispose()

    p = "volcengine_speech"
    api_key = store.get(p, "api_key")
    app_key = store.get(p, "app_key")
    app_token = store.get(p, "app_token")
    res = store.get(p, "tts_resource_id")
    if api_key:
        os.environ.setdefault("ISALES_VOLC_API_KEY", api_key)
    if app_key:
        os.environ.setdefault("ISALES_VOLC_APP_KEY", app_key)
        os.environ.setdefault("ISALES_VOLC_APP_ID", app_key)
    if app_token:
        os.environ.setdefault("ISALES_VOLC_ACCESS_KEY", app_token)
    if res and not os.environ.get("ISALES_VOLC_TTS_RESOURCE_ID"):
        os.environ["ISALES_VOLC_TTS_RESOURCE_ID"] = res
    print(f"  creds from DB: api_key={'set' if api_key else '-'} "
          f"app_key={'set' if app_key else '-'} "
          f"app_token={'set' if app_token else '-'} "
          f"resource_id={os.environ.get('ISALES_VOLC_TTS_RESOURCE_ID', 'seed-tts-2.0')}")


def _headers() -> dict[str, str]:
    api_key = os.environ.get("ISALES_VOLC_API_KEY", "").strip()
    app_key = os.environ.get("ISALES_VOLC_APP_KEY", "").strip()
    access_key = os.environ.get("ISALES_VOLC_ACCESS_KEY", "").strip()
    app_id = os.environ.get("ISALES_VOLC_APP_ID", "").strip()
    resource_id = os.environ.get("ISALES_VOLC_TTS_RESOURCE_ID", "seed-tts-2.0").strip()

    if not api_key and not (app_key and access_key):
        sys.exit(
            "ERROR: set ISALES_VOLC_API_KEY (new console) OR "
            "ISALES_VOLC_APP_KEY + ISALES_VOLC_ACCESS_KEY (legacy console)."
        )
    h = {
        "X-Api-Resource-Id": resource_id,
        "X-Api-Request-Id": str(uuid.uuid4()),
        "X-Api-Connect-Id": str(uuid.uuid4()),
    }
    if api_key:
        h["X-Api-Key"] = api_key
    if app_key:
        h["X-Api-App-Key"] = app_key
    if access_key:
        h["X-Api-Access-Key"] = access_key
    if app_id:
        h["X-Api-App-Id"] = app_id
    return h


def _req_params(text: str | None) -> dict:
    voice = os.environ.get("ISALES_VOLC_VOICE", "zh_female_xiaohe_uranus_bigtts")
    fmt = os.environ.get("ISALES_VOLC_FORMAT", "pcm")
    sr = int(os.environ.get("ISALES_VOLC_SAMPLE_RATE", "16000"))
    speech_rate = int(os.environ.get("ISALES_VOLC_SPEECH_RATE", "0"))
    # lib-exact base: format + sample_rate + speech_rate are always present
    ap: dict = {"format": fmt, "sample_rate": sr, "speech_rate": speech_rate}
    emo = os.environ.get("ISALES_VOLC_EMOTION", "").strip()
    if emo:
        ap["emotion"] = emo
    emo_raw = os.environ.get("ISALES_VOLC_EMOTION_SCALE", "").strip()
    if emo_raw and emo_raw.lower() != "none":
        ap["emotion_scale"] = int(emo_raw)
    rp: dict = {"speaker": voice, "audio_params": ap}
    if text is not None:
        rp["text"] = text
    return rp


async def main() -> int:
    try:
        import websockets
    except ImportError:
        sys.exit("websockets not installed; run from isales-engine/.venv")

    if os.environ.get("ISALES_VOLC_FROM_DB") == "1":
        await _load_creds_from_db()

    headers = _headers()
    uid = headers["X-Api-Request-Id"]
    session_id = str(uuid.uuid4())
    text = os.environ.get(
        "ISALES_VOLC_TEXT",
        "您好，我是小何，很高兴为您服务。今天给您来电是想了解一下您的需求。"
        "我们最近有一个很适合您的方案。方便的话，我简单为您介绍一下。",
    )
    sentences = [s.strip() for s in text.replace("；", ";").split(";") if s.strip()]
    if len(sentences) == 1:
        # fall back to splitting on Chinese full stops so we exercise multi-task
        import re
        sentences = [s for s in re.split(r"(?<=[。！？])", text) if s.strip()]
    out_path = os.environ.get("ISALES_VOLC_OUT", "/tmp/tts_bidi_probe.wav")
    sr = int(os.environ.get("ISALES_VOLC_SAMPLE_RATE", "16000"))

    print(f"→ connecting {BIDI_URL}")
    print(f"  resource_id={headers['X-Api-Resource-Id']} voice={_req_params(None)['speaker']} "
          f"audio_params={_req_params(None)['audio_params']}")
    print(f"  sentences ({len(sentences)}): {sentences}")

    pcm = bytearray()
    try:
        async with websockets.connect(
            BIDI_URL, additional_headers=headers, max_size=None, open_timeout=15
        ) as ws:
            logid = ws.response.headers.get("X-Tt-Logid") if hasattr(ws, "response") else None
            print(f"✓ WS upgraded (logid={logid})")

            async def expect(label: str) -> dict:
                raw = await asyncio.wait_for(ws.recv(), timeout=15)
                m = _decode(raw if isinstance(raw, bytes) else raw.encode())
                ev = m["event"]
                name = EVENT_NAMES.get(ev, str(ev))
                if m["msg_type"] == MSG_ERROR or ev in (EV_CONNECTION_FAILED, EV_SESSION_FAILED):
                    body = m["payload"].decode("utf-8", "replace")
                    raise RuntimeError(
                        f"{label}: server error code={m['error_code']} event={name} body={body}"
                    )
                print(f"  ← {label}: event={name} ({ev}) payload={m['payload'][:120]!r}")
                return m

            # 1. StartConnection
            await ws.send(_encode(EV_START_CONNECTION, session_id, b"{}"))
            await expect("ConnectionStarted")

            # 2. StartSession
            ss = {"user": {"uid": uid}, "namespace": "BidirectionalTTS",
                  "req_params": _req_params(None), "event": EV_START_SESSION}
            await ws.send(_encode(EV_START_SESSION, session_id, json.dumps(ss).encode()))
            await expect("SessionStarted")

            # 3. one TaskRequest per sentence — all in the SAME session
            for i, sent in enumerate(sentences):
                tr = {"user": {"uid": uid}, "namespace": "BidirectionalTTS",
                      "req_params": _req_params(sent), "event": EV_TASK_REQUEST}
                await ws.send(_encode(EV_TASK_REQUEST, session_id, json.dumps(tr).encode()))
                print(f"  → TaskRequest[{i}]: {sent}")

            # 4. FinishSession (no more text)
            await ws.send(_encode(EV_FINISH_SESSION, session_id, b"{}"))

            # 5. drain audio until terminal — FULL diagnostic dump of every frame
            import base64
            frame_no = 0
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=20)
                raw = raw if isinstance(raw, bytes) else raw.encode()
                m = _decode(raw)
                ev, mt = m["event"], m["msg_type"]
                name = EVENT_NAMES.get(ev, str(ev))
                pl = m["payload"]
                frame_no += 1
                # raw header bytes + decoded fields — this is what tells us where audio is
                print(f"  ← #{frame_no} raw[:8]={raw[:8].hex()} mt={mt:#06b} "
                      f"ser={m['serialization']} comp={m['compression']} ev={name} "
                      f"plen={len(pl)} head={pl[:48]!r}")

                if mt == MSG_ERROR or ev in (EV_CONNECTION_FAILED, EV_SESSION_FAILED):
                    raise RuntimeError(
                        f"server error code={m['error_code']} event={name} "
                        f"body={pl.decode('utf-8', 'replace')}"
                    )

                # collect audio: AudioOnlyServer payload is raw PCM; some builds
                # ship PCM inside a TTSResponse(352) frame (raw, or base64 in JSON).
                if mt == MSG_AUDIO_SERVER and pl:
                    pcm += pl
                    continue
                if ev == EV_TTS_RESPONSE and pl:
                    if m["serialization"] == SER_JSON:
                        try:
                            obj = json.loads(pl)
                            b64 = (obj.get("data") if isinstance(obj, dict) else None)
                            if b64:
                                pcm += base64.b64decode(b64)
                        except Exception:  # noqa: BLE001
                            pass
                    else:
                        pcm += pl
                    continue
                if ev in (EV_SESSION_FINISHED, EV_TTS_ENDED):
                    break

            # 6. FinishConnection (best-effort)
            try:
                await ws.send(_encode(EV_FINISH_CONNECTION, session_id, b"{}"))
                await asyncio.wait_for(ws.recv(), timeout=5)
            except Exception:  # noqa: BLE001
                pass
    except Exception as exc:  # noqa: BLE001
        print(f"\n✗ FAIL: {type(exc).__name__}: {exc}")
        print("\nIf this is an auth/grant error (45000001/45000002/403):")
        print("  • try ISALES_VOLC_TTS_RESOURCE_ID=volc.service_type.10029")
        print("  • verify the same speaker works on the unidirectional path first")
        return 1

    if not pcm:
        print("\n✗ FAIL: handshake OK but zero audio bytes returned "
              f"(format={os.environ.get('ISALES_VOLC_FORMAT', 'pcm')} "
              f"resource_id={os.environ.get('ISALES_VOLC_TTS_RESOURCE_ID', 'seed-tts-2.0')}).")
        return 2

    fmt = os.environ.get("ISALES_VOLC_FORMAT", "pcm")
    if fmt == "pcm":
        with wave.open(out_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(bytes(pcm))
        dur = f"{len(pcm) / 2 / sr:.2f}s"
        final = out_path
    else:
        final = out_path.rsplit(".", 1)[0] + f".{fmt.replace('ogg_opus', 'ogg')}"
        with open(final, "wb") as f:
            f.write(bytes(pcm))
        dur = "?"
    print(f"\n✓ PASS — bidi session granted & synthesized ({fmt}).")
    print(f"  {len(pcm)} audio bytes  ≈ {dur}  →  {final}")
    print(f"  Listen for cross-sentence continuity:  afplay {final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
