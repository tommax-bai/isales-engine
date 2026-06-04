"""Continuous-connection ASR probe — test Volcengine V3 SAUC multi-utterance
behavior on a SINGLE websocket (no per-turn close).

Question this answers (for the barge-in redesign): if we keep ONE ASR
connection open for the whole call and feed audio continuously (multiple
utterances with silence gaps) WITHOUT sending last-packet / closing between
turns, does the vendor:
  - emit a clean ``definite=true`` per utterance (natural silence segmentation)?
  - reset or accumulate ``result.text`` across utterances?
  - index ``utterances[]`` per-segment or cumulatively?

This determines whether the engine can do its own turn segmentation on a
live connection (promote stable partials as finals + dedupe the vendor's
late definites) instead of the fragile close+reconnect-per-turn dance.

Run on ECS (has provider_credential in DB + network to Volcengine):
    set -a; source /etc/isales/env/engine.env; set +a
    /opt/isales/current/venv/bin/python scripts/continuous_asr_probe.py

Audio is TTS-synthesized (clean, deterministic) for 3 short phrases, fed at
real-time pace with ~1.2s silence gaps. Every vendor server response is
logged with its text + per-utterance definite/text/time.
"""

from __future__ import annotations

import asyncio
import json
import os
import time

import websockets

from isales_engine.db import get_engine, get_sessionmaker
from isales_engine.providers import asr_volcengine as A
from isales_engine.providers.tts_volcengine import VolcengineTTSProvider

PHRASES = ["你好，我是张三", "我现在不太方便接电话", "下午三点以后再打给我"]
SR = 16000
FRAME_BYTES = SR // 100 * 2  # 10 ms @ 16k mono int16 = 320 bytes
SILENCE_GAP_S = 1.2


async def _synth(tts: VolcengineTTSProvider, text: str) -> bytes:
    buf = bytearray()
    async for chunk in tts.synthesize_stream(text, "default"):
        buf += chunk
    return bytes(buf)


async def main() -> None:
    db_url = os.environ["ISALES_DATABASE_URL"]
    engine = get_engine(db_url)
    sm = get_sessionmaker(engine)
    from isales_common.credentials import CredentialStore

    async with sm() as db:
        store = await CredentialStore.from_db(db)
    api_key = store.get("volcengine", "api_key")
    if not api_key:
        raise SystemExit("no volcengine api_key in provider_credential")

    tts = VolcengineTTSProvider(
        api_key=api_key,
        resource_id=store.get("volcengine", "tts_resource_id") or "seed-tts-2.0",
        sample_rate=SR,
    )
    asr = A.VolcengineASRProvider(
        api_key=api_key,
        resource_id=store.get("volcengine", "asr_resource_id")
        or A.DEFAULT_ASR_RESOURCE_ID,
    )

    print("synthesizing phrases via TTS ...", flush=True)
    pcms = [await _synth(tts, p) for p in PHRASES]
    for p, pcm in zip(PHRASES, pcms):
        print(f"  '{p}' -> {len(pcm)} bytes ({len(pcm)/SR/2:.2f}s)", flush=True)
    await tts.aclose()

    silence = b"\x00" * (int(SR * SILENCE_GAP_S) * 2)

    headers = asr._headers()
    print(f"\nopening ONE ASR connection to {A.V3_SAUC_BIDI_URL}", flush=True)
    ws = await websockets.connect(A.V3_SAUC_BIDI_URL, additional_headers=headers)

    # config frame (same as _stream_one_connection)
    config = {
        "user": {"uid": "isales-probe"},
        "audio": {"format": "pcm", "rate": SR, "bits": 16, "channel": 1,
                  "language": "zh-CN"},
        "request": {"model_name": "bigmodel", "enable_itn": True,
                    "enable_punc": True, "result_type": "single",
                    "show_utterances": True,
                    # KEY: force a stop (→ definite) after this much silence
                    # WITHOUT closing the connection. Disables semantic
                    # segmentation (vad_segment_duration). min 200ms.
                    "end_window_size": 500},
    }
    await ws.send(A._encode_frame(
        msg_type=A.MSG_FULL_CLIENT_REQUEST, flags=A.FLAGS_NO_SEQ,
        serialization=A.SERIALIZATION_JSON, compression=A.COMPRESSION_NONE,
        payload=json.dumps(config).encode(),
    ))

    t0 = time.monotonic()

    async def _recv() -> None:
        try:
            async for raw in ws:
                if isinstance(raw, str):
                    continue
                d = A._decode_frame(raw)
                if d["msg_type"] == A.MSG_ERROR_RESPONSE:
                    print(f"[{time.monotonic()-t0:6.2f}] ERROR code={d.get('error_code')} "
                          f"{d['payload'][:200]!r}", flush=True)
                    continue
                if d["msg_type"] != A.MSG_FULL_SERVER_RESPONSE or not d["payload"]:
                    continue
                try:
                    data = json.loads(d["payload"].decode())
                except Exception:  # noqa: BLE001
                    continue
                res = data.get("result") or {}
                text = res.get("text", "")
                utts = res.get("utterances") or []
                utt_str = " | ".join(
                    f"def={u.get('definite')} [{u.get('start_time')}-{u.get('end_time')}] "
                    f"{u.get('text')!r}" for u in utts
                )
                # only log frames that carry text (skip empty acks)
                if text or utts:
                    print(f"[{time.monotonic()-t0:6.2f}] text={text!r}  utt=[{utt_str}]",
                          flush=True)
        except websockets.exceptions.ConnectionClosed:
            print(f"[{time.monotonic()-t0:6.2f}] (recv) connection closed", flush=True)

    recv_task = asyncio.create_task(_recv())

    async def _feed(seg: bytes, label: str) -> None:
        print(f"[{time.monotonic()-t0:6.2f}] >>> feeding {label} ({len(seg)} bytes)", flush=True)
        for i in range(0, len(seg), FRAME_BYTES):
            frame = seg[i:i + FRAME_BYTES]
            if len(frame) < FRAME_BYTES:
                frame = frame + b"\x00" * (FRAME_BYTES - len(frame))
            await ws.send(A._encode_frame(
                msg_type=A.MSG_AUDIO_ONLY_REQUEST, flags=A.FLAGS_NO_SEQ,
                serialization=A.SERIALIZATION_RAW, compression=A.COMPRESSION_NONE,
                payload=frame,
            ))
            await asyncio.sleep(0.01)  # real-time pace

    # Feed: phrase1, silence, phrase2, silence, phrase3, trailing silence —
    # NO last-packet / close between phrases.
    for idx, pcm in enumerate(pcms):
        await _feed(pcm, f"PHRASE{idx+1} {PHRASES[idx]!r}")
        await _feed(silence, f"SILENCE after phrase{idx+1}")

    print(f"[{time.monotonic()-t0:6.2f}] all phrases fed; waiting 5s for late definites ...",
          flush=True)
    await asyncio.sleep(5)

    # Now end the session: last-packet + close.
    print(f"[{time.monotonic()-t0:6.2f}] sending last-packet + close", flush=True)
    await ws.send(A._encode_frame(
        msg_type=A.MSG_AUDIO_ONLY_REQUEST, flags=A.FLAGS_NO_SEQ_LAST_PACKET,
        serialization=A.SERIALIZATION_RAW, compression=A.COMPRESSION_NONE, payload=b"",
    ))
    await asyncio.sleep(3)
    await ws.close()
    await asyncio.wait_for(recv_task, timeout=5)
    await engine.dispose()
    print("\n=== DONE ===", flush=True)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level="WARNING")  # silence provider INFO noise
    asyncio.run(main())
