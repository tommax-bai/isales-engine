"""Volcengine 豆包流式语音识别大模型 V3 SAUC ASR provider.

Spec: provider-abc § Requirement: ASR Provider 异步流式接口;
      § Scenario "v1 默认 ASR Provider" (火山豆包 default).

Implements ``stream_recognize(audio_chunks) -> AsyncIterator[ASRResult]`` over
the Volcengine **V3 SAUC WebSocket binary** endpoint:

    wss://openspeech.bytedance.com/api/v3/sauc/bigmodel

This is the 豆包流式语音识别大模型 (1.0 / 2.0 family). The legacy V1
``/api/v1/asr`` / ``/api/v3/asr`` paths are unrelated and return HTTP 404
(verified 2026-05-27 mac smoke). The V3 SAUC binary protocol is the
correct path for the grants actually available in console (``volc.bigasr.
sauc.duration`` / ``volc.seedasr.sauc.duration`` etc.).

Two auth modes (vendor docs § "鉴权"):

- **New console (preferred)**: ``X-Api-Key`` UUID + ``X-Api-Resource-Id``
  + ``X-Api-Request-Id`` UUID + ``X-Api-Sequence: -1`` (fixed).
- **Legacy console (fallback)**: ``X-Api-App-Key`` (APP ID) +
  ``X-Api-Access-Key`` (Access Token) + same other headers.

Resource-Id selects the model + billing SKU:

- ``volc.bigasr.sauc.duration`` — 1.0 小时版 (default)
- ``volc.bigasr.sauc.concurrent`` — 1.0 并发版
- ``volc.seedasr.sauc.duration`` — 2.0 小时版 (SeedASR)
- ``volc.seedasr.sauc.concurrent`` — 2.0 并发版

Binary frame protocol (vendor docs § "WebSocket 二进制协议"):

    Byte 0: protocol_version(4) | header_size(4)        # 0b0001_0001
    Byte 1: message_type(4)     | flags(4)
    Byte 2: serialization(4)    | compression(4)
    Byte 3: reserved 0x00
    [Sequence (4 bytes, big-endian int32) iff flags & 0b01]
    Payload size (4 bytes, big-endian uint32)
    Payload (raw bytes; JSON UTF-8 / audio PCM / gzipped per compression)

Message type values:
- 0b0001 = Full client request (JSON config)
- 0b0010 = Audio-only request
- 0b1001 = Full server response (carries result JSON)
- 0b1111 = Error response (carries 4-byte error code, then UTF-8 message)

Flags values:
- 0b0000 = no sequence
- 0b0001 = positive sequence (server response normal frame)
- 0b0010 = no sequence + last-packet marker (client end-of-audio)
- 0b0011 = negative sequence + last-packet (client end-of-audio with seq)

Response JSON shape (V3 SAUC, vendor docs § "full server response"):

    {
      "result": {
        "text": "<aggregated text>",
        "utterances": [
          {
            "definite": <bool>,   # ← is_final equivalent
            "start_time": <ms>,
            "end_time": <ms>,
            "text": "<sentence>",
            "words": [{"text": "<char>", "start_time": <ms>, "end_time": <ms>}, ...]
          }
        ]
      },
      "audio_info": {"duration": <ms>}
    }

Mapping to ``ASRResult(text, is_final, timestamp_ms)``: we yield one result
per utterance — ``is_final`` ← ``utterance.definite``; ``text`` ←
``utterance.text``; ``timestamp_ms`` ← ``utterance.end_time``. Partial
(non-definite) utterances drive the interruption-detector / partial-monitor
in run_loop; definite utterances flush to ``asr_finals_q``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import struct
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from isales_common.providers._models import ASRResult
from isales_common.providers.asr import ASRProvider

from isales_engine.providers._errors import (
    ProviderInvalidRequest,
    ProviderServerError,
)

logger = logging.getLogger(__name__)

DEFAULT_RECONNECT_BACKOFFS_S = (0.2, 0.4, 0.8)

# V3 SAUC endpoint family. The cred-supplied "endpoint" field is ignored
# for V3 — path determines transport (bidirectional / nostream / async)
# and is part of the API contract, not a deployment knob.
V3_SAUC_BIDI_URL = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel"

# Default model SKU. ``volc.bigasr.sauc.duration`` is 1.0 小时版 — granted
# in most accounts that have any ASR product enabled. If the account uses
# 2.0 SeedASR exclusively, set ``ISALES_VOLCENGINE_ASR_RESOURCE_ID=
# volc.seedasr.sauc.duration`` in env.
DEFAULT_ASR_RESOURCE_ID = "volc.bigasr.sauc.duration"

# Default ASR EOS endpoint (pipeline-latency-tail § A): seconds of
# text-stable vendor partial output before we promote-to-final + EOS.
# Lowered from the old hardwired 0.7 to 0.4 (≈ campaign default
# asr_eos_silence_ms=400). Smaller opens the AI's turn faster but risks
# clipping a hesitating caller's pause as "done"; per-campaign override
# via campaign.asr_eos_silence_ms.
DEFAULT_PARTIAL_STABLE_S = 0.4

# Frame protocol constants -----------------------------------------------------

PROTO_VERSION = 0b0001
HEADER_SIZE_UNITS = 0b0001  # 1 * 4 bytes = 4-byte header

MSG_FULL_CLIENT_REQUEST = 0b0001
MSG_AUDIO_ONLY_REQUEST = 0b0010
MSG_FULL_SERVER_RESPONSE = 0b1001
MSG_ERROR_RESPONSE = 0b1111

FLAGS_NO_SEQ = 0b0000
FLAGS_POS_SEQ = 0b0001
FLAGS_NO_SEQ_LAST_PACKET = 0b0010
FLAGS_NEG_SEQ_LAST_PACKET = 0b0011

SERIALIZATION_RAW = 0b0000
SERIALIZATION_JSON = 0b0001

COMPRESSION_NONE = 0b0000
COMPRESSION_GZIP = 0b0001


def _encode_frame(
    *,
    msg_type: int,
    flags: int,
    serialization: int,
    compression: int,
    payload: bytes,
    sequence: int | None = None,
) -> bytes:
    """Encode a binary frame per vendor docs § 'header 数据格式'."""
    header = bytes([
        (PROTO_VERSION << 4) | HEADER_SIZE_UNITS,
        (msg_type << 4) | flags,
        (serialization << 4) | compression,
        0,
    ])
    parts: list[bytes] = [header]
    if sequence is not None:
        parts.append(struct.pack(">i", sequence))  # signed int32 BE
    parts.append(struct.pack(">I", len(payload)))  # uint32 BE payload size
    parts.append(payload)
    return b"".join(parts)


def _decode_frame(raw: bytes) -> dict[str, Any]:
    """Decode a binary server frame. Returns dict with msg_type / flags /
    sequence / error_code / payload fields (whichever present per docs)."""
    if len(raw) < 4:
        raise ValueError(f"frame too short: {len(raw)} bytes")
    b0, b1, b2, b3 = raw[0], raw[1], raw[2], raw[3]
    msg_type = b1 >> 4
    flags = b1 & 0x0F
    serialization = b2 >> 4
    compression = b2 & 0x0F
    header_size_bytes = (b0 & 0x0F) * 4
    cursor = header_size_bytes

    sequence: int | None = None
    error_code: int | None = None

    if msg_type == MSG_ERROR_RESPONSE:
        # Error frame: header + 4-byte error code + 4-byte msg size + msg.
        error_code = struct.unpack(">i", raw[cursor:cursor + 4])[0]
        cursor += 4
    elif flags in (FLAGS_POS_SEQ, FLAGS_NEG_SEQ_LAST_PACKET):
        sequence = struct.unpack(">i", raw[cursor:cursor + 4])[0]
        cursor += 4

    payload_size = struct.unpack(">I", raw[cursor:cursor + 4])[0]
    cursor += 4
    payload = raw[cursor:cursor + payload_size]

    if compression == COMPRESSION_GZIP and payload:
        import gzip  # noqa: PLC0415
        try:
            payload = gzip.decompress(payload)
        except Exception:  # noqa: BLE001
            logger.warning("asr_v3_frame gzip decompress failed; keeping raw")

    return {
        "msg_type": msg_type,
        "flags": flags,
        "serialization": serialization,
        "compression": compression,
        "sequence": sequence,
        "error_code": error_code,
        "payload": payload,
    }


class VolcengineASRProvider(ASRProvider):
    """Per-call WebSocket connection to Volcengine 豆包 V3 SAUC ASR.

    Each call to ``stream_recognize(audio_chunks)`` opens a fresh WebSocket,
    sends the auth + audio config frame, then concurrently:

    - pushes audio chunks as binary frames (msg_type 0b0010)
    - receives server frames (msg_type 0b1001) and parses utterances
    - emits ``ASRResult`` per utterance (definite → is_final=True)

    On disconnection, retries up to ``len(reconnect_backoffs_s)`` times.
    """

    def __init__(
        self,
        *,
        # New-console auth (preferred):
        api_key: str | None = None,
        # Legacy-console auth (fallback):
        app_key: str | None = None,
        access_key: str | None = None,
        # Both modes share:
        resource_id: str = DEFAULT_ASR_RESOURCE_ID,
        sample_rate: int = 16000,
        reconnect_backoffs_s: tuple[float, ...] = DEFAULT_RECONNECT_BACKOFFS_S,
        partial_stable_s: float | None = None,
        override_url: str | None = None,
    ) -> None:
        if not api_key and not (app_key and access_key):
            raise ValueError(
                "VolcengineASRProvider needs either api_key (new console) "
                "or app_key + access_key (legacy console)"
            )
        self._api_key = api_key
        self._app_key = app_key
        self._access_key = access_key
        self._resource_id = resource_id
        self._sample_rate = sample_rate
        self._reconnect_backoffs_s = reconnect_backoffs_s
        # EOS endpoint: how long vendor partial output must be text-stable
        # before we promote it as final + send EOS (pipeline-latency-tail § A).
        # Default lowered 0.7 → 0.4 to open the AI's turn ~300ms sooner;
        # campaign.asr_eos_silence_ms overrides this per-campaign via
        # load_runtime_config → factory build_asr. None → DEFAULT.
        self._partial_stable_s = (
            partial_stable_s if partial_stable_s is not None
            else DEFAULT_PARTIAL_STABLE_S
        )
        self._url = override_url or V3_SAUC_BIDI_URL

    def _headers(self) -> dict[str, str]:
        common = {
            "X-Api-Resource-Id": self._resource_id,
            "X-Api-Request-Id": str(uuid.uuid4()),
            "X-Api-Sequence": "-1",
        }
        if self._api_key:
            return {"X-Api-Key": self._api_key, **common}
        return {
            "X-Api-App-Key": self._app_key or "",
            "X-Api-Access-Key": self._access_key or "",
            **common,
        }

    async def stream_recognize(
        self,
        audio_chunks: AsyncIterator[bytes],
    ) -> AsyncIterator[ASRResult]:
        """Recognize over a SINGLE persistent connection per call.

        Redesign (asr-speaking-ear-close-timeout, validated by
        scripts/continuous_asr_probe.py): the old design closed the vendor
        websocket after every user turn (promote partial + ws.close) and
        reconnected, which left ASR deaf for ~10s during the AI's SPEAKING
        turn so barge-in never fired. Instead we keep ONE connection open for
        the whole call and let the vendor segment turns itself via
        ``end_window_size`` — it force-stops + emits ``definite=true`` after
        that much silence WITHOUT closing. The connection stays "listening"
        while the AI speaks, so the partial monitor can detect a barge-in.

        Yields one ASRResult per vendor utterance: ``definite=true`` →
        ``is_final=True`` (turn final); ``definite=false`` → ``is_final=False``
        (partial; drives barge-in detection in run_loop). Reconnects only on
        an actual disconnection (network resilience), not per turn.
        """
        try:
            import websockets  # noqa: PLC0415
            from websockets.exceptions import ConnectionClosed  # noqa: PLC0415
        except ImportError as exc:
            raise ProviderInvalidRequest(
                "websockets package not installed; run: pip install websockets",
                provider="volcengine_asr",
            ) from exc

        attempt_index = 0
        ts0 = time.monotonic()

        while True:
            try:
                async for result in self._stream_one_connection(
                    websockets, audio_chunks, ts0=ts0,
                ):
                    attempt_index = 0  # any successful frame resets backoff
                    yield result
                # Clean exit only if audio_chunks is exhausted (call teardown).
                return
            except ConnectionClosed as exc:
                if attempt_index >= len(self._reconnect_backoffs_s):
                    raise ProviderServerError(
                        f"volcengine_asr disconnected after {attempt_index} retries: {exc}",
                        provider="volcengine_asr",
                    ) from exc
                backoff = self._reconnect_backoffs_s[attempt_index]
                logger.warning(
                    "volcengine_asr_reconnect attempt=%s backoff=%s reason=%s",
                    attempt_index + 1, backoff, exc,
                )
                await asyncio.sleep(backoff)
                attempt_index += 1
            except Exception as exc:
                raise ProviderServerError(
                    f"volcengine_asr unexpected: {exc}", provider="volcengine_asr",
                ) from exc

    async def _stream_one_connection(
        self,
        websockets_module: Any,
        audio_chunks: AsyncIterator[bytes],
        *,
        ts0: float,
    ) -> AsyncIterator[ASRResult]:
        headers = self._headers()
        # end_window_size (V3 SAUC request param, min 200ms): the vendor
        # force-stops + emits definite after this much silence, WITHOUT
        # closing the connection. Derived from the configured EOS endpoint
        # (campaign.asr_eos_silence_ms → partial_stable_s seconds → ms here).
        end_window_ms = max(200, int(self._partial_stable_s * 1000))
        logger.info(
            "volcengine_asr_connecting endpoint=%s resource_id=%s end_window_ms=%s",
            self._url, self._resource_id, end_window_ms,
        )
        ws_ctx = websockets_module.connect(self._url, additional_headers=headers)
        ws = await ws_ctx.__aenter__()
        try:
            _hdrs = getattr(getattr(ws, "response", None), "headers", None)
            logid = _hdrs.get("x-tt-logid", "?") if _hdrs else "?"
            logger.info("volcengine_asr_connected logid=%s", logid)

            config_payload = {
                "user": {"uid": "isales-engine"},
                "audio": {
                    "format": "pcm",
                    "rate": self._sample_rate,
                    "bits": 16,
                    "channel": 1,
                    "language": "zh-CN",
                },
                "request": {
                    "model_name": "bigmodel",
                    "enable_itn": True,
                    "enable_punc": True,
                    # incremental: don't re-send already-finalized sentences.
                    "result_type": "single",
                    "show_utterances": True,
                    # silence → definite WITHOUT closing the connection.
                    "end_window_size": end_window_ms,
                },
            }
            await ws.send(_encode_frame(
                msg_type=MSG_FULL_CLIENT_REQUEST,
                flags=FLAGS_NO_SEQ,
                serialization=SERIALIZATION_JSON,
                compression=COMPRESSION_NONE,
                payload=json.dumps(config_payload).encode("utf-8"),
            ))

            async def _push_audio() -> None:
                """Push inbound PCM to the vendor batched into ~100ms packets.

                Vendor SAUC doc 明确: 单包 100~200ms、发包间隔 100~200ms,
                "不能过大或者过小,否则均会影响性能"。上游 DingRTC 是 10ms 帧,
                若 1:1 转发 (~100 包/s) 比规格小 10-20×, 实测 (call 150/151)
                导致 definite finalize 拖到 1-4s + keepalive ping timeout 重连。
                这里攒到 ~100ms (sample_rate × 0.1 × 2B) 成一包再发。
                连续音频保证 buffer 每 100ms 满一次, 不会饿死。
                No per-turn EOS / close: 连接活满整通, vendor 按 end_window_size
                切句 (配合上游 asr-noise-gate 喂真静音让 end_window 及时触发)。
                """
                target_bytes = int(self._sample_rate * 0.1) * 2  # 100ms mono int16
                buf = bytearray()
                try:
                    async for chunk in audio_chunks:
                        if not chunk:
                            continue
                        buf += chunk
                        while len(buf) >= target_bytes:
                            await ws.send(_encode_frame(
                                msg_type=MSG_AUDIO_ONLY_REQUEST,
                                flags=FLAGS_NO_SEQ,
                                serialization=SERIALIZATION_RAW,
                                compression=COMPRESSION_NONE,
                                payload=bytes(buf[:target_bytes]),
                            ))
                            del buf[:target_bytes]
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001
                    logger.exception("volcengine_asr_audio_push_failed")

            push_task = asyncio.create_task(_push_audio(), name="asr_push")
            last_final_sig: tuple[str, int] | None = None
            try:
                async for raw in ws:
                    if isinstance(raw, str):
                        continue
                    try:
                        decoded = _decode_frame(raw)
                    except ValueError as exc:
                        logger.warning("volcengine_asr_bad_frame: %s", exc)
                        continue

                    if decoded["msg_type"] == MSG_ERROR_RESPONSE:
                        err_code = decoded.get("error_code")
                        err_msg = (
                            decoded["payload"].decode("utf-8", errors="replace")
                            if decoded["payload"] else ""
                        )
                        logger.error(
                            "volcengine_asr_error code=%s message=%s",
                            err_code, err_msg,
                        )
                        raise ProviderInvalidRequest(
                            f"volcengine_asr error code={err_code} message={err_msg}",
                            provider="volcengine_asr",
                            vendor_code=str(err_code) if err_code is not None else None,
                        )

                    if decoded["msg_type"] != MSG_FULL_SERVER_RESPONSE or not decoded["payload"]:
                        continue
                    try:
                        data = json.loads(decoded["payload"].decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        logger.warning("volcengine_asr_bad_payload: %s", exc)
                        continue

                    for asr_result in _parse_v3_response(data, ts0=ts0):
                        if asr_result.is_final:
                            sig = (asr_result.text, asr_result.timestamp_ms)
                            if sig == last_final_sig:
                                continue  # dedup a repeated definite
                            last_final_sig = sig
                            logger.info(
                                "volcengine_asr_FINAL text=%r", asr_result.text
                            )
                        yield asr_result
            finally:
                push_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await push_task
        finally:
            with contextlib.suppress(Exception):
                await ws_ctx.__aexit__(None, None, None)


def _parse_v3_response(data: dict[str, Any], *, ts0: float) -> list[ASRResult]:
    """Translate a V3 SAUC response into zero or more ``ASRResult``s.

    Each ``utterance`` becomes one ``ASRResult``; ``definite`` → ``is_final``.
    Aggregated ``result.text`` is also surfaced as a partial when no utterances
    are present (some accounts / SKUs return only aggregate text).
    """
    out: list[ASRResult] = []
    if not isinstance(data, dict):
        return out
    result = data.get("result")
    if not isinstance(result, dict):
        return out

    utterances = result.get("utterances")
    if isinstance(utterances, list) and utterances:
        for utt in utterances:
            if not isinstance(utt, dict):
                continue
            text = utt.get("text")
            if not isinstance(text, str) or not text:
                continue
            is_final = bool(utt.get("definite", False))
            end_ms = utt.get("end_time")
            timestamp_ms = (
                int(end_ms) if isinstance(end_ms, int | float)
                else int((time.monotonic() - ts0) * 1000)
            )
            out.append(ASRResult(text=text, is_final=is_final, timestamp_ms=timestamp_ms))
        return out

    # No utterances: surface aggregate text as a partial result.
    text = result.get("text")
    if isinstance(text, str) and text:
        timestamp_ms = int((time.monotonic() - ts0) * 1000)
        out.append(ASRResult(text=text, is_final=False, timestamp_ms=timestamp_ms))
    return out
