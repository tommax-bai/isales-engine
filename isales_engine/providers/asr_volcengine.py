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
                    if result.is_final:
                        logger.info(
                            "volcengine_asr_stream_recognize_yielding_FINAL "
                            "text=%r",
                            result.text,
                        )
                    yield result
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
        logger.info(
            "volcengine_asr_connecting endpoint=%s resource_id=%s",
            self._url, self._resource_id,
        )
        try:
            ws_ctx = websockets_module.connect(self._url, additional_headers=headers)
            ws = await ws_ctx.__aenter__()
        except Exception as exc:  # noqa: BLE001
            logger.exception("volcengine_asr_connect_failed: %s", exc)
            raise
        try:
            logid = getattr(getattr(ws, "response", None), "headers", {})
            if logid:
                try:
                    logid = logid.get("x-tt-logid", "?")
                except Exception:  # noqa: BLE001
                    logid = "?"
            logger.info("volcengine_asr_connected logid=%s", logid)

            # Step 1: send Full client request with audio + request config.
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
                    "result_type": "single",      # incremental
                    "show_utterances": True,       # surface definite + word timing
                },
            }
            config_frame = _encode_frame(
                msg_type=MSG_FULL_CLIENT_REQUEST,
                flags=FLAGS_NO_SEQ,
                serialization=SERIALIZATION_JSON,
                compression=COMPRESSION_NONE,
                payload=json.dumps(config_payload).encode("utf-8"),
            )
            await ws.send(config_frame)

            # Step 2: concurrently push audio frames + recv server frames.
            push_done = asyncio.Event()
            # Q-fix: partial-promote on silence-EOS. Shared state between
            # _push_audio (writer) and the recv loop (reader) so EOS can
            # immediately surface the latest partial as a final, without
            # waiting for vendor's own finalize (observed 7+ s delay on
            # 2026-06-01 mac dev-no-modem). After promote, vendor's eventual
            # final is suppressed (avoid double-yield).
            _latest_partial_text: list[str | None] = [None]
            _latest_partial_update_at: list[float] = [0.0]  # monotonic ts
            _promote_yielded: list[bool] = [False]
            _pending_promote: list[ASRResult] = []  # max 1 entry, drained by recv loop

            async def _push_audio() -> None:
                # DIAG-REMOVE-AFTER-MIC-DEBUG: push chunks/bytes per 1s + dump first 5s
                import time as _t  # noqa: PLC0415
                _last = _t.monotonic()
                _chunks = 0
                _bytes = 0
                # DIAG-REMOVE: dump first ~5s (100 chunks × 320B = 16kHz mono = 5s)
                # to verify what's actually going to vendor
                _dump_path = "/tmp/asr_push_dump.pcm"
                _dump_fh: Any = None
                _dump_max_bytes = 5 * 16000 * 2  # 5s @ 16k mono int16
                _dump_written = 0
                try:
                    _dump_fh = open(_dump_path, "wb")
                except Exception:  # noqa: BLE001
                    pass

                # Client-side silence-driven EOS (PERMANENT, not DIAG).
                # Vendor V3 SAUC does not emit ``definite=true`` until either
                # (a) it sees absolute silence in its own server-side VAD or
                # (b) the client sends ``FLAGS_NO_SEQ_LAST_PACKET`` EOS.
                # The (a) path is unreliable under low-energy comfort noise
                # (dev-no-modem mixed-playback self-loopback measured 50-350
                # final_rms after the WAV ends; noisy real-mic environments
                # also stay above absolute silence). So drive (b) actively:
                # after ``SILENCE_DURATION_S`` of chunks with RMS below
                # ``SILENCE_RMS_THRESHOLD``, send EOS to force finalize.
                # Vendor then emits the final + closes the connection; the
                # outer ``stream_recognize`` ``while True`` reconnects when
                # new audio arrives. Threshold + duration tuned against
                # 2026-06-01 mac dev-no-modem dump data (rtc_inbound_1s log).
                try:
                    import audioop  # noqa: PLC0415
                except ImportError:  # pragma: no cover  - Py 3.13+ removal
                    audioop = None  # type: ignore[assignment]
                # Hysteresis: low threshold to enter silence, high to exit.
                # Single-chunk RMS jitter is high (observed 125 → 335 → 1152
                # → 2881 in 1 s on dev-no-modem self-loopback), so reset
                # ONLY on clear speech (> _SPEECH_RMS_RESET); chunks in the
                # 500-1500 band keep whatever silence state we're already in.
                _SILENCE_RMS_THRESHOLD = 500
                _SPEECH_RMS_RESET = 1500
                _SILENCE_DURATION_S = 1.5
                _silence_started_at: float | None = None

                try:
                    async for chunk in audio_chunks:
                        if not chunk:
                            continue
                        # DIAG-REMOVE: dump
                        if _dump_fh and _dump_written < _dump_max_bytes:
                            _dump_fh.write(chunk)
                            _dump_written += len(chunk)
                            if _dump_written >= _dump_max_bytes:
                                _dump_fh.close()
                                _dump_fh = None
                                logger.info(
                                    "volcengine_asr_push_dump_complete path=%s bytes=%s",
                                    _dump_path, _dump_written,
                                )

                        _now = _t.monotonic()
                        # Silence-driven EOS check (before send) — if this
                        # chunk is silence and we've accumulated enough, fire
                        # EOS and return without sending this chunk; vendor
                        # finalizes on the EOS marker.
                        _chunk_rms = 0
                        if audioop is not None:
                            try:
                                _chunk_rms = audioop.rms(chunk, 2)
                            except Exception:  # noqa: BLE001
                                _chunk_rms = 0
                        if _chunk_rms < _SILENCE_RMS_THRESHOLD:
                            if _silence_started_at is None:
                                _silence_started_at = _now
                            elif (_now - _silence_started_at) >= _SILENCE_DURATION_S:
                                logger.info(
                                    "volcengine_asr_silence_eos: "
                                    "silence_duration_s=%.2f chunk_rms=%s "
                                    "low_thr=%s high_reset=%s — sending EOS",
                                    _now - _silence_started_at,
                                    _chunk_rms, _SILENCE_RMS_THRESHOLD,
                                    _SPEECH_RMS_RESET,
                                )
                                # Q-fix: send EOS, wait for vendor partial
                                # to catch up (vendor server-side processing
                                # lags push-side silence detection by ~500ms
                                # observed on 2026-06-01), then promote
                                # latest partial as final. Recv loop yields
                                # promoted via _pending_promote on ws.close.
                                _silence_eos = _encode_frame(
                                    msg_type=MSG_AUDIO_ONLY_REQUEST,
                                    flags=FLAGS_NO_SEQ_LAST_PACKET,
                                    serialization=SERIALIZATION_RAW,
                                    compression=COMPRESSION_NONE,
                                    payload=b"",
                                )
                                try:
                                    await ws.send(_silence_eos)
                                except Exception as _exc:  # noqa: BLE001
                                    logger.warning(
                                        "volcengine_asr_silence_eos_send_failed: %s",
                                        _exc,
                                    )
                                # Wait for partial to STABILIZE (vendor stops
                                # changing partial text — typically means
                                # utterance complete). Stable = 0.3 s no
                                # text change. MAX 5 s — vendor server-side
                                # processing latency is unstable across days
                                # (observed 1 s on 2026-06-01 smoke #20 vs
                                # 4 s on 2026-06-02 smoke #21 same setup);
                                # need headroom so post-EOS catchup doesn't
                                # under-shoot. campaign silence_threshold_ms
                                # must be ≥ silence_acc (1.5) + MAX_WAIT (5)
                                # + LLM RTT (~3-5) = ~10 s minimum for full
                                # chain. If no partial ever arrives, skip
                                # promote.
                                _STABLE_S = 0.3
                                _MAX_WAIT_S = 5.0
                                _wait_start = _t.monotonic()
                                while (_t.monotonic() - _wait_start) < _MAX_WAIT_S:
                                    if _latest_partial_text[0]:
                                        _since_update = (
                                            _t.monotonic()
                                            - _latest_partial_update_at[0]
                                        )
                                        if _since_update >= _STABLE_S:
                                            break
                                    await asyncio.sleep(0.05)
                                _pt = _latest_partial_text[0]
                                _waited_s = _t.monotonic() - _wait_start
                                if _pt:
                                    _promoted = ASRResult(
                                        text=_pt,
                                        is_final=True,
                                        timestamp_ms=int(
                                            (_t.monotonic() - ts0) * 1000,
                                        ),
                                    )
                                    _pending_promote.append(_promoted)
                                    _promote_yielded[0] = True
                                    logger.info(
                                        "volcengine_asr_promote_partial_to_final "
                                        "text=%r catchup_waited_s=%.2f",
                                        _pt, _waited_s,
                                    )
                                else:
                                    logger.warning(
                                        "volcengine_asr_silence_eos_no_partial "
                                        "catchup_waited_s=%.2f — vendor never "
                                        "emitted partial",
                                        _waited_s,
                                    )
                                # Close ws → recv loop wakes up → drain
                                # _pending_promote → outer reconnect.
                                try:
                                    await ws.close()
                                except Exception:  # noqa: BLE001
                                    pass
                                return
                        elif _chunk_rms > _SPEECH_RMS_RESET:
                            _silence_started_at = None  # clear speech resets
                        # else: 500 <= rms <= 1500, in-between — maintain state

                        audio_frame = _encode_frame(
                            msg_type=MSG_AUDIO_ONLY_REQUEST,
                            flags=FLAGS_NO_SEQ,
                            serialization=SERIALIZATION_RAW,
                            compression=COMPRESSION_NONE,
                            payload=chunk,
                        )
                        await ws.send(audio_frame)
                        # DIAG-REMOVE-AFTER-MIC-DEBUG ------------------------
                        _chunks += 1
                        _bytes += len(chunk)
                        if _now - _last >= 1.0:
                            logger.info(
                                "volcengine_asr_push_1s chunks=%s bytes=%s "
                                "first_chunk_len=%s chunk_rms=%s silence_acc_s=%.2f",
                                _chunks, _bytes, len(chunk), _chunk_rms,
                                (_now - _silence_started_at) if _silence_started_at else 0.0,
                            )
                            _last = _now
                            _chunks = 0
                            _bytes = 0
                        # DIAG-REMOVE-AFTER-MIC-DEBUG end --------------------
                    # End-of-stream: empty audio frame with last-packet flag.
                    eos_frame = _encode_frame(
                        msg_type=MSG_AUDIO_ONLY_REQUEST,
                        flags=FLAGS_NO_SEQ_LAST_PACKET,
                        serialization=SERIALIZATION_RAW,
                        compression=COMPRESSION_NONE,
                        payload=b"",
                    )
                    await ws.send(eos_frame)
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001
                    logger.exception("volcengine_asr_audio_push_failed")
                finally:
                    push_done.set()

            push_task = asyncio.create_task(_push_audio(), name="asr_push")

            # DIAG-REMOVE-AFTER-MIC-DEBUG: recv frame count + msg_type histogram
            import time as _t  # noqa: PLC0415
            _last_recv_log = _t.monotonic()
            _recv_count = 0
            _msg_type_seen: dict[int, int] = {}
            _first_payload_preview: str | None = None
            try:
                # Q-fix: replaced `async for raw in ws` with a 100 ms polling
                # loop so we can check `_pending_promote` even when vendor
                # does not respond to ws.close() immediately. Original
                # `async for` blocks on next ws frame; vendor sometimes
                # keeps the connection open after EOS for 7+ s draining
                # buffered audio, during which a queued promoted-final
                # would be stuck waiting for ConnectionClosed that never
                # arrives. Now: wake every 100 ms, drain any pending
                # promote, then re-enter recv.
                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=0.1)
                    except asyncio.TimeoutError:
                        if _pending_promote:
                            while _pending_promote:
                                _yp = _pending_promote.pop(0)
                                logger.info(
                                    "volcengine_asr_yield_promoted_in_poll "
                                    "text=%r is_final=%s",
                                    _yp.text, _yp.is_final,
                                )
                                yield _yp
                            # ws.close() was called by _push_audio; recv
                            # connection is dead — break out of poll loop
                            # so outer reconnects on the natural next
                            # utterance.
                            break
                        continue
                    if isinstance(raw, str):
                        # Text frames are unexpected on this protocol but log
                        # them for diagnostics rather than crashing.
                        logger.warning(
                            "volcengine_asr_unexpected_text_frame len=%s",
                            len(raw),
                        )
                        continue
                    try:
                        decoded = _decode_frame(raw)
                    except ValueError as exc:
                        logger.warning("volcengine_asr_bad_frame: %s", exc)
                        continue

                    msg_type = decoded["msg_type"]
                    payload = decoded["payload"]
                    # DIAG-REMOVE-AFTER-MIC-DEBUG begin ----------------------
                    _recv_count += 1
                    _msg_type_seen[msg_type] = _msg_type_seen.get(msg_type, 0) + 1
                    if _first_payload_preview is None and payload:
                        try:
                            _first_payload_preview = payload[:200].decode(
                                "utf-8", errors="replace",
                            )
                            logger.info(
                                "volcengine_asr_first_recv msg_type=0x%x "
                                "payload_len=%s preview=%r",
                                msg_type, len(payload), _first_payload_preview,
                            )
                        except Exception:  # noqa: BLE001
                            pass
                    # Log every frame with non-empty text (filter handshake
                    # ack where text=""). Caller can grep `NONEMPTY_TEXT` to
                    # see if vendor ever emits a real partial.
                    if payload:
                        try:
                            _txt = payload[:500].decode("utf-8", errors="replace")
                            if '"text":"' in _txt and '"text":""' not in _txt:
                                logger.info(
                                    "volcengine_asr_NONEMPTY_TEXT msg_type=0x%x "
                                    "preview=%r",
                                    msg_type, _txt,
                                )
                        except Exception:  # noqa: BLE001
                            pass
                    _now = _t.monotonic()
                    if _now - _last_recv_log >= 1.0:
                        logger.info(
                            "volcengine_asr_recv_1s total=%s msg_types=%s",
                            _recv_count,
                            {f"0x{k:x}": v for k, v in _msg_type_seen.items()},
                        )
                        _last_recv_log = _now
                    # DIAG-REMOVE-AFTER-MIC-DEBUG end ------------------------

                    if msg_type == MSG_ERROR_RESPONSE:
                        err_code = decoded.get("error_code")
                        err_msg = (
                            payload.decode("utf-8", errors="replace")
                            if payload else ""
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

                    if msg_type != MSG_FULL_SERVER_RESPONSE:
                        continue

                    if not payload:
                        continue
                    try:
                        data = json.loads(payload.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        logger.warning(
                            "volcengine_asr_bad_payload: %s len=%s",
                            exc, len(payload),
                        )
                        continue

                    for asr_result in _parse_v3_response(data, ts0=ts0):
                        # Q-fix: track latest partial text. Update timestamp
                        # ONLY when text actually changes — vendor pushes the
                        # same partial every ~20 ms even when content is
                        # unchanged, so a timestamp-on-every-frame model
                        # would never see "stable" (silence-EOS catchup loop
                        # relies on `(now - update_at) >= STABLE_S` to detect
                        # utterance completion).
                        if not asr_result.is_final and asr_result.text:
                            import time as _t2  # noqa: PLC0415
                            if asr_result.text != _latest_partial_text[0]:
                                _latest_partial_text[0] = asr_result.text
                                _latest_partial_update_at[0] = _t2.monotonic()
                        # Suppress vendor's eventual real final after we've
                        # already promoted the partial — avoid double-final
                        # yield to engine state machine (would trigger 2nd
                        # LLM turn on the same utterance content).
                        if _promote_yielded[0] and asr_result.is_final:
                            continue
                        yield asr_result
            except ConnectionClosed as _cc_exc:
                # Drain any promoted final queued by _push_audio before
                # letting outer stream_recognize reconnect. ws.close() was
                # called by _push_audio after silence-EOS to skip vendor's
                # 7+ s real-finalize delay.
                logger.info(
                    "volcengine_asr_recv_connectionclosed pending_promote_len=%s "
                    "exc=%r",
                    len(_pending_promote), _cc_exc,
                )
                while _pending_promote:
                    _p_yield = _pending_promote.pop(0)
                    logger.info(
                        "volcengine_asr_yield_promoted_in_except "
                        "text=%r is_final=%s",
                        _p_yield.text, _p_yield.is_final,
                    )
                    yield _p_yield
                logger.info("volcengine_asr_recv_reraising_connectionclosed")
                raise
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
