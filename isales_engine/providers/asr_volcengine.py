"""Volcengine streaming ASR provider (WebSocket).

Spec: provider-abc § Requirement: ASR Provider 异步流式接口;
      § Scenario "v1 默认 ASR Provider" (火山豆包 default).

Implements ``stream_recognize(audio_chunks) -> AsyncIterator[ASRResult]``
over the publicly-documented Volcengine real-time ASR WebSocket endpoint.

**Vendor verification required before production**: the exact wire format
(initial config message + audio binary frame headers + result JSON shape)
follows publicly available Volcengine 实时语音识别 documentation but the
field names + binary layouts vary by API version. Smoke-test against your
account's WSS URL + run ``tests/test_real_providers.py::test_volcengine_asr_smoke``
with a recorded 8 kHz wav before relying on the partial / final stream.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
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


class VolcengineASRProvider(ASRProvider):
    """Per-call WebSocket connection to Volcengine real-time ASR.

    Each call to ``stream_recognize(audio_chunks)`` opens a fresh WebSocket,
    sends the auth + audio config, then concurrently pushes audio chunks
    and yields recognition results. Disconnections retry up to
    ``len(reconnect_backoffs_s)`` times before giving up with a
    ``ProviderServerError``.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        app_key: str,
        app_token: str,
        sample_rate: int = 8000,
        reconnect_backoffs_s: tuple[float, ...] = DEFAULT_RECONNECT_BACKOFFS_S,
    ) -> None:
        self._endpoint = endpoint
        self._app_key = app_key
        self._app_token = app_token
        self._sample_rate = sample_rate
        self._reconnect_backoffs_s = reconnect_backoffs_s

    async def stream_recognize(
        self,
        audio_chunks: AsyncIterator[bytes],
    ) -> AsyncIterator[ASRResult]:
        """Stream PCM frames + yield partial / final ``ASRResult`` objects."""

        try:
            import websockets
            from websockets.exceptions import ConnectionClosed
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
                    websockets, audio_chunks, ts0=ts0
                ):
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
                    attempt_index + 1,
                    backoff,
                    exc,
                )
                await asyncio.sleep(backoff)
                attempt_index += 1
            except Exception as exc:
                raise ProviderServerError(
                    f"volcengine_asr unexpected: {exc}", provider="volcengine_asr"
                ) from exc

    async def _stream_one_connection(
        self,
        websockets_module: Any,
        audio_chunks: AsyncIterator[bytes],
        *,
        ts0: float,
    ) -> AsyncIterator[ASRResult]:
        headers = {
            "Authorization": f"Bearer; {self._app_token}",
            "App-Key": self._app_key,
            "Resource-Id": "asr",
        }

        logger.info(
            "volcengine_asr_connecting endpoint=%s headers=%s",
            self._endpoint, list(headers.keys()),
        )
        try:
            ws_ctx = websockets_module.connect(
                self._endpoint, additional_headers=headers
            )
            ws = await ws_ctx.__aenter__()
        except Exception as exc:  # noqa: BLE001
            logger.exception("volcengine_asr_connect_failed: %s", exc)
            raise
        logger.info("volcengine_asr_connected")
        try:
            # Send config frame (JSON). Field names follow Volcengine 实时
            # 语音识别 docs — verify against the vendor account before
            # production.
            config = {
                "audio": {
                    "encoding": "pcm",
                    "rate": self._sample_rate,
                    "bits": 16,
                    "channel": 1,
                },
                "request": {
                    "reqid": _request_id(),
                    "model_name": "bigmodel",
                    "enable_punc": True,
                    "result_type": "full",
                },
            }
            await ws.send(json.dumps(config))

            # Producer task: forward audio chunks → ws as binary frames.
            async def _push_audio() -> None:
                try:
                    async for chunk in audio_chunks:
                        if not chunk:
                            continue
                        await ws.send(chunk)
                    # End-of-stream sentinel — vendor docs require either an
                    # empty binary frame or a JSON {"is_last": true} frame.
                    await ws.send(json.dumps({"request": {"is_last": True}}))
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("volcengine_asr_audio_push_failed")

            push_task = asyncio.create_task(_push_audio(), name="asr_push")

            try:
                async for raw in ws:
                    parsed = _parse_result_frame(raw, ts0=ts0)
                    if parsed is not None:
                        yield parsed
            finally:
                push_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await push_task
        finally:
            with contextlib.suppress(Exception):
                await ws_ctx.__aexit__(None, None, None)


def _parse_result_frame(raw: bytes | str, *, ts0: float) -> ASRResult | None:
    """Parse a vendor result frame into an ``ASRResult``.

    Frame shape per Volcengine 实时语音识别 (verify with vendor):

    ::

        {
          "result": [{"text": "你好", "is_final": false}],
          "code": 0
        }

    Returns ``None`` for non-result frames (heartbeats, ack, etc.).
    """

    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError:
            return None
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None

    # Most vendors put recognition results inside a `result` array; some put
    # text directly at the top level. Accept both shapes.
    result = data.get("result")
    text: object = None
    is_final = False
    if isinstance(result, list) and result:
        entry = result[0]
        if isinstance(entry, dict):
            text = entry.get("text")
            is_final = bool(entry.get("is_final", False))
    elif isinstance(result, dict):
        text = result.get("text")
        is_final = bool(result.get("is_final", False))
    else:
        text = data.get("text")
        is_final = bool(data.get("is_final", False))

    if not isinstance(text, str):
        return None

    timestamp_ms = int((time.monotonic() - ts0) * 1000)
    return ASRResult(text=text, is_final=is_final, timestamp_ms=timestamp_ms)


def _request_id() -> str:
    import uuid

    return uuid.uuid4().hex
