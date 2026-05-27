"""Volcengine 豆包语音合成大模型 V3 SSE TTS provider.

Spec: provider-abc § Requirement: TTS Provider 异步流式合成接口;
      impl-engine-providers proposal § "真实 TTS Provider 实现".

Implements ``synthesize_stream(text, voice_id) -> AsyncIterator[bytes]``
over the Volcengine **V3 HTTP SSE 单向流式** endpoint:

    POST https://openspeech.bytedance.com/api/v3/tts/unidirectional/sse

This is the 豆包 SeedTTS 2.0 / BigTTS / 声音复刻 family. The legacy V1
OpenSpeech protocol (``/api/v1/tts`` with ``App-Key`` + ``Authorization:
Bearer; <token>`` + ``Resource-Id: tts`` + body ``app.{appid,token,cluster}
+ user + audio + request``) is a **different service** that this account
does NOT have grants for — attempting it returns 401 "grant not found in
SaaS storage" or 403 "[resource_id=] requested resource not granted". The
V3 SSE protocol is the one we want for the voice models actually granted
in the console (``zh_female_xiaohe_uranus_bigtts`` etc.; voice_type
suffix ``*_uranus_bigtts`` and ``saturn_*_tob`` are V3-only).

Two auth modes (per vendor docs § 2.1, 新版/旧版控制台):

- **New console (preferred)**: ``X-Api-Key`` + ``X-Api-Resource-Id`` —
  single API Key UUID, no separate APP ID / Access Token.
- **Legacy console (fallback)**: ``X-Api-App-Id`` + ``X-Api-Access-Key``
  + ``X-Api-Resource-Id`` — three headers; values come from the same
  console's "服务接口认证信息" panel (APP ID + Access Token).

Resource-Id selects the underlying model + billing SKU:

- ``seed-tts-2.0`` — 豆包语音合成模型 2.0 (default)
- ``seed-tts-1.0`` / ``seed-tts-1.0-concurr`` — 豆包语音合成模型 1.0
- ``seed-icl-2.0`` / ``seed-icl-1.0`` — 声音复刻

SSE response framing (vendor docs § 3.4):

    event: 352
    data: {"code":0,"message":"","data":"<base64-encoded PCM bytes>"}

    event: 351
    data: {"code":0,"message":"","data":null,"sentence":{...}}

    event: 152
    data: {"code":20000000,"message":"OK","data":null,"usage":{...}}

Event 352 = TTSResponse (audio chunk; ``data.data`` is base64 of raw PCM /
mp3 / ogg-opus depending on ``audio_params.format``). Event 351 =
TTSSentenceEnd (sentence + word-level timestamps; not consumed here). Event
152 = SessionFinish (terminal OK). Event 153 = SessionFailed (terminal
error, ``data.code`` carries the vendor error code per § 4).
"""

from __future__ import annotations

import base64
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator

import httpx
from isales_common.providers.tts import TTSProvider

from isales_engine.providers._errors import (
    ProviderInvalidRequest,
    map_http_error,
    map_transport_error,
)

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 30.0

# V3 endpoint is fixed by the protocol. The cred-supplied "endpoint" field
# (e.g. "tts_endpoint") is ignored for V3 — the path determines the
# transport (chunked / SSE / WebSocket) and is part of the API contract,
# not a deployment knob. If a future vendor swaps host (e.g. private
# region), override via ``override_url=`` kwarg.
V3_SSE_URL = "https://openspeech.bytedance.com/api/v3/tts/unidirectional/sse"

# Fallback voice when the campaign-bound voice_id is empty / "default" /
# unset — the V3 service rejects "default" with a parameter error since it
# expects a real speaker name from the granted voice list. xiaohe is in
# every account's grant list (通用场景 voice, 豆包 TTS 2.0 default).
DEFAULT_VOICE = "zh_female_xiaohe_uranus_bigtts"

# PCM output sample rate. DingRTC engine-side ``join(send_sample_rate=)``
# defaults to 16000 (RtcSession ABC default; isales-engine main.py
# rtc_factory does not override). Matching that here avoids a downstream
# resample. If the modem path (8 kHz over GSM) becomes the consumer
# instead, override via ``sample_rate=8000``.
DEFAULT_SAMPLE_RATE = 16000


class VolcengineTTSProvider(TTSProvider):
    """Streaming TTS over the Volcengine 豆包 V3 SSE API.

    Yields raw PCM bytes (16-bit LE mono at ``sample_rate``) as they arrive
    via SSE. Transient (5xx / timeout) failures are retried once before
    surfacing; that matches the impl-engine-providers design § 3 stance —
    aggressive retries only stretch user-facing silence.
    """

    def __init__(
        self,
        *,
        # New-console auth (preferred):
        api_key: str | None = None,
        # Legacy-console auth (3-tuple fallback):
        app_id: str | None = None,
        access_key: str | None = None,
        # Both modes share:
        resource_id: str = "seed-tts-2.0",
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        audio_format: str = "pcm",
        timeout_s: float = DEFAULT_TIMEOUT_S,
        override_url: str | None = None,
    ) -> None:
        if not api_key and not (app_id and access_key):
            raise ValueError(
                "VolcengineTTSProvider needs either api_key (new console) "
                "or app_id + access_key (legacy console)"
            )
        self._api_key = api_key
        self._app_id = app_id
        self._access_key = access_key
        self._resource_id = resource_id
        self._sample_rate = sample_rate
        self._audio_format = audio_format
        self._timeout_s = timeout_s
        self._url = override_url or V3_SSE_URL

    def _headers(self) -> dict[str, str]:
        common = {
            "Content-Type": "application/json",
            "X-Api-Resource-Id": self._resource_id,
            "X-Api-Request-Id": str(uuid.uuid4()),
        }
        if self._api_key:
            return {"X-Api-Key": self._api_key, **common}
        # Legacy 3-tuple
        return {
            "X-Api-App-Id": self._app_id or "",
            "X-Api-Access-Key": self._access_key or "",
            **common,
        }

    def _resolve_voice(self, voice_id: str) -> str:
        """Map empty / placeholder voice ids onto a real granted voice."""
        if not voice_id or voice_id == "default":
            return DEFAULT_VOICE
        return voice_id

    async def synthesize_stream(
        self,
        text: str,
        voice_id: str,
    ) -> AsyncIterator[bytes]:
        last_error: Exception | None = None
        # One initial attempt + 1 retry on transient errors.
        for attempt in range(2):
            try:
                async for chunk in self._stream_once(text, voice_id):
                    yield chunk
                return
            except httpx.HTTPError as exc:
                last_error = map_transport_error(exc, provider="volcengine_tts")
                if attempt == 0 and _is_retryable(last_error):
                    logger.warning(
                        "volcengine_tts_retrying after transport error: %s", exc
                    )
                    continue
                raise last_error from exc

        if last_error is not None:
            raise last_error

    async def _stream_once(
        self, text: str, voice_id: str
    ) -> AsyncIterator[bytes]:
        speaker = self._resolve_voice(voice_id)
        payload = {
            "user": {"uid": "isales-engine"},
            "req_params": {
                "text": text,
                "speaker": speaker,
                "audio_params": {
                    "format": self._audio_format,
                    "sample_rate": self._sample_rate,
                },
            },
        }
        start = time.monotonic()
        first_byte_logged = False

        async with httpx.AsyncClient(timeout=self._timeout_s) as client, client.stream(
            "POST", self._url, headers=self._headers(), json=payload
        ) as response:
            if response.status_code >= 400:
                body = await response.aread()
                response = httpx.Response(
                    status_code=response.status_code,
                    content=body,
                    headers=response.headers,
                )
                raise map_http_error(response, provider="volcengine_tts")

            # SSE framing: event lines separated by \n\n. Each frame has
            # ``event: <id>`` then ``data: <json>``. httpx aiter_lines
            # yields one line at a time, stripped of trailing newline.
            current_event: str | None = None
            async for line in response.aiter_lines():
                if not line:
                    # Blank line = frame terminator. Reset event state.
                    current_event = None
                    continue
                if line.startswith("event:"):
                    current_event = line[len("event:"):].strip()
                    continue
                if not line.startswith("data:"):
                    continue
                data_raw = line[len("data:"):].strip()
                if not data_raw:
                    continue
                try:
                    data_obj = json.loads(data_raw)
                except json.JSONDecodeError:
                    logger.warning(
                        "volcengine_tts_bad_sse_frame event=%s len=%s",
                        current_event, len(data_raw),
                    )
                    continue

                # Event 352 = TTSResponse audio chunk.
                if current_event == "352":
                    audio_b64 = data_obj.get("data")
                    if audio_b64:
                        try:
                            chunk = base64.b64decode(audio_b64)
                        except (ValueError, TypeError) as exc:
                            logger.warning(
                                "volcengine_tts_bad_b64 err=%s", exc,
                            )
                            continue
                        if not first_byte_logged:
                            latency_ms = int((time.monotonic() - start) * 1000)
                            logger.info(
                                "volcengine_tts_first_byte latency_ms=%s "
                                "text_len=%s speaker=%s",
                                latency_ms, len(text), speaker,
                            )
                            first_byte_logged = True
                        yield chunk
                    continue

                # Event 152 = SessionFinish (terminal OK).
                if current_event == "152":
                    code = data_obj.get("code")
                    if code != 20000000:
                        logger.warning(
                            "volcengine_tts_session_finish_unexpected_code "
                            "code=%s message=%s",
                            code, data_obj.get("message"),
                        )
                    return

                # Event 153 = SessionFailed.
                if current_event == "153":
                    code = data_obj.get("code")
                    message = data_obj.get("message", "")
                    raise ProviderInvalidRequest(
                        f"volcengine_tts session_failed code={code} message={message}",
                        provider="volcengine_tts",
                    )

                # Event 151 = SessionCancel; 351 = TTSSentenceEnd timestamps;
                # other events are informational and ignored here.


def _is_retryable(error: Exception) -> bool:
    """Only transient errors (5xx / timeouts) get the one retry."""

    from isales_engine.providers._errors import (
        ProviderServerError,
        ProviderTimeout,
    )

    return isinstance(error, ProviderServerError | ProviderTimeout)
