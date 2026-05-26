"""Cloud-side RtcSession implementation backed by DingRTC 3.x Linux C++ SDK.

Replaces the legacy ``isales_engine.transport.aliyun_rtc`` module (which
wrapped the wrong product line — ApsaraVideo Live ARTC SDK — and was
deleted as part of openspec change ``engine-rtc-dingrtc-migration``).

Public surface:

- :class:`DingRtcSession` — implements ``isales_common.audio.rtc.RtcSession``
  ABC. Construct via :meth:`DingRtcSession.production` for the real SDK
  path, or with an injected :class:`DingRtcChannelProtocol` for tests.
- :class:`InMemoryDingRtcChannel` — test double mirroring the production
  channel's external behavior without loading the pybind11 binding.
- :class:`SdkLoadError` — raised on import / instantiation if the
  ``dingrtc_pywrap`` extension is unavailable on this host. Tests should
  construct ``DingRtcSession(channel=InMemoryDingRtcChannel(...))``
  directly to avoid the load.

The C++ pybind11 binding source lives under
``isales-engine/deploy/cloud/pybind/dingrtc_pywrap/`` and is built by
``deploy/cloud/scripts/build-dingrtc-binding.sh``. Build prerequisites
(see ``deploy/cloud/STATE.md`` § "DingRTC SDK vendor"):

1. DingRTC Linux SDK 3.9.0 unpacked at
   ``/opt/isales/vendor/DingRTC_Linux_SDK_3_9_0/`` (or wherever
   ``ISALES_DINGRTC_LINUX_SDK_PATH`` points).
2. ``pip install pybind11`` in the active venv.
3. ``apt install build-essential cmake python3-dev`` (or equivalent dnf).

After build, ``import dingrtc_pywrap`` should resolve from venv
site-packages.
"""

from __future__ import annotations

from isales_engine.transport.dingrtc._in_memory import InMemoryDingRtcChannel
from isales_engine.transport.dingrtc._session import (
    DingRtcChannelProtocol,
    DingRtcSession,
    SdkLoadError,
)

__all__ = [
    "DingRtcChannelProtocol",
    "DingRtcSession",
    "InMemoryDingRtcChannel",
    "SdkLoadError",
]
