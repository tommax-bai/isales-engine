"""Cloud-side transport implementations for the cloud-edge control plane.

This package hosts cloud-side concrete implementations of the ABCs in
:mod:`isales_common.transport.cloud_edge` and
:mod:`isales_common.audio.rtc`:

- :mod:`isales_engine.transport.dingrtc` — :class:`RtcSession` implementation
  backed by the DingRTC Linux C++ SDK + project-internal pybind11 binding
  (server-side). Replaces the legacy :mod:`isales_engine.transport.aliyun_rtc`
  module — see openspec change ``engine-rtc-dingrtc-migration``.

The cloud-edge gRPC server (CloudEdgeServer impl) lives alongside as
``grpc_server.py``.

Spec: arch-cloud-edge-split / design.md Decision 2 (audio topology) and
Decision 5 (control plane), service-communication § 云-边媒体面 (阿里 RTC PaaS,
DingRTC 3.x); engine-rtc-dingrtc-migration § 6 (cloud Linux switch).
"""
