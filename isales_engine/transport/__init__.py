"""Cloud-side transport implementations for the cloud-edge control plane.

This package hosts cloud-side concrete implementations of the ABCs in
:mod:`isales_common.transport.cloud_edge` and
:mod:`isales_common.audio.rtc`:

- :mod:`isales_engine.transport.aliyun_rtc` — :class:`RtcSession` implementation
  backed by the Aliyun ARTC SDK for Linux Python (server-side).

The cloud-edge gRPC server (CloudEdgeServer impl) lives alongside as
``grpc_server.py`` once landed.

Spec: arch-cloud-edge-split / design.md Decision 2 (audio topology) and
Decision 5 (control plane), service-communication § 云-边媒体面 (阿里 RTC PaaS).
"""
