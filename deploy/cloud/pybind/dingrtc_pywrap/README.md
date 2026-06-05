# dingrtc_pywrap

Project-internal pybind11 binding for the DingRTC Linux C++ SDK 3.x.

Mirrors the structure of isales-telephony's `windows-artc-pybind11`
binding (`deploy/edge/windows/pybind/aliyun_artc_pywrap/`), but targets
**DingRTC** (RTC PaaS new-generation, OSS bucket
`dingrtc.oss-cn-zhangjiakou.aliyuncs.com`) — NOT the legacy ApsaraVideo
Live ARTC SDK (`alivc-demo-cms.alicdn.com`), which is from a different
product line and does not interoperate with DingRTC channel rooms.

See `openspec/changes/engine-rtc-dingrtc-migration/` (in the iSales
meta-repo) for the full design + rationale.

## Build

The binding is built out-of-source by
`deploy/cloud/scripts/build-dingrtc-binding.sh`:

```bash
# Prerequisites (one-time per machine):
sudo apt install build-essential cmake python3-dev   # Debian/Ubuntu
# or
sudo dnf install gcc gcc-c++ cmake python3-devel     # AL3 / RHEL

# In an active venv:
pip install pybind11

# Place the DingRTC SDK extraction at the expected path:
#   /opt/isales/vendor/DingRTC_Linux_SDK_3_9_0/{api,lib/x86_64}/
# Or set ISALES_DINGRTC_LINUX_SDK_PATH to override.

# Then build:
deploy/cloud/scripts/build-dingrtc-binding.sh
```

The script:
1. Locates the active venv Python (must have `pybind11` installed).
2. Invokes CMake with the venv Python and DingRTC vendor path.
3. Builds the `dingrtc_pywrap.so` extension module (Release config).
4. Copies the resulting `.so` into the venv's `site-packages/`.

After build, `python -c "import dingrtc_pywrap"` should succeed.

## API surface

`dingrtc_pywrap` exposes a minimal C++→Python facade that the higher-level
`isales_engine.transport.dingrtc.DingRtcSession` wraps to satisfy
`isales-common`'s `RtcSession` ABC.

- `EngineHandle` — lifecycle (`create` / `destroy`), channel ops
  (`join_channel` / `leave_channel`), pub/sub flags
  (`publish_local_audio_stream` / `subscribe_all_remote_audio_streams`),
  external audio source (`set_external_audio_source` /
  `push_external_audio`), observer registration
  (`register_audio_observer` / `enable_audio_frame_observer`).
- `EngineListener` — engine event trampoline with Python callable hooks
  (join / leave / bye / error / connection-status).
- `AudioObserver` + `FrameRingBuffer` — SDK-thread-safe inbound PCM
  collection. Python drainer pulls frames via `FrameRingBuffer.drain`.
- `PcmFrame` — one inbound audio frame (pcm bytes + sample_rate +
  channels + bytes_per_sample + num_samples + remote_uid + timestamp).
- `set_log_dir_path` — static helper for DingRTC SDK log routing.
- `DingRtcError(code, message)` — exception raised on any SDK call
  returning non-zero. `error_code` accessible via `.args[0]` in
  Python-side exception.

## Vendor SDK reference

Headers + struct shapes + selectors are captured in
`openspec/changes/engine-rtc-dingrtc-migration/notes/sdk_api_ground_truth.md`.
That file is the API reference — vendor docs may lag actual 3.9.0 SDK.

## Layout deviation from spec/design.md

The spec/design originally proposed putting the C++ binding source inside
the Python package (`isales-engine/isales_engine/transport/dingrtc/`).
For consistency with the existing windows-artc-pybind11 layout
(binding source under `deploy/`, Python wrapper inside the package) and
to avoid switching the isales-engine project from hatchling backend to
scikit-build, we placed:

- C++ source: `isales-engine/deploy/cloud/pybind/dingrtc_pywrap/` (this dir)
- Python wrapper: `isales-engine/isales_engine/transport/dingrtc/`

`dingrtc_pywrap.so` is installed into venv site-packages by the build
script, so the wrapper's `import dingrtc_pywrap` works the same way the
Windows binding does for `aliyun_artc_pywrap`.

This deviation is recorded in `tasks.md` § 2.1 of the openspec change.
