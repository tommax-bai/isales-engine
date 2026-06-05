"""ECS-side PCM loopback listener (pybind §9.5 对端).

Spec: joint-mvp-gate-13301035545 § 4；windows-artc-pybind11 §9.5；
      engine-rtc-dingrtc-migration § 5.4.

ECS 上 join 同一 DingRTC channel 作为对端，counter inbound PCM frames +
反推 silence。运行在 DingRTC Linux C++ SDK + 项目内 pybind11 binding
(vendor SDK 路径见 ``ISALES_DINGRTC_LINUX_SDK_PATH`` env 或默认
``/opt/isales/vendor/DingRTC_Linux_SDK_3_9_0/``).

启动顺序: **ECS 端先跑本脚本** → Windows 端再跑
``pybind_pcm_loopback_smoke.py``。本脚本默认 listen `--duration` 秒后退出
+ 打印 stats JSON。

Usage::

    sudo -u isales -H -E env $(cat /etc/isales/env/engine.env | \\
        grep -v '^#' | grep -v '^$' | xargs) \\
        /opt/isales/current/venv/bin/python \\
        /opt/isales/current/isales-engine/scripts/ecs_pcm_loopback_listen.py \\
        --channel smoke-channel-9-5 --duration 30
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

from isales_engine.transport.dingrtc import DingRtcSession, SdkLoadError
from isales_engine.transport.rtc_token import RtcTokenIssuer


async def listen(
    *,
    channel: str,
    user_id: str,
    duration: float,
    silence_chunk_ms: int = 100,
) -> dict:
    app_id = os.environ.get("ISALES_RTC_APP_ID", "")
    app_key = os.environ.get("ISALES_RTC_APP_KEY", "")
    if not app_id or not app_key:
        sys.exit("error: ISALES_RTC_APP_ID + ISALES_RTC_APP_KEY env must be set")

    issuer = RtcTokenIssuer(app_id=app_id, app_key=app_key)
    creds = issuer.sign(channel=channel, user_id=user_id, ttl_seconds=int(duration) + 600)
    print(
        f"[1/4] minted token; app_id={app_id} channel={channel} user_id={user_id}",
        file=sys.stderr,
    )

    try:
        session = DingRtcSession.production(app_id=app_id)
    except SdkLoadError as e:
        sys.exit(
            f"error: DingRTC Linux SDK / binding not loadable: {e}\n"
            "提示: 1) 跑 deploy/cloud/scripts/build-dingrtc-binding.sh 编译 "
            "dingrtc_pywrap; 2) 检查 ISALES_DINGRTC_LINUX_SDK_PATH env 或默认 "
            "/opt/isales/vendor/DingRTC_Linux_SDK_3_9_0/ 含 api/ + lib/x86_64/; "
            "3) 检查 LD_LIBRARY_PATH 含 lib/x86_64/"
        )

    print(f"[2/4] join channel={channel!r} as uid={user_id!r} ...", file=sys.stderr)
    await session.join(channel, creds.token, user_id, send_sample_rate=8000, send_channels=1)

    # silence push task: 推 silence PCM 让对端听到东西，counter 通路是否反向通
    sample_rate = 8000
    chunk_samples = int(sample_rate * silence_chunk_ms / 1000)
    silence_pcm = b"\x00\x00" * chunk_samples  # 16-bit mono silence
    push_started = time.monotonic()
    push_count = 0

    async def push_silence_loop() -> None:
        nonlocal push_count
        ts = 0
        while True:
            try:
                await session.push_audio(silence_pcm, timestamp_ms=ts)
            except Exception as e:  # noqa: BLE001
                print(f"      push error: {e}", file=sys.stderr)
                return
            push_count += 1
            ts += silence_chunk_ms
            await asyncio.sleep(silence_chunk_ms / 1000)

    push_task = asyncio.create_task(push_silence_loop())

    # listen task: 收 inbound frames
    inbound_frames = 0
    inbound_bytes = 0
    first_frame_at: float | None = None

    async def consume_inbound() -> None:
        nonlocal inbound_frames, inbound_bytes, first_frame_at
        async for frame in session.audio_frames():
            inbound_frames += 1
            inbound_bytes += len(frame.pcm)
            if first_frame_at is None:
                first_frame_at = time.monotonic()
                print(
                    f"      first inbound frame at t+"
                    f"{first_frame_at - push_started:.2f}s "
                    f"({len(frame.pcm)} bytes)",
                    file=sys.stderr,
                )

    consume_task = asyncio.create_task(consume_inbound())

    print(f"[3/4] listening {duration}s ...", file=sys.stderr)
    try:
        await asyncio.sleep(duration)
    finally:
        push_task.cancel()
        consume_task.cancel()
        for t in (push_task, consume_task):
            try:
                await t
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

    print("[4/4] leaving ...", file=sys.stderr)
    await session.leave()

    return {
        "ok": True,
        "channel": channel,
        "duration_s": duration,
        "push_count": push_count,
        "inbound_frames": inbound_frames,
        "inbound_bytes": inbound_bytes,
        "first_inbound_frame_at_s": (
            first_frame_at - push_started if first_frame_at else None
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ECS PCM loopback listener (pybind §9.5)")
    parser.add_argument("--channel", default="smoke-channel-9-5")
    parser.add_argument("--user-id", dest="user_id", default="engine-smoke")
    parser.add_argument("--duration", type=float, default=30.0)
    args = parser.parse_args(argv)

    stats = asyncio.run(
        listen(channel=args.channel, user_id=args.user_id, duration=args.duration)
    )
    print(json.dumps(stats))
    return 0 if stats.get("inbound_frames", 0) > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
