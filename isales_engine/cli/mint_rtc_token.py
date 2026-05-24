"""isales-engine-mint-rtc-token — offline RTC token mint for smoke scripts.

Spec: joint-mvp-gate-13301035545 § "RTC client token 签发 — smoke 走 ECS CLI"。

生产 dial 流程的 RTC token 由 engine 在 push DialCommand 之前用既有
:class:`isales_engine.transport.rtc_token.RtcTokenIssuer.sign_for_call`
签出，随 ``DialCommand.rtc_token`` 字段下发给 edge。本 CLI 仅服务**离线
smoke 场景** (pybind §9.4 真 RTC join / §9.5 PCM loopback)：smoke 脚本
通过 ssh 在 ECS 跑本 CLI 拿 token，AppKey 不离 ECS。

不在 cloud-edge proto 加 unary RPC 的理由见 design § D1。

使用:

    ssh root@121.89.85.150 'sudo -u isales -H -E env $(cat \\
        /etc/isales/env/engine.env | grep -v ^# | xargs) \\
        /opt/isales/current/venv/bin/isales-engine-mint-rtc-token \\
        --channel smoke-channel-9-4 --user-id edge-smoke --ttl 600'

输出 JSON 单行到 stdout (其他日志走 stderr)。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable

from isales_engine.transport.rtc_token import RtcTokenIssuer

ENV_APP_ID = "ISALES_RTC_APP_ID"
ENV_APP_KEY = "ISALES_RTC_APP_KEY"


def _read_env() -> tuple[str, str]:
    """Read ISALES_RTC_APP_ID + ISALES_RTC_APP_KEY；缺失 sys.exit。"""
    app_id = os.environ.get(ENV_APP_ID, "")
    app_key = os.environ.get(ENV_APP_KEY, "")
    if not app_id:
        sys.exit(
            f"error: {ENV_APP_ID} env var is empty; expected from "
            "/etc/isales/env/engine.env (cloud) — RTC AppId 是 cloud-only "
            "secret, smoke 必须在 ECS 上跑"
        )
    if not app_key:
        sys.exit(
            f"error: {ENV_APP_KEY} env var is empty; same caveat as "
            f"{ENV_APP_ID} — 不要试图从 dev box 跑本 CLI"
        )
    return app_id, app_key


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="isales-engine-mint-rtc-token",
        description=(
            "Mint a single Aliyun RTC channel token using "
            "ISALES_RTC_APP_ID + ISALES_RTC_APP_KEY env. Intended for "
            "offline smoke scripts (pybind §9.4 / §9.5); production dial "
            "path uses DialCommand.rtc_token (engine signs inline)."
        ),
    )
    parser.add_argument("--channel", required=True, help="RTC 房间名 (smoke 自取)")
    parser.add_argument(
        "--user-id",
        required=True,
        dest="user_id",
        help="RTC 参与者 uid (e.g. edge-smoke / engine-smoke)",
    )
    parser.add_argument(
        "--ttl",
        type=int,
        default=600,
        help="token TTL 秒数 (默认 600，够一通 smoke)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=True,
        help="(默认开) 输出单行 JSON 到 stdout",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.ttl <= 0:
        sys.exit("error: --ttl must be positive")

    app_id, app_key = _read_env()
    issuer = RtcTokenIssuer(app_id=app_id, app_key=app_key)
    creds = issuer.sign(channel=args.channel, user_id=args.user_id, ttl_seconds=args.ttl)

    payload = {
        "app_id": creds.app_id,
        "channel": creds.channel,
        "user_id": creds.user_id,
        "nonce": creds.nonce,
        "token": creds.token,
        "expires_at": creds.expires_at,
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
