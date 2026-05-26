"""DingRTC PoC smoke — isolated SDK / binding / token verification.

Spec: engine-rtc-dingrtc-migration § 4.1-4.4.

Step 1 of the 3-stage PoC (per design.md D5):

    1. SDK + binding + token algo verification ← THIS SCRIPT (demo appid)
    2. Real credentials verification (real appid, this script with --real)
    3. End-to-end multi-peer verification (mac/Windows + cloud join同房间)

Uses DingRTC's documented demo appid ``a4zfr1hn`` (no real account needed)
to isolate whether failure is in:

    * SDK + binding: if join times out / OnJoinChannelResult returns
      non-zero with demo appid, the binding or SDK install is wrong.
    * Token algorithm: if the binding works but DingRTC roomserver
      rejects token, rtc_token.py output is malformed.
    * Real credentials: only verifiable after demo path passes.

The demo AppKey ships in DingRTC's vendor sample packages; capture it
from your local copy of the sample (NOT committed to git).

Usage:

    # Demo appid path (isolation step 1):
    ISALES_DINGRTC_DEMO_APPKEY=<from-vendor-sample> \\
        python scripts/dingrtc_demo_smoke.py

    # Real appid path (isolation step 2 — same code, different creds):
    ISALES_RTC_APP_ID=o6dpsan9... \\
        ISALES_RTC_APP_KEY=<real-key> \\
        python scripts/dingrtc_demo_smoke.py --real --channel rtc-smoke-$(date +%s)

Expected output on success:

    [1/4] minted token...
    [2/4] DingRtcSession.production()...
    [3/4] join(channel=...)...
        OnJoinChannelResult: rc=0
    [4/4] PASS — joined channel for 5s, then left cleanly.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

from isales_engine.transport.dingrtc import DingRtcSession, SdkLoadError
from isales_engine.transport.rtc_token import RtcTokenIssuer

DEMO_APP_ID = "a4zfr1hn"  # DingRTC public demo appid (vendor download page)


async def run(args: argparse.Namespace) -> int:
    if args.real:
        app_id = os.environ.get("ISALES_RTC_APP_ID", "")
        app_key = os.environ.get("ISALES_RTC_APP_KEY", "")
        if not app_id or not app_key:
            print("error: --real requires ISALES_RTC_APP_ID + ISALES_RTC_APP_KEY env",
                  file=sys.stderr)
            return 2
    else:
        app_id = DEMO_APP_ID
        app_key = os.environ.get("ISALES_DINGRTC_DEMO_APPKEY", "")
        if not app_key:
            print(
                "error: ISALES_DINGRTC_DEMO_APPKEY env must be set.\n"
                "Locate the demo AppKey in your local DingRTC vendor sample\n"
                "package (do NOT commit to git).",
                file=sys.stderr,
            )
            return 2

    channel = args.channel or f"dingrtc-poc-{int(time.time())}"
    user_id = args.user_id

    issuer = RtcTokenIssuer(app_id=app_id, app_key=app_key)
    creds = issuer.sign(channel=channel, user_id=user_id, ttl_seconds=600)
    print(f"[1/4] minted token: app_id={app_id} channel={channel} user_id={user_id}",
          file=sys.stderr)

    try:
        session = DingRtcSession.production(app_id=app_id)
    except SdkLoadError as e:
        print(f"error: dingrtc_pywrap binding not loadable:\n  {e}", file=sys.stderr)
        return 3

    print(f"[2/4] DingRtcSession.production(app_id={app_id!r})", file=sys.stderr)

    print(f"[3/4] join(channel={channel!r}, uid={user_id!r}) ...", file=sys.stderr)
    t0 = time.monotonic()
    try:
        await session.join(channel, creds.token, user_id, send_sample_rate=16000)
    except Exception as e:  # noqa: BLE001
        print(f"  ❌ FAIL join: {e}", file=sys.stderr)
        return 4
    elapsed = (time.monotonic() - t0) * 1000
    print(f"  ✅ joined in {elapsed:.0f} ms", file=sys.stderr)

    # Hold the channel briefly so OnRemoteUserOnLineNotify can fire if
    # another peer is in the same channel.
    hold_s = args.hold
    print(f"[4/4] hold channel for {hold_s}s then leave ...", file=sys.stderr)
    await asyncio.sleep(hold_s)
    await session.leave()
    print("  ✅ left cleanly. PASS.", file=sys.stderr)
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--real", action="store_true",
                   help="use real AppId/AppKey from ISALES_RTC_APP_ID/KEY env "
                        "(default: demo appid a4zfr1hn + ISALES_DINGRTC_DEMO_APPKEY)")
    p.add_argument("--channel", default=None,
                   help="RTC channel id (default: dingrtc-poc-<timestamp>)")
    p.add_argument("--user-id", default="smoke-uid-01")
    p.add_argument("--hold", type=float, default=5.0,
                   help="seconds to hold the channel before leaving")
    args = p.parse_args()
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
