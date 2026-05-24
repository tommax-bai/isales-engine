"""Tests for isales-engine-mint-rtc-token CLI.

Spec: joint-mvp-gate-13301035545 § 2.2。
"""

from __future__ import annotations

import json
import time

import pytest

from isales_engine.cli.mint_rtc_token import main


class TestEnvMissing:
    def test_missing_app_id_exits(self, monkeypatch, capsys):
        monkeypatch.delenv("ISALES_RTC_APP_ID", raising=False)
        monkeypatch.setenv("ISALES_RTC_APP_KEY", "key-x")
        with pytest.raises(SystemExit) as exc_info:
            main(["--channel", "smoke", "--user-id", "u1"])
        # message goes to stderr (sys.exit() with str writes to stderr)
        captured = capsys.readouterr()
        assert "ISALES_RTC_APP_ID" in str(exc_info.value) or "ISALES_RTC_APP_ID" in captured.err

    def test_missing_app_key_exits(self, monkeypatch):
        monkeypatch.setenv("ISALES_RTC_APP_ID", "app-x")
        monkeypatch.delenv("ISALES_RTC_APP_KEY", raising=False)
        with pytest.raises(SystemExit):
            main(["--channel", "smoke", "--user-id", "u1"])


class TestSignHappyPath:
    def test_returns_json_with_expected_keys(self, monkeypatch, capsys):
        monkeypatch.setenv("ISALES_RTC_APP_ID", "app-test")
        monkeypatch.setenv("ISALES_RTC_APP_KEY", "key-test")
        before = int(time.time())
        rc = main(["--channel", "smoke-channel", "--user-id", "edge-1", "--ttl", "60"])
        assert rc == 0
        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert payload["app_id"] == "app-test"
        assert payload["channel"] == "smoke-channel"
        assert payload["user_id"] == "edge-1"
        assert payload["nonce"] == ""  # v3 默认 nonce 空 (Aliyun doc 推荐)
        assert len(payload["token"]) == 64  # sha256 hex digest length
        # token 不包含明文 app_key
        assert "key-test" not in payload["token"]
        # expires_at ≈ now + ttl，宽容 5s 时钟漂移
        assert before + 60 <= payload["expires_at"] <= before + 60 + 5

    def test_default_ttl_is_600(self, monkeypatch, capsys):
        monkeypatch.setenv("ISALES_RTC_APP_ID", "app-test")
        monkeypatch.setenv("ISALES_RTC_APP_KEY", "key-test")
        before = int(time.time())
        main(["--channel", "smoke", "--user-id", "u1"])
        payload = json.loads(capsys.readouterr().out)
        assert before + 600 <= payload["expires_at"] <= before + 600 + 5


class TestArgValidation:
    def test_negative_ttl_rejected(self, monkeypatch):
        monkeypatch.setenv("ISALES_RTC_APP_ID", "app-test")
        monkeypatch.setenv("ISALES_RTC_APP_KEY", "key-test")
        with pytest.raises(SystemExit):
            main(["--channel", "smoke", "--user-id", "u1", "--ttl", "-5"])

    def test_missing_channel_rejected(self, monkeypatch):
        monkeypatch.setenv("ISALES_RTC_APP_ID", "app-test")
        monkeypatch.setenv("ISALES_RTC_APP_KEY", "key-test")
        with pytest.raises(SystemExit):
            main(["--user-id", "u1"])

    def test_missing_user_id_rejected(self, monkeypatch):
        monkeypatch.setenv("ISALES_RTC_APP_ID", "app-test")
        monkeypatch.setenv("ISALES_RTC_APP_KEY", "key-test")
        with pytest.raises(SystemExit):
            main(["--channel", "smoke"])
