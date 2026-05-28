"""Alert helpers for AFES live execution.

All alerting is optional.  Configure one or both environment variables:

    AFES_TELEGRAM_BOT_TOKEN
    AFES_TELEGRAM_CHAT_ID
    AFES_SLACK_WEBHOOK_URL

If neither is set, alerts are printed only.  Failures never block trading
logic; they are logged to stdout so the orchestrator log captures them.
"""
from __future__ import annotations

import os
from typing import Any

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None


def send_alert(message: str, *, level: str = "INFO", extra: dict[str, Any] | None = None) -> None:
    """Send a best-effort alert to Telegram/Slack and always print it."""
    text = f"[AFES {level}] {message}"
    if extra:
        text = text + "\n" + "\n".join(f"{k}: {v}" for k, v in extra.items())
    print(text)
    if requests is None:
        return

    slack = os.environ.get("AFES_SLACK_WEBHOOK_URL", "").strip()
    if slack:
        try:
            requests.post(slack, json={"text": text}, timeout=5)
        except Exception as exc:
            print(f"[alert] Slack send failed: {exc}")

    token = os.environ.get("AFES_TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("AFES_TELEGRAM_CHAT_ID", "").strip()
    if token and chat_id:
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=5)
        except Exception as exc:
            print(f"[alert] Telegram send failed: {exc}")
