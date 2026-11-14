"""Error alerting for automated pipeline failures.

Supports:
1. Telegram Bot webhook notifications (completely free).
2. SMTP Email alerts (via standard library smtplib).
"""

from __future__ import annotations

import json
import os
import smtplib
from email.message import EmailMessage
from urllib.request import Request, urlopen

from newscheck.config import AlertingConfig, Settings
from newscheck.log import get_logger

logger = get_logger(__name__)


def send_pipeline_failure_alert(settings: Settings, error_message: str, run_id: int | None = None) -> bool:
    """Send an alert if pipeline alerting is enabled."""
    cfg: AlertingConfig = settings.alerting
    if not cfg.enabled:
        return False

    title = f"NewsCheck Alert: Pipeline Failure (Run #{run_id or 'unknown'})"
    body = (
        f"A pipeline execution error occurred:\n\n"
        f"Error: {error_message}\n"
        f"Site status: Preserved last-good published build.\n"
    )

    if cfg.method == "telegram":
        return _send_telegram(title, body)
    if cfg.method == "email":
        return _send_email(title, body)

    return False


def _send_telegram(title: str, body: str) -> bool:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        logger.warning("telegram_alerting_missing_credentials")
        return False

    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = json.dumps(
            {
                "chat_id": chat_id,
                "text": f"🚨 *{title}*\n\n{body}",
                "parse_mode": "Markdown",
            }
        ).encode("utf-8")
        req = Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=10) as resp:
            return bool(resp.status == 200)
    except Exception:
        logger.exception("telegram_alert_send_failed")
        return False


def _send_email(subject: str, content: str) -> bool:
    host = os.getenv("SMTP_HOST", "")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER", "")
    password = os.getenv("SMTP_PASSWORD", "")
    to_email = os.getenv("ALERT_EMAIL_TO", "")

    if not host or not to_email:
        logger.warning("email_alerting_missing_configuration")
        return False

    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = user or "alerts@newscheck.local"
        msg["To"] = to_email
        msg.set_content(content)

        with smtplib.SMTP(host, port, timeout=15) as server:
            server.starttls()
            if user and password:
                server.login(user, password)
            server.send_message(msg)
        return True
    except Exception:
        logger.exception("email_alert_send_failed")
        return False
