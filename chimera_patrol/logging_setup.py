"""Rotating file and console logging, observability helpers, and a redaction guard.

The setup function takes plain arguments that mirror the ``logging`` section of
config.yaml (design defaults: level INFO, dir ./data/logs, max_bytes 10485760,
backup_count 5). Callers pass the values from the loaded config.

Helpers log only whitelisted summary fields. The redaction guard additionally
removes API key values, ``data:...;base64,`` URLs, and long base64 runs from every
record, so image payloads and keys are not written even if passed in by mistake.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import re
import sys
from pathlib import Path
from typing import Any

LOG_FILE_NAME = "chimera_patrol.log"
SUMMARY_MAX_CHARS = 500
REDACTED_IMAGE = "[REDACTED_IMAGE]"
REDACTED_SECRET = "[REDACTED_SECRET]"
VALID_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
_MIN_SECRET_LENGTH = 8
_MANAGED_MARK = "_chimera_managed_handler"

_DATA_URL_RE = re.compile(r"data:[A-Za-z0-9.+/-]+;base64,[A-Za-z0-9+/=]*")
_BASE64_RUN_RE = re.compile(r"[A-Za-z0-9+/]{120,}={0,2}")
_SK_KEY_RE = re.compile(r"sk-[A-Za-z0-9_\-]{12,}")
_BEARER_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{12,}")
_API_KEY_ASSIGN_RE = re.compile(r"(?i)\b(api[_-]?key)(\s*[=:]\s*)\S+")

_registered_secrets: set[str] = set()

_events_log = logging.getLogger("chimera_patrol.events")
_vlm_log = logging.getLogger("chimera_patrol.vlm")
_status_log = logging.getLogger("chimera_patrol.status")


def register_secret(value: str | None) -> None:
    """Add an exact secret value (for example an API key read from env) to the redaction list."""
    if value and len(value) >= _MIN_SECRET_LENGTH:
        _registered_secrets.add(value)


def redact(text: str) -> str:
    """Remove secrets and image payloads from text. Returns the redacted text."""
    for secret in sorted(_registered_secrets, key=len, reverse=True):
        text = text.replace(secret, REDACTED_SECRET)
    text = _DATA_URL_RE.sub(REDACTED_IMAGE, text)
    text = _BASE64_RUN_RE.sub(REDACTED_IMAGE, text)
    text = _SK_KEY_RE.sub(REDACTED_SECRET, text)
    text = _BEARER_RE.sub(f"Bearer {REDACTED_SECRET}", text)
    text = _API_KEY_ASSIGN_RE.sub(lambda m: m.group(1) + m.group(2) + REDACTED_SECRET, text)
    return text


def truncate(text: str, limit: int = SUMMARY_MAX_CHARS) -> str:
    """Cap text at ``limit`` characters, marking the cut with an ellipsis."""
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


class RedactingFormatter(logging.Formatter):
    """Formatter that redacts the final line, including any exception traceback."""

    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


def setup_logging(
    *,
    level: str = "INFO",
    log_dir: str | Path = "./data/logs",
    max_bytes: int = 10485760,
    backup_count: int = 5,
    console: bool = True,
) -> Path:
    """Configure the root logger with a rotating file handler and optional console handler.

    Calling this again replaces the handlers installed by an earlier call.
    Returns the path of the active log file.
    """
    level_name = str(level).upper()
    if level_name not in VALID_LEVELS:
        raise ValueError(f"logging.level must be one of {', '.join(VALID_LEVELS)}, got {level!r}")
    if max_bytes <= 0:
        raise ValueError(f"logging.max_bytes must be > 0, got {max_bytes}")
    if backup_count < 0:
        raise ValueError(f"logging.backup_count must be >= 0, got {backup_count}")

    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / LOG_FILE_NAME

    root = logging.getLogger()
    root.setLevel(level_name)
    for handler in list(root.handlers):
        if getattr(handler, _MANAGED_MARK, False):
            root.removeHandler(handler)
            handler.close()

    formatter = RedactingFormatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    setattr(file_handler, _MANAGED_MARK, True)
    root.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(formatter)
        setattr(console_handler, _MANAGED_MARK, True)
        root.addHandler(console_handler)

    return log_path


def _fmt(value: Any) -> str:
    """Render one field for a key=value log line. Strings are redacted, capped, and quoted."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, int):
        return str(value)
    text = truncate(redact(str(value)))
    return json.dumps(text, ensure_ascii=False)


def log_event_generated(
    *,
    event_id: str,
    camera_id: str,
    event_type: str,
    start_ms: int,
    end_ms: int,
) -> None:
    """Log that a candidate event was created. Logs identifiers and times only."""
    _events_log.info(
        "event generated id=%s camera=%s type=%s start_ms=%s end_ms=%s",
        _fmt(event_id),
        _fmt(camera_id),
        _fmt(event_type),
        _fmt(start_ms),
        _fmt(end_ms),
    )


def log_vlm_request(*, event_id: str, image_bytes: int | bytes | bytearray) -> None:
    """Log a VLM request summary. Image content is never logged; only its byte count is."""
    size = len(image_bytes) if isinstance(image_bytes, (bytes, bytearray)) else int(image_bytes)
    _vlm_log.info("vlm request id=%s image_bytes=%s", _fmt(event_id), _fmt(size))


def log_vlm_response(
    *,
    event_id: str,
    verdict: str | None,
    confidence: float | None,
    duration_ms: int,
    summary: str | None = None,
) -> None:
    """Log a VLM response summary. The summary is redacted and capped at 500 characters."""
    summary_text = truncate(redact(summary), SUMMARY_MAX_CHARS) if summary else None
    _vlm_log.info(
        "vlm response id=%s verdict=%s confidence=%s duration_ms=%s summary=%s",
        _fmt(event_id),
        _fmt(verdict),
        _fmt(confidence),
        _fmt(duration_ms),
        _fmt(summary_text),
    )


def log_status_change(
    *,
    event_id: str,
    from_status: str | None,
    to_status: str,
    actor: str,
    reason: str | None = None,
) -> None:
    """Log an event status transition with its actor and reason."""
    _status_log.info(
        "status change id=%s from=%s to=%s actor=%s reason=%s",
        _fmt(event_id),
        _fmt(from_status),
        _fmt(to_status),
        _fmt(actor),
        _fmt(reason),
    )
