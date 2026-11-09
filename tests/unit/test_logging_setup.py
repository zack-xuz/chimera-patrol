"""Unit tests for chimera_patrol.logging_setup (task 1.4)."""

from __future__ import annotations

import base64
import logging
import logging.handlers
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from chimera_patrol import logging_setup as ls

DUMMY_KEY = "sk-DUMMYKEY0123456789abcdef"


@pytest.fixture(autouse=True)
def _reset_logging() -> Iterator[None]:
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, ls._MANAGED_MARK, False):
            root.removeHandler(handler)
            handler.close()
    ls._registered_secrets.clear()


def _read(path: Path) -> str:
    for handler in logging.getLogger().handlers:
        handler.flush()
    return path.read_text(encoding="utf-8")


def test_setup_creates_rotating_file_and_console_handlers(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_path = ls.setup_logging(level="DEBUG", log_dir=log_dir, max_bytes=1024, backup_count=2)

    assert log_path == log_dir / "chimera_patrol.log"
    assert log_dir.is_dir()
    managed = [h for h in logging.getLogger().handlers if getattr(h, ls._MANAGED_MARK, False)]
    rotating = [h for h in managed if isinstance(h, logging.handlers.RotatingFileHandler)]
    assert len(rotating) == 1
    assert rotating[0].maxBytes == 1024
    assert rotating[0].backupCount == 2
    assert len(managed) == 2


def test_setup_is_idempotent(tmp_path: Path) -> None:
    ls.setup_logging(log_dir=tmp_path)
    ls.setup_logging(log_dir=tmp_path)
    managed = [h for h in logging.getLogger().handlers if getattr(h, ls._MANAGED_MARK, False)]
    assert len(managed) == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"level": "TRACE"},
        {"max_bytes": 0},
        {"backup_count": -1},
    ],
)
def test_setup_rejects_invalid_arguments(tmp_path: Path, kwargs: dict) -> None:
    with pytest.raises(ValueError):
        ls.setup_logging(log_dir=tmp_path, **kwargs)


def test_example_of_each_kind_is_written_without_key_or_image_data(tmp_path: Path) -> None:
    ls.register_secret(DUMMY_KEY)
    log_path = ls.setup_logging(level="INFO", log_dir=tmp_path, console=False)

    image = os.urandom(600)
    b64 = base64.b64encode(image).decode("ascii")
    data_url = f"data:image/jpeg;base64,{b64}"

    ls.log_event_generated(
        event_id="evt_0123456789ab",
        camera_id="cam_a",
        event_type="intrusion",
        start_ms=1790000000000,
        end_ms=1790000001000,
    )
    ls.log_vlm_request(event_id="evt_0123456789ab", image_bytes=image)
    ls.log_vlm_response(
        event_id="evt_0123456789ab",
        verdict="false_positive",
        confidence=0.91,
        duration_ms=1234,
        summary="描述 " + data_url + " key=" + DUMMY_KEY + " " + "x" * 800,
    )
    ls.log_status_change(
        event_id="evt_0123456789ab",
        from_status="pending_review",
        to_status="dismissed",
        actor="张三",
        reason="误报",
    )
    # Mistakes: raw key and raw base64 passed straight into a message.
    logging.getLogger("chimera_patrol.test").warning("leak %s and %s", DUMMY_KEY, b64)

    content = _read(log_path)

    assert DUMMY_KEY not in content
    assert b64[:60] not in content
    assert "base64," not in content
    assert "event generated" in content
    assert "vlm request" in content and "image_bytes=600" in content
    assert "vlm response" in content and "duration_ms=1234" in content
    assert "status change" in content and "to=\"dismissed\"" in content
    assert "[REDACTED_SECRET]" in content
    assert "[REDACTED_IMAGE]" in content


def test_response_summary_is_capped_at_500_characters(tmp_path: Path) -> None:
    log_path = ls.setup_logging(log_dir=tmp_path, console=False)
    ls.log_vlm_response(
        event_id="evt_x",
        verdict="uncertain",
        confidence=0.5,
        duration_ms=10,
        summary="字" * 2000,
    )
    line = [ln for ln in _read(log_path).splitlines() if "vlm response" in ln][0]
    summary_field = line.split("summary=", 1)[1]
    # The quoted, capped summary is at most 500 characters plus quotes.
    assert len(summary_field) <= 502
    assert "..." in summary_field


def test_image_bytes_argument_logs_only_length(tmp_path: Path) -> None:
    log_path = ls.setup_logging(log_dir=tmp_path, console=False)
    payload = b"\xff\xd8\xff" + b"A" * 4096
    ls.log_vlm_request(event_id="evt_y", image_bytes=payload)
    content = _read(log_path)
    assert "image_bytes=4099" in content
    assert "\\xff" not in content and "AAAA" not in content


def test_registered_secret_is_redacted_from_any_field(tmp_path: Path) -> None:
    ls.register_secret(DUMMY_KEY)
    log_path = ls.setup_logging(log_dir=tmp_path, console=False)
    ls.log_status_change(
        event_id="evt_z",
        from_status=None,
        to_status="needs_human",
        actor=DUMMY_KEY,
        reason=None,
    )
    content = _read(log_path)
    assert DUMMY_KEY not in content
    assert "from=null" in content
