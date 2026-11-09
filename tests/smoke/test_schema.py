"""SQLite schema 冒烟测试：在临时库中应用 schema.sql，检查表与关键索引存在。"""

import sqlite3
from pathlib import Path

import pytest

SCHEMA = Path(__file__).resolve().parents[2] / "chimera_patrol" / "store" / "schema.sql"

EXPECTED_TABLES = {"cameras", "events", "event_status_log", "sop_runs", "sop_step_results"}


@pytest.fixture
def conn(tmp_path: Path):
    connection = sqlite3.connect(tmp_path / "chimera.db")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript(SCHEMA.read_text(encoding="utf-8"))
    yield connection
    connection.close()


def test_schema_creates_all_tables(conn: sqlite3.Connection) -> None:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    tables = {name for (name,) in rows}

    assert EXPECTED_TABLES <= tables


def test_running_per_camera_unique_index_exists(conn: sqlite3.Connection) -> None:
    row = conn.execute(
        "SELECT tbl_name, sql FROM sqlite_master "
        "WHERE type = 'index' AND name = 'uq_sop_runs_running_per_camera'"
    ).fetchone()

    assert row is not None
    assert row[0] == "sop_runs"
    assert "UNIQUE" in row[1].upper()
    assert "WHERE status = 'running'" in row[1]


def test_schema_is_idempotent(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA.read_text(encoding="utf-8"))

    count = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'events'"
    ).fetchone()[0]
    assert count == 1
