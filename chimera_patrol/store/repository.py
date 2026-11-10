"""SQLite 事件存储：EventRepository。

所有方法都是阻塞调用。异步层通过 ``asyncio.to_thread`` 调用它们。
连接在多线程间共享，因此每次操作都持有同一把 ``threading.Lock``。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Sequence

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

EVENT_TYPES = ("intrusion", "loitering", "smoke_fire", "no_helmet", "sop_violation")
EVENT_STATUSES = ("pending_review", "confirmed", "dismissed", "needs_human", "review_failed")
CLIP_STATUSES = ("pending", "ready", "failed")

LIST_LIMIT_MIN = 1
LIST_LIMIT_MAX = 200


@dataclass
class NewEvent:
    """写入 events 表的一条新事件。

    关键帧路径为相对 ``evidence.base_dir`` 的路径。
    """

    id: str
    camera_id: str
    event_type: str
    detection_source: str
    start_ms: int
    end_ms: int
    keyframe_annotated_path: str
    keyframe_raw_path: str
    created_ms: int
    cv_confidence: float | None = None
    track_id: int | None = None
    roi_id: str | None = None
    roi_type: str | None = None
    cv_detail: dict[str, Any] = field(default_factory=dict)
    trajectory: list[list[float]] = field(default_factory=list)
    sop_run_id: str | None = None


def _normalize_rel_path(value: str) -> str:
    """校验证据路径为相对路径，并统一为正斜杠。"""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("evidence path must be a non-empty string")
    normalized = value.replace("\\", "/")
    if (
        PurePosixPath(normalized).is_absolute()
        or PureWindowsPath(value).drive
        or ".." in normalized.split("/")
    ):
        raise ValueError(f"evidence path must be relative to base_dir: {value!r}")
    return normalized


def _parse_multi(value: str | None, allowed: Iterable[str] | None, name: str) -> list[str] | None:
    """解析逗号分隔的多值过滤条件。空值表示不过滤。"""
    if value is None or not value.strip():
        return None
    items = [part.strip() for part in value.split(",") if part.strip()]
    if not items:
        return None
    if allowed is not None:
        allowed_set = set(allowed)
        invalid = [item for item in items if item not in allowed_set]
        if invalid:
            raise ValueError(f"invalid {name}: {', '.join(invalid)}")
    return items


def _row_to_event(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    data["trajectory"] = json.loads(data.pop("trajectory_json"))
    data["cv_detail"] = json.loads(data.pop("cv_detail_json"))
    return data


class StatusConflictError(RuntimeError):
    """``transition_status`` 发现事件当前状态不等于调用方给出的 ``from_status``。"""


class EventRepository:
    """事件表与状态日志的读写入口。

    构造时打开数据库、设置连接参数并幂等地执行 schema.sql。
    """

    def __init__(self, db_path: str | Path) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(db_path), timeout=5.0, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def insert_event(self, event: NewEvent, *, creation_actor: str = "cv_pipeline") -> None:
        """写入事件（pending_review）及其首条状态日志，两者在同一事务内提交。

        首条日志为 ``NULL -> pending_review``，``actor_type`` 为 ``system``。
        ``camera_id`` 必须已存在于 cameras 表，否则抛出 ``sqlite3.IntegrityError``。
        """
        annotated = _normalize_rel_path(event.keyframe_annotated_path)
        raw = _normalize_rel_path(event.keyframe_raw_path)
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO events (
                    id, camera_id, event_type, detection_source, status,
                    start_ms, end_ms, cv_confidence, track_id, roi_id, roi_type,
                    cv_detail_json, trajectory_json,
                    keyframe_annotated_path, keyframe_raw_path,
                    sop_run_id, created_ms, updated_ms
                ) VALUES (?, ?, ?, ?, 'pending_review', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.id,
                    event.camera_id,
                    event.event_type,
                    event.detection_source,
                    event.start_ms,
                    event.end_ms,
                    event.cv_confidence,
                    event.track_id,
                    event.roi_id,
                    event.roi_type,
                    json.dumps(event.cv_detail, ensure_ascii=False),
                    json.dumps(event.trajectory),
                    annotated,
                    raw,
                    event.sop_run_id,
                    event.created_ms,
                    event.created_ms,
                ),
            )
            self._conn.execute(
                """
                INSERT INTO event_status_log
                    (event_id, from_status, to_status, actor_type, actor, reason, created_ms)
                VALUES (?, NULL, 'pending_review', 'system', ?, NULL, ?)
                """,
                (event.id, creation_actor, event.created_ms),
            )

    def get_event(self, event_id: str) -> dict[str, Any] | None:
        """按 ID 读取事件，不存在时返回 None。JSON 列已解析为 Python 对象。"""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM events WHERE id = ?", (event_id,)
            ).fetchone()
        return _row_to_event(row) if row is not None else None

    def list_events(
        self,
        *,
        camera_id: str | None = None,
        event_type: str | None = None,
        status: str | None = None,
        from_ms: int | None = None,
        to_ms: int | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """按条件筛选事件，返回 ``(当前页事件, 满足条件的总数)``。

        ``camera_id``、``event_type``、``status`` 支持逗号分隔的多值（取并集）。
        ``from_ms`` 与 ``to_ms`` 对 ``created_ms`` 做闭区间过滤。
        结果按 ``created_ms DESC, id DESC`` 排序。无 status 过滤时包含 dismissed 事件。
        """
        if not LIST_LIMIT_MIN <= limit <= LIST_LIMIT_MAX:
            raise ValueError(f"limit must be in [{LIST_LIMIT_MIN}, {LIST_LIMIT_MAX}]")
        if offset < 0:
            raise ValueError("offset must be >= 0")
        if from_ms is not None and to_ms is not None and from_ms > to_ms:
            raise ValueError("from_ms must be <= to_ms")

        clauses: list[str] = []
        params: list[Any] = []

        cameras = _parse_multi(camera_id, None, "camera_id")
        if cameras:
            clauses.append(f"camera_id IN ({','.join('?' * len(cameras))})")
            params.extend(cameras)
        types = _parse_multi(event_type, EVENT_TYPES, "event_type")
        if types:
            clauses.append(f"event_type IN ({','.join('?' * len(types))})")
            params.extend(types)
        statuses = _parse_multi(status, EVENT_STATUSES, "status")
        if statuses:
            clauses.append(f"status IN ({','.join('?' * len(statuses))})")
            params.extend(statuses)
        if from_ms is not None:
            clauses.append("created_ms >= ?")
            params.append(from_ms)
        if to_ms is not None:
            clauses.append("created_ms <= ?")
            params.append(to_ms)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        with self._lock:
            total = self._conn.execute(
                f"SELECT COUNT(*) FROM events {where}", params
            ).fetchone()[0]
            rows = self._conn.execute(
                f"SELECT * FROM events {where} "
                "ORDER BY created_ms DESC, id DESC LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
        return [_row_to_event(row) for row in rows], int(total)

    def update_evidence(
        self,
        event_id: str,
        *,
        keyframe_annotated_path: str | None = None,
        keyframe_raw_path: str | None = None,
        clip_path: str | None = None,
        clip_status: str | None = None,
        updated_ms: int | None = None,
    ) -> bool:
        """更新证据字段，只写入传入的非 None 字段。

        路径必须为相对 ``evidence.base_dir`` 的路径。
        事件存在并已更新时返回 True，事件不存在时返回 False。
        ``clip_status='ready'`` 而无 ``clip_path`` 时由数据库 CHECK 约束拒绝。
        """
        updates: dict[str, Any] = {}
        if keyframe_annotated_path is not None:
            updates["keyframe_annotated_path"] = _normalize_rel_path(keyframe_annotated_path)
        if keyframe_raw_path is not None:
            updates["keyframe_raw_path"] = _normalize_rel_path(keyframe_raw_path)
        if clip_path is not None:
            updates["clip_path"] = _normalize_rel_path(clip_path)
        if clip_status is not None:
            if clip_status not in CLIP_STATUSES:
                raise ValueError(f"invalid clip_status: {clip_status!r}")
            updates["clip_status"] = clip_status
        if not updates:
            raise ValueError("update_evidence requires at least one field")

        updates["updated_ms"] = updated_ms if updated_ms is not None else int(time.time() * 1000)
        assignments = ", ".join(f"{column} = ?" for column in updates)
        with self._lock, self._conn:
            cursor = self._conn.execute(
                f"UPDATE events SET {assignments} WHERE id = ?",
                [*updates.values(), event_id],
            )
            return cursor.rowcount > 0

    def update_vlm_result(
        self,
        event_id: str,
        *,
        verdict: str | None = None,
        confidence: float | None = None,
        description: str | None = None,
        suggestion: str | None = None,
        error: str | None = None,
        reviewed_ms: int | None = None,
        review_skipped: bool = False,
        updated_ms: int | None = None,
    ) -> bool:
        """写入事件的 VLM 复核块，只更新 vlm_* 列、review_skipped 与 updated_ms。

        每次调用都会整体覆盖这些列；未传入的值写为 NULL，因此不会残留旧的
        复核结果。事件本身的字段（类型、区域、CV 置信度、关键帧路径）保持不变。
        事件存在时返回 True，不存在时返回 False。
        """
        timestamp = updated_ms if updated_ms is not None else int(time.time() * 1000)
        with self._lock, self._conn:
            cursor = self._conn.execute(
                """
                UPDATE events SET
                    vlm_verdict = ?,
                    vlm_confidence = ?,
                    vlm_description = ?,
                    vlm_suggestion = ?,
                    vlm_error = ?,
                    vlm_reviewed_ms = ?,
                    review_skipped = ?,
                    updated_ms = ?
                WHERE id = ?
                """,
                (
                    verdict,
                    confidence,
                    description,
                    suggestion,
                    error,
                    reviewed_ms,
                    1 if review_skipped else 0,
                    timestamp,
                    event_id,
                ),
            )
            return cursor.rowcount > 0

    def update_vlm_error(
        self,
        event_id: str,
        vlm_error: str,
        *,
        from_status: str = "pending_review",
    ) -> bool:
        """只写入 ``vlm_error`` 字段，不改动其他列（包括 ``updated_ms``）。

        仅当事件当前状态等于 ``from_status`` 时写入。事件存在且状态匹配并已更新时返回 True，
        否则返回 False。用于启动恢复等需要记录失败原因、又不能改变其他字段的场景。
        """
        if from_status not in EVENT_STATUSES:
            raise ValueError(f"invalid from_status: {from_status!r}")
        if not isinstance(vlm_error, str) or not vlm_error.strip():
            raise ValueError("vlm_error must be a non-empty string")
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "UPDATE events SET vlm_error = ? WHERE id = ? AND status = ?",
                (vlm_error, event_id, from_status),
            )
            return cursor.rowcount > 0

    def transition_status(
        self,
        event_id: str,
        *,
        from_status: str,
        to_status: str,
        actor_type: str,
        actor: str,
        reason: str | None,
        now_ms: int,
    ) -> None:
        """在一个事务内更新 ``events.status`` 并追加一行 ``event_status_log``。

        仅当事件当前状态等于 ``from_status`` 时执行。状态校验与写入在同一条
        UPDATE 语句中完成（``WHERE id = ? AND status = ?``），因此不会与其他连接竞争。
        不校验迁移是否合法，合法性由调用方（``EventStateMachine``）判断。

        - 事件不存在：抛出 ``LookupError``。
        - 当前状态不等于 ``from_status``：抛出 ``StatusConflictError``，不写入任何内容。
        """
        if from_status not in EVENT_STATUSES:
            raise ValueError(f"invalid from_status: {from_status!r}")
        if to_status not in EVENT_STATUSES:
            raise ValueError(f"invalid to_status: {to_status!r}")
        if actor_type not in ("system", "inspector"):
            raise ValueError(f"invalid actor_type: {actor_type!r}")
        if not isinstance(actor, str) or not actor:
            raise ValueError("actor must be a non-empty string")

        with self._lock, self._conn:
            cursor = self._conn.execute(
                "UPDATE events SET status = ?, updated_ms = ? WHERE id = ? AND status = ?",
                (to_status, now_ms, event_id, from_status),
            )
            if cursor.rowcount == 0:
                row = self._conn.execute(
                    "SELECT status FROM events WHERE id = ?", (event_id,)
                ).fetchone()
                if row is None:
                    raise LookupError(f"event not found: {event_id}")
                raise StatusConflictError(
                    f"event {event_id} status is {row['status']!r}, expected {from_status!r}"
                )
            self._conn.execute(
                """
                INSERT INTO event_status_log
                    (event_id, from_status, to_status, actor_type, actor, reason, created_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (event_id, from_status, to_status, actor_type, actor, reason, now_ms),
            )
    def upsert_camera(
        self,
        camera_id: str,
        *,
        name: str,
        source_type: str,
        source: str,
        status: str,
        now_ms: int,
    ) -> None:
        """登记或更新 Camera 行（启动时调用，满足 events 的外键约束）。

        ``source`` 应为脱敏后的地址（RTSP 凭据已去除）。
        """
        if source_type not in ("file", "rtsp"):
            raise ValueError(f"invalid source_type: {source_type!r}")
        if status not in ("online", "offline"):
            raise ValueError(f"invalid status: {status!r}")
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO cameras (id, name, source_type, source, status, status_changed_ms)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    source_type = excluded.source_type,
                    source = excluded.source
                """,
                (camera_id, name, source_type, source, status, now_ms),
            )

    def set_camera_status(self, camera_id: str, status: str, now_ms: int) -> bool:
        """更新 Camera 的在线状态。Camera 存在并已更新时返回 True。"""
        if status not in ("online", "offline"):
            raise ValueError(f"invalid status: {status!r}")
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "UPDATE cameras SET status = ?, status_changed_ms = ? WHERE id = ?",
                (status, now_ms, camera_id),
            )
            return cursor.rowcount > 0
    def list_cameras(self) -> list[dict[str, Any]]:
        """返回全部 Camera 行，按 id 排序。"""
        with self._lock:
            rows = self._conn.execute("SELECT * FROM cameras ORDER BY id").fetchall()
        return [dict(row) for row in rows]

    def get_camera(self, camera_id: str) -> dict[str, Any] | None:
        """按 ID 读取 Camera 行，不存在时返回 None。"""
        with self._lock:
            row = self._conn.execute("SELECT * FROM cameras WHERE id = ?", (camera_id,)).fetchone()
        return dict(row) if row is not None else None

    def list_status_log(self, event_id: str) -> list[dict[str, Any]]:
        """返回事件的全部状态日志，按写入顺序（id 升序）排列。"""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT from_status, to_status, actor_type, actor, reason, created_ms
                FROM event_status_log WHERE event_id = ? ORDER BY id
                """,
                (event_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """持锁开启一个事务，供同包内的仓储（如 SOP 仓储）执行写入。

        正常退出时提交，抛出异常时回滚。调用方不得在事务内调用本类的其他方法（会死锁）。
        """
        with self._lock, self._conn:
            yield self._conn

    def read_rows(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        """在持锁状态下执行只读查询并返回字典列表。

        仅供统计等只读聚合使用，调用方须保证 ``sql`` 不含写操作。
        """
        with self._lock:
            rows = self._conn.execute(sql, tuple(params)).fetchall()
        return [dict(row) for row in rows]
