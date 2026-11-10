"""SOP 运行记录与步骤结果的读写（任务 15.7）。

表结构见 schema.sql 中的 ``sop_runs`` 与 ``sop_step_results``。
- 同一 Camera 同时只允许一个 ``running``，由部分唯一索引保证；
  ``insert_run`` 违反该约束时抛出 ``RunInProgressError``。
- ``sop_violation`` 事件通过 ``events.sop_run_id`` 关联到运行记录，本模块只负责查询关联。
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Sequence

from chimera_patrol.store.repository import EventRepository

RUN_STATUSES = ("running", "completed", "check_failed_retry")
FAILURE_REASONS = ("vlm_timeout", "vlm_error", "response_invalid", "no_frames", "service_restart")
STEP_STATUSES = ("pass", "fail", "unknown")


class RunInProgressError(RuntimeError):
    """同一 Camera 已有 running 状态的检查。"""


@dataclass(frozen=True)
class NewSopRun:
    id: str
    sop_id: str
    sop_name: str
    camera_id: str
    operator: str
    triggered_ms: int
    window_start_ms: int
    window_end_ms: int
    sample_count: int
    sample_times_ms: list[int]
    created_ms: int
    retry_of: str | None = None


@dataclass(frozen=True)
class StepResult:
    index: int
    step_id: str
    description: str
    status: str
    evidence: str
    approximate: bool


@dataclass
class SopRunRecord:
    """sop_runs 的一行，附带步骤结果与关联事件 ID。"""

    row: dict[str, Any]
    steps: list[dict[str, Any]] = field(default_factory=list)
    event_id: str | None = None


class SopRepository:
    """SOP 运行记录的读写入口，基于 EventRepository 的连接与锁。"""

    def __init__(self, repository: EventRepository) -> None:
        self._repo = repository

    def insert_run(self, run: NewSopRun) -> None:
        """插入一条 running 运行记录。同一 Camera 已有 running 时抛出 RunInProgressError。"""
        if run.sample_count < 1:
            raise ValueError("sample_count must be >= 1")
        try:
            with self._repo.transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO sop_runs (
                        id, sop_id, sop_name, camera_id, status, operator,
                        triggered_ms, window_start_ms, window_end_ms,
                        sample_count, sample_times_json, retry_of, created_ms
                    ) VALUES (?, ?, ?, ?, 'running', ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run.id,
                        run.sop_id,
                        run.sop_name,
                        run.camera_id,
                        run.operator,
                        run.triggered_ms,
                        run.window_start_ms,
                        run.window_end_ms,
                        run.sample_count,
                        json.dumps(run.sample_times_ms),
                        run.retry_of,
                        run.created_ms,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            if "uq_sop_runs_running_per_camera" in str(exc) or "UNIQUE" in str(exc):
                raise RunInProgressError(f"camera {run.camera_id} already has a running SOP check") from exc
            raise

    def finish_completed(self, run_id: str, steps: Sequence[StepResult], finished_ms: int) -> None:
        """写入步骤结果并把运行置为 completed，两者在同一事务内完成。"""
        for step in steps:
            if step.status not in STEP_STATUSES:
                raise ValueError(f"invalid step status: {step.status!r}")
        with self._repo.transaction() as conn:
            cursor = conn.execute(
                "UPDATE sop_runs SET status = 'completed', finished_ms = ? "
                "WHERE id = ? AND status = 'running'",
                (finished_ms, run_id),
            )
            if cursor.rowcount == 0:
                raise LookupError(f"running SOP run not found: {run_id}")
            conn.executemany(
                """
                INSERT INTO sop_step_results
                    (run_id, step_index, step_id, description, status, evidence, approximate)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_id,
                        step.index,
                        step.step_id,
                        step.description,
                        step.status,
                        step.evidence,
                        1 if step.approximate else 0,
                    )
                    for step in steps
                ],
            )

    def finish_check_failed(self, run_id: str, reason: str, finished_ms: int) -> bool:
        """把 running 运行置为 check_failed_retry，并写入失败原因。仅更新 running 记录。"""
        if reason not in FAILURE_REASONS:
            raise ValueError(f"invalid failure_reason: {reason!r}")
        with self._repo.transaction() as conn:
            cursor = conn.execute(
                "UPDATE sop_runs SET status = 'check_failed_retry', failure_reason = ?, finished_ms = ? "
                "WHERE id = ? AND status = 'running'",
                (reason, finished_ms, run_id),
            )
            return cursor.rowcount > 0

    def recover_running(self, finished_ms: int) -> int:
        """启动恢复：把全部 running 运行置为 check_failed_retry（service_restart）。返回数量。"""
        with self._repo.transaction() as conn:
            cursor = conn.execute(
                "UPDATE sop_runs SET status = 'check_failed_retry', failure_reason = 'service_restart', "
                "finished_ms = ? WHERE status = 'running'",
                (finished_ms,),
            )
            return int(cursor.rowcount)

    def get_run(self, run_id: str) -> SopRunRecord | None:
        """按 ID 读取运行记录（含步骤结果与关联事件 ID）。"""
        rows = self._repo.read_rows("SELECT * FROM sop_runs WHERE id = ?", [run_id])
        if not rows:
            return None
        return self._hydrate(rows[0])

    def list_runs(
        self,
        *,
        camera_id: str | None = None,
        status: str | None = None,
        limit: int = 20,
    ) -> list[SopRunRecord]:
        """按 created_ms 降序返回运行记录。"""
        clauses: list[str] = []
        params: list[Any] = []
        if camera_id is not None:
            clauses.append("camera_id = ?")
            params.append(camera_id)
        if status is not None:
            if status not in RUN_STATUSES:
                raise ValueError(f"invalid status: {status!r}")
            clauses.append("status = ?")
            params.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._repo.read_rows(
            f"SELECT * FROM sop_runs {where} ORDER BY created_ms DESC, id DESC LIMIT ?",
            [*params, limit],
        )
        return [self._hydrate(row) for row in rows]

    def find_running(self, camera_id: str) -> dict[str, Any] | None:
        """返回该 Camera 当前 running 的运行记录（若有）。"""
        rows = self._repo.read_rows(
            "SELECT * FROM sop_runs WHERE camera_id = ? AND status = 'running' LIMIT 1", [camera_id]
        )
        return rows[0] if rows else None

    def _hydrate(self, row: dict[str, Any]) -> SopRunRecord:
        run_id = row["id"]
        row = dict(row)
        row["sample_times_ms"] = json.loads(row.pop("sample_times_json") or "[]")
        steps = self._repo.read_rows(
            "SELECT step_index, step_id, description, status, evidence, approximate "
            "FROM sop_step_results WHERE run_id = ? ORDER BY step_index",
            [run_id],
        )
        for step in steps:
            step["approximate"] = bool(step["approximate"])
        events = self._repo.read_rows(
            "SELECT id FROM events WHERE sop_run_id = ? LIMIT 1", [run_id]
        )
        return SopRunRecord(
            row=row,
            steps=steps,
            event_id=events[0]["id"] if events else None,
        )
