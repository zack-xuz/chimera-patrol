"""SQL 聚合统计测试（需求 8.1、8.2；设计 StatsDTO、A6、A13、A14）。

使用临时 SQLite 文件与真实 EventRepository，不加载模型权重。
"""

from __future__ import annotations

import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pytest

from chimera_patrol.api.routes_stats import get_stats
from chimera_patrol.api.stats import compute_stats
from chimera_patrol.store.repository import EventRepository, NewEvent
from chimera_patrol.store.stats_sql import compute_stats_sql

CAMERAS = ("cam_a", "cam_b", "cam_c")
CV_TYPES = ("intrusion", "loitering", "smoke_fire", "no_helmet")
STATUSES = ("pending_review", "confirmed", "dismissed", "needs_human", "review_failed")


@dataclass
class Store:
    repo: EventRepository
    db_path: Path
    counter: int = 0

    def next_id(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}_{self.counter:012x}"

    def add_event(
        self,
        camera: str,
        event_type: str,
        status: str,
        created_ms: int,
    ) -> None:
        """写入一条事件并迁移到目标状态。sop_violation 同时写入对应的 SOP 运行记录。"""
        sop_run_id = None
        detection_source = "yolo"
        if event_type == "sop_violation":
            sop_run_id = self.next_id("sop")
            detection_source = "sop"
            conn = sqlite3.connect(str(self.db_path))
            conn.execute(
                "INSERT INTO sop_runs (id, sop_id, sop_name, camera_id, status, operator, "
                "triggered_ms, window_start_ms, window_end_ms, sample_count, created_ms) "
                "VALUES (?, 'entry_prep', '入场作业准备', ?, 'completed', 'tester', ?, ?, ?, 1, ?)",
                (sop_run_id, camera, created_ms, created_ms, created_ms, created_ms),
            )
            conn.commit()
            conn.close()
        event_id = self.next_id("evt")
        self.repo.insert_event(
            NewEvent(
                id=event_id,
                camera_id=camera,
                event_type=event_type,
                detection_source=detection_source,
                start_ms=created_ms,
                end_ms=created_ms,
                keyframe_annotated_path=f"{camera}/{event_id}/annotated.jpg",
                keyframe_raw_path=f"{camera}/{event_id}/raw.jpg",
                created_ms=created_ms,
                sop_run_id=sop_run_id,
            )
        )
        if status != "pending_review":
            self.repo.transition_status(
                event_id,
                from_status="pending_review",
                to_status=status,
                actor_type="system",
                actor="test_fixture",
                reason=None,
                now_ms=created_ms + 1,
            )


@pytest.fixture
def store(tmp_path: Path):
    db_path = tmp_path / "stats.db"
    repository = EventRepository(db_path)
    conn = sqlite3.connect(str(db_path))
    for cam in CAMERAS:
        conn.execute(
            "INSERT INTO cameras (id, name, source_type, source, status_changed_ms) "
            "VALUES (?, ?, 'file', ?, 0)",
            (cam, cam, f"{cam}.mp4"),
        )
    conn.commit()
    conn.close()
    yield Store(repo=repository, db_path=db_path)
    repository.close()


def _fixed_dataset(s: Store) -> None:
    """覆盖多摄像头、多类型、全部状态与 sop_violation 的确定性数据集。"""
    t = 1_000
    for cam in CAMERAS:
        for event_type in CV_TYPES:
            for status in STATUSES:
                t += 10
                s.add_event(cam, event_type, status, t)
    for cam in CAMERAS:
        for status in STATUSES:
            t += 10
            s.add_event(cam, "sop_violation", status, t)


def _random_dataset(s: Store, seed: int, count: int) -> None:
    rng = random.Random(seed)
    for _ in range(count):
        camera = rng.choice(CAMERAS)
        event_type = rng.choice(CV_TYPES + ("sop_violation",))
        status = rng.choice(STATUSES)
        created_ms = rng.randint(0, 50_000)
        s.add_event(camera, event_type, status, created_ms)


def test_sql_matches_compute_stats_on_mixed_data(store: Store):
    _fixed_dataset(store)
    _random_dataset(store, seed=20240601, count=40)
    events, total = store.repo.list_events(limit=200)
    assert total == len(events) <= 200

    for include_sop in (False, True):
        expected = compute_stats(events, include_sop=include_sop)
        actual = compute_stats_sql(store.repo, include_sop=include_sop)
        assert actual == expected, f"include_sop={include_sop}"


def test_zero_alarm_rate_is_none_on_empty_store(store: Store):
    stats = compute_stats_sql(store.repo)
    assert stats["totals"]["total_alarms"] == 0
    assert stats["totals"]["false_positive_reduction_rate"] is None
    assert stats["by_camera"] == []
    assert stats["by_type"] == []


def test_zero_alarm_rate_is_none_when_only_sop_present(store: Store):
    store.add_event("cam_a", "sop_violation", "dismissed", 100)
    stats = compute_stats_sql(store.repo, include_sop=False)
    assert stats["totals"]["total_alarms"] == 0
    assert stats["totals"]["false_positive_reduction_rate"] is None


def test_include_sop_switch(store: Store):
    store.add_event("cam_a", "intrusion", "confirmed", 100)
    store.add_event("cam_a", "sop_violation", "needs_human", 110)

    default = compute_stats_sql(store.repo)
    assert default["totals"]["total_alarms"] == 1
    assert default["totals"]["needs_human"] == 0
    assert [b["event_type"] for b in default["by_type"]] == ["intrusion"]

    included = compute_stats_sql(store.repo, include_sop=True)
    assert included["totals"]["total_alarms"] == 2
    assert included["totals"]["needs_human"] == 1
    assert [b["event_type"] for b in included["by_type"]] == ["intrusion", "sop_violation"]


def test_camera_filter_accepts_comma_list(store: Store):
    store.add_event("cam_a", "intrusion", "confirmed", 100)
    store.add_event("cam_b", "intrusion", "dismissed", 110)
    store.add_event("cam_c", "intrusion", "needs_human", 120)

    stats = compute_stats_sql(store.repo, camera_id="cam_a,cam_c")
    assert [b["camera_id"] for b in stats["by_camera"]] == ["cam_a", "cam_c"]
    assert stats["totals"]["total_alarms"] == 2
    assert stats["totals"]["dismissed"] == 0


def test_time_range_is_inclusive_on_created_ms(store: Store):
    for created_ms in (100, 200, 300, 400):
        store.add_event("cam_a", "intrusion", "confirmed", created_ms)

    stats = compute_stats_sql(store.repo, from_ms=200, to_ms=300)
    assert stats["totals"]["total_alarms"] == 2

    stats = compute_stats_sql(store.repo, from_ms=201, to_ms=299)
    assert stats["totals"]["total_alarms"] == 0


def test_every_status_is_counted(store: Store):
    for i, status in enumerate(STATUSES):
        store.add_event("cam_a", "smoke_fire", status, 100 + i)
    totals = compute_stats_sql(store.repo)["totals"]
    for status in STATUSES:
        assert totals[status] == 1, status
    assert totals["total_alarms"] == 5
    assert totals["pending_human"] == 2
    assert totals["false_positive_reduction_rate"] == pytest.approx(0.2)


def test_get_stats_rejects_from_after_to(store: Store):
    with pytest.raises(ValueError):
        get_stats(store.repo, from_ms=500, to_ms=100)


def test_get_stats_delegates_to_sql_aggregation(store: Store):
    store.add_event("cam_a", "intrusion", "dismissed", 100)
    store.add_event("cam_a", "intrusion", "confirmed", 150)
    stats = get_stats(store.repo, camera_id="cam_a", from_ms=0, to_ms=200, include_sop=False)
    assert stats["totals"]["total_alarms"] == 2
    assert stats["totals"]["false_positive_reduction_rate"] == pytest.approx(0.5)
