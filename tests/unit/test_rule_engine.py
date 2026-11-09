"""RuleEngine.process 的确定性端到端示例。

时间单位为毫秒，每个示例按固定时间序列喂入已处理帧。
"""

from __future__ import annotations

import pytest

from chimera_patrol.config import RulesSection
from chimera_patrol.rules.engine import RuleEngine
from chimera_patrol.rules.smoke_fire import SmokeFireDetection
from chimera_patrol.rules.types import (
    CandidateEvent,
    FrameResult,
    HelmetDetection,
    PersonDetection,
    RoiSpec,
)

SQUARE = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
BIG_WORK_ZONE = [(0.0, 0.0), (300.0, 0.0), (300.0, 400.0), (0.0, 400.0)]

# 人员框高 200 像素，参考点（底边中点）分别为 (50, 90) 与 (125, 300)。
INSIDE_SQUARE = (40.0, 50.0, 60.0, 90.0)
OUTSIDE_SQUARE = (200.0, 50.0, 220.0, 90.0)
STANDING_IN_ZONE = (100.0, 100.0, 150.0, 300.0)
STANDING_OUTSIDE_ZONE = (400.0, 100.0, 450.0, 300.0)


def make_engine(rois: list[RoiSpec] | None = None, **rules: object) -> RuleEngine:
    """用给定的 rules 覆盖项构造单路规则引擎，其余字段取配置默认值。"""
    camera_rules = RulesSection.model_validate(rules)
    return RuleEngine("cam1", camera_rules, rois or [])


def frame(ts_ms: int, persons: list[PersonDetection] | None = None,
          smoke: list[SmokeFireDetection] | None = None,
          helmets: list[HelmetDetection] | None = None) -> FrameResult:
    return FrameResult(
        ts_ms=ts_ms,
        frame_index=ts_ms // 100,
        persons=persons or [],
        smoke_fire=smoke or [],
        helmets=helmets or [],
    )


def person(track_id: int, box: tuple[float, float, float, float]) -> PersonDetection:
    return PersonDetection(track_id=track_id, box=box, conf=0.9)


def test_intrusion_fires_once_and_cooldown_dedups_reentry() -> None:
    engine = make_engine([RoiSpec("zone", "intrusion", SQUARE)])
    sequence = [
        (0, [person(1, OUTSIDE_SQUARE)]),
        (100, [person(1, INSIDE_SQUARE)]),
        (200, [person(1, INSIDE_SQUARE)]),
        (300, [person(1, OUTSIDE_SQUARE)]),
        (2000, [person(1, OUTSIDE_SQUARE)]),  # 距最后在内 1800 ms，超过 roi_exit_grace_s，停留结束
        (2100, [person(1, INSIDE_SQUARE)]),  # 重新进入，但仍在 cooldown_s 内
    ]

    fired: list[tuple[int, str, int]] = []
    for ts, persons in sequence:
        for ev in engine.process(frame(ts, persons)):
            fired.append((ev.start_ms, ev.event_type, ev.track_id))

    assert fired == [(100, "intrusion", 1)]


def test_smoke_fire_fires_after_consecutive_frames_only_once() -> None:
    engine = make_engine(smoke_fire={"consecutive_frames": 3})
    box = (10.0, 10.0, 50.0, 50.0)
    smoke = [SmokeFireDetection(label="smoke", box=box, conf=0.9)]

    assert engine.process(frame(0, smoke=smoke)) == []
    assert engine.process(frame(100, smoke=smoke)) == []
    events = engine.process(frame(200, smoke=smoke))
    assert len(events) == 1
    ev = events[0]
    assert ev.event_type == "smoke_fire"
    assert ev.detection_source == "yolo"
    assert (ev.start_ms, ev.end_ms) == (0, 200)
    assert ev.cv_confidence == pytest.approx(0.9)
    assert ev.detail["label"] == "smoke"
    assert ev.track_id is None
    # 同一连续点位不重复触发
    assert engine.process(frame(300, smoke=smoke)) == []


def test_no_helmet_fires_after_duration_with_trajectory() -> None:
    engine = make_engine(helmet={"no_helmet_duration_s": 1.0})
    standing = [person(1, STANDING_IN_ZONE)]

    assert engine.process(frame(0, standing)) == []
    assert engine.process(frame(500, standing)) == []
    events = engine.process(frame(1000, standing))
    assert len(events) == 1
    ev = events[0]
    assert ev.event_type == "no_helmet"
    assert ev.detection_source == "yolo"
    assert (ev.start_ms, ev.end_ms, ev.track_id) == (0, 1000, 1)
    assert [point[0] for point in ev.trajectory] == [0, 500, 1000]
    # 同一段连续违规不重复触发
    assert engine.process(frame(1500, standing)) == []


def test_work_zone_scopes_helmet_checks_to_zone() -> None:
    frames = [
        frame(0, [person(1, STANDING_IN_ZONE), person(2, STANDING_OUTSIDE_ZONE)]),
        frame(500, [person(1, STANDING_IN_ZONE), person(2, STANDING_OUTSIDE_ZONE)]),
        frame(1000, [person(1, STANDING_IN_ZONE), person(2, STANDING_OUTSIDE_ZONE)]),
    ]
    rules = {"helmet": {"no_helmet_duration_s": 1.0}}

    scoped = make_engine([RoiSpec("site", "work_zone", BIG_WORK_ZONE)], **rules)
    scoped_events: list[CandidateEvent] = []
    for f in frames:
        scoped_events.extend(scoped.process(f))
    assert [(ev.event_type, ev.track_id) for ev in scoped_events] == [("no_helmet", 1)]

    # 无 work_zone 的对照组：全画面人员都参与判定（D1）。
    unscoped = make_engine(**rules)
    unscoped_events: list[CandidateEvent] = []
    for f in frames:
        unscoped_events.extend(unscoped.process(f))
    assert sorted(ev.track_id for ev in unscoped_events) == [1, 2]


def test_person_cooldown_suppresses_then_releases_per_track() -> None:
    engine = make_engine([RoiSpec("zone", "intrusion", SQUARE)], cooldown_s=5)
    sequence = [
        (100, [person(1, INSIDE_SQUARE)]),
        (1300, [person(1, OUTSIDE_SQUARE)]),  # 超过宽限，停留结束
        (2100, [person(1, INSIDE_SQUARE), person(2, (70.0, 50.0, 90.0, 90.0))]),
        (2200, [person(1, OUTSIDE_SQUARE), person(2, (70.0, 50.0, 90.0, 90.0))]),
        (6000, [person(1, INSIDE_SQUARE), person(2, (70.0, 50.0, 90.0, 90.0))]),
    ]

    fired: list[tuple[int, int]] = []
    for ts, persons in sequence:
        for ev in engine.process(frame(ts, persons)):
            fired.append((ev.start_ms, ev.track_id))

    # track 1 在 2100 被冷却丢弃（距 100 ms 仅 2 秒 < 5 秒）；
    # track 2 的冷却键独立，2100 正常产生；track 1 在 6000 满足冷却（5.9 秒 ≥ 5 秒）。
    assert fired == [(100, 1), (2100, 2), (6000, 1)]
