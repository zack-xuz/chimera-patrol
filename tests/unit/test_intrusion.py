"""区域闯入与人员滞留的确定性冒烟测试（示例级，非属性测试）。"""

import pytest

from chimera_patrol.rules.intrusion import IntrusionRule, RoiSpec, TrackObservation

ZONE = RoiSpec(id="gate", type="intrusion", polygon=[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)])
INSIDE = (40.0, 50.0, 60.0, 90.0)  # 底边中点 (50, 90) 在 ZONE 内
OUTSIDE = (200.0, 50.0, 220.0, 90.0)  # 底边中点 (210, 90) 在 ZONE 外


def _rule(grace_s: float = 1.0, dwell_s: float = 2.0) -> IntrusionRule:
    return IntrusionRule(
        reference_point="bottom_center",
        roi_exit_grace_s=grace_s,
        dwell_threshold_s=dwell_s,
        rois=[ZONE],
    )


def _obs(track_id: int, box: tuple) -> TrackObservation:
    return TrackObservation(track_id=track_id, box=box)


def test_entering_roi_emits_one_intrusion_with_entry_timestamp() -> None:
    rule = _rule(dwell_s=100.0)
    assert rule.process(0, [_obs(1, OUTSIDE)]) == []
    events = rule.process(1000, [_obs(1, INSIDE)])
    assert [(e.type, e.roi_id, e.track_id, e.start_ms, e.end_ms) for e in events] == [
        ("intrusion", "gate", 1, 1000, 1000)
    ]
    assert rule.process(2000, [_obs(1, INSIDE)]) == []


def test_jitter_shorter_than_grace_does_not_refire_intrusion() -> None:
    rule = _rule(grace_s=1.0, dwell_s=100.0)
    rule.process(0, [_obs(1, INSIDE)])
    assert rule.process(500, [_obs(1, OUTSIDE)]) == []
    assert rule.process(1000, [_obs(1, INSIDE)]) == []


def test_leaving_longer_than_grace_allows_new_intrusion() -> None:
    rule = _rule(grace_s=1.0, dwell_s=100.0)
    rule.process(0, [_obs(1, INSIDE)])
    rule.process(1500, [_obs(1, OUTSIDE)])
    events = rule.process(2000, [_obs(1, INSIDE)])
    assert [(e.type, e.start_ms, e.end_ms) for e in events] == [("intrusion", 2000, 2000)]


def test_dwell_reaching_threshold_emits_one_loitering_event() -> None:
    rule = _rule(grace_s=1.0, dwell_s=2.0)
    assert [e.type for e in rule.process(0, [_obs(1, INSIDE)])] == ["intrusion"]
    assert rule.process(1000, [_obs(1, INSIDE)]) == []
    events = rule.process(2000, [_obs(1, INSIDE)])
    assert [(e.type, e.start_ms, e.end_ms) for e in events] == [("loitering", 0, 2000)]
    assert rule.process(3000, [_obs(1, INSIDE)]) == []


def test_detection_gap_within_grace_keeps_dwell_running() -> None:
    rule = _rule(grace_s=1.0, dwell_s=2.0)
    rule.process(0, [_obs(1, INSIDE)])
    assert rule.process(500, []) == []
    assert rule.process(1000, [_obs(1, INSIDE)]) == []  # 缺失 1000ms，等于宽限，停留保持
    events = rule.process(2000, [_obs(1, INSIDE)])
    assert [(e.type, e.start_ms, e.end_ms) for e in events] == [("loitering", 0, 2000)]


def test_gap_beyond_grace_restarts_stay_and_dwell() -> None:
    rule = _rule(grace_s=1.0, dwell_s=2.0)
    rule.process(0, [_obs(1, INSIDE)])
    assert rule.process(1500, []) == []
    assert [e.type for e in rule.process(2000, [_obs(1, INSIDE)])] == ["intrusion"]
    assert rule.process(3000, [_obs(1, INSIDE)]) == []
    events = rule.process(4000, [_obs(1, INSIDE)])
    assert [(e.type, e.start_ms, e.end_ms) for e in events] == [("loitering", 2000, 4000)]


def test_non_intrusion_roi_is_rejected() -> None:
    work = RoiSpec(id="zone_w", type="work_zone", polygon=ZONE.polygon)
    with pytest.raises(ValueError):
        IntrusionRule(reference_point="center", roi_exit_grace_s=1.0, dwell_threshold_s=2.0, rois=[work])


def test_reset_clears_stays_so_next_entry_fires_again() -> None:
    rule = _rule(dwell_s=100.0)
    rule.process(0, [_obs(1, INSIDE)])
    rule.reset()
    assert [e.type for e in rule.process(500, [_obs(1, INSIDE)])] == ["intrusion"]
