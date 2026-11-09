"""compute_stats 单元示例测试（需求 8.1、8.2；设计 StatsDTO、A6、A13、A14）。"""

import pytest

from chimera_patrol.api.stats import compute_stats


def _ev(camera_id: str, event_type: str, status: str) -> dict:
    return {"camera_id": camera_id, "event_type": event_type, "status": status}


def _sums_match(stats: dict) -> None:
    totals = stats["totals"]
    for key in (
        "total_alarms",
        "confirmed",
        "dismissed",
        "needs_human",
        "review_failed",
        "pending_review",
        "pending_human",
    ):
        assert sum(b[key] for b in stats["by_camera"]) == totals[key], key
        assert sum(b[key] for b in stats["by_type"]) == totals[key], key


def test_zero_alarms_gives_null_rate():
    stats = compute_stats([], include_sop=False)
    assert stats["totals"]["total_alarms"] == 0
    assert stats["totals"]["false_positive_reduction_rate"] is None
    assert stats["by_camera"] == []
    assert stats["by_type"] == []


def test_zero_alarms_after_sop_exclusion_gives_null_rate():
    events = [_ev("cam_a", "sop_violation", "dismissed")]
    stats = compute_stats(events, include_sop=False)
    assert stats["totals"]["total_alarms"] == 0
    assert stats["totals"]["false_positive_reduction_rate"] is None


def test_sop_excluded_by_default_path():
    events = [
        _ev("cam_a", "intrusion", "confirmed"),
        _ev("cam_a", "sop_violation", "needs_human"),
    ]
    stats = compute_stats(events, include_sop=False)
    assert stats["totals"]["total_alarms"] == 1
    assert stats["totals"]["needs_human"] == 0
    assert [b["event_type"] for b in stats["by_type"]] == ["intrusion"]


def test_sop_included_when_requested():
    events = [
        _ev("cam_a", "intrusion", "confirmed"),
        _ev("cam_a", "sop_violation", "needs_human"),
    ]
    stats = compute_stats(events, include_sop=True)
    assert stats["totals"]["total_alarms"] == 2
    assert stats["totals"]["needs_human"] == 1
    assert [b["event_type"] for b in stats["by_type"]] == ["intrusion", "sop_violation"]


def test_rate_is_dismissed_over_total():
    events = [
        _ev("cam_a", "no_helmet", "dismissed"),
        _ev("cam_a", "no_helmet", "dismissed"),
        _ev("cam_a", "no_helmet", "confirmed"),
        _ev("cam_b", "intrusion", "pending_review"),
    ]
    stats = compute_stats(events, include_sop=False)
    assert stats["totals"]["total_alarms"] == 4
    assert stats["totals"]["false_positive_reduction_rate"] == pytest.approx(0.5)


def test_pending_human_is_needs_human_plus_review_failed():
    events = [
        _ev("cam_a", "intrusion", "needs_human"),
        _ev("cam_a", "intrusion", "needs_human"),
        _ev("cam_b", "loitering", "review_failed"),
        _ev("cam_b", "loitering", "pending_review"),
        _ev("cam_b", "loitering", "confirmed"),
    ]
    stats = compute_stats(events, include_sop=False)
    totals = stats["totals"]
    assert totals["needs_human"] == 2
    assert totals["review_failed"] == 1
    assert totals["pending_human"] == 3
    assert totals["total_alarms"] == 5
    by_cam = {b["camera_id"]: b for b in stats["by_camera"]}
    assert by_cam["cam_a"]["pending_human"] == 2
    assert by_cam["cam_b"]["pending_human"] == 1


def test_buckets_sorted_and_sums_match_totals():
    events = [
        _ev("cam_b", "no_helmet", "dismissed"),
        _ev("cam_a", "intrusion", "confirmed"),
        _ev("cam_a", "smoke_fire", "pending_review"),
        _ev("cam_b", "intrusion", "review_failed"),
        _ev("cam_a", "sop_violation", "needs_human"),
        _ev("cam_b", "loitering", "needs_human"),
    ]
    stats = compute_stats(events, include_sop=True)
    assert [b["camera_id"] for b in stats["by_camera"]] == ["cam_a", "cam_b"]
    assert [b["event_type"] for b in stats["by_type"]] == [
        "intrusion",
        "loitering",
        "smoke_fire",
        "no_helmet",
        "sop_violation",
    ]
    _sums_match(stats)


def test_bucket_rates_use_bucket_denominator():
    events = [
        _ev("cam_a", "intrusion", "dismissed"),
        _ev("cam_b", "intrusion", "confirmed"),
        _ev("cam_b", "intrusion", "confirmed"),
    ]
    stats = compute_stats(events, include_sop=False)
    by_cam = {b["camera_id"]: b for b in stats["by_camera"]}
    assert by_cam["cam_a"]["false_positive_reduction_rate"] == pytest.approx(1.0)
    assert by_cam["cam_b"]["false_positive_reduction_rate"] == pytest.approx(0.0)
    _sums_match(stats)


def test_unknown_status_raises():
    with pytest.raises(ValueError):
        compute_stats([_ev("cam_a", "intrusion", "bogus")], include_sop=False)
