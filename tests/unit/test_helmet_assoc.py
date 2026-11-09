"""Unit tests for chimera_patrol.rules.helmet_assoc (task 4.7)."""

from __future__ import annotations

import pytest

from chimera_patrol.rules.helmet_assoc import associate_helmets

# person 框高 200，ratio=0.25 时头部区域为 (x1, y1, x2, y1 + 50)。
PERSON_A = (0.0, 0.0, 100.0, 200.0)  # head (0, 0, 100, 50)
PERSON_B = (50.0, 0.0, 150.0, 200.0)  # head (50, 0, 150, 50)


def _assoc(persons, helmets, metric="ioh", overlap=0.5, conf=0.5, ratio=0.25):
    return associate_helmets(persons, helmets, ratio, metric, overlap, conf)


def test_low_confidence_helmet_is_ignored() -> None:
    helmet_in_head = ((10.0, 10.0, 50.0, 40.0), 0.3, "helmet")
    assert _assoc([("t1", PERSON_A)], [helmet_in_head], conf=0.5) == {}

    helmet_conf_ok = ((10.0, 10.0, 50.0, 40.0), 0.9, "helmet")
    assert _assoc([("t1", PERSON_A)], [helmet_conf_ok], conf=0.5) == {"t1": 0}


def test_no_helmet_detections_do_not_associate() -> None:
    no_helmet_in_head = ((10.0, 10.0, 50.0, 40.0), 0.99, "no_helmet")
    assert _assoc([("t1", PERSON_A)], [no_helmet_in_head]) == {}


def test_ioh_and_iou_differ_for_small_helmet_inside_head() -> None:
    # 安全帽完全位于头部区域内：ioh = 1，而 iou = 1200 / 5000 = 0.24。
    helmet = ((10.0, 10.0, 50.0, 40.0), 0.9, "helmet")

    assert _assoc([("t1", PERSON_A)], [helmet], metric="ioh", overlap=0.5) == {"t1": 0}
    assert _assoc([("t1", PERSON_A)], [helmet], metric="iou", overlap=0.5) == {}
    assert _assoc([("t1", PERSON_A)], [helmet], metric="iou", overlap=0.2) == {"t1": 0}


def test_assignment_is_one_to_one_and_greedy_by_overlap() -> None:
    # h0 完全在 B 的头部内（ioh=1），与 A 的头部重叠 0.5。
    # h1 完全在 A 的头部内（ioh=1），与 B 无重叠。
    # 贪心应先分配 B-h0，再把 h1 分给 A，而不是让 A 先占 h0。
    helmets = [
        ((80.0, 10.0, 120.0, 40.0), 0.9, "helmet"),
        ((10.0, 10.0, 40.0, 40.0), 0.9, "helmet"),
    ]
    result = _assoc([("A", PERSON_A), ("B", PERSON_B)], helmets, overlap=0.5)
    assert result == {"B": 0, "A": 1}
    assert len(set(result.values())) == len(result)


def test_one_helmet_is_used_at_most_once_on_equal_overlap() -> None:
    # 同一安全帽同时完全覆盖两个 person 的头部，只能分给先出现的 person。
    helmet = ((60.0, 10.0, 90.0, 40.0), 0.9, "helmet")
    assert _assoc([("A", PERSON_A), ("B", PERSON_B)], [helmet]) == {"A": 0}


def test_helmet_above_head_region_is_not_associated() -> None:
    # person 框顶部为 y=100，头部区域为 y in [100, 150]；安全帽位于 y in [10, 60]。
    person = (0.0, 100.0, 100.0, 300.0)
    helmet_above = ((10.0, 10.0, 50.0, 60.0), 0.9, "helmet")
    assert _assoc([("t1", person)], [helmet_above]) == {}


def test_invalid_metric_raises() -> None:
    with pytest.raises(ValueError):
        _assoc([("t1", PERSON_A)], [], metric="dice")


@pytest.mark.parametrize("ratio", [0.0, -0.1, 1.01])
def test_invalid_head_region_ratio_raises(ratio: float) -> None:
    with pytest.raises(ValueError):
        _assoc([("t1", PERSON_A)], [], ratio=ratio)


@pytest.mark.parametrize(("overlap", "conf"), [(-0.1, 0.5), (1.5, 0.5), (0.5, -0.1), (0.5, 1.2)])
def test_out_of_range_thresholds_raise(overlap: float, conf: float) -> None:
    with pytest.raises(ValueError):
        _assoc([("t1", PERSON_A)], [], overlap=overlap, conf=conf)
