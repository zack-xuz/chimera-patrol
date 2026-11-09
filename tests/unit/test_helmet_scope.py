"""helmet_scope：作用域（D1）与人员高度过滤的示例测试。"""

from __future__ import annotations

from chimera_patrol.rules.helmet_scope import filter_scoped_persons, in_work_scope
from chimera_patrol.rules.types import PersonDetection

ZONE = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]


def test_no_work_zone_means_every_person_is_in_scope() -> None:
    assert in_work_scope((0.0, 0.0, 10.0, 10.0), [], "bottom_center") is True
    assert in_work_scope((900.0, 900.0, 950.0, 990.0), [], "center") is True


def test_in_scope_by_bottom_center_reference_point() -> None:
    # 底边中点 (50, 90) 在区域内，框中心 (50, 60) 也在内。
    assert in_work_scope((40.0, 40.0, 60.0, 90.0), [ZONE], "bottom_center") is True
    # 底边中点 (50, 140) 在区域外。
    assert in_work_scope((40.0, 40.0, 60.0, 140.0), [ZONE], "bottom_center") is False


def test_bottom_center_and_center_disagree_on_straddling_box() -> None:
    # 框 y 范围 40~140：底边中点 (50, 140) 在区域外，框中心 (50, 90) 在区域内。
    box = (40.0, 40.0, 60.0, 140.0)
    assert in_work_scope(box, [ZONE], "bottom_center") is False
    assert in_work_scope(box, [ZONE], "center") is True


def test_point_on_zone_boundary_counts_as_inside() -> None:
    # 底边中点恰好位于区域上边界 y=100。
    assert in_work_scope((40.0, 60.0, 60.0, 100.0), [ZONE], "bottom_center") is True


def test_point_in_any_of_several_zones_is_in_scope() -> None:
    other = [(200.0, 200.0), (300.0, 200.0), (300.0, 300.0), (200.0, 300.0)]
    assert in_work_scope((240.0, 240.0, 260.0, 280.0), [ZONE, other], "bottom_center") is True


def test_filter_splits_scoped_and_outside_by_zone() -> None:
    persons = {
        1: PersonDetection(track_id=1, box=(40.0, 40.0, 60.0, 90.0), conf=0.9),
        2: PersonDetection(track_id=2, box=(400.0, 40.0, 420.0, 90.0), conf=0.9),
    }
    scoped, outside = filter_scoped_persons(persons, [ZONE], "bottom_center", 10)
    assert list(scoped) == [1]
    assert list(outside) == [2]


def test_filter_without_zones_puts_all_tall_persons_in_scope() -> None:
    persons = {
        1: PersonDetection(track_id=1, box=(40.0, 40.0, 60.0, 90.0), conf=0.9),
        2: PersonDetection(track_id=2, box=(400.0, 40.0, 420.0, 90.0), conf=0.9),
    }
    scoped, outside = filter_scoped_persons(persons, [], "bottom_center", 10)
    assert list(scoped) == [1, 2]
    assert outside == {}


def test_height_threshold_drops_short_persons_from_both_groups() -> None:
    # 高度 59 低于阈值 60，丢弃；高度 60 等于阈值，保留。
    persons = {
        1: PersonDetection(track_id=1, box=(40.0, 0.0, 60.0, 59.0), conf=0.9),
        2: PersonDetection(track_id=2, box=(40.0, 0.0, 60.0, 60.0), conf=0.9),
        3: PersonDetection(track_id=3, box=(400.0, 0.0, 420.0, 30.0), conf=0.9),
    }
    scoped, outside = filter_scoped_persons(persons, [ZONE], "bottom_center", 60)
    assert list(scoped) == [2]
    assert list(outside) == []
