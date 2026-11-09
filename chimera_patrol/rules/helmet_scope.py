"""安全帽判定的作用域与人员高度过滤（需求 3.5，决策 D1，Rule_Engine「安全帽」步骤 1）。

纯函数，不依赖配置。阈值与参考点模式由调用方以普通参数传入。

规则：

- 无 work_zone 时，全画面每个 person 都在作用域内（D1）。
- 有 work_zone 时，参考点落在任一 work_zone 多边形内（含边界）的 person 在作用域内。
- 框高度 ``y2 - y1`` 小于 ``min_person_height_px`` 的 person 不参与判定：既不进入
  ``scoped``，也不进入 ``outside``。头部区域过小时关联结果不可靠。
"""

from __future__ import annotations

from typing import Mapping, Sequence, Tuple, TypeVar

from chimera_patrol.rules import geometry
from chimera_patrol.rules.geometry import Box, Polygon, RefMode

__all__ = ["in_work_scope", "filter_scoped_persons"]

K = TypeVar("K")
V = TypeVar("V")


def in_work_scope(person_box: Box, work_zones: Sequence[Polygon], reference_point: RefMode) -> bool:
    """判断 person 是否在安全帽作用域内。

    Args:
        person_box: person 像素框 ``(x1, y1, x2, y2)``。
        work_zones: work_zone 多边形列表。为空表示无作用域限制。
        reference_point: 参考点模式，``"bottom_center"`` 或 ``"center"``。

    Returns:
        ``work_zones`` 为空时返回 True；否则参考点位于任一多边形内（含边界）时返回 True。
    """
    if not work_zones:
        return True
    point = geometry.ref_point(person_box, reference_point)
    return any(geometry.point_in_polygon(point, zone) for zone in work_zones)


def filter_scoped_persons(
    persons: Mapping[K, V],
    work_zones: Sequence[Polygon],
    reference_point: RefMode,
    min_person_height_px: int,
) -> Tuple[dict[K, V], dict[K, V]]:
    """按高度与作用域把 person 分为两组。

    Args:
        persons: ``{track_id: person}``。每个 person 须有 ``box`` 属性，
            即 ``(x1, y1, x2, y2)`` 像素框。
        work_zones: work_zone 多边形列表，空列表表示全画面。
        reference_point: 参考点模式。
        min_person_height_px: 框高度下限（像素）。高度严格小于该值的 person 被丢弃。

    Returns:
        ``(scoped, outside)``。``scoped`` 为作用域内且高度达标的 person；
        ``outside`` 为作用域外且高度达标的 person。两者保持入参的键与顺序。
    """
    scoped: dict[K, V] = {}
    outside: dict[K, V] = {}
    for key, person in persons.items():
        box = person.box  # type: ignore[attr-defined]
        x1, y1, x2, y2 = box
        if (y2 - y1) < min_person_height_px:
            continue
        if in_work_scope(box, work_zones, reference_point):
            scoped[key] = person
        else:
            outside[key] = person
    return scoped, outside
