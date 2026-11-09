"""规则引擎使用的几何纯函数。

坐标约定：
- 框为 ``(x1, y1, x2, y2)`` 浮点数，单位为像素，要求 ``x1 <= x2`` 且 ``y1 <= y2``。
- 多边形为 ``(x, y)`` 顶点列表。本模块不做归一化，调用方使用的坐标空间即为判定空间。

本模块不依赖配置，所有函数均无副作用。
"""

from __future__ import annotations

from typing import Literal, Sequence, Tuple

Box = Tuple[float, float, float, float]
Point = Tuple[float, float]
Polygon = Sequence[Point]
RefMode = Literal["bottom_center", "center"]

__all__ = ["Box", "Point", "Polygon", "RefMode", "ref_point", "point_in_polygon", "iou", "ioh", "head_region"]


def _area(box: Box) -> float:
    """返回框的面积，框退化或翻转时按 0 计。"""
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def ref_point(box: Box, mode: RefMode) -> Point:
    """计算框的参考点。

    Args:
        box: ``(x1, y1, x2, y2)`` 像素框。
        mode: ``"bottom_center"`` 取底边中点 ``((x1+x2)/2, y2)``；
            ``"center"`` 取框中心 ``((x1+x2)/2, (y1+y2)/2)``。

    Returns:
        参考点 ``(x, y)``。

    Raises:
        ValueError: ``mode`` 不是受支持的取值。
    """
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    if mode == "bottom_center":
        return (cx, y2)
    if mode == "center":
        return (cx, (y1 + y2) / 2.0)
    raise ValueError(f"unsupported ref_point mode: {mode!r}")


def _on_segment(p: Point, a: Point, b: Point) -> bool:
    """判断点 p 是否位于线段 ab 上（含端点）。"""
    px, py = p
    ax, ay = a
    bx, by = b
    cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    if cross != 0:
        return False
    return min(ax, bx) <= px <= max(ax, bx) and min(ay, by) <= py <= max(ay, by)


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    """用射线法判断点是否在多边形内。

    落在任一边上的点（含顶点）视为在内。顶点少于 3 个时视为无面积，返回 False。

    Args:
        point: 待判定点 ``(x, y)``。
        polygon: 按顺序排列的顶点列表，首尾不必重复。

    Returns:
        点在多边形内或边界上返回 True，否则 False。
    """
    n = len(polygon)
    if n < 3:
        return False

    px, py = point
    inside = False
    for i in range(n):
        a = polygon[i]
        b = polygon[(i + 1) % n]
        if _on_segment(point, a, b):
            return True
        ax, ay = a
        bx, by = b
        # 半开区间判定，避免顶点被重复计数。
        if (ay > py) != (by > py):
            x_cross = ax + (py - ay) * (bx - ax) / (by - ay)
            if px < x_cross:
                inside = not inside
    return inside


def _intersection_area(a: Box, b: Box) -> float:
    """返回两框交集面积，无交集时为 0。"""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def iou(a: Box, b: Box) -> float:
    """计算两框的交并比 ``area(a∩b) / area(a∪b)``。

    两框并集面积为 0（均退化）时返回 0。

    Returns:
        取值范围 [0, 1] 的浮点数，对称。
    """
    inter = _intersection_area(a, b)
    union = _area(a) + _area(b) - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def ioh(helmet: Box, head: Box) -> float:
    """计算安全帽框被头部区域覆盖的比例 ``area(helmet∩head) / area(helmet)``。

    安全帽框面积为 0 时返回 0，避免除零。

    Returns:
        取值范围 [0, 1] 的浮点数。
    """
    helmet_area = _area(helmet)
    if helmet_area <= 0.0:
        return 0.0
    return _intersection_area(helmet, head) / helmet_area


def head_region(person_box: Box, ratio: float) -> Box:
    """返回 person 框顶部的头部区域 ``(x1, y1, x2, y1 + ratio·(y2−y1))``。

    Args:
        person_box: person 像素框。
        ratio: 头部区域占 person 框高度的比例，设计取值范围为 (0, 1]。

    Returns:
        头部区域框，与 person 框同左右边界、同顶边。
    """
    x1, y1, x2, y2 = person_box
    return (x1, y1, x2, y1 + ratio * (y2 - y1))
