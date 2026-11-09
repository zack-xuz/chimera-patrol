"""规则引擎几何函数的属性测试（设计 Property 6：头部区域几何不变量）。

**Validates: Requirements 3.5, 3.6**
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from chimera_patrol.rules.geometry import head_region, ioh, iou, point_in_polygon

COORD = st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False)
SIZE = st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False)
RATIO = st.floats(min_value=1e-6, max_value=1.0, allow_nan=False, allow_infinity=False)
EPS = 1e-6


@st.composite
def positive_boxes(draw):
    """生成 x1<x2、y1<y2 且面积为正的框。"""
    x1 = draw(COORD)
    y1 = draw(COORD)
    x2 = x1 + draw(SIZE)
    y2 = y1 + draw(SIZE)
    assume(x2 > x1 and y2 > y1)
    return (x1, y1, x2, y2)


@st.composite
def any_boxes(draw):
    """生成可能退化（零宽或零高）的框，仍满足 x1<=x2、y1<=y2。"""
    x1 = draw(COORD)
    y1 = draw(COORD)
    dx = draw(st.one_of(st.just(0.0), SIZE))
    dy = draw(st.one_of(st.just(0.0), SIZE))
    return (x1, y1, x1 + dx, y1 + dy)


@st.composite
def helmet_inside_head(draw):
    """生成完全位于 person 框头部区域内、面积为正的 helmet 框，返回 (person, ratio, helmet)。"""
    person = draw(positive_boxes())
    ratio = draw(RATIO)
    head = head_region(person, ratio)
    hx_a, hx_b = draw(st.floats(min_value=0.0, max_value=1.0)), draw(st.floats(min_value=0.0, max_value=1.0))
    hy_a, hy_b = draw(st.floats(min_value=0.0, max_value=1.0)), draw(st.floats(min_value=0.0, max_value=1.0))
    hx1, hx2 = sorted((hx_a, hx_b))
    hy1, hy2 = sorted((hy_a, hy_b))
    # 在头部区域内按比例取点，保证 helmet 与 head 边界一致时仍为正面积。
    w_head = head[2] - head[0]
    h_head = head[3] - head[1]
    helmet = (
        head[0] + hx1 * w_head,
        head[1] + hy1 * h_head,
        head[0] + hx2 * w_head,
        head[1] + hy2 * h_head,
    )
    assume(helmet[2] > helmet[0] and helmet[3] > helmet[1])
    return person, ratio, helmet


@settings(max_examples=200, deadline=None)
@given(person=positive_boxes(), ratio=RATIO)
def test_head_region_shares_left_right_top_edges_and_scales_height(person, ratio) -> None:
    """head_region 与 person 框同左右边界、同顶边，高度为 ratio 倍 person 高度。"""
    hx1, hy1, hx2, hy2 = head_region(person, ratio)
    px1, py1, px2, py2 = person

    assert hx1 == px1
    assert hx2 == px2
    assert hy1 == py1
    assert hy2 - hy1 == pytest.approx(ratio * (py2 - py1), rel=1e-9, abs=EPS)


@settings(max_examples=200, deadline=None)
@given(person=positive_boxes(), ratio=RATIO)
def test_head_region_is_contained_in_person_box(person, ratio) -> None:
    """head_region 包含于 person 框内。"""
    hx1, hy1, hx2, hy2 = head_region(person, ratio)
    px1, py1, px2, py2 = person

    assert px1 <= hx1 <= hx2 <= px2
    assert py1 <= hy1 <= hy2 <= py2 + EPS


@settings(max_examples=200, deadline=None)
@given(a=any_boxes(), b=any_boxes())
def test_iou_and_ioh_stay_within_unit_interval(a, b) -> None:
    """iou 与 ioh 取值在 [0, 1]，包括退化框。"""
    assert 0.0 <= iou(a, b) <= 1.0
    assert 0.0 <= ioh(a, b) <= 1.0


@settings(max_examples=200, deadline=None)
@given(a=any_boxes(), b=any_boxes())
def test_iou_is_symmetric(a, b) -> None:
    """iou(a, b) == iou(b, a)。"""
    assert iou(a, b) == iou(b, a)


@settings(max_examples=200, deadline=None)
@given(data=helmet_inside_head())
def test_ioh_is_one_when_helmet_lies_inside_head_region(data) -> None:
    """helmet 框完全位于头部区域内时 ioh == 1。"""
    person, ratio, helmet = data
    head = head_region(person, ratio)

    assert ioh(helmet, head) == pytest.approx(1.0, rel=1e-12)


def test_point_in_polygon_treats_boundary_as_inside() -> None:
    """边界上的点（含顶点、边上的点）视为在多边形内。"""
    square = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]

    assert point_in_polygon((2.0, 0.0), square)  # 边上
    assert point_in_polygon((0.0, 0.0), square)  # 顶点
    assert point_in_polygon((2.0, 2.0), square)  # 内部
    assert not point_in_polygon((5.0, 2.0), square)  # 外部
