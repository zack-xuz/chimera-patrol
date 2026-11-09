"""Property 7: 安全帽关联（helmet association）。

对任意 person 框集合、helmet 检测集合（含任意置信度）、``head_region_ratio``、
度量（``ioh`` 或 ``iou``）和阈值，``associate_helmets`` 满足：

(a) 置信度低于 ``conf_threshold`` 的 helmet 不会被关联；
(b) 一个 person 被关联，当且仅当存在与其头部区域重叠度 ``>= overlap_threshold`` 的
    达标 helmet，且该 helmet 未被其他 person 占用；
(c) 一个 helmet 至多关联一个 person（一对一）。

参考实现独立于被测模块的排序逻辑：重叠度直接用 ``geometry.head_region``、``geometry.ioh``、
``geometry.iou`` 重算，并用“候选全部被他人占用”的极大性条件检验 (b)。

**Validates: Requirements 3.6**
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from hypothesis import given, settings, strategies as st

from chimera_patrol.rules import geometry
from chimera_patrol.rules.helmet_assoc import associate_helmets

Box = Tuple[float, float, float, float]

# 坐标取整数网格上的值，使退化框（零宽或零高）与恰好等于阈值的重叠度频繁出现。
_COORD = st.integers(min_value=0, max_value=12).map(float)


@st.composite
def boxes(draw) -> Box:
    """生成 x1 <= x2、y1 <= y2 的框，允许零面积的退化框。"""
    x1 = draw(_COORD)
    x2 = draw(st.integers(min_value=int(x1), max_value=12).map(float))
    y1 = draw(_COORD)
    y2 = draw(st.integers(min_value=int(y1), max_value=12).map(float))
    return (x1, y1, x2, y2)


# 置信度与阈值混合离散值（含边界 0 与 1）和连续值。
_UNIT = st.one_of(
    st.sampled_from([0.0, 0.25, 0.5, 0.75, 1.0]),
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)


@st.composite
def helmet_detection(draw) -> Tuple[Box, float, str]:
    box = draw(boxes())
    conf = draw(_UNIT)
    kind = draw(st.sampled_from(["helmet", "helmet", "no_helmet"]))
    return (box, conf, kind)


@st.composite
def scenarios(draw):
    persons: List[Tuple[int, Box]] = [
        (track_id, draw(boxes())) for track_id in range(draw(st.integers(0, 6)))
    ]
    helmets = draw(st.lists(helmet_detection(), min_size=0, max_size=8))
    ratio = draw(st.floats(min_value=0.01, max_value=1.0, allow_nan=False))
    metric = draw(st.sampled_from(["ioh", "iou"]))
    overlap_thr = draw(_UNIT)
    conf_thr = draw(_UNIT)
    return persons, helmets, ratio, metric, overlap_thr, conf_thr


def _overlap(helmet_box: Box, head: Box, metric: str) -> float:
    fn = geometry.ioh if metric == "ioh" else geometry.iou
    return fn(helmet_box, head)


@settings(max_examples=200, deadline=None)
@given(scenarios())
def test_helmet_association_properties(scenario) -> None:
    persons, helmets, ratio, metric, overlap_thr, conf_thr = scenario

    result: Dict[int, int] = associate_helmets(
        persons, helmets, ratio, metric, overlap_thr, conf_thr
    )

    person_ids = {tid for tid, _ in persons}
    eligible = {
        idx
        for idx, (_box, conf, kind) in enumerate(helmets)
        if kind == "helmet" and conf >= conf_thr
    }

    # 结果只包含入参中的 person，且下标指向入参 helmets 中的有效检测。
    assert set(result.keys()) <= person_ids
    for tid, h_idx in result.items():
        assert 0 <= h_idx < len(helmets)

    # (a) 低于置信度阈值（或非 helmet 类型）的检测不会出现在结果中。
    for h_idx in result.values():
        _box, conf, kind = helmets[h_idx]
        assert conf >= conf_thr, f"low-confidence helmet {h_idx} associated"
        assert kind == "helmet", f"non-helmet detection {h_idx} associated"

    # (c) 一对一：每个 helmet 至多被一个 person 关联。
    values = list(result.values())
    assert len(values) == len(set(values)), "helmet associated to more than one person"

    # (b) 关联的正确性：被关联的 (person, helmet) 重叠度达到阈值。
    for tid, h_idx in result.items():
        person_box = dict(persons)[tid]
        head = geometry.head_region(person_box, ratio)
        overlap = _overlap(helmets[h_idx][0], head, metric)
        assert overlap >= overlap_thr, (
            f"person {tid} linked to helmet {h_idx} with overlap {overlap} < {overlap_thr}"
        )

    # (b) 完备性（极大性）：未被关联的 person，其所有达标候选 helmet 都必须已被他人占用。
    # 等价于：不存在“未被关联且未被占用”的达标候选，即 person 未关联就一定是候选全被占用。
    taken = set(values)
    for tid, person_box in persons:
        if tid in result:
            continue
        head = geometry.head_region(person_box, ratio)
        for h_idx in eligible:
            overlap = _overlap(helmets[h_idx][0], head, metric)
            if overlap >= overlap_thr:
                assert h_idx in taken, (
                    f"person {tid} unassociated although helmet {h_idx} "
                    f"(overlap {overlap}) is eligible and free"
                )


@settings(max_examples=200, deadline=None)
@given(
    st.lists(boxes(), min_size=1, max_size=5),
    st.lists(helmet_detection(), min_size=0, max_size=6),
    st.sampled_from(["ioh", "iou"]),
    _UNIT,
    _UNIT,
)
def test_helmet_association_with_degenerate_helmets(
    person_boxes: Sequence[Box],
    helmets: List[Tuple[Box, float, str]],
    metric: str,
    overlap_thr: float,
    conf_thr: float,
) -> None:
    """零面积 helmet 的 ioh/iou 为 0，只有阈值为 0 时才可能被关联，且仍需满足一对一。"""
    persons = list(enumerate(person_boxes))
    result = associate_helmets(persons, helmets, 0.5, metric, overlap_thr, conf_thr)

    for _tid, h_idx in result.items():
        box, conf, kind = helmets[h_idx]
        assert kind == "helmet" and conf >= conf_thr
        if (box[2] - box[0]) * (box[3] - box[1]) <= 0:
            assert overlap_thr == 0.0, "zero-area helmet associated with positive threshold"

    values = list(result.values())
    assert len(values) == len(set(values))
