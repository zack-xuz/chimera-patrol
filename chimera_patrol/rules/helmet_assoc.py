"""安全帽关联：把 person 的头部区域与 helmet 检测框做一对一贪心配对（需求 3.5、3.6，决策 D2/D5）。

本模块是纯函数，不读取配置。所有阈值与度量由调用方以普通参数传入。

输入约定：

- ``persons``：``(track_id, person_box)`` 序列。``person_box`` 为 ``(x1, y1, x2, y2)``。
- ``helmets``：``(box, conf, kind)`` 序列。``box`` 同上，``conf`` 为 0~1 的置信度，
  ``kind`` 为 ``"helmet"`` 或 ``"no_helmet"``。只有 ``kind == "helmet"`` 的检测参与关联，
  ``"no_helmet"`` 检测直接忽略（它不用于关联，由违规帧逻辑另行使用）。

输出约定：

- 返回 ``{track_id: helmet_index}``，``helmet_index`` 为该 helmet 在入参 ``helmets``
  序列中的下标（从 0 开始，与是否被过滤无关）。未关联到安全帽的 person 不出现在结果中。
"""

from __future__ import annotations

from typing import Dict, Hashable, Literal, Sequence, Tuple

from chimera_patrol.rules import geometry
from chimera_patrol.rules.geometry import Box

AssocMetric = Literal["ioh", "iou"]
HelmetKind = Literal["helmet", "no_helmet"]
HelmetDetection = Tuple[Box, float, str]
PersonEntry = Tuple[Hashable, Box]

__all__ = ["associate_helmets", "AssocMetric", "HelmetDetection", "PersonEntry"]

_VALID_METRICS = ("ioh", "iou")


def _check_unit_interval(name: str, value: float) -> None:
    """要求 ``value`` 位于闭区间 [0, 1]，否则抛出 ValueError。"""
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")


def associate_helmets(
    persons: Sequence[PersonEntry],
    helmets: Sequence[HelmetDetection],
    head_region_ratio: float,
    metric: str,
    overlap_threshold: float,
    conf_threshold: float,
) -> Dict[Hashable, int]:
    """按头部区域重叠度，把安全帽一对一关联到 person。

    步骤：
      1. 丢弃 ``kind != "helmet"`` 或 ``conf < conf_threshold`` 的检测。
      2. 对每个 person 计算头部区域 ``head_region(person_box, head_region_ratio)``。
      3. 对每对（person, helmet）计算重叠度：``metric == "ioh"`` 时为
         ``ioh(helmet_box, head)``，``metric == "iou"`` 时为 ``iou(helmet_box, head)``。
      4. 重叠度不低于 ``overlap_threshold`` 的配对为候选。
      5. 候选按重叠度降序排列（并列时按 person、helmet 的入参顺序），依次分配：
         person 已有关联或 helmet 已被使用则跳过。

    Args:
        persons: person 列表，元素为 ``(track_id, box)``。
        helmets: 检测列表，元素为 ``(box, conf, kind)``。
        head_region_ratio: 头部区域占 person 框高度的比例，取值须在 (0, 1]。
        metric: 重叠度量，``"ioh"`` 或 ``"iou"``。
        overlap_threshold: 关联重叠阈值，取值须在 [0, 1]。
        conf_threshold: helmet 置信度阈值，取值须在 [0, 1]。

    Returns:
        ``{track_id: helmet_index}``，helmet_index 指向入参 ``helmets``。

    Raises:
        ValueError: 参数越界，或 ``metric`` 不是 ``"ioh"`` 与 ``"iou"`` 之一。
    """
    if not (0.0 < head_region_ratio <= 1.0):
        raise ValueError(f"head_region_ratio must be in (0, 1], got {head_region_ratio!r}")
    _check_unit_interval("overlap_threshold", overlap_threshold)
    _check_unit_interval("conf_threshold", conf_threshold)
    if metric not in _VALID_METRICS:
        raise ValueError(f"metric must be one of {_VALID_METRICS}, got {metric!r}")

    overlap_fn = geometry.ioh if metric == "ioh" else geometry.iou

    # 步骤 1：保留入参下标，过滤非 helmet 或低置信度检测。
    kept: list[Tuple[int, Box]] = [
        (idx, box)
        for idx, (box, conf, kind) in enumerate(helmets)
        if kind == "helmet" and conf >= conf_threshold
    ]

    # 步骤 2~4：收集达到阈值的候选配对。
    # 排序键为 (-重叠度, person 序号, helmet 下标)，保证结果确定。
    candidates: list[Tuple[float, int, int, Hashable]] = []
    for p_order, (track_id, person_box) in enumerate(persons):
        head = geometry.head_region(person_box, head_region_ratio)
        for helmet_idx, helmet_box in kept:
            overlap = overlap_fn(helmet_box, head)
            if overlap >= overlap_threshold:
                candidates.append((-overlap, p_order, helmet_idx, track_id))

    # 步骤 5：一对一贪心分配。
    candidates.sort(key=lambda c: (c[0], c[1], c[2]))
    assigned: Dict[Hashable, int] = {}
    used_helmets: set[int] = set()
    for _neg_overlap, _p_order, helmet_idx, track_id in candidates:
        if track_id in assigned or helmet_idx in used_helmets:
            continue
        assigned[track_id] = helmet_idx
        used_helmets.add(helmet_idx)
    return assigned
