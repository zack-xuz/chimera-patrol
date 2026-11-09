"""统计口径的纯函数实现（设计 StatsDTO / 假设 A6、A13、A14）。

本模块不访问数据库。输入为 ``EventRepository.list_events`` 返回的事件字典，
至少包含 ``camera_id``、``event_type``、``status`` 三个键。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

SOP_EVENT_TYPE = "sop_violation"

# 设计中的事件类型顺序；未知类型排在其后按字典序。
EVENT_TYPE_ORDER: tuple[str, ...] = (
    "intrusion",
    "loitering",
    "smoke_fire",
    "no_helmet",
    SOP_EVENT_TYPE,
)

COUNT_KEYS: tuple[str, ...] = (
    "confirmed",
    "dismissed",
    "needs_human",
    "review_failed",
    "pending_review",
)


def _empty_counts() -> dict[str, int]:
    return {key: 0 for key in COUNT_KEYS}


def _bucket_dto(counts: Mapping[str, int]) -> dict[str, Any]:
    """由各状态计数派生出 StatsDTO 中的一个统计桶（含派生字段）。"""
    total_alarms = sum(counts[key] for key in COUNT_KEYS)
    dismissed = counts["dismissed"]
    return {
        "total_alarms": total_alarms,
        **{key: counts[key] for key in COUNT_KEYS},
        # A13：待人工数 = needs_human + review_failed
        "pending_human": counts["needs_human"] + counts["review_failed"],
        # A14：total_alarms 为 0 时返回 None（前端显示“—”）
        "false_positive_reduction_rate": (
            dismissed / total_alarms if total_alarms > 0 else None
        ),
    }


def _sort_key_type(event_type: str) -> tuple[int, str]:
    try:
        return (EVENT_TYPE_ORDER.index(event_type), "")
    except ValueError:
        return (len(EVENT_TYPE_ORDER), event_type)


def compute_stats(events: Iterable[Mapping[str, Any]], include_sop: bool) -> dict[str, Any]:
    """按设计口径计算 totals、by_camera、by_type。

    - ``include_sop`` 为 False（默认）时排除 ``sop_violation`` 事件（A6）。
    - 状态不在五种统计状态内时抛出 ``ValueError``，避免总数与分桶口径悄悄偏离。
    - ``by_camera`` 按 ``camera_id`` 排序，``by_type`` 按设计类型顺序排列。
    """
    totals = _empty_counts()
    by_camera: dict[str, dict[str, int]] = {}
    by_type: dict[str, dict[str, int]] = {}

    for event in events:
        event_type = event["event_type"]
        if event_type == SOP_EVENT_TYPE and not include_sop:
            continue
        status = event["status"]
        if status not in COUNT_KEYS:
            raise ValueError(f"未知事件状态，无法纳入统计口径: {status!r}")
        camera_id = event["camera_id"]

        totals[status] += 1
        by_camera.setdefault(camera_id, _empty_counts())[status] += 1
        by_type.setdefault(event_type, _empty_counts())[status] += 1

    return {
        "totals": _bucket_dto(totals),
        "by_camera": [
            {"camera_id": camera_id, **_bucket_dto(by_camera[camera_id])}
            for camera_id in sorted(by_camera)
        ],
        "by_type": [
            {"event_type": event_type, **_bucket_dto(by_type[event_type])}
            for event_type in sorted(by_type, key=_sort_key_type)
        ],
    }
