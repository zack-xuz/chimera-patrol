"""SQL 聚合版统计（设计 StatsDTO / 假设 A6、A13、A14）。

与纯函数 ``chimera_patrol.api.stats.compute_stats`` 采用同一过滤口径与派生字段，
两者对同一数据的输出必须一致。分桶由 GROUP BY 得到，派生字段复用纯函数模块中的实现。
"""

from __future__ import annotations

from typing import Any

from chimera_patrol.api.stats import (
    COUNT_KEYS,
    SOP_EVENT_TYPE,
    _bucket_dto,
    _empty_counts,
    _sort_key_type,
)
from chimera_patrol.store.repository import EventRepository, _parse_multi

STATS_GROUP_SQL = """
SELECT camera_id, event_type, status, COUNT(*) AS n
FROM events
{where}
GROUP BY camera_id, event_type, status
ORDER BY camera_id, event_type, status
"""


def _build_where(
    camera_id: str | None,
    from_ms: int | None,
    to_ms: int | None,
    include_sop: bool,
) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    cameras = _parse_multi(camera_id, None, "camera_id")
    if cameras:
        clauses.append(f"camera_id IN ({','.join('?' * len(cameras))})")
        params.extend(cameras)
    if from_ms is not None:
        clauses.append("created_ms >= ?")
        params.append(from_ms)
    if to_ms is not None:
        clauses.append("created_ms <= ?")
        params.append(to_ms)
    if not include_sop:
        clauses.append("event_type <> ?")
        params.append(SOP_EVENT_TYPE)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    return where, params


def compute_stats_sql(
    repo: EventRepository,
    *,
    camera_id: str | None = None,
    from_ms: int | None = None,
    to_ms: int | None = None,
    include_sop: bool = False,
) -> dict[str, Any]:
    """用 SQL GROUP BY 聚合事件，返回与 ``compute_stats`` 相同结构的字典。

    过滤口径：
    - ``camera_id`` 支持逗号分隔多值（取并集）。
    - ``from_ms`` / ``to_ms`` 作用于 ``created_ms``，两端均为闭区间。
    - ``include_sop`` 为 False 时排除 ``sop_violation``（A6）。
    调用方需保证 ``from_ms <= to_ms``。
    """
    if from_ms is not None and to_ms is not None and from_ms > to_ms:
        raise ValueError("from_ms must be <= to_ms")

    where, params = _build_where(camera_id, from_ms, to_ms, include_sop)
    rows = repo.read_rows(STATS_GROUP_SQL.format(where=where), params)

    totals = _empty_counts()
    by_camera: dict[str, dict[str, int]] = {}
    by_type: dict[str, dict[str, int]] = {}
    for row in rows:
        status = row["status"]
        if status not in COUNT_KEYS:
            raise ValueError(f"未知事件状态，无法纳入统计口径: {status!r}")
        count = int(row["n"])
        event_type = row["event_type"]
        totals[status] += count
        by_camera.setdefault(row["camera_id"], _empty_counts())[status] += count
        by_type.setdefault(event_type, _empty_counts())[status] += count

    return {
        "totals": _bucket_dto(totals),
        "by_camera": [
            {"camera_id": cid, **_bucket_dto(by_camera[cid])} for cid in sorted(by_camera)
        ],
        "by_type": [
            {"event_type": et, **_bucket_dto(by_type[et])}
            for et in sorted(by_type, key=_sort_key_type)
        ],
    }
