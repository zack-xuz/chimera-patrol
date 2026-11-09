"""/api/stats 的框架无关处理函数（需求 8.1、8.2；设计 StatsDTO）。

本模块不构建 FastAPI 应用，路由注册由 13.1 完成。参数非法时抛出 ``ValueError``，
由调用方映射为 422 ``validation_error``。
"""

from __future__ import annotations

from typing import Any

from chimera_patrol.store.repository import EventRepository
from chimera_patrol.store.stats_sql import compute_stats_sql


def get_stats(
    repo: EventRepository,
    camera_id: str | None = None,
    from_ms: int | None = None,
    to_ms: int | None = None,
    include_sop: bool = False,
) -> dict[str, Any]:
    """校验查询参数并返回统计字典（totals、by_camera、by_type）。

    - ``from_ms`` 与 ``to_ms`` 均提供时，要求 ``from_ms <= to_ms``。
    - ``include_sop`` 默认 False，即排除 ``sop_violation``。
    """
    if from_ms is not None and to_ms is not None and from_ms > to_ms:
        raise ValueError("from_ms must be <= to_ms")
    return compute_stats_sql(
        repo,
        camera_id=camera_id,
        from_ms=from_ms,
        to_ms=to_ms,
        include_sop=include_sop,
    )
