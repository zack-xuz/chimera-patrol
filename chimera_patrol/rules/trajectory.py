"""每个 track 的轨迹缓冲与下采样（需求 3.10，design.md「轨迹记录（3.10）」）。

每个点为 ``(ts_ms, x1, y1, x2, y2)``。同一 track 的点按追加顺序存储，追加时间须单调不减。

淘汰：track 的最后一个点距最新时间超过 ``INACTIVE_HORIZON_MS``（10 分钟）即整体删除，
防止长时间运行时内存持续增长。淘汰在 ``append`` 时按当前时间节流执行。
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from typing import Dict, Hashable, List, Sequence, Tuple, TypeVar

from chimera_patrol.rules.geometry import Box

__all__ = ["TrajectoryPoint", "TrajectoryBuffer", "downsample", "INACTIVE_HORIZON_MS"]

TrajectoryPoint = Tuple[int, float, float, float, float]
_T = TypeVar("_T")

INACTIVE_HORIZON_MS = 600_000  # 10 分钟
_PRUNE_INTERVAL_MS = 60_000


class TrajectoryBuffer:
    """按 track 保存 ``(ts_ms, x1, y1, x2, y2)`` 轨迹点。"""

    def __init__(self) -> None:
        self._ts: Dict[Hashable, List[int]] = {}
        self._points: Dict[Hashable, List[TrajectoryPoint]] = {}
        self._latest_ms: int | None = None
        self._last_prune_ms: int | None = None

    def append(self, track_id: Hashable, ts_ms: int, box: Box) -> None:
        """追加一个轨迹点。

        Raises:
            ValueError: ``ts_ms`` 早于该 track 最后一个已记录点的时间。
        """
        timestamps = self._ts.setdefault(track_id, [])
        points = self._points.setdefault(track_id, [])
        if timestamps and ts_ms < timestamps[-1]:
            raise ValueError(
                f"ts_ms {ts_ms} is earlier than last point {timestamps[-1]} "
                f"for track {track_id!r}"
            )
        x1, y1, x2, y2 = box
        timestamps.append(ts_ms)
        points.append((ts_ms, x1, y1, x2, y2))
        if self._latest_ms is None or ts_ms > self._latest_ms:
            self._latest_ms = ts_ms
        self._maybe_prune()

    def prune(self, now_ms: int) -> int:
        """删除最后一个点距 ``now_ms`` 超过 10 分钟的 track，返回删除的 track 数。"""
        stale = [
            track_id
            for track_id, timestamps in self._ts.items()
            if timestamps and now_ms - timestamps[-1] > INACTIVE_HORIZON_MS
        ]
        for track_id in stale:
            del self._ts[track_id]
            del self._points[track_id]
        self._last_prune_ms = now_ms
        return len(stale)

    def _maybe_prune(self) -> None:
        if self._latest_ms is None:
            return
        if self._last_prune_ms is None or self._latest_ms - self._last_prune_ms >= _PRUNE_INTERVAL_MS:
            self.prune(self._latest_ms)

    def window(self, track_id: Hashable, from_ms: int, to_ms: int) -> List[TrajectoryPoint]:
        """返回 track 在 ``[from_ms, to_ms]``（含两端）内的轨迹点，按时间升序。

        未知 track 或区间内无点时返回空列表。
        """
        timestamps = self._ts.get(track_id)
        if not timestamps or from_ms > to_ms:
            return []
        lo = bisect_left(timestamps, from_ms)
        hi = bisect_right(timestamps, to_ms)
        return list(self._points[track_id][lo:hi])

    def clear(self) -> None:
        """清空所有 track 的轨迹（循环重启时调用）。"""
        self._ts.clear()
        self._points.clear()
        self._latest_ms = None
        self._last_prune_ms = None


def downsample(points: Sequence[_T], max_points: int) -> List[_T]:
    """等距下采样，保留首尾点，结果长度不超过 ``max_points``。

    若 ``len(points) <= max_points``，原样返回副本。否则选取索引
    ``round(k * (n-1) / (max_points-1))``（半数向上取整），``k = 0..max_points-1``。
    这些索引严格递增，因此首尾必然保留且无重复点。

    Raises:
        ValueError: ``max_points < 2``。
    """
    if max_points < 2:
        raise ValueError(f"max_points must be >= 2, got {max_points}")
    n = len(points)
    if n <= max_points:
        return list(points)
    span = n - 1
    steps = max_points - 1
    indices = [(2 * k * span + steps) // (2 * steps) for k in range(max_points)]
    return [points[i] for i in indices]