"""冷却去重状态表（需求 3.9，design.md「冷却去重（3.9）」）。

``CooldownTable`` 以 ``(event_type, target_key)`` 记录上次被接受的触发时间：

- person 类事件的 ``target_key`` 由调用方给出，形如 ``track:{track_id}``；
- 烟火事件按点位去重：``try_accept_spot`` 在同类事件的已记录点位中查找与
  当前框 IoU ``≥ same_spot_iou_threshold`` 的点位，命中则套用同一冷却规则；
  未命中则新建点位。点位记录最近一次被接受事件的框。

规则：``ts - last < cooldown_s`` 的触发被丢弃，且被丢弃的触发不更新记录。
同一键的触发时间早于上次被接受的时间视为非法输入，抛出 ``ValueError``。

本模块为纯状态，不依赖配置。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Hashable, List, Tuple

from chimera_patrol.rules.geometry import Box, iou

__all__ = ["CooldownTable"]

_log = logging.getLogger(__name__)

# 不活跃淘汰：点位记录超过 10 分钟未被命中即删除；键记录仅在超过 max(10 分钟, cooldown) 后删除，不会缩短冷却。
INACTIVE_HORIZON_MS = 600_000
_PRUNE_INTERVAL_MS = 60_000


@dataclass
class _SpotRecord:
    """烟火点位记录：最近一次被接受事件的时间与框。"""

    last_ms: int
    last_box: Box


class CooldownTable:
    """按 ``(event_type, target_key)`` 的冷却去重表。

    Args:
        cooldown_s: 冷却时长（秒），必须 ``≥ 0``。

    Raises:
        ValueError: ``cooldown_s`` 为负数或非有限值。
    """

    def __init__(self, cooldown_s: float) -> None:
        if not (cooldown_s >= 0.0) or cooldown_s == float("inf"):
            raise ValueError(f"cooldown_s must be a finite value >= 0, got {cooldown_s!r}")
        # 规整浮点误差：例如 2.007 * 1000 得到 2007.0000000000002，会使恰好等于冷却时长的触发被拒绝。
        self._cooldown_ms: float = round(float(cooldown_s) * 1000.0, 6)
        self._last_ms: Dict[Tuple[str, Hashable], int] = {}
        self._spots: Dict[str, List[_SpotRecord]] = {}
        self._last_prune_ms: int | None = None

    @property
    def cooldown_ms(self) -> float:
        """冷却时长（毫秒）。"""
        return self._cooldown_ms

    def try_accept(self, event_type: str, target_key: Hashable, ts_ms: int) -> bool:
        """对非点位目标（如 ``track:{id}``）尝试接受一次触发。

        首次触发或 ``ts_ms - last >= cooldown`` 时接受并记录 ``ts_ms``；
        否则丢弃且不改变记录。

        Args:
            event_type: 事件类型。
            target_key: 目标键，任意可哈希值。
            ts_ms: 触发时间（毫秒）。

        Returns:
            接受返回 True，丢弃返回 False。

        Raises:
            ValueError: ``ts_ms`` 早于该键上次被接受的时间。
        """
        self._maybe_prune(ts_ms)
        key = (event_type, target_key)
        last = self._last_ms.get(key)
        if last is None:
            self._last_ms[key] = ts_ms
            return True
        if ts_ms < last:
            raise ValueError(
                f"ts_ms {ts_ms} is earlier than last accepted {last} for {key!r}"
            )
        if ts_ms - last >= self._cooldown_ms:
            self._last_ms[key] = ts_ms
            return True
        _log.debug(
            "cooldown discard: event_type=%s target=%r ts_ms=%d last_ms=%d",
            event_type, target_key, ts_ms, last,
        )
        return False

    def try_accept_spot(
        self,
        event_type: str,
        box: Box,
        ts_ms: int,
        same_spot_iou_threshold: float,
    ) -> bool:
        """对烟火等点位目标尝试接受一次触发。

        在同 ``event_type`` 的已记录点位中，选取与 ``box`` 的 IoU 最大且
        ``≥ same_spot_iou_threshold`` 的点位（IoU 相同时取先记录者）。

        - 命中：按冷却规则判定；接受则更新该点位的时间与框，丢弃则不更新。
        - 未命中：接受并新建点位记录。

        Args:
            event_type: 事件类型。
            box: 本次事件的像素框 ``(x1, y1, x2, y2)``。
            ts_ms: 触发时间（毫秒）。
            same_spot_iou_threshold: 判定同一点位的 IoU 阈值。

        Returns:
            接受返回 True，丢弃返回 False。

        Raises:
            ValueError: 命中点位的上次接受时间晚于 ``ts_ms``。
        """
        self._maybe_prune(ts_ms)
        records = self._spots.setdefault(event_type, [])

        best: _SpotRecord | None = None
        best_iou = -1.0
        for rec in records:
            overlap = iou(rec.last_box, box)
            if overlap >= same_spot_iou_threshold and overlap > best_iou:
                best = rec
                best_iou = overlap

        if best is None:
            records.append(_SpotRecord(last_ms=ts_ms, last_box=box))
            return True

        if ts_ms < best.last_ms:
            raise ValueError(
                f"ts_ms {ts_ms} is earlier than last accepted {best.last_ms} "
                f"for spot of {event_type!r}"
            )
        if ts_ms - best.last_ms >= self._cooldown_ms:
            best.last_ms = ts_ms
            best.last_box = box
            return True
        _log.debug(
            "cooldown discard: event_type=%s spot ts_ms=%d last_ms=%d",
            event_type, ts_ms, best.last_ms,
        )
        return False

    def prune(self, now_ms: int) -> int:
        """删除不活跃的记录，返回删除的记录数。

        点位记录：`now_ms - last_ms > max(10 分钟, cooldown)` 时删除。
        键记录：`now_ms - last > max(10 分钟, cooldown)` 时删除，因此淘汰不会缩短任何冷却。
        """
        horizon = max(float(INACTIVE_HORIZON_MS), self._cooldown_ms)
        removed = 0
        for event_type, records in self._spots.items():
            kept = [rec for rec in records if now_ms - rec.last_ms <= horizon]
            removed += len(records) - len(kept)
            self._spots[event_type] = kept
        stale_keys = [key for key, last in self._last_ms.items() if now_ms - last > horizon]
        for key in stale_keys:
            del self._last_ms[key]
        removed += len(stale_keys)
        self._last_prune_ms = now_ms
        return removed

    def _maybe_prune(self, now_ms: int) -> None:
        if self._last_prune_ms is None or now_ms - self._last_prune_ms >= _PRUNE_INTERVAL_MS:
            self.prune(now_ms)
    def reset(self) -> None:
        """清空所有记录（循环重启时调用）。"""
        self._last_ms.clear()
        self._spots.clear()
