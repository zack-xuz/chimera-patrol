"""安全帽 VLM 兜底路径（需求 3.7，任务 11.2）。

设计：RuleEngine.process 保持同步。兜底分三步，由调用方串起来：

1. ``RuleEngine.collect_fallback_requests(frame_result)`` 返回本帧需要检查头部的作用域内 person
   （未关联到安全帽、且该 track 距上次请求已满 ``interval_s``）。调用方不需要做任何额外判断。
2. 调用方用 ``crop_head_region`` 从帧图像裁剪头部区域，调用 ``VLM_Adapter.check_helmet_crop``。
   调用失败或超时只记录日志，不产生事件。
3. ``RuleEngine.apply_fallback_result(request, verdict, confidence)`` 把结果转为候选事件。
   仅当 verdict 为 ``no_helmet`` 且置信度不低于 ``min_confidence`` 时产生 ``detection_source="vlm_fallback"``
   的 ``no_helmet`` 候选，并与权重路径共用同一冷却键 ``track:{id}``。

间隔按已处理帧的时间戳计算（ts_ms），不使用墙钟。开关：``rules.helmet.vlm_fallback.enabled``
且 ``vlm.review_enabled`` 同时为真时才会产生请求。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Hashable, Optional

import numpy as np

from chimera_patrol.rules.geometry import Box, head_region

__all__ = ["FallbackRequest", "FallbackSchedule", "crop_head_region"]


@dataclass(frozen=True)
class FallbackRequest:
    """一次头部检查请求。``box`` 为 person 像素框，``ts_ms`` 为已处理帧时间戳。"""

    track_id: int
    ts_ms: int
    box: Box


class FallbackSchedule:
    """按 track 限制兜底请求频率：同一 track 两次请求至少相隔 ``interval_ms``。"""

    def __init__(self, interval_ms: int) -> None:
        if isinstance(interval_ms, bool) or not isinstance(interval_ms, int) or interval_ms <= 0:
            raise ValueError(f"interval_ms must be a positive int, got {interval_ms!r}")
        self._interval_ms = interval_ms
        self._last_ms: Dict[Hashable, int] = {}

    def due(self, track_id: Hashable, ts_ms: int) -> bool:
        """若该 track 尚未请求过，或距上次请求已达到间隔，则记录并返回 True。"""
        last = self._last_ms.get(track_id)
        if last is None or ts_ms - last >= self._interval_ms:
            self._last_ms[track_id] = ts_ms
            return True
        return False

    def reset(self) -> None:
        self._last_ms.clear()


def crop_head_region(
    frame: np.ndarray,
    person_box: Box,
    head_region_ratio: float,
    padding_ratio: float,
) -> Optional[np.ndarray]:
    """从帧中裁剪头部区域，四周按 ``padding_ratio`` 外扩并裁到画面范围内。

    返回裁剪图副本；裁剪结果为空（区域完全在画面外）时返回 None。
    """
    if frame is None or frame.ndim < 2:
        return None
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = head_region(person_box, head_region_ratio)
    pad_x = (x2 - x1) * padding_ratio
    pad_y = (y2 - y1) * padding_ratio
    left = max(0, int(math.floor(x1 - pad_x)))
    top = max(0, int(math.floor(y1 - pad_y)))
    right = min(width, int(math.ceil(x2 + pad_x)))
    bottom = min(height, int(math.ceil(y2 + pad_y)))
    if right <= left or bottom <= top:
        return None
    return frame[top:bottom, left:right].copy()