"""每路 Camera 独立的 person 跟踪器（需求 2.2，design 决策：每路独立 ByteTrack 实例）。

API 依据：supervision 0.31.5 已移除 sv.ByteTrack（见 design.md 调研结论），因此改用独立包
``trackers`` 2.7.0 的 ``ByteTrackTracker``。其 ``__init__`` 参数与 ``update`` 签名已在该版本上核对：

- ``ByteTrackTracker(lost_track_buffer: int = 30, frame_rate: float = 30.0, ...)``：
  ``lost_track_buffer`` 以 30 FPS 为单位，库内按 ``frame_rate`` 缩放。
- ``update(detections: supervision.Detections, frame=None, timestamp=None) -> Detections``，
  返回的 ``Detections.tracker_id`` 为稳定跟踪 ID。

``lost_buffer_s``（配置 detector.tracker.lost_buffer_s，单位秒）换算规则：
``lost_track_buffer = round(lost_buffer_s * 30)``，即以 30 FPS 计的帧数，``frame_rate`` 原样传入库。
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import supervision as sv
from trackers import ByteTrackTracker

Box = Tuple[float, float, float, float]
PersonDetection = Tuple[Box, float]
TrackedPerson = Tuple[int, Box, float]

_LIBRARY_BASE_FPS = 30


def lost_buffer_frames(lost_buffer_s: float) -> int:
    """把秒换算为库所需的 30 FPS 单位缓冲帧数（至少为 1）。"""
    if lost_buffer_s <= 0:
        raise ValueError("lost_buffer_s must be > 0")
    return max(1, round(lost_buffer_s * _LIBRARY_BASE_FPS))


class PersonTracker:
    """单路 Camera 的 person 跟踪器。不同实例之间互不共享状态。"""

    def __init__(self, frame_rate: float, lost_buffer_s: float) -> None:
        if frame_rate <= 0:
            raise ValueError("frame_rate must be > 0")
        self._frame_rate = float(frame_rate)
        self._lost_buffer_s = float(lost_buffer_s)
        self._tracker = self._make()

    def _make(self) -> ByteTrackTracker:
        return ByteTrackTracker(
            lost_track_buffer=lost_buffer_frames(self._lost_buffer_s),
            frame_rate=self._frame_rate,
        )

    def reset(self) -> None:
        """丢弃全部跟踪状态，下次 update 从新的 ID 开始（视频循环重启，假设 A4）。"""
        self._tracker = self._make()

    def update(self, detections: Sequence[PersonDetection]) -> list[TrackedPerson]:
        """输入一帧的 person 检测 ``(box, conf)``，返回 ``(track_id, box, conf)``。"""
        if not detections:
            empty = sv.Detections.empty()
            self._tracker.update(empty)
            return []

        xyxy = np.array([box for box, _ in detections], dtype=np.float32).reshape(-1, 4)
        conf = np.array([score for _, score in detections], dtype=np.float32)
        det = sv.Detections(
            xyxy=xyxy,
            confidence=conf,
            class_id=np.zeros(len(detections), dtype=int),
        )
        tracked = self._tracker.update(det)

        out: list[TrackedPerson] = []
        if tracked.tracker_id is None:
            return out
        for i in range(len(tracked)):
            tid = tracked.tracker_id[i]
            if tid is None or int(tid) < 0:
                continue
            x1, y1, x2, y2 = (float(v) for v in tracked.xyxy[i])
            out.append((int(tid), (x1, y1, x2, y2), float(tracked.confidence[i])))
        return out