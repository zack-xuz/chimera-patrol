"""烟火点位跟踪（需求 3.4，设计决策 D4、D5）。

每个已处理帧输入一组烟火检测。同一 ``label`` 且 IoU 不低于
``same_spot_iou_threshold`` 的检测视为同一处（点位）。点位连续
``consecutive_frames`` 个已处理帧命中后，触发一次 ``smoke_fire`` 事件；
同一连续点位之后不再重复触发。

本模块为纯状态，不读取配置。阈值由调用方以参数传入。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

from chimera_patrol.rules.geometry import Box, iou

__all__ = ["SmokeFireDetection", "SmokeFireEvent", "SmokeFireTracker"]


@dataclass(frozen=True)
class SmokeFireDetection:
    """单帧内的一个烟火检测结果。"""

    label: str
    box: Box
    conf: float


@dataclass(frozen=True)
class SmokeFireEvent:
    """点位连续命中达到阈值时产生的事件。

    Attributes:
        label: 检测类别（smoke 或 fire）。
        spot_id: 点位编号，同一 tracker 内唯一。
        start_ms: 连续段首帧时间戳（毫秒）。
        end_ms: 触发帧时间戳（毫秒）。
        conf: 触发帧上匹配检测的置信度。
        box: 触发帧上匹配检测的框。
    """

    label: str
    spot_id: int
    start_ms: int
    end_ms: int
    conf: float
    box: Box


class _Spot:
    """内部点位状态：类别、最近框、连续命中帧数、起始时间与是否已触发。"""

    __slots__ = ("spot_id", "label", "last_box", "start_ms", "streak", "fired")

    def __init__(self, spot_id: int, label: str, box: Box, start_ms: int) -> None:
        self.spot_id = spot_id
        self.label = label
        self.last_box = box
        self.start_ms = start_ms
        self.streak = 0
        self.fired = False


class SmokeFireTracker:
    """按已处理帧维护烟火点位并产生 ``smoke_fire`` 事件。

    Args:
        conf_threshold: 烟火置信度阈值，低于该值的检测被丢弃（D5，与安全帽阈值独立）。
        consecutive_frames: 点位需连续命中的已处理帧数，至少为 1。
        same_spot_iou_threshold: 判定为同一点位的最小 IoU（D4），取值 [0, 1]。

    Raises:
        ValueError: 任一参数越界。
    """

    def __init__(
        self,
        conf_threshold: float,
        consecutive_frames: int,
        same_spot_iou_threshold: float,
    ) -> None:
        if not 0.0 <= conf_threshold <= 1.0:
            raise ValueError(f"conf_threshold must be in [0, 1], got {conf_threshold!r}")
        if isinstance(consecutive_frames, bool) or not isinstance(consecutive_frames, int):
            raise ValueError(f"consecutive_frames must be an int, got {consecutive_frames!r}")
        if consecutive_frames < 1:
            raise ValueError(f"consecutive_frames must be >= 1, got {consecutive_frames!r}")
        if not 0.0 <= same_spot_iou_threshold <= 1.0:
            raise ValueError(
                f"same_spot_iou_threshold must be in [0, 1], got {same_spot_iou_threshold!r}"
            )
        self._conf_threshold = conf_threshold
        self._consecutive_frames = consecutive_frames
        self._iou_threshold = same_spot_iou_threshold
        self._spots: List[_Spot] = []
        self._next_spot_id = 1

    def process(
        self, ts_ms: int, detections: Sequence[SmokeFireDetection]
    ) -> List[SmokeFireEvent]:
        """处理一个已处理帧，返回本帧新触发的事件。

        Args:
            ts_ms: 该帧的时间戳（毫秒）。
            detections: 该帧的烟火检测列表，可包含低置信度项（会被过滤）。

        Returns:
            本帧触发的 ``SmokeFireEvent`` 列表，无触发时为空列表。
        """
        kept = [d for d in detections if d.conf >= self._conf_threshold]

        # 收集所有合格的 (IoU, 点位下标, 检测下标) 配对，按 IoU 降序做一对一贪心匹配。
        candidates: List[Tuple[float, int, int]] = []
        for si, spot in enumerate(self._spots):
            for di, det in enumerate(kept):
                if det.label != spot.label:
                    continue
                overlap = iou(spot.last_box, det.box)
                if overlap >= self._iou_threshold:
                    candidates.append((overlap, si, di))
        candidates.sort(key=lambda c: (-c[0], c[1], c[2]))

        spot_to_det: Dict[int, int] = {}
        matched_dets: Set[int] = set()
        for _, si, di in candidates:
            if si in spot_to_det or di in matched_dets:
                continue
            spot_to_det[si] = di
            matched_dets.add(di)

        events: List[SmokeFireEvent] = []
        next_spots: List[_Spot] = []

        # 未匹配的点位在此被丢弃（连续被中断）。
        for si, spot in enumerate(self._spots):
            di = spot_to_det.get(si)
            if di is None:
                continue
            det = kept[di]
            spot.streak += 1
            spot.last_box = det.box
            event = self._maybe_fire(spot, ts_ms, det)
            if event is not None:
                events.append(event)
            next_spots.append(spot)

        # 未匹配的检测新建点位，连续段从本帧开始。
        for di, det in enumerate(kept):
            if di in matched_dets:
                continue
            spot = _Spot(self._next_spot_id, det.label, det.box, start_ms=ts_ms)
            self._next_spot_id += 1
            spot.streak = 1
            event = self._maybe_fire(spot, ts_ms, det)
            if event is not None:
                events.append(event)
            next_spots.append(spot)

        self._spots = next_spots
        return events

    def _maybe_fire(
        self, spot: _Spot, ts_ms: int, det: SmokeFireDetection
    ) -> SmokeFireEvent | None:
        """点位达到连续帧数且未触发过时，标记为已触发并返回事件。"""
        if spot.fired or spot.streak < self._consecutive_frames:
            return None
        spot.fired = True
        return SmokeFireEvent(
            label=spot.label,
            spot_id=spot.spot_id,
            start_ms=spot.start_ms,
            end_ms=ts_ms,
            conf=det.conf,
            box=det.box,
        )
