"""区域闯入与人员滞留规则（纯状态逻辑）。

本模块不读取配置，所有参数以普通参数传入。参考点与 ROI 判定复用 ``geometry``。

按 ``(track_id, roi_id)`` 维护一段“停留”（stay）：

- 参考点由 ROI 外变为 ROI 内时开始一段停留，并产生一条 ``intrusion`` 事件。
- 参考点在 ROI 内的已处理帧刷新停留的最后在内时间。
- 距最后在内时间超过 ``roi_exit_grace_s`` 后，停留结束（离开确认）。宽限内的抖动或检测缺失不会结束停留。
- 停留期间，在内帧上若 ``ts - entry_ms >= dwell_threshold_s``，产生一条 ``loitering`` 事件，每段停留最多一次。

时间戳为 UTC 毫秒整数，调用方需保证按时间顺序传入帧。循环重启后应调用 ``reset``。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Sequence, Tuple

from .geometry import Box, Point, RefMode, point_in_polygon, ref_point

ZoneEventType = Literal["intrusion", "loitering"]

__all__ = ["RoiSpec", "TrackObservation", "ZoneEvent", "IntrusionRule"]


@dataclass(frozen=True)
class RoiSpec:
    """一个待判定 ROI。``type`` 必须为 ``"intrusion"``，由调用方先行过滤。"""

    id: str
    type: str
    polygon: Sequence[Point]


@dataclass(frozen=True)
class TrackObservation:
    """单帧内某个 person 轨迹的检测框。"""

    track_id: int
    box: Box


@dataclass(frozen=True)
class ZoneEvent:
    """闯入或滞留事件。``intrusion`` 的 start_ms 与 end_ms 相同，均为进入帧时间。"""

    type: ZoneEventType
    roi_id: str
    track_id: int
    start_ms: int
    end_ms: int


@dataclass
class _Stay:
    entry_ms: int
    last_inside_ms: int
    loitering_fired: bool = False


class IntrusionRule:
    """区域闯入与人员滞留的状态机，每路 Camera 一个实例。"""

    def __init__(
        self,
        *,
        reference_point: RefMode,
        roi_exit_grace_s: float,
        dwell_threshold_s: float,
        rois: Sequence[RoiSpec],
    ) -> None:
        """
        Args:
            reference_point: 参考点模式，``"bottom_center"`` 或 ``"center"``。
            roi_exit_grace_s: 离开宽限秒数，必须不小于 0。
            dwell_threshold_s: 滞留阈值秒数，必须不小于 0。
            rois: 仅含 ``type == "intrusion"`` 的 ROI 列表。

        Raises:
            ValueError: 参考点模式非法、时长为负、ROI 类型不是 intrusion、或 ROI id 重复。
        """
        ref_point((0.0, 0.0, 0.0, 0.0), reference_point)  # 非法模式在此处抛出
        if roi_exit_grace_s < 0:
            raise ValueError(f"roi_exit_grace_s must be >= 0, got {roi_exit_grace_s}")
        if dwell_threshold_s < 0:
            raise ValueError(f"dwell_threshold_s must be >= 0, got {dwell_threshold_s}")
        for roi in rois:
            if roi.type != "intrusion":
                raise ValueError(f"IntrusionRule only accepts 'intrusion' ROIs, got {roi.type!r} for {roi.id!r}")
        roi_ids = [roi.id for roi in rois]
        if len(set(roi_ids)) != len(roi_ids):
            raise ValueError(f"duplicate ROI ids: {roi_ids}")

        self._mode: RefMode = reference_point
        self._grace_ms = round(roi_exit_grace_s * 1000)
        self._dwell_ms = round(dwell_threshold_s * 1000)
        self._rois: Tuple[RoiSpec, ...] = tuple(rois)
        self._stays: Dict[Tuple[int, str], _Stay] = {}

    def reset(self) -> None:
        """清空全部停留状态，用于视频循环重启。"""
        self._stays.clear()

    def process(self, ts_ms: int, observations: Sequence[TrackObservation]) -> List[ZoneEvent]:
        """处理一个已处理帧，返回本帧新产生的事件。

        先结束已超过宽限未在内的停留，再依次处理本帧的每个轨迹。
        """
        self._expire(ts_ms)
        events: List[ZoneEvent] = []
        for obs in observations:
            point = ref_point(obs.box, self._mode)
            for roi in self._rois:
                if not point_in_polygon(point, roi.polygon):
                    continue
                key = (obs.track_id, roi.id)
                stay = self._stays.get(key)
                if stay is None:
                    stay = _Stay(entry_ms=ts_ms, last_inside_ms=ts_ms)
                    self._stays[key] = stay
                    events.append(ZoneEvent("intrusion", roi.id, obs.track_id, ts_ms, ts_ms))
                else:
                    stay.last_inside_ms = ts_ms
                if not stay.loitering_fired and ts_ms - stay.entry_ms >= self._dwell_ms:
                    stay.loitering_fired = True
                    events.append(ZoneEvent("loitering", roi.id, obs.track_id, stay.entry_ms, ts_ms))
        return events

    def _expire(self, ts_ms: int) -> None:
        """删除距最后在内时间超过宽限的停留。"""
        expired = [key for key, stay in self._stays.items() if ts_ms - stay.last_inside_ms > self._grace_ms]
        for key in expired:
            del self._stays[key]
