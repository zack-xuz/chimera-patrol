"""Rule_Engine 的输入与输出类型（design.md「核心内部类型」「关键接口」）。

坐标约定：所有框为像素坐标 ``(x1, y1, x2, y2)``，ROI 多边形与检测框处于同一坐标空间。
本模块不依赖配置。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

from chimera_patrol.rules.geometry import Box, Point
from chimera_patrol.rules.smoke_fire import SmokeFireDetection

__all__ = [
    "EventType",
    "EventStatus",
    "DetectionSource",
    "RoiType",
    "HelmetKind",
    "PersonDetection",
    "HelmetDetection",
    "FrameResult",
    "RoiSpec",
    "OverlayItem",
    "CandidateEvent",
    "SmokeFireDetection",
]

EventType = Literal["intrusion", "loitering", "smoke_fire", "no_helmet", "sop_violation"]
EventStatus = Literal["pending_review", "confirmed", "dismissed", "needs_human", "review_failed"]
DetectionSource = Literal["yolo", "vlm_fallback", "sop"]
RoiType = Literal["intrusion", "work_zone"]
HelmetKind = Literal["helmet", "no_helmet"]


@dataclass(frozen=True)
class PersonDetection:
    """单帧内一个 person 检测（已跟踪）。"""

    track_id: int
    box: Box
    conf: float


@dataclass(frozen=True)
class HelmetDetection:
    """单帧内一个安全帽相关检测。``kind`` 为 ``helmet`` 或 ``no_helmet``。"""

    kind: HelmetKind
    box: Box
    conf: float


@dataclass(frozen=True)
class FrameResult:
    """一个已处理帧的检测结果，``ts_ms`` 为 UTC 毫秒时间戳。"""

    ts_ms: int
    frame_index: int
    persons: list[PersonDetection] = field(default_factory=list)
    smoke_fire: list[SmokeFireDetection] = field(default_factory=list)
    helmets: list[HelmetDetection] = field(default_factory=list)


@dataclass(frozen=True)
class RoiSpec:
    """一个 ROI。``polygon`` 为按顺序排列的顶点，顶点数不少于 3。"""

    id: str
    type: RoiType
    polygon: Sequence[Point]


@dataclass(frozen=True)
class OverlayItem:
    """关键帧上的一个标注框。``roi_polygon`` 为可选的 ROI 多边形。"""

    box: Box
    label: str
    conf: float | None = None
    roi_polygon: Sequence[Point] | None = None


@dataclass
class CandidateEvent:
    """规则引擎产生的候选事件，尚未落库。

    ``trajectory`` 中每个点为 ``[ts_ms, x1, y1, x2, y2]``。
    ``sop_run_id`` 仅由 SOP 检查填写，规则引擎不设置。
    """

    camera_id: str
    event_type: EventType
    detection_source: DetectionSource
    start_ms: int
    end_ms: int
    cv_confidence: float | None
    track_id: int | None
    roi_id: str | None
    roi_type: RoiType | None
    detail: dict[str, Any]
    trajectory: list[list[float]]
    overlay: list[OverlayItem]
    sop_run_id: str | None = None
