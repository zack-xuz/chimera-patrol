"""RuleEngine：每路 Camera 一个实例，把已处理帧的检测转换为候选事件。

每帧的处理顺序：

1. 轨迹：每个 person 观测追加到 ``TrajectoryBuffer``（所有 person，含高度过滤掉的）。
2. 闯入与滞留：``IntrusionRule``，只使用 ``intrusion`` 类型的 ROI。
3. 烟火：``SmokeFireTracker``，点位去重使用 ``CooldownTable.try_accept_spot``。
4. 安全帽：作用域（D1）→ 高度过滤 → ``associate_helmets``（D2）→ ``HelmetTimer``（D3）。
5. 冷却：闯入、滞留、未佩戴安全帽按 ``track:{track_id}`` 去重，在产生候选时应用。

只返回被冷却接受的事件。四类 CV 事件的 ``detection_source`` 均为 ``"yolo"``。

坐标约定：ROI 多边形与检测框必须处于同一像素坐标空间。``config.yaml`` 中的
归一化多边形须由调用方乘以帧宽高后再传入。

时间约定：``process`` 的 ``ts_ms`` 必须单调不减，否则抛出 ``ValueError``。

VLM 兜底（3.7）不在 ``process`` 内执行（它需要异步调用 VLM）。调用方使用
``collect_fallback_requests`` 与 ``apply_fallback_result`` 完成兜底，见 ``helmet_fallback`` 模块。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from chimera_patrol.rules import geometry
from chimera_patrol.rules.cooldown import CooldownTable
from chimera_patrol.rules.geometry import Box, Point
from chimera_patrol.rules.helmet_assoc import associate_helmets
from chimera_patrol.rules.helmet_fallback import FallbackRequest, FallbackSchedule
from chimera_patrol.rules.helmet_scope import filter_scoped_persons
from chimera_patrol.rules.helmet_timer import HelmetObservation, HelmetTimer
from chimera_patrol.rules.intrusion import IntrusionRule, RoiSpec as _ZoneSpec, TrackObservation
from chimera_patrol.rules.smoke_fire import SmokeFireTracker
from chimera_patrol.rules.trajectory import TrajectoryBuffer, downsample
from chimera_patrol.rules.types import (
    CandidateEvent,
    FrameResult,
    OverlayItem,
    PersonDetection,
    RoiSpec,
)

if TYPE_CHECKING:  # 仅用于类型标注；规则模块本身不依赖配置。
    from chimera_patrol.config import RulesSection

__all__ = ["RuleEngine"]


def seconds_to_ms(value: float) -> int:
    """配置秒数一次性换算为整数毫秒（round，避免浮点乘法的误差累积在计时中）。"""
    return round(value * 1000)

_CV_SOURCE = "yolo"
_VALID_ROI_TYPES = ("intrusion", "work_zone")


def _validate_rois(rois: Sequence[RoiSpec]) -> None:
    """检查 ROI 类型、id 唯一性与顶点数。非法时抛出 ValueError。"""
    seen: set[str] = set()
    for roi in rois:
        if roi.type not in _VALID_ROI_TYPES:
            raise ValueError(f"ROI {roi.id!r} has unsupported type {roi.type!r}")
        if roi.id in seen:
            raise ValueError(f"duplicate ROI id: {roi.id!r}")
        seen.add(roi.id)
        if len(roi.polygon) < 3:
            raise ValueError(f"ROI {roi.id!r} needs at least 3 vertices, got {len(roi.polygon)}")


def _in_box(point: Point, box: Box) -> bool:
    """判断点是否位于框内（含边界）。"""
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


class RuleEngine:
    """单路 Camera 的规则引擎。状态仅由该路已处理帧驱动。

    Args:
        camera_id: Camera ID，写入每个候选事件。
        rules: ``config.yaml`` 中 ``rules`` 节（``RulesSection``）。
        rois: 该 Camera 的 ROI 列表，类型为 ``intrusion`` 或 ``work_zone``。

    Raises:
        ValueError: ROI 类型不受支持、id 重复或顶点数不足 3。
    """

    def __init__(
        self,
        camera_id: str,
        rules: RulesSection,
        rois: Sequence[RoiSpec],
        *,
        vlm_review_enabled: bool = True,
    ) -> None:
        _validate_rois(rois)
        self._camera_id = camera_id
        self._rules = rules
        self._rois: dict[str, RoiSpec] = {roi.id: roi for roi in rois}
        self._work_zones = [roi for roi in rois if roi.type == "work_zone"]
        intrusion_rois = [
            _ZoneSpec(id=roi.id, type=roi.type, polygon=roi.polygon)
            for roi in rois
            if roi.type == "intrusion"
        ]

        self._intrusion = IntrusionRule(
            reference_point=rules.reference_point,
            roi_exit_grace_s=rules.roi_exit_grace_s,
            dwell_threshold_s=rules.dwell_threshold_s,
            rois=intrusion_rois,
        )
        self._smoke = SmokeFireTracker(
            conf_threshold=rules.smoke_fire.conf_threshold,
            consecutive_frames=rules.smoke_fire.consecutive_frames,
            same_spot_iou_threshold=rules.smoke_fire.same_spot_iou_threshold,
        )
        self._helmet_timer = HelmetTimer(
            no_helmet_duration_ms=seconds_to_ms(rules.helmet.no_helmet_duration_s),
            track_gap_reset_ms=seconds_to_ms(rules.helmet.track_gap_reset_s),
            roi_exit_grace_ms=seconds_to_ms(rules.roi_exit_grace_s),
        )
        self._cooldown = CooldownTable(rules.cooldown_s)
        self._trajectory = TrajectoryBuffer()
        self._last_ts_ms: int | None = None
        fallback = rules.helmet.vlm_fallback
        self._fallback_enabled = bool(fallback.enabled and vlm_review_enabled)
        self._fallback_min_confidence = fallback.min_confidence
        self._fallback_padding = fallback.crop_padding_ratio
        self._fallback_schedule = FallbackSchedule(seconds_to_ms(fallback.interval_s))

    @property
    def camera_id(self) -> str:
        return self._camera_id

    def reset(self) -> None:
        """清空全部状态，用于视频循环重启（A4）。"""
        self._intrusion.reset()
        self._smoke = SmokeFireTracker(
            conf_threshold=self._rules.smoke_fire.conf_threshold,
            consecutive_frames=self._rules.smoke_fire.consecutive_frames,
            same_spot_iou_threshold=self._rules.smoke_fire.same_spot_iou_threshold,
        )
        self._helmet_timer = HelmetTimer(
            no_helmet_duration_ms=seconds_to_ms(self._rules.helmet.no_helmet_duration_s),
            track_gap_reset_ms=seconds_to_ms(self._rules.helmet.track_gap_reset_s),
            roi_exit_grace_ms=seconds_to_ms(self._rules.roi_exit_grace_s),
        )
        self._cooldown.reset()
        self._trajectory.clear()
        self._fallback_schedule.reset()
        self._last_ts_ms = None

    def process(self, frame_result: FrameResult) -> list[CandidateEvent]:
        """处理一个已处理帧，返回本帧被冷却接受的候选事件。

        Raises:
            ValueError: ``ts_ms`` 早于上一帧，或同一帧中出现重复的 ``track_id``。
        """
        ts = frame_result.ts_ms
        if self._last_ts_ms is not None and ts < self._last_ts_ms:
            raise ValueError(f"ts_ms {ts} is earlier than previous frame {self._last_ts_ms}")
        self._last_ts_ms = ts

        persons: dict[int, PersonDetection] = {}
        for person in frame_result.persons:
            if person.track_id in persons:
                raise ValueError(f"duplicate track_id {person.track_id} in frame {frame_result.frame_index}")
            persons[person.track_id] = person
            self._trajectory.append(person.track_id, ts, person.box)

        events: list[CandidateEvent] = []
        events.extend(self._process_zones(ts, persons))
        events.extend(self._process_smoke_fire(frame_result))
        events.extend(self._process_helmets(ts, frame_result, persons))
        return events

    # ------------------------------------------------------------------ 闯入与滞留

    def _process_zones(self, ts: int, persons: dict[int, PersonDetection]) -> list[CandidateEvent]:
        observations = [TrackObservation(p.track_id, p.box) for p in persons.values()]
        out: list[CandidateEvent] = []
        for zone_event in self._intrusion.process(ts, observations):
            if not self._cooldown.try_accept(zone_event.type, f"track:{zone_event.track_id}", zone_event.end_ms):
                continue
            roi = self._rois[zone_event.roi_id]
            person = persons[zone_event.track_id]
            detail: dict[str, object] = {
                "box": list(person.box),
                "reference_point": list(geometry.ref_point(person.box, self._rules.reference_point)),
            }
            if zone_event.type == "loitering":
                detail["dwell_threshold_s"] = self._rules.dwell_threshold_s
            out.append(
                self._person_candidate(
                    event_type=zone_event.type,
                    start_ms=zone_event.start_ms,
                    end_ms=zone_event.end_ms,
                    person=person,
                    roi_id=roi.id,
                    roi_type=roi.type,
                    roi_polygon=roi.polygon,
                    detail=detail,
                )
            )
        return out

    # ------------------------------------------------------------------ 烟火

    def _process_smoke_fire(self, frame_result: FrameResult) -> list[CandidateEvent]:
        out: list[CandidateEvent] = []
        iou_threshold = self._rules.smoke_fire.same_spot_iou_threshold
        for ev in self._smoke.process(frame_result.ts_ms, frame_result.smoke_fire):
            if not self._cooldown.try_accept_spot("smoke_fire", ev.box, ev.end_ms, iou_threshold):
                continue
            out.append(
                CandidateEvent(
                    camera_id=self._camera_id,
                    event_type="smoke_fire",
                    detection_source=_CV_SOURCE,
                    start_ms=ev.start_ms,
                    end_ms=ev.end_ms,
                    cv_confidence=ev.conf,
                    track_id=None,
                    roi_id=None,
                    roi_type=None,
                    detail={"label": ev.label, "box": list(ev.box), "spot_id": ev.spot_id},
                    trajectory=[],
                    overlay=[OverlayItem(box=ev.box, label=ev.label, conf=ev.conf)],
                )
            )
        return out

    # ------------------------------------------------------------------ 安全帽

    def _process_helmets(
        self,
        ts: int,
        frame_result: FrameResult,
        persons: dict[int, PersonDetection],
    ) -> list[CandidateEvent]:
        helmet_rules = self._rules.helmet

        # 作用域（D1）与高度过滤：高度不足的 person 不参与判定，也不进入计时器。
        scoped, outside = filter_scoped_persons(
            persons,
            [zone.polygon for zone in self._work_zones],
            self._rules.reference_point,
            helmet_rules.min_person_height_px,
        )

        helmet_inputs = [(d.box, d.conf, d.kind) for d in frame_result.helmets]
        assigned = associate_helmets(
            persons=[(p.track_id, p.box) for p in scoped.values()],
            helmets=helmet_inputs,
            head_region_ratio=helmet_rules.head_region_ratio,
            metric=helmet_rules.assoc_overlap_metric,
            overlap_threshold=helmet_rules.assoc_overlap_threshold,
            conf_threshold=helmet_rules.conf_threshold,
        )

        observations: list[HelmetObservation] = []
        for track_id, person in scoped.items():
            compliant = track_id in assigned
            if not compliant and helmet_rules.require_no_helmet_evidence:
                # 无 no_helmet 证据时视为非违规（保守），计时器按已关联处理。
                compliant = not self._has_no_helmet_evidence(person, frame_result)
            observations.append(HelmetObservation(track_id, in_scope=True, has_associated_helmet=compliant))
        for track_id in outside:
            observations.append(HelmetObservation(track_id, in_scope=False, has_associated_helmet=False))

        # VLM 兜底（3.7）见 collect_fallback_requests / apply_fallback_result。

        out: list[CandidateEvent] = []
        for ev in self._helmet_timer.update(ts, observations):
            if not self._cooldown.try_accept("no_helmet", f"track:{ev.track_id}", ev.end_ms):
                continue
            person = scoped[ev.track_id]
            out.append(
                self._person_candidate(
                    event_type="no_helmet",
                    start_ms=ev.start_ms,
                    end_ms=ev.end_ms,
                    person=person,
                    roi_id=None,
                    roi_type=None,
                    roi_polygon=None,
                    detail={
                        "box": list(person.box),
                        "no_helmet_duration_s": helmet_rules.no_helmet_duration_s,
                    },
                )
            )
        return out

    def _has_no_helmet_evidence(self, person: PersonDetection, frame_result: FrameResult) -> bool:
        """头部区域内是否有置信度达标的 no_helmet 检测（检测框中心落在头部区域内）。"""
        helmet_rules = self._rules.helmet
        head = geometry.head_region(person.box, helmet_rules.head_region_ratio)
        for det in frame_result.helmets:
            if det.kind != "no_helmet" or det.conf < helmet_rules.conf_threshold:
                continue
            if _in_box(geometry.ref_point(det.box, "center"), head):
                return True
        return False

    # ------------------------------------------------------------------ VLM 兜底（3.7）

    def collect_fallback_requests(self, frame_result: FrameResult) -> list[FallbackRequest]:
        """返回本帧需要做头部兜底检查的作用域内 person（未关联安全帽，且满足间隔）。

        开关关闭（`rules.helmet.vlm_fallback.enabled` 或 `vlm.review_enabled`）时返回空列表。
        """
        if not self._fallback_enabled:
            return []
        helmet_rules = self._rules.helmet
        persons = {
            p.track_id: PersonDetection(track_id=p.track_id, box=p.box, conf=p.conf)
            for p in frame_result.persons
        }
        scoped, _outside = filter_scoped_persons(
            persons,
            [zone.polygon for zone in self._work_zones],
            self._rules.reference_point,
            helmet_rules.min_person_height_px,
        )
        assigned = associate_helmets(
            persons=[(p.track_id, p.box) for p in scoped.values()],
            helmets=[(d.box, d.conf, d.kind) for d in frame_result.helmets],
            head_region_ratio=helmet_rules.head_region_ratio,
            metric=helmet_rules.assoc_overlap_metric,
            overlap_threshold=helmet_rules.assoc_overlap_threshold,
            conf_threshold=helmet_rules.conf_threshold,
        )
        requests: list[FallbackRequest] = []
        for track_id, person in scoped.items():
            if track_id in assigned:
                continue
            if self._fallback_schedule.due(track_id, frame_result.ts_ms):
                requests.append(FallbackRequest(track_id=track_id, ts_ms=frame_result.ts_ms, box=person.box))
        return requests

    def apply_fallback_result(
        self, request: FallbackRequest, verdict: str, confidence: float
    ) -> CandidateEvent | None:
        """把一次兜底检查结果转为候选事件；不满足条件时返回 None。

        仅当 verdict 为 `no_helmet` 且 confidence 不低于 `min_confidence` 时产生候选，
        并与权重路径共用冷却键 `track:{track_id}`。
        """
        if not self._fallback_enabled:
            return None
        if verdict != "no_helmet" or confidence < self._fallback_min_confidence:
            return None
        if not self._cooldown.try_accept("no_helmet", f"track:{request.track_id}", request.ts_ms):
            return None
        points = self._trajectory.window(request.track_id, request.ts_ms, request.ts_ms)
        return CandidateEvent(
            camera_id=self._camera_id,
            event_type="no_helmet",
            detection_source="vlm_fallback",
            start_ms=request.ts_ms,
            end_ms=request.ts_ms,
            cv_confidence=float(confidence),
            track_id=request.track_id,
            roi_id=None,
            roi_type=None,
            detail={"box": list(request.box), "verdict": verdict, "source": "vlm_fallback"},
            trajectory=[list(p) for p in downsample(points, self._rules.trajectory_max_points)],
            overlay=[OverlayItem(box=request.box, label="person", conf=float(confidence))],
        )
    # ------------------------------------------------------------------ 公共构造

    def _person_candidate(
        self,
        *,
        event_type: str,
        start_ms: int,
        end_ms: int,
        person: PersonDetection,
        roi_id: str | None,
        roi_type: str | None,
        roi_polygon: Sequence[Point] | None,
        detail: dict[str, object],
    ) -> CandidateEvent:
        """构造 person 类候选事件（intrusion / loitering / no_helmet）。

        轨迹窗口为候选事件自身的 ``[start_ms, end_ms]``，超过 ``trajectory_max_points``
        时等距下采样。闯入事件的 start 与 end 相同，因此其轨迹只含触发帧一个点。
        """
        points = self._trajectory.window(person.track_id, start_ms, end_ms)
        sampled = downsample(points, self._rules.trajectory_max_points)
        return CandidateEvent(
            camera_id=self._camera_id,
            event_type=event_type,  # type: ignore[arg-type]
            detection_source=_CV_SOURCE,
            start_ms=start_ms,
            end_ms=end_ms,
            cv_confidence=person.conf,
            track_id=person.track_id,
            roi_id=roi_id,
            roi_type=roi_type,  # type: ignore[arg-type]
            detail=dict(detail),
            trajectory=[list(p) for p in sampled],
            overlay=[
                OverlayItem(
                    box=person.box,
                    label="person",
                    conf=person.conf,
                    roi_polygon=roi_polygon,
                )
            ],
        )
