"""安全帽 VLM 兜底路径测试（任务 11.2）。"""

from __future__ import annotations

import numpy as np

from chimera_patrol.config import RulesSection
from chimera_patrol.rules.engine import RuleEngine
from chimera_patrol.rules.helmet_fallback import FallbackSchedule, crop_head_region
from chimera_patrol.rules.types import FrameResult, PersonDetection

STANDING = (100.0, 100.0, 150.0, 300.0)  # 高 200 像素，参考点 (125, 300)


def _rules(**helmet_fallback: object) -> RulesSection:
    fallback = {"enabled": True, "interval_s": 10, "min_confidence": 0.6, "crop_padding_ratio": 0.2}
    fallback.update(helmet_fallback)
    return RulesSection.model_validate({"helmet": {"vlm_fallback": fallback}})


def _frame(ts_ms: int) -> FrameResult:
    return FrameResult(ts_ms=ts_ms, frame_index=ts_ms // 100, persons=[PersonDetection(1, STANDING, 0.9)])


def test_schedule_gates_per_track_by_interval() -> None:
    schedule = FallbackSchedule(10_000)
    assert schedule.due(1, 0) is True
    assert schedule.due(1, 9_999) is False
    assert schedule.due(1, 10_000) is True
    assert schedule.due(2, 9_999) is True  # 另一个 track 独立计时


def test_collect_requests_only_when_enabled_and_due() -> None:
    engine = RuleEngine("cam1", _rules(), [])
    first = engine.collect_fallback_requests(_frame(0))
    assert [r.track_id for r in first] == [1]
    assert engine.collect_fallback_requests(_frame(1_000)) == []
    assert [r.track_id for r in engine.collect_fallback_requests(_frame(10_000))] == [1]


def test_disabled_switches_produce_no_requests() -> None:
    off_by_config = RuleEngine("cam1", _rules(enabled=False), [])
    assert off_by_config.collect_fallback_requests(_frame(0)) == []
    off_by_review = RuleEngine("cam1", _rules(), [], vlm_review_enabled=False)
    assert off_by_review.collect_fallback_requests(_frame(0)) == []


def test_associated_helmet_skips_fallback_request() -> None:
    from chimera_patrol.rules.types import HelmetDetection

    engine = RuleEngine("cam1", _rules(), [])
    head_helmet = HelmetDetection(kind="helmet", box=(100.0, 100.0, 150.0, 150.0), conf=0.9)
    frame = FrameResult(ts_ms=0, frame_index=0, persons=[PersonDetection(1, STANDING, 0.9)], helmets=[head_helmet])
    assert engine.collect_fallback_requests(frame) == []


def test_no_helmet_result_creates_vlm_fallback_event() -> None:
    engine = RuleEngine("cam1", _rules(), [])
    (request,) = engine.collect_fallback_requests(_frame(0))
    event = engine.apply_fallback_result(request, "no_helmet", 0.85)
    assert event is not None
    assert event.event_type == "no_helmet"
    assert event.detection_source == "vlm_fallback"
    assert event.cv_confidence == 0.85
    assert event.start_ms == event.end_ms == 0
    assert event.track_id == 1


def test_helmet_uncertain_and_low_confidence_produce_nothing() -> None:
    engine = RuleEngine("cam1", _rules(), [])
    (request,) = engine.collect_fallback_requests(_frame(0))
    assert engine.apply_fallback_result(request, "helmet", 0.99) is None
    assert engine.apply_fallback_result(request, "uncertain", 0.99) is None
    assert engine.apply_fallback_result(request, "no_helmet", 0.59) is None


def test_fallback_shares_cooldown_with_weight_path() -> None:
    engine = RuleEngine("cam1", _rules(), [])
    (request,) = engine.collect_fallback_requests(_frame(0))
    assert engine.apply_fallback_result(request, "no_helmet", 0.9) is not None
    # 同一 track 在冷却时间（默认 30 秒）内再次产生的 no_helmet 被丢弃。
    (later,) = engine.collect_fallback_requests(_frame(10_000))
    assert engine.apply_fallback_result(later, "no_helmet", 0.9) is None


def test_crop_head_region_clamps_and_pads() -> None:
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    crop = crop_head_region(frame, (100.0, 100.0, 150.0, 300.0), 0.25, 0.2)
    assert crop is not None and crop.shape[0] > 0 and crop.shape[1] > 0
    edge = crop_head_region(frame, (0.0, 0.0, 20.0, 40.0), 0.5, 0.5)
    assert edge is not None and edge.shape[0] >= 1
    outside = crop_head_region(frame, (900.0, 900.0, 950.0, 990.0), 0.25, 0.2)
    assert outside is None