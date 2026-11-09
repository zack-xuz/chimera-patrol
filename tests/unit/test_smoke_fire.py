"""SmokeFireTracker 示例测试：确定性输入，覆盖阈值、同位匹配、中断与单次触发。"""

import pytest

from chimera_patrol.rules.smoke_fire import SmokeFireDetection, SmokeFireTracker

SMOKE_BOX = (100.0, 100.0, 200.0, 200.0)
FAR_BOX = (500.0, 500.0, 600.0, 600.0)


def _det(box=SMOKE_BOX, conf=0.9, label="smoke") -> SmokeFireDetection:
    return SmokeFireDetection(label=label, box=box, conf=conf)


def _tracker(consecutive: int = 3) -> SmokeFireTracker:
    return SmokeFireTracker(
        conf_threshold=0.5,
        consecutive_frames=consecutive,
        same_spot_iou_threshold=0.3,
    )


def test_fires_once_on_consecutive_frames_with_streak_start_and_end() -> None:
    tracker = _tracker(consecutive=3)

    assert tracker.process(1000, [_det()]) == []
    assert tracker.process(1100, [_det()]) == []
    events = tracker.process(1200, [_det(box=(102.0, 100.0, 202.0, 200.0), conf=0.8)])

    assert len(events) == 1
    event = events[0]
    assert event.label == "smoke"
    assert event.start_ms == 1000
    assert event.end_ms == 1200
    assert event.conf == 0.8
    assert event.box == (102.0, 100.0, 202.0, 200.0)


def test_does_not_refire_while_same_spot_stays_matched() -> None:
    tracker = _tracker(consecutive=2)

    fired = []
    for ts in range(0, 10 * 100, 100):
        fired.extend(tracker.process(ts, [_det()]))

    assert len(fired) == 1
    assert fired[0].start_ms == 0
    assert fired[0].end_ms == 100


def test_low_confidence_detection_is_dropped_and_breaks_nothing_new() -> None:
    tracker = _tracker(consecutive=2)

    assert tracker.process(0, [_det(conf=0.49)]) == []
    assert tracker.process(100, [_det(conf=0.49)]) == []
    # 低置信度不计入连续段，第一次达标帧只是新点位的第 1 帧。
    assert tracker.process(200, [_det(conf=0.5)]) == []
    events = tracker.process(300, [_det(conf=0.5)])

    assert len(events) == 1
    assert events[0].start_ms == 200
    assert events[0].end_ms == 300


def test_missing_frame_breaks_streak_and_restarts_count() -> None:
    tracker = _tracker(consecutive=3)

    assert tracker.process(0, [_det()]) == []
    assert tracker.process(100, [_det()]) == []
    assert tracker.process(200, []) == []  # 未命中，点位被删除
    assert tracker.process(300, [_det()]) == []
    assert tracker.process(400, [_det()]) == []
    events = tracker.process(500, [_det()])

    assert len(events) == 1
    assert events[0].start_ms == 300
    assert events[0].end_ms == 500


def test_non_overlapping_box_starts_a_new_spot() -> None:
    tracker = _tracker(consecutive=2)

    assert tracker.process(0, [_det()]) == []
    # 位置跳到远处，IoU 为 0，原点位中断，远处为新点位（streak=1）。
    assert tracker.process(100, [_det(box=FAR_BOX)]) == []
    events = tracker.process(200, [_det(box=FAR_BOX)])

    assert len(events) == 1
    assert events[0].start_ms == 100
    assert events[0].box == FAR_BOX


def test_label_mismatch_does_not_match_same_spot() -> None:
    tracker = _tracker(consecutive=2)

    assert tracker.process(0, [_det(label="smoke")]) == []
    assert tracker.process(100, [_det(label="fire")]) == []
    events = tracker.process(200, [_det(label="fire")])

    assert len(events) == 1
    assert events[0].label == "fire"
    assert events[0].start_ms == 100


def test_matching_is_one_to_one_so_duplicate_boxes_do_not_double_count() -> None:
    tracker = _tracker(consecutive=2)

    # 同帧两个框都覆盖同一点位：只有一个能匹配，另一个新建点位。
    assert tracker.process(0, [_det(), _det(box=(101.0, 101.0, 201.0, 201.0))]) == []
    events = tracker.process(100, [_det()])

    # 上一帧两个点位中，仅一个被本帧命中，另一个被删除，因此只触发一次且为原点位。
    assert len(events) == 1
    assert events[0].start_ms == 0


def test_greedy_matching_prefers_higher_iou_regardless_of_input_order() -> None:
    tracker = _tracker(consecutive=3)

    assert tracker.process(0, [_det()]) == []
    # 两个框都超过 0.3 阈值，先给出 IoU 较低的框。点位应匹配 IoU 较高的框，
    # 较低的框成为新点位，之后未被命中而删除。
    lower = _det(box=(120.0, 120.0, 220.0, 220.0))  # IoU ≈ 0.47
    higher = _det(box=(102.0, 102.0, 202.0, 202.0))  # IoU ≈ 0.92
    assert tracker.process(100, [lower, higher]) == []
    events = tracker.process(200, [higher])

    assert len(events) == 1
    assert events[0].start_ms == 0
    assert events[0].end_ms == 200


def test_consecutive_frames_of_one_fires_on_first_frame() -> None:
    tracker = _tracker(consecutive=1)

    events = tracker.process(42, [_det()])

    assert len(events) == 1
    assert events[0].start_ms == 42
    assert events[0].end_ms == 42


@pytest.mark.parametrize(
    "kwargs",
    [
        {"conf_threshold": 1.5, "consecutive_frames": 3, "same_spot_iou_threshold": 0.3},
        {"conf_threshold": 0.5, "consecutive_frames": 0, "same_spot_iou_threshold": 0.3},
        {"conf_threshold": 0.5, "consecutive_frames": 3, "same_spot_iou_threshold": -0.1},
    ],
)
def test_invalid_constructor_arguments_are_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        SmokeFireTracker(**kwargs)
