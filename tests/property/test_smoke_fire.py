"""烟火连续同位判定的属性测试。

**Validates: Requirements 3.4, 3.11**

对应设计文档 Property 5：对任意按时间排列的烟火检测序列，当且仅当存在连续
``consecutive_frames`` 个已处理帧，每帧都有置信度 ``>= conf_threshold`` 且同 ``label``
的检测，并且相邻两帧检测框 IoU ``>= same_spot_iou_threshold`` 时，才产生事件；
同一连续点位至多触发一次。

参考模型是一个独立的连续计数器，不复用 ``SmokeFireTracker`` 的实现。IoU 直接调用
``geometry.iou``，该函数已有独立的属性测试覆盖。

适用范围限制：
- 单点位测试：每帧至多一个检测，因此不涉及多点位之间的贪心匹配。
- 双点位测试：两个点位相距很远（框之间 IoU 恒为 0），且 ``same_spot_iou_threshold > 0``，
  因此两个点位互不影响，可分别用参考模型计算。
"""
from __future__ import annotations

from itertools import accumulate
from typing import List, Optional, Sequence, Tuple

from hypothesis import given, settings
from hypothesis import strategies as st

from chimera_patrol.rules.geometry import Box, iou
from chimera_patrol.rules.smoke_fire import SmokeFireDetection, SmokeFireTracker

LABELS = ["smoke", "fire"]
NEAR_A: Box = (100.0, 100.0, 140.0, 160.0)
NEAR_B: Box = (400.0, 100.0, 440.0, 160.0)
FAR_A: Box = (700.0, 500.0, 740.0, 560.0)
FAR_B: Box = (900.0, 500.0, 940.0, 560.0)

# 参考模型的输出：(label, start_ms, end_ms)
Fire = Tuple[str, int, int]
# 单帧的检测记录：(label, conf, box)，None 表示该帧无检测。
Raw = Optional[Tuple[str, float, Box]]


def _jittered(base: Box):
    """围绕 base 做整数像素抖动，保证 x1<=x2 且 y1<=y2。"""
    return st.tuples(*[st.integers(-10, 10) for _ in range(4)]).map(
        lambda d: (base[0] + d[0], base[1] + d[1], base[2] + d[2], base[3] + d[3])
    )


def _box_for(base: Box, far: Box):
    """大部分时间在 base 附近抖动，偶尔跳到远处以打断连续段。"""
    return st.one_of(_jittered(base), _jittered(base), _jittered(base), st.just(far))


def _detection(conf_thr: float, base: Box, far: Box):
    """单个检测：置信度既包含任意值，也包含恰好等于阈值的边界值。"""
    conf = st.one_of(st.floats(0.0, 1.0), st.just(conf_thr))
    return st.tuples(st.sampled_from(LABELS), conf, _box_for(base, far))


def _reference_fires(
    frames: Sequence[Raw],
    timestamps: Sequence[int],
    conf_thr: float,
    n_required: int,
    iou_thr: float,
) -> List[Fire]:
    """单点位参考模型：连续计数器。

    计数器在某帧达到 ``n_required`` 时记一次触发。之后同一段继续命中时计数器超过
    ``n_required``，不会再次等于它，因此每段连续命中至多触发一次。
    """
    fires: List[Fire] = []
    streak = 0
    prev_label: Optional[str] = None
    prev_box: Optional[Box] = None
    start_ms = 0
    for ts, frame in zip(timestamps, frames):
        if frame is None or frame[1] < conf_thr:
            streak = 0
            prev_label = None
            prev_box = None
            continue
        label, _, box = frame
        if prev_label == label and prev_box is not None and iou(prev_box, box) >= iou_thr:
            streak += 1
        else:
            streak = 1
            start_ms = ts
        prev_label, prev_box = label, box
        if streak == n_required:
            fires.append((label, start_ms, ts))
    return fires


def _run_tracker(
    frames: Sequence[Sequence[SmokeFireDetection]],
    timestamps: Sequence[int],
    conf_thr: float,
    n_required: int,
    iou_thr: float,
) -> List[Fire]:
    tracker = SmokeFireTracker(
        conf_threshold=conf_thr,
        consecutive_frames=n_required,
        same_spot_iou_threshold=iou_thr,
    )
    out: List[Fire] = []
    for ts, dets in zip(timestamps, frames):
        for ev in tracker.process(ts, dets):
            out.append((ev.label, ev.start_ms, ev.end_ms))
    return out


def _to_dets(frame: Raw) -> List[SmokeFireDetection]:
    if frame is None:
        return []
    label, conf, box = frame
    return [SmokeFireDetection(label=label, box=box, conf=conf)]


@st.composite
def _single_spot_case(draw):
    conf_thr = draw(st.floats(0.0, 1.0))
    iou_thr = draw(st.floats(0.0, 1.0))
    n_required = draw(st.integers(1, 6))
    frames = draw(
        st.lists(
            st.one_of(st.none(), _detection(conf_thr, NEAR_A, FAR_A)),
            min_size=0,
            max_size=40,
        )
    )
    gaps = draw(st.lists(st.integers(1, 5000), min_size=len(frames), max_size=len(frames)))
    timestamps = list(accumulate(gaps))
    return conf_thr, iou_thr, n_required, frames, timestamps


@st.composite
def _two_spot_case(draw):
    conf_thr = draw(st.floats(0.0, 1.0))
    # 必须大于 0，否则两个相距很远的点位（IoU=0）也会被判为同一处。
    iou_thr = draw(st.floats(0.05, 1.0))
    n_required = draw(st.integers(1, 6))
    frames = draw(
        st.lists(
            st.tuples(
                st.one_of(st.none(), _detection(conf_thr, NEAR_A, FAR_A)),
                st.one_of(st.none(), _detection(conf_thr, NEAR_B, FAR_B)),
            ),
            min_size=0,
            max_size=40,
        )
    )
    gaps = draw(st.lists(st.integers(1, 5000), min_size=len(frames), max_size=len(frames)))
    timestamps = list(accumulate(gaps))
    return conf_thr, iou_thr, n_required, frames, timestamps


@settings(max_examples=200, deadline=None)
@given(case=_single_spot_case())
def test_single_spot_fires_iff_consecutive_matching_run(case) -> None:
    """Property 5：单点位的触发时刻与参考连续计数器完全一致。

    Validates: Requirements 3.4, 3.11
    """
    conf_thr, iou_thr, n_required, frames, timestamps = case
    expected = _reference_fires(frames, timestamps, conf_thr, n_required, iou_thr)
    actual = _run_tracker(
        [_to_dets(f) for f in frames], timestamps, conf_thr, n_required, iou_thr
    )
    assert actual == expected


@settings(max_examples=200, deadline=None)
@given(case=_two_spot_case())
def test_two_separated_spots_fire_independently(case) -> None:
    """两个相距很远的点位各自独立计数、各自至多触发一次。

    Validates: Requirements 3.4, 3.11
    """
    conf_thr, iou_thr, n_required, frames, timestamps = case
    stream_a = [f[0] for f in frames]
    stream_b = [f[1] for f in frames]
    expected = sorted(
        _reference_fires(stream_a, timestamps, conf_thr, n_required, iou_thr)
        + _reference_fires(stream_b, timestamps, conf_thr, n_required, iou_thr)
    )
    actual = sorted(
        _run_tracker(
            [_to_dets(a) + _to_dets(b) for a, b in frames],
            timestamps,
            conf_thr,
            n_required,
            iou_thr,
        )
    )
    assert actual == expected


def test_two_separated_spots_fire_at_their_own_frames() -> None:
    """确定性示例：A 在帧 0-2 命中，B 在帧 1-3 命中，N=3，两者各在自己的第 3 帧触发。"""
    tracker = SmokeFireTracker(
        conf_threshold=0.5, consecutive_frames=3, same_spot_iou_threshold=0.5
    )
    det_a = SmokeFireDetection("smoke", NEAR_A, 0.9)
    det_b = SmokeFireDetection("fire", NEAR_B, 0.9)
    frames = [[det_a], [det_a, det_b], [det_a, det_b], [det_b]]
    fired = []
    for ts, dets in zip([0, 100, 200, 300], frames):
        fired.extend((e.label, e.start_ms, e.end_ms) for e in tracker.process(ts, dets))
    assert sorted(fired) == [("fire", 100, 300), ("smoke", 0, 200)]


def test_iou_exactly_at_threshold_counts_as_same_spot() -> None:
    """边界：IoU 恰好等于阈值时视为同一处（判定为 >=）。"""
    box_a: Box = (0.0, 0.0, 10.0, 10.0)
    box_b: Box = (0.0, 0.0, 10.0, 5.0)  # IoU = 50 / 100 = 0.5
    assert iou(box_a, box_b) == 0.5
    fires = _run_tracker(
        [[SmokeFireDetection("smoke", box_a, 0.9)], [SmokeFireDetection("smoke", box_b, 0.9)]],
        [0, 100],
        conf_thr=0.5,
        n_required=2,
        iou_thr=0.5,
    )
    assert fires == [("smoke", 0, 100)]
