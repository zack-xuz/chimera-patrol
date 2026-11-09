"""Property 8: 未佩戴安全帽连续计时（HelmetTimer）。

**Validates: Requirements 3.5**

对任意单个 person track 的已处理帧序列（每帧标注是否有关联安全帽、是否在作用域内，
时间戳严格递增，间隔任意）：计时器产生的 ``no_helmet`` 事件列表与独立参考模型一致。
参考模型体现以下规则：

- 违规帧（作用域内且无关联安全帽）记录 ``t0``；``ts - t0 >= no_helmet_duration_s`` 时触发。
- 关联安全帽的帧清零计时，之后需重新累计满时长。
- 相对上次观测间隔严格大于 ``track_gap_reset_s`` 时清零（等于限值不清零）。
- 同一段连续违规至多触发一次。
- 作用域外帧：宽限期 ``roi_exit_grace_s`` 内保留 ``t0``，超出宽限期清零。

秒级参数取 0.25 的整数倍，乘以 1000 后为精确整数毫秒，使实现中的秒→毫秒换算
不引入浮点误差，边界比较（间隔等于限值）可以精确对照。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from hypothesis import given, settings, strategies as st

from chimera_patrol.rules.helmet_timer import HelmetObservation, HelmetTimer, NoHelmetEvent

TRACK_ID = 7
_Q = 250  # 毫秒；参数网格为 0.25 秒的整数倍

# (ts_ms, in_scope, has_associated_helmet)
Frame = Tuple[int, bool, bool]
Case = Tuple[int, int, int, List[Frame]]  # (duration_ms, gap_ms, grace_ms, frames)


def _reference_events(
    frames: List[Frame], duration_ms: int, gap_ms: int, grace_ms: int
) -> List[NoHelmetEvent]:
    """独立参考模型：逐帧循环维护 t0 / 已触发标记 / 离开作用域起点。"""
    events: List[NoHelmetEvent] = []
    t0: Optional[int] = None
    fired = False
    out_since: Optional[int] = None
    last_seen: Optional[int] = None

    for ts, in_scope, helmet in frames:
        if last_seen is not None and ts - last_seen > gap_ms:
            t0, fired, out_since = None, False, None
        last_seen = ts

        if not in_scope:
            if out_since is None:
                out_since = ts
            if ts - out_since > grace_ms:
                t0, fired = None, False
            continue

        out_since = None
        if helmet:
            t0, fired = None, False
            continue

        if t0 is None:
            t0, fired = ts, False
        if not fired and ts - t0 >= duration_ms:
            fired = True
            events.append(NoHelmetEvent(track_id=TRACK_ID, start_ms=t0, end_ms=ts))

    return events


def _impl_events(frames: List[Frame], duration_ms: int, gap_ms: int, grace_ms: int) -> List[NoHelmetEvent]:
    timer = HelmetTimer(
        no_helmet_duration_ms=duration_ms,
        track_gap_reset_ms=gap_ms,
        roi_exit_grace_ms=grace_ms,
    )
    events: List[NoHelmetEvent] = []
    for ts, in_scope, helmet in frames:
        events.extend(timer.update(ts, [HelmetObservation(TRACK_ID, in_scope, helmet)]))
    return events


@st.composite
def _timer_case(draw, mixed_scope: bool) -> Case:
    duration_ms = draw(st.integers(min_value=1, max_value=20)) * _Q
    gap_ms = draw(st.integers(min_value=1, max_value=20)) * _Q
    grace_ms = draw(st.integers(min_value=0, max_value=12)) * _Q

    # 时间间隔：包含边界附近的取值（1ms、时长、间隔限值及其 ±1ms），以及任意间隔。
    delta_strategy = st.one_of(
        st.sampled_from([1, duration_ms - 1, duration_ms, gap_ms - 1, gap_ms, gap_ms + 1]),
        st.integers(min_value=1, max_value=3000),
    )

    ts = draw(st.integers(min_value=0, max_value=10_000))
    n = draw(st.integers(min_value=0, max_value=80))
    frames: List[Frame] = []
    for _ in range(n):
        ts += draw(delta_strategy)
        in_scope = draw(st.booleans()) if mixed_scope else True
        helmet = draw(st.booleans())
        frames.append((ts, in_scope, helmet))
    return duration_ms, gap_ms, grace_ms, frames


@settings(max_examples=200, deadline=None)
@given(case=_timer_case(mixed_scope=False))
def test_events_match_reference_all_in_scope(case: Case) -> None:
    """全部帧位于作用域内时，事件列表与参考模型逐项一致。"""
    duration_ms, gap_ms, grace_ms, frames = case
    assert _impl_events(frames, duration_ms, gap_ms, grace_ms) == _reference_events(
        frames, duration_ms, gap_ms, grace_ms
    )


@settings(max_examples=200, deadline=None)
@given(case=_timer_case(mixed_scope=True))
def test_events_match_reference_with_out_of_scope_frames(case: Case) -> None:
    """作用域外帧与宽限期混入时，事件列表仍与参考模型（含 roi_exit_grace_s）一致。"""
    duration_ms, gap_ms, grace_ms, frames = case
    assert _impl_events(frames, duration_ms, gap_ms, grace_ms) == _reference_events(
        frames, duration_ms, gap_ms, grace_ms
    )
