"""Unit tests for chimera_patrol.rules.helmet_timer (task 4.9, design D3)."""

from __future__ import annotations

import pytest

from chimera_patrol.rules.helmet_timer import HelmetObservation, HelmetTimer, NoHelmetEvent

DURATION_S = 5.0
GAP_S = 2.0
GRACE_S = 1.0


def _timer() -> HelmetTimer:
    return HelmetTimer(no_helmet_duration_ms=round(DURATION_S * 1000), track_gap_reset_ms=round(GAP_S * 1000), roi_exit_grace_ms=round(GRACE_S * 1000))


def _violation(track_id: int = 1) -> HelmetObservation:
    return HelmetObservation(track_id=track_id, in_scope=True, has_associated_helmet=False)


def _helmet(track_id: int = 1) -> HelmetObservation:
    return HelmetObservation(track_id=track_id, in_scope=True, has_associated_helmet=True)


def _out_of_scope(track_id: int = 1) -> HelmetObservation:
    return HelmetObservation(track_id=track_id, in_scope=False, has_associated_helmet=False)


def _run(timer: HelmetTimer, frames: list[tuple[int, list[HelmetObservation]]]) -> list[NoHelmetEvent]:
    events: list[NoHelmetEvent] = []
    for ts_ms, observations in frames:
        events.extend(timer.update(ts_ms, observations))
    return events


def test_continuous_violation_fires_once() -> None:
    timer = _timer()
    frames = [(t, [_violation()]) for t in range(0, 9000, 1000)]

    events = _run(timer, frames)

    assert events == [NoHelmetEvent(track_id=1, start_ms=0, end_ms=5000)]


def test_threshold_exactly_reached_fires_on_that_frame() -> None:
    timer = _timer()

    # Observations stay within the 2 s track gap, so only the threshold decides.
    assert timer.update(0, [_violation()]) == []
    assert timer.update(2000, [_violation()]) == []
    assert timer.update(4000, [_violation()]) == []
    assert timer.update(4999, [_violation()]) == []
    assert timer.update(5000, [_violation()]) == [NoHelmetEvent(1, 0, 5000)]


def test_helmet_in_middle_resets_and_full_duration_is_needed_again() -> None:
    timer = _timer()
    frames = [(t, [_violation()]) for t in (0, 1000, 2000, 3000)]
    frames.append((4000, [_helmet()]))
    frames += [(t, [_violation()]) for t in (5000, 6000, 7000, 8000, 9000)]

    events = _run(timer, frames)

    # The violation restarts at 5000, so 9000 is only 4 s in and must not fire yet.
    assert events == []
    assert timer.update(10000, [_violation()]) == [NoHelmetEvent(1, 5000, 10000)]


def test_track_gap_resets_timer() -> None:
    timer = _timer()
    assert _run(timer, [(t, [_violation()]) for t in (0, 1000, 2000, 3000)]) == []

    # Track is absent at 4000 and 5000; reappears at 6000, gap of 3 s > 2 s.
    frames = [(t, [_violation()]) for t in range(6000, 12000, 1000)]

    events = _run(timer, frames)

    # Must fire from the new t0 = 6000. Without the reset, t0 = 0 would fire at 6000.
    assert events == [NoHelmetEvent(1, 6000, 11000)]


def test_out_of_scope_beyond_grace_resets_timer() -> None:
    timer = _timer()
    assert _run(timer, [(t, [_violation()]) for t in (0, 1000, 2000, 3000)]) == []

    # Out of scope from 4000; grace is 1 s, so by 7000 the timer is cleared.
    assert _run(timer, [(t, [_out_of_scope()]) for t in (4000, 5000, 6000, 7000)]) == []

    # Back in scope at 8000. Without the reset, t0 = 0 would fire at 8000.
    assert _run(timer, [(t, [_violation()]) for t in (8000, 9000, 10000, 11000, 12000)]) == []
    assert timer.update(13000, [_violation()]) == [NoHelmetEvent(1, 8000, 13000)]


def test_out_of_scope_within_grace_keeps_timer() -> None:
    timer = _timer()
    assert _run(timer, [(t, [_violation()]) for t in (0, 1000, 2000, 3000)]) == []

    # Out of scope at 4000 and 5000 (1 s, not beyond grace), then back in scope.
    assert _run(timer, [(4000, [_out_of_scope()]), (5000, [_out_of_scope()])]) == []

    # t0 is still 0, so the threshold is reached at 6000 with start_ms 0.
    assert timer.update(6000, [_violation()]) == [NoHelmetEvent(1, 0, 6000)]


def test_tracks_are_timed_independently() -> None:
    timer = _timer()
    frames = [(t, [_violation(1), _helmet(2)]) for t in range(0, 6000, 1000)]

    events = _run(timer, frames)

    assert events == [NoHelmetEvent(track_id=1, start_ms=0, end_ms=5000)]


def test_non_positive_duration_is_rejected() -> None:
    with pytest.raises(ValueError):
        HelmetTimer(no_helmet_duration_ms=0, track_gap_reset_ms=round(GAP_S * 1000), roi_exit_grace_ms=round(GRACE_S * 1000))


def test_track_gap_equal_to_limit_does_not_reset_but_one_ms_more_does() -> None:
    # 1.001 s 经 round(1.001 * 1000) 换算为 1001 ms；浮点乘法 1.001 * 1000.0 会得到 1000.9999...，边界须精确。
    limit_ms = round(1.001 * 1000)
    assert limit_ms == 1001
    timer = HelmetTimer(no_helmet_duration_ms=10_000, track_gap_reset_ms=limit_ms, roi_exit_grace_ms=0)
    assert timer.update(0, [_violation()]) == []
    # 间隔恰为 1001 ms，等于限值，不清零：t0 仍为 0。
    assert timer.update(1001, [_violation()]) == []
    assert timer._states[1].t0_ms == 0
    # 间隔 1002 ms 超过限值，清零，t0 变为 2003。
    assert timer.update(2003, [_violation()]) == []
    assert timer._states[1].t0_ms == 2003
