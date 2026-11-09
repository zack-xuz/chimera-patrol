"""Unit tests for chimera_patrol.evidence.ring_buffer (task 5.1)."""
from __future__ import annotations

import pytest

from chimera_patrol.evidence.ring_buffer import RingBuffer, select_frame


def _jpeg(ts: int) -> bytes:
    return f"jpeg-{ts}".encode()


def test_push_enforces_rate_cap() -> None:
    # 10 fps -> frames must be at least 100 ms apart.
    buf = RingBuffer(window_s=10, max_fps=10)
    assert buf.push(0, _jpeg(0)) is True
    assert buf.push(50, _jpeg(50)) is False  # 50 ms after last stored frame
    assert buf.push(100, _jpeg(100)) is True  # exactly 100 ms is allowed
    assert buf.push(150, _jpeg(150)) is False
    assert [ts for ts, _ in buf.snapshot(0, 1000)] == [0, 100]


def test_push_rejects_out_of_order_timestamp() -> None:
    buf = RingBuffer(window_s=10, max_fps=30)
    buf.push(500, _jpeg(500))
    with pytest.raises(ValueError):
        buf.push(400, _jpeg(400))
    assert [ts for ts, _ in buf.snapshot(0, 1000)] == [500]


def test_eviction_drops_frames_older_than_window() -> None:
    # 1 s window at 10 fps: after pushing ts=1100, cutoff is 100 ms.
    buf = RingBuffer(window_s=1, max_fps=10)
    for ts in range(0, 1001, 100):
        buf.push(ts, _jpeg(ts))
    # cutoff = 1000 - 1000 = 0, so ts=0 is kept (not strictly older).
    assert [ts for ts, _ in buf.snapshot(0, 2000)][0] == 0

    # cutoff = 1100 - 1000 = 100, so ts=0 is evicted.
    buf.push(1100, _jpeg(1100))
    remaining = [ts for ts, _ in buf.snapshot(0, 2000)]
    assert remaining == list(range(100, 1101, 100))


def test_snapshot_returns_inclusive_ascending_range() -> None:
    buf = RingBuffer(window_s=60, max_fps=10)
    for ts in (0, 200, 400, 600, 800):
        buf.push(ts, _jpeg(ts))

    assert buf.snapshot(200, 600) == [(200, _jpeg(200)), (400, _jpeg(400)), (600, _jpeg(600))]
    assert buf.snapshot(250, 350) == []  # no stored frame in this gap
    assert buf.snapshot(-100, 50) == [(0, _jpeg(0))]


def test_snapshot_with_inverted_range_is_empty() -> None:
    buf = RingBuffer(window_s=60, max_fps=10)
    buf.push(0, _jpeg(0))
    assert buf.snapshot(500, 100) == []


def test_snapshot_of_empty_buffer_is_empty() -> None:
    buf = RingBuffer(window_s=20, max_fps=10)
    assert buf.snapshot(0, 10_000) == []


def test_select_frame_picks_nearest_to_midpoint() -> None:
    frames = [(0, _jpeg(0)), (300, _jpeg(300)), (700, _jpeg(700))]
    assert select_frame(frames, 400) == (300, _jpeg(300))
    assert select_frame(frames, 650) == (700, _jpeg(700))
    assert select_frame(frames, -50) == (0, _jpeg(0))


def test_select_frame_breaks_ties_toward_earlier_frame() -> None:
    frames = [(100, _jpeg(100)), (200, _jpeg(200))]
    assert select_frame(frames, 150) == (100, _jpeg(100))


def test_select_frame_rejects_empty_list() -> None:
    with pytest.raises(ValueError):
        select_frame([], 100)


def test_constructor_rejects_non_positive_arguments() -> None:
    with pytest.raises(ValueError):
        RingBuffer(window_s=0, max_fps=10)
    with pytest.raises(ValueError):
        RingBuffer(window_s=10, max_fps=0)
