"""TrajectoryBuffer 与 downsample 单元测试（需求 3.10）。"""

import pytest

from chimera_patrol.rules.trajectory import TrajectoryBuffer, downsample


def test_window_returns_points_in_range_in_time_order():
    buf = TrajectoryBuffer()
    for ts in (0, 100, 200, 300, 400):
        buf.append("track:1", ts, (ts, ts, ts + 10, ts + 10))
    pts = buf.window("track:1", 100, 300)
    assert [p[0] for p in pts] == [100, 200, 300]
    assert pts[0] == (100, 100, 100, 110, 110)


def test_window_unknown_track_or_empty_range_is_empty():
    buf = TrajectoryBuffer()
    buf.append("track:1", 100, (0, 0, 1, 1))
    assert buf.window("track:2", 0, 1000) == []
    assert buf.window("track:1", 200, 300) == []
    assert buf.window("track:1", 300, 200) == []


def test_tracks_are_kept_separately():
    buf = TrajectoryBuffer()
    buf.append("track:1", 0, (0, 0, 1, 1))
    buf.append("track:2", 0, (5, 5, 6, 6))
    assert buf.window("track:1", 0, 0) == [(0, 0, 0, 1, 1)]
    assert buf.window("track:2", 0, 0) == [(0, 5, 5, 6, 6)]


def test_append_out_of_order_raises():
    buf = TrajectoryBuffer()
    buf.append("track:1", 100, (0, 0, 1, 1))
    with pytest.raises(ValueError):
        buf.append("track:1", 99, (0, 0, 1, 1))


def test_downsample_returns_unchanged_when_within_cap():
    pts = [0, 1, 2, 3]
    assert downsample(pts, 4) == [0, 1, 2, 3]
    assert downsample(pts, 10) == [0, 1, 2, 3]


def test_downsample_keeps_endpoints_and_length_cap():
    pts = list(range(100))
    out = downsample(pts, 10)
    assert len(out) == 10
    assert out[0] == 0
    assert out[-1] == 99
    assert out == sorted(out)
    assert len(set(out)) == len(out)


def test_downsample_is_evenly_spaced():
    pts = list(range(11))  # indices 0..10
    assert downsample(pts, 6) == [0, 2, 4, 6, 8, 10]


def test_downsample_to_two_points_keeps_first_and_last():
    pts = list(range(7))
    assert downsample(pts, 2) == [0, 6]


def test_downsample_rejects_cap_below_two():
    with pytest.raises(ValueError):
        downsample([1, 2, 3], 1)


def test_track_idle_over_ten_minutes_is_dropped_and_active_track_kept():
    buf = TrajectoryBuffer()
    buf.append('idle', 0, (0, 0, 1, 1))
    buf.append('active', 0, (0, 0, 1, 1))
    # 'active' keeps moving; 'idle' stops at t=0. Advance 11 minutes.
    buf.append('active', 11 * 60_000, (0, 0, 1, 1))
    assert buf.window('idle', 0, 11 * 60_000) == []
    assert len(buf.window('active', 0, 11 * 60_000)) == 2


def test_prune_returns_removed_count():
    buf = TrajectoryBuffer()
    buf.append('a', 0, (0, 0, 1, 1))
    buf.append('b', 0, (0, 0, 1, 1))
    assert buf.prune(10 * 60_000 + 1) == 2
