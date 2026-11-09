"""PersonTracker 单元测试（任务 3.2，库为 trackers 2.7.0 的 ByteTrackTracker）。

ByteTrack 默认需要 minimum_consecutive_frames 帧确认，前几帧可能没有输出轨迹，因此测试先预热再断言。
"""

from __future__ import annotations

from chimera_patrol.detect.tracker import PersonTracker, lost_buffer_frames


def _moving_box(i: int) -> tuple[tuple[float, float, float, float], float]:
    x = 100.0 + 2.0 * i
    return (x, 100.0, x + 60.0, 220.0), 0.9


def _ids_after_warmup(tracker: PersonTracker, frames: int, start: int = 0) -> list[int]:
    ids: list[int] = []
    for i in range(start, start + frames):
        for tid, _box, _conf in tracker.update([_moving_box(i)]):
            ids.append(tid)
    return ids


def test_same_person_keeps_track_id_over_30_frames() -> None:
    tracker = PersonTracker(frame_rate=25.0, lost_buffer_s=2.0)
    ids = _ids_after_warmup(tracker, 30)
    assert len(ids) >= 20, "track should be confirmed after warm-up"
    assert len(set(ids)) == 1


def test_two_instances_do_not_share_state() -> None:
    a = PersonTracker(frame_rate=25.0, lost_buffer_s=2.0)
    b = PersonTracker(frame_rate=25.0, lost_buffer_s=2.0)
    a_ids = _ids_after_warmup(a, 15)
    b_ids = _ids_after_warmup(b, 15)
    assert a_ids and b_ids
    assert set(a_ids) == set(b_ids) == {a_ids[0]}, "independent trackers both start from the same id"


def test_reset_starts_a_fresh_track_id_sequence() -> None:
    tracker = PersonTracker(frame_rate=25.0, lost_buffer_s=2.0)
    before = _ids_after_warmup(tracker, 15)
    tracker.reset()
    after = _ids_after_warmup(tracker, 15)
    assert before and after
    assert after[0] == before[0], "a reset tracker restarts its id sequence"


def test_empty_frame_returns_no_tracks() -> None:
    tracker = PersonTracker(frame_rate=25.0, lost_buffer_s=2.0)
    assert tracker.update([]) == []


def test_lost_buffer_conversion_to_30fps_units() -> None:
    assert lost_buffer_frames(2.0) == 60
    assert lost_buffer_frames(1.0) == 30
    assert lost_buffer_frames(0.01) == 1