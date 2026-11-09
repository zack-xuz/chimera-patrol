"""CaptureThread、FrameSlot、DetectQueue 单元测试。

使用注入的假 VideoCapture（不依赖真实视频文件），等待函数同样注入，测试无需真实休眠。
"""

from __future__ import annotations

import queue
import time
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np
import pytest

from chimera_patrol.config import VideoSection
from chimera_patrol.ingest.camera import Camera
from chimera_patrol.ingest.capture import CaptureThread, DetectItem, DetectQueue, FrameSlot


def _frame(value: int) -> np.ndarray:
    return np.full((8, 8, 3), value, dtype=np.uint8)


class FakeCapture:
    """按列表依次返回帧；列表耗尽后若设置了 endless 则重复返回该帧，否则读取失败。

    ``on_read(n)`` 在第 n 次 read 时调用，用于在测试中停止线程。
    """

    def __init__(
        self,
        frames: list[np.ndarray] | tuple[()] = (),
        *,
        opened: bool = True,
        fps: float = 25.0,
        endless: np.ndarray | None = None,
        on_read: Callable[[int], None] | None = None,
    ) -> None:
        self._frames = list(frames)
        self._opened = opened
        self._fps = fps
        self._endless = endless
        self._on_read = on_read
        self._pos = 0
        self._reads = 0
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802
        return self._opened

    def read(self) -> tuple[bool, Any]:
        self._reads += 1
        if self._on_read is not None:
            self._on_read(self._reads)
        if self._pos < len(self._frames):
            frame = self._frames[self._pos]
            self._pos += 1
            return True, frame
        if self._endless is not None:
            return True, self._endless
        return False, None

    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_FPS:
            return self._fps
        if prop_id == cv2.CAP_PROP_POS_FRAMES:
            return float(self._pos)
        return 0.0

    def set(self, prop_id: int, value: float) -> bool:
        if prop_id == cv2.CAP_PROP_POS_FRAMES:
            self._pos = int(value)
            return True
        return False

    def release(self) -> None:
        self.released = True


class FakeRing:
    def __init__(self) -> None:
        self.pushes: list[tuple[int, bytes]] = []

    def push(self, ts_ms: int, jpeg: bytes) -> None:
        self.pushes.append((ts_ms, jpeg))


class Recorder:
    def __init__(self) -> None:
        self.statuses: list[tuple[str, str]] = []
        self.restarts: list[str] = []

    def on_status(self, camera_id: str, status: str, ts_ms: int) -> None:
        assert ts_ms > 0
        self.statuses.append((camera_id, status))

    def on_restart(self, camera_id: str, ts_ms: int) -> None:
        self.restarts.append(camera_id)


def _video(**overrides: Any) -> VideoSection:
    values: dict[str, Any] = {
        "loop": False,
        "default_fps": 25,
        "offline_after_failures": 3,
        "reconnect_interval_s": 7,
    }
    values.update(overrides)
    return VideoSection(**values)


def _file_camera() -> Camera:
    return Camera(id="cam_file", name="cam_file", source_type="file", source="clip.mp4")


def _rtsp_camera(camera_id: str = "cam_rtsp") -> Camera:
    return Camera(
        id=camera_id,
        name=camera_id,
        source_type="rtsp",
        source="rtsp://user:secret@example.invalid/stream",
    )


def _drain(dq: DetectQueue) -> list[DetectItem]:
    items: list[DetectItem] = []
    while True:
        try:
            items.append(dq.get(timeout=0.01))
        except queue.Empty:
            return items


def _wait_until(predicate: Callable[[], bool], timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_frames_reach_frame_slot_and_ring_buffer_then_file_end_marks_offline() -> None:
    frames = [_frame(i * 10) for i in range(5)]
    slot = FrameSlot()
    ring = FakeRing()
    rec = Recorder()
    thread = CaptureThread(
        _file_camera(),
        _video(),
        frame_skip_n=1,
        frame_slot=slot,
        detect_queue=DetectQueue(maxsize=10),
        on_status_change=rec.on_status,
        ring_buffer=ring,
        ring_fps=10,
        capture_factory=lambda source: FakeCapture(frames),
        wait=lambda seconds: None,
    )

    thread.run()

    latest = slot.get_latest()
    assert latest is not None
    assert latest.frame_index == 4
    assert np.array_equal(latest.image, frames[4])
    assert ring.pushes, "首帧应写入 RingBuffer"
    assert ring.pushes[0][1][:2] == b"\xff\xd8", "写入的应为 JPEG"
    # 未开启循环的文件读到末尾后采集结束，状态置为离线
    assert rec.statuses == [("cam_file", "online"), ("cam_file", "offline")]


def test_every_nth_frame_is_queued_for_detection() -> None:
    frames = [_frame(i) for i in range(7)]
    dq = DetectQueue(maxsize=10)
    thread = CaptureThread(
        _file_camera(),
        _video(),
        frame_skip_n=3,
        frame_slot=FrameSlot(),
        detect_queue=dq,
        on_status_change=Recorder().on_status,
        capture_factory=lambda source: FakeCapture(frames),
        wait=lambda seconds: None,
    )

    thread.run()

    items = _drain(dq)
    assert [item.frame_index for item in items] == [0, 3, 6]
    assert all(item.camera_id == "cam_file" for item in items)
    assert np.array_equal(items[-1].image, frames[6])


def test_repeated_read_failures_go_offline_once_then_recover_online() -> None:
    holder: dict[str, CaptureThread] = {}
    bad = FakeCapture(())  # 打开成功但读不到帧
    good = FakeCapture(
        [_frame(1), _frame(2), _frame(3)],
        on_read=lambda n: holder["thread"].stop() if n >= 3 else None,
    )
    sources = [bad, good]

    def factory(source: str) -> FakeCapture:
        if sources:
            return sources.pop(0)
        holder["thread"].stop()
        return FakeCapture(opened=False)

    rec = Recorder()
    waits: list[float] = []
    slot = FrameSlot()
    thread = CaptureThread(
        _rtsp_camera(),
        _video(),
        frame_skip_n=1,
        frame_slot=slot,
        detect_queue=DetectQueue(maxsize=10),
        on_status_change=rec.on_status,
        capture_factory=factory,
        wait=waits.append,
    )
    holder["thread"] = thread

    thread.run()

    assert rec.statuses == [("cam_rtsp", "offline"), ("cam_rtsp", "online")]
    assert bad.released, "离线后应释放旧源"
    # 前两次失败按源帧间隔（25 fps）等待，第 3 次失败置离线后按 reconnect_interval_s 等待
    assert waits == [pytest.approx(0.04), pytest.approx(0.04), 7]
    latest = slot.get_latest()
    assert latest is not None and latest.frame_index == 2


def test_loop_rewinds_file_source_and_notifies_restart() -> None:
    holder: dict[str, CaptureThread] = {}
    frames = [_frame(1), _frame(2), _frame(3)]
    cap = FakeCapture(
        frames,
        on_read=lambda n: holder["thread"].stop() if n >= 9 else None,
    )
    rec = Recorder()
    slot = FrameSlot()
    thread = CaptureThread(
        _file_camera(),
        _video(loop=True),
        frame_skip_n=1,
        frame_slot=slot,
        detect_queue=DetectQueue(maxsize=10),
        on_status_change=rec.on_status,
        on_loop_restart=rec.on_restart,
        capture_factory=lambda source: cap,
        wait=lambda seconds: None,
    )
    holder["thread"] = thread

    thread.run()

    # 读完 3 帧后回绕两次，每次回绕后读到的首帧都触发一次重启回调
    assert rec.restarts == ["cam_file", "cam_file"]
    latest = slot.get_latest()
    assert latest is not None
    assert latest.frame_index == 6
    assert np.array_equal(latest.image, frames[0])
    assert rec.statuses == [("cam_file", "online")]


def test_drop_oldest_keeps_newest_item() -> None:
    dq = DetectQueue(maxsize=1)
    items = [DetectItem("cam", i, i * 100, _frame(i)) for i in range(3)]

    dropped = [dq.put_drop_oldest(item) for item in items]

    assert dropped == [False, True, True]
    assert dq.get(timeout=0.1).frame_index == 2
    with pytest.raises(queue.Empty):
        dq.get(timeout=0.01)


def test_broken_source_does_not_stop_other_capture_threads() -> None:
    slot_ok = FrameSlot()
    slot_bad = FrameSlot()
    rec_bad = Recorder()
    rec_ok = Recorder()
    bad = CaptureThread(
        _rtsp_camera("cam_bad"),
        _video(reconnect_interval_s=1),
        frame_skip_n=1,
        frame_slot=slot_bad,
        detect_queue=DetectQueue(maxsize=1),
        on_status_change=rec_bad.on_status,
        capture_factory=lambda source: FakeCapture(opened=False),
    )
    good = CaptureThread(
        _rtsp_camera("cam_ok"),
        _video(),
        frame_skip_n=1,
        frame_slot=slot_ok,
        detect_queue=DetectQueue(maxsize=1),
        on_status_change=rec_ok.on_status,
        capture_factory=lambda source: FakeCapture(endless=_frame(9)),
    )

    bad.start()
    good.start()
    try:
        assert _wait_until(lambda: bad.status == "offline")
        assert _wait_until(
            lambda: slot_ok.get_latest() is not None and slot_ok.get_latest().frame_index >= 5
        )
        assert good.is_alive()
        assert ("cam_ok", "online") in rec_ok.statuses
    finally:
        bad.stop()
        good.stop()
        bad.join(5)
        good.join(5)

    assert not bad.is_alive()
    assert not good.is_alive()
