"""Video_Ingest 采集线程（design.md「线程与异步模型」「视频接入（Video_Ingest）」）。

- ``FrameSlot``：每路最新原始帧（覆盖写），供 MJPEG 读取；可选保存最近一次检测叠加。
- ``DetectQueue``：``maxsize=1`` 的队列，满时丢弃旧帧，检测慢不会阻塞采集。
- ``CaptureThread``：每路 Camera 一个守护线程。读帧、按源 fps 节拍、写 FrameSlot 与
  RingBuffer、按抽帧间隔投递到 DetectQueue。连续读帧失败置为离线并按间隔重连；
  文件源读到末尾时循环回绕并通知重置（假设 A4）。线程内不抛出异常。

依赖（VideoCapture 工厂、墙钟、单调时钟、等待函数、RingBuffer、回调）全部通过参数注入。
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol

import cv2

from chimera_patrol.ingest.camera import mask_source

if TYPE_CHECKING:
    import numpy as np

    from chimera_patrol.config import VideoSection
    from chimera_patrol.ingest.camera import Camera

log = logging.getLogger(__name__)

# (camera_id, "online" | "offline", ts_ms)
StatusCallback = Callable[[str, str, int], None]
# (camera_id, ts_ms)
LoopCallback = Callable[[str, int], None]


class CaptureLike(Protocol):
    """cv2.VideoCapture 所需的最小接口，便于测试注入假源。"""

    def isOpened(self) -> bool: ...  # noqa: N802 - 与 OpenCV 命名保持一致
    def read(self) -> tuple[bool, Any]: ...
    def get(self, prop_id: int) -> float: ...
    def set(self, prop_id: int, value: float) -> bool: ...
    def release(self) -> None: ...


class RingBufferLike(Protocol):
    """环形缓存接口：按时间戳保存一帧 JPEG。"""

    def push(self, ts_ms: int, jpeg: bytes) -> None: ...


def _open_video_capture(source: str) -> CaptureLike:
    return cv2.VideoCapture(source)


def _wall_ms() -> int:
    """采集时刻的 UTC 毫秒时间戳（假设 A4）。"""
    return time.time_ns() // 1_000_000


# ---------------------------------------------------------------- FrameSlot


@dataclass(frozen=True)
class LatestFrame:
    image: np.ndarray
    ts_ms: int
    frame_index: int


class FrameSlot:
    """每路最新原始帧的线程安全容器。

    ``put`` 不复制图像，调用方写入后不得再修改该数组。
    ``overlay`` 保存 DetectWorker 最近一次发布的检测结果，默认 None。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: LatestFrame | None = None
        self._overlay: Any = None

    def put(self, image: np.ndarray, ts_ms: int, frame_index: int) -> None:
        with self._lock:
            self._frame = LatestFrame(image=image, ts_ms=ts_ms, frame_index=frame_index)

    def get_latest(self) -> LatestFrame | None:
        with self._lock:
            return self._frame

    @property
    def overlay(self) -> Any:
        with self._lock:
            return self._overlay

    @overlay.setter
    def overlay(self, value: Any) -> None:
        with self._lock:
            self._overlay = value


# ---------------------------------------------------------------- DetectQueue


class DetectItem(NamedTuple):
    camera_id: str
    frame_index: int
    ts_ms: int
    image: np.ndarray


class DetectQueue:
    """``queue.Queue`` 的包装，满时丢弃最旧的项，保证生产者永不阻塞。"""

    def __init__(self, maxsize: int = 1) -> None:
        if maxsize < 1:
            raise ValueError("DetectQueue maxsize 必须大于 0")
        self._q: queue.Queue[DetectItem] = queue.Queue(maxsize=maxsize)

    def put_drop_oldest(self, item: DetectItem) -> bool:
        """放入 item；队列满时先丢弃最旧的项。返回 True 表示丢弃了旧项。"""
        dropped = False
        while True:
            try:
                self._q.put_nowait(item)
                return dropped
            except queue.Full:
                try:
                    self._q.get_nowait()
                    dropped = True
                except queue.Empty:
                    pass

    def get(self, timeout: float | None = None) -> DetectItem:
        """取出一项；超时抛出 ``queue.Empty``。"""
        return self._q.get(timeout=timeout)


# ---------------------------------------------------------------- CaptureThread


def _read_fps(cap: CaptureLike) -> float | None:
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
    except Exception:
        return None
    if not math.isfinite(fps) or fps <= 0:
        return None
    return fps


def _safe_release(cap: CaptureLike) -> None:
    try:
        cap.release()
    except Exception:
        log.debug("释放视频源时出错", exc_info=True)


class CaptureThread(threading.Thread):
    """每路 Camera 一个采集线程。

    参数:
        camera: 视频源描述（file 或 rtsp）。
        video_cfg: ``video`` 配置段，使用 loop、default_fps、offline_after_failures、reconnect_interval_s。
        frame_skip_n: 每 N 帧投递一次检测（frame_index % N == 0）。
        frame_slot: 写入最新原始帧。
        detect_queue: 投递检测帧，满时丢旧帧。
        on_status_change: 状态变化回调 ``(camera_id, status, ts_ms)``。
        ring_buffer: 可选，``push(ts_ms, jpeg)``；为 None 时不写缓存。
        ring_fps: 写入 RingBuffer 的最大帧率（evidence.ring_fps）。
        ring_jpeg_quality: RingBuffer 的 JPEG 质量（evidence.ring_jpeg_quality）。
        on_loop_restart: 文件源循环回绕时调用 ``(camera_id, ts_ms)``，用于重置 tracker 与规则状态（A4）。
        capture_factory: 由源地址创建 VideoCapture 的工厂，测试时注入假源。
        wall_ms: 返回采集时刻的 UTC 毫秒时间戳。
        clock: 单调时钟，用于文件源节拍。
        wait: 可选的等待函数 ``wait(seconds)``；None 时使用可被 stop() 唤醒的等待。
    """

    def __init__(
        self,
        camera: Camera,
        video_cfg: VideoSection,
        frame_skip_n: int,
        frame_slot: FrameSlot,
        detect_queue: DetectQueue,
        on_status_change: StatusCallback,
        *,
        ring_buffer: RingBufferLike | None = None,
        ring_fps: int = 10,
        ring_jpeg_quality: int = 90,
        on_loop_restart: LoopCallback | None = None,
        capture_factory: Callable[[str], CaptureLike] = _open_video_capture,
        wall_ms: Callable[[], int] = _wall_ms,
        clock: Callable[[], float] = time.monotonic,
        wait: Callable[[float], None] | None = None,
    ) -> None:
        super().__init__(name=f"capture-{camera.id}", daemon=True)
        if frame_skip_n < 1:
            raise ValueError("frame_skip_n 必须大于 0")
        if ring_fps < 1:
            raise ValueError("ring_fps 必须大于 0")
        self._camera = camera
        self._video = video_cfg
        self._frame_skip_n = frame_skip_n
        self._frame_slot = frame_slot
        self._detect_queue = detect_queue
        self._on_status_change = on_status_change
        self._ring_buffer = ring_buffer
        self._ring_interval_ms = 1000.0 / ring_fps
        self._ring_jpeg_quality = int(ring_jpeg_quality)
        self._on_loop_restart = on_loop_restart
        self._factory = capture_factory
        self._wall_ms = wall_ms
        self._clock = clock
        self._wait_fn = wait

        self._is_file = camera.source_type == "file"
        self._default_period = 1.0 / video_cfg.default_fps
        self._frame_period = self._default_period
        self._stop_event = threading.Event()
        self._cap: CaptureLike | None = None
        self._status: str | None = None
        self._consec_failures = 0
        self._frame_index = 0
        self._next_due: float | None = None
        self._last_ring_ts: int | None = None

    @property
    def status(self) -> str | None:
        """当前状态 ``online`` / ``offline``，尚未读到过帧时为 None。"""
        return self._status

    def stop(self) -> None:
        """请求线程退出；线程结束时释放 VideoCapture。"""
        self._stop_event.set()

    def run(self) -> None:
        try:
            self._loop()
        except Exception:
            log.exception("采集线程异常退出 camera=%s", self._camera.id)
        finally:
            self._release()

    # ------------------------------------------------------------ main loop

    def _loop(self) -> None:
        video = self._video
        while not self._stop_event.is_set():
            if self._cap is None:
                self._cap = self._open()
            if self._cap is not None:
                if self._is_file:
                    self._pace()
                frame, rewound = self._read()
                if frame is not None:
                    self._consec_failures = 0
                    self._set_status("online")
                    if rewound:
                        self._notify_loop_restart()
                    self._handle_frame(frame)
                    continue
                if self._is_file and not video.loop:
                    log.info("文件源已读到末尾且未开启循环，采集结束 camera=%s", self._camera.id)
                    self._set_status("offline")
                    return
            self._on_failure()
            if self._status == "offline":
                # 离线后释放源，等待 reconnect_interval_s 再重新打开。
                self._release()
                self._idle(video.reconnect_interval_s)
            else:
                self._idle(self._default_period)

    def _on_failure(self) -> None:
        self._consec_failures += 1
        if (
            self._status != "offline"
            and self._consec_failures >= self._video.offline_after_failures
        ):
            log.warning(
                "连续读帧失败 %d 次，置为离线 camera=%s", self._consec_failures, self._camera.id
            )
            self._set_status("offline")

    def _open(self) -> CaptureLike | None:
        """打开视频源；失败返回 None 并计入失败次数，不抛出异常。"""
        source = self._camera.source
        try:
            cap = self._factory(source)
            opened = bool(cap.isOpened())
        except Exception:
            log.warning("打开视频源异常 camera=%s", self._camera.id, exc_info=True)
            return None
        if not opened:
            log.warning(
                "无法打开视频源 camera=%s source=%s", self._camera.id, mask_source(source)
            )
            _safe_release(cap)
            return None
        if self._is_file:
            fps = _read_fps(cap) or self._video.default_fps
            self._frame_period = 1.0 / fps
            self._next_due = None
        return cap

    def _read(self) -> tuple[Any, bool]:
        """读取一帧。返回 (frame, rewound)；失败返回 (None, False)。

        文件源开启循环时，读取失败（即到达末尾）后回绕到开头再读一次；
        回绕后读到的帧 rewound 为 True，由调用方触发循环重启回调。
        """
        cap = self._cap
        if cap is None:
            return None, False
        try:
            ok, frame = cap.read()
            if ok and frame is not None:
                return frame, False
            if self._is_file and self._video.loop:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
                if ok and frame is not None:
                    return frame, True
        except Exception:
            log.warning("读帧异常 camera=%s", self._camera.id, exc_info=True)
        return None, False

    def _pace(self) -> None:
        """文件源按源 fps 节拍播放（A4）。落后超过一帧时重新对齐，避免连续突发。"""
        period = self._frame_period
        now = self._clock()
        if self._next_due is None:
            self._next_due = now
        delay = self._next_due - now
        if delay > 0:
            self._idle(delay)
        self._next_due += period
        if self._clock() - self._next_due > period:
            self._next_due = self._clock()

    def _handle_frame(self, frame: Any) -> None:
        try:
            ts = self._wall_ms()
            idx = self._frame_index
            self._frame_index += 1
            self._frame_slot.put(frame, ts, idx)

            if self._ring_buffer is not None and (
                self._last_ring_ts is None or ts - self._last_ring_ts >= self._ring_interval_ms
            ):
                ok, buf = cv2.imencode(
                    ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self._ring_jpeg_quality]
                )
                if ok:
                    self._ring_buffer.push(ts, buf.tobytes())
                    self._last_ring_ts = ts

            if idx % self._frame_skip_n == 0:
                self._detect_queue.put_drop_oldest(
                    DetectItem(self._camera.id, idx, ts, frame)
                )
        except Exception:
            log.exception("处理采集帧失败 camera=%s frame_index=%s", self._camera.id, self._frame_index)

    # ------------------------------------------------------------ helpers

    def _set_status(self, status: str) -> None:
        if status == self._status:
            return
        self._status = status
        log.info("视频源状态变化 camera=%s status=%s", self._camera.id, status)
        try:
            self._on_status_change(self._camera.id, status, self._wall_ms())
        except Exception:
            log.exception("状态回调异常 camera=%s", self._camera.id)

    def _notify_loop_restart(self) -> None:
        # 回绕后时间戳不连续，清空缓存节流状态，让新一轮的首帧立即入缓存。
        self._last_ring_ts = None
        if self._on_loop_restart is None:
            return
        try:
            self._on_loop_restart(self._camera.id, self._wall_ms())
        except Exception:
            log.exception("循环重启回调异常 camera=%s", self._camera.id)

    def _idle(self, seconds: float) -> None:
        if self._wait_fn is not None:
            self._wait_fn(seconds)
        else:
            self._stop_event.wait(seconds)

    def _release(self) -> None:
        cap = self._cap
        self._cap = None
        self._next_due = None
        if cap is not None:
            _safe_release(cap)
