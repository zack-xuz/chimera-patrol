"""每路 Camera 的检测暂停状态（只存内存，不落库，进程重启后全部为运行状态）。

暂停只影响检测与报警：DetectWorker 在暂停期间丢弃检测帧，不产生候选事件。
采集线程与 MJPEG 视频流不受影响。已经投递到流水线的候选事件与在途的 VLM 复核照常完成。
"""

from __future__ import annotations

import threading


class CameraControl:
    """线程安全的暂停登记表。检测线程读取，API 线程写入。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._paused: set[str] = set()

    def is_paused(self, camera_id: str) -> bool:
        with self._lock:
            return camera_id in self._paused

    def pause(self, camera_id: str) -> None:
        with self._lock:
            self._paused.add(camera_id)

    def resume(self, camera_id: str) -> None:
        with self._lock:
            self._paused.discard(camera_id)

    def snapshot(self) -> set[str]:
        with self._lock:
            return set(self._paused)
