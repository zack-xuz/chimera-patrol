"""RingBuffer: per-camera in-memory cache of recent JPEG frames (task 5.1).

The buffer keeps frames as ``(ts_ms, jpeg_bytes)`` tuples in ascending time
order. It is independent of configuration: callers pass ``window_s`` and
``max_fps`` explicitly. Methods are synchronous and guarded by a lock, because
the capture thread pushes frames while other threads take snapshots.
"""
from __future__ import annotations

import bisect
import threading
from collections.abc import Sequence

# A stored frame record: (timestamp in milliseconds, JPEG bytes).
Frame = tuple[int, bytes]


class RingBuffer:
    """Keeps the last ``window_s`` seconds of frames, capped at ``max_fps`` frames per second."""

    def __init__(self, window_s: float, max_fps: float) -> None:
        if window_s <= 0:
            raise ValueError(f"window_s must be > 0, got {window_s}")
        if max_fps <= 0:
            raise ValueError(f"max_fps must be > 0, got {max_fps}")
        self._window_ms: float = window_s * 1000.0
        self._min_interval_ms: float = 1000.0 / max_fps
        self._timestamps: list[int] = []
        self._frames: list[Frame] = []
        self._lock = threading.Lock()

    def push(self, ts_ms: int, jpeg_bytes: bytes) -> bool:
        """Store a frame if the rate cap allows it. Return True when stored.

        Raises ValueError when ``ts_ms`` is earlier than the newest stored frame.
        A frame arriving sooner than ``1000 / max_fps`` ms after the last stored
        frame is dropped and False is returned.
        """
        with self._lock:
            if self._frames:
                last_ts = self._timestamps[-1]
                if ts_ms < last_ts:
                    raise ValueError(
                        f"out-of-order frame: ts_ms={ts_ms} is earlier than newest {last_ts}"
                    )
                if ts_ms - last_ts < self._min_interval_ms:
                    return False

            self._timestamps.append(ts_ms)
            self._frames.append((ts_ms, jpeg_bytes))

            cutoff = ts_ms - self._window_ms
            evict = bisect.bisect_left(self._timestamps, cutoff)
            if evict:
                del self._timestamps[:evict]
                del self._frames[:evict]
            return True

    def snapshot(self, from_ms: int, to_ms: int) -> list[Frame]:
        """Return frames with ``from_ms <= ts_ms <= to_ms`` in ascending time order.

        Returns an empty list when ``from_ms > to_ms``.
        """
        if from_ms > to_ms:
            return []
        with self._lock:
            lo = bisect.bisect_left(self._timestamps, from_ms)
            hi = bisect.bisect_right(self._timestamps, to_ms)
            return self._frames[lo:hi]


def select_frame(frames: Sequence[Frame], mid_ms: float) -> Frame:
    """Return the frame whose timestamp is closest to ``mid_ms``.

    Ties go to the earlier frame. Raises ValueError when ``frames`` is empty.
    """
    if not frames:
        raise ValueError("select_frame requires at least one frame")
    return min(frames, key=lambda frame: (abs(frame[0] - mid_ms), frame[0]))
