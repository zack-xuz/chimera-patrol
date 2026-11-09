"""DetectWorker 单元测试（任务 3.3）：异常帧隔离、候选投递、循环重启重置。"""

from __future__ import annotations

import queue
import time

import numpy as np

from chimera_patrol.config import RulesSection
from chimera_patrol.config import RoiConfig
from chimera_patrol.detect.tracker import PersonTracker
from chimera_patrol.detect.worker import DetectWorker, scale_rois
from chimera_patrol.ingest.capture import DetectItem, DetectQueue

BOX = (100.0, 50.0, 160.0, 230.0)
FULL = [[0, 0], [1, 0], [1, 1], [0, 1]]


class Detector:
    def __init__(self, fail_first: bool = False) -> None:
        self.calls = 0
        self._fail_first = fail_first

    def detect_persons(self, frame: np.ndarray):
        self.calls += 1
        if self._fail_first and self.calls == 1:
            raise RuntimeError("inference failed")
        return [(BOX, 0.9)]

    def detect_fire_smoke(self, frame: np.ndarray):
        return []

    def detect_helmets(self, frame: np.ndarray):
        return []


def _item(index: int) -> DetectItem:
    return DetectItem("cam", index, 1_000 + index * 40, np.zeros((240, 320, 3), dtype=np.uint8))


def _worker(detector: Detector, submitted: list, rois=None) -> DetectWorker:
    roi_list = rois if rois is not None else [RoiConfig(id="z", type="intrusion", polygon=FULL)]
    return DetectWorker(
        camera_id="cam",
        detect_queue=DetectQueue(maxsize=4),
        detector=detector,
        tracker=PersonTracker(frame_rate=25.0, lost_buffer_s=2.0),
        rules=RulesSection(),
        rois=roi_list,
        submit=submitted.append,
        poll_s=0.01,
    )


def test_scale_rois_maps_normalized_polygon_to_pixels() -> None:
    (roi,) = scale_rois([RoiConfig(id="z", type="intrusion", polygon=[[0.5, 0.5], [1, 0], [0, 1]])], 320, 240)
    assert roi.polygon == [(160.0, 120.0), (320.0, 0.0), (0.0, 240.0)]


def test_failed_frame_is_dropped_and_later_frames_still_processed() -> None:
    submitted: list = []
    detector = Detector(fail_first=True)
    worker = _worker(detector, submitted)
    worker.start()
    try:
        for index in range(10):
            worker._queue.put_drop_oldest(_item(index))
            time.sleep(0.05)
        deadline = time.monotonic() + 5
        while not submitted and time.monotonic() < deadline:
            time.sleep(0.02)
    finally:
        worker.stop()
        worker.join(5)
    assert detector.calls >= 3, "worker must keep consuming after the failed frame"
    assert "intrusion" in [c.event_type for c in submitted]


def test_request_reset_clears_tracker_and_rules_before_next_frame() -> None:
    submitted: list = []
    worker = _worker(Detector(), submitted)
    for index in range(5):
        worker._handle(_item(index))
    # ByteTrack confirms the track after warm-up; the zone entry fires once, then cooldown suppresses repeats.
    assert len(submitted) == 1
    worker.request_reset()
    for index in range(5, 10):
        worker._handle(_item(index))
    # After reset the rule engine and cooldown are fresh, so the entry fires again.
    assert len(submitted) == 2