"""Unit tests for chimera_patrol.detect.detector (task 3.1).

所有测试都注入假模型（fake model），不下载也不加载真实 YOLO 权重。
"""

from __future__ import annotations

import threading
from pathlib import Path
import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from chimera_patrol.config import DetectorSection
from chimera_patrol.detect.detector import Detector

FRAME = np.zeros((48, 64, 3), dtype=np.uint8)


class FakeBoxes:
    """模拟 ultralytics.engine.results.Boxes 的 xyxy / conf / cls 字段（numpy 实现）。"""

    def __init__(self, rows: Sequence[tuple[float, float, float, float, float, int]]) -> None:
        data = np.asarray(rows, dtype=np.float32).reshape(-1, 6)
        self.xyxy = data[:, :4]
        self.conf = data[:, 4]
        self.cls = data[:, 5].astype(np.int64)

    def __len__(self) -> int:
        return int(self.conf.shape[0])


class FakeResult:
    def __init__(self, names: dict[int, str], rows: Sequence[tuple[float, float, float, float, float, int]]) -> None:
        self.names = names
        self.boxes = FakeBoxes(rows)


class FakeModel:
    """记录 predict 调用；返回由 responder 决定的 Results 列表。"""

    def __init__(self, names: dict[int, str], responder: Callable[[Any], list[FakeResult]] | None = None) -> None:
        self.names = names
        self.calls: list[dict[str, Any]] = []
        self._responder = responder or (lambda _frame: [FakeResult(names, [])])

    def predict(self, source: Any, **kwargs: Any) -> list[FakeResult]:
        self.calls.append({"source": source, **kwargs})
        return self._responder(source)


def _detector_cfg(**overrides: Any) -> DetectorSection:
    base: dict[str, Any] = {
        "person": {"weights": "weights/person.pt", "classes": ["person"], "conf": 0.4},
        "fire_smoke": {"weights": "weights/fire_smoke.pt", "classes": ["smoke", "fire"]},
        "helmet": {
            "weights": "weights/helmet.pt",
            "helmet_classes": ["helmet"],
            "no_helmet_classes": ["no-helmet", "head"],
        },
    }
    base.update(overrides)
    return DetectorSection.model_validate(base)


class CountingFactory:
    """记录每次加载的权重路径，并为每个路径返回一个假模型。"""

    def __init__(self, responders: dict[str, Callable[[Any], list[FakeResult]]] | None = None) -> None:
        self.paths: list[str] = []
        self.models: dict[str, FakeModel] = {}
        self._responders = responders or {}

    def __call__(self, weights: str) -> FakeModel:
        self.paths.append(weights)
        model = FakeModel(
            {0: "person", 1: "smoke", 2: "fire", 3: "helmet", 4: "no-helmet", 5: "head", 6: "hat"},
            self._responders.get(Path(weights).name),
        )
        self.models[Path(weights).name] = model
        return model


def test_shared_weights_are_loaded_once() -> None:
    cfg = _detector_cfg(
        fire_smoke={"weights": "weights/person.pt", "classes": ["smoke", "fire"]},
    )
    factory = CountingFactory()

    detector = Detector(cfg, model_factory=factory)

    assert len(factory.paths) == 2
    assert detector.loaded_weight_count == 2
    assert detector._person_model is detector._fire_smoke_model


def test_all_three_roles_on_one_file_load_once() -> None:
    cfg = _detector_cfg(
        person={"weights": "weights/same.pt", "classes": ["person"], "conf": 0.4},
        fire_smoke={"weights": "weights/same.pt", "classes": ["smoke", "fire"]},
        helmet={"weights": "weights/same.pt", "helmet_classes": ["helmet"], "no_helmet_classes": ["head"]},
    )
    factory = CountingFactory()

    detector = Detector(cfg, model_factory=factory)

    assert len(factory.paths) == 1
    assert detector._person_model is detector._fire_smoke_model is detector._helmet_model


def test_distinct_weights_load_once_each() -> None:
    factory = CountingFactory()

    detector = Detector(_detector_cfg(), model_factory=factory)

    assert len(factory.paths) == 3
    assert len(set(factory.paths)) == 3
    assert detector._person_model is not detector._fire_smoke_model
    assert detector._fire_smoke_model is not detector._helmet_model


def test_person_filters_by_class_and_conf() -> None:
    def responder(_frame: Any) -> list[FakeResult]:
        return [
            FakeResult(
                {0: "person", 6: "hat"},
                [
                    (1, 2, 11, 22, 0.9, 0),  # person，保留
                    (3, 4, 13, 24, 0.3, 0),  # person 但 conf 低于 0.4，丢弃
                    (5, 6, 15, 26, 0.8, 6),  # 非 person 类，丢弃
                ],
            )
        ]

    factory = CountingFactory({"person.pt": responder})
    detector = Detector(_detector_cfg(), model_factory=factory)

    persons = detector.detect_persons(FRAME)

    assert len(persons) == 1
    (box, conf) = persons[0]
    assert box == (1.0, 2.0, 11.0, 22.0)
    assert abs(conf - 0.9) < 1e-6


def test_fire_smoke_returns_label_box_conf_for_configured_classes() -> None:
    def responder(_frame: Any) -> list[FakeResult]:
        return [
            FakeResult(
                {1: "smoke", 2: "fire", 7: "car"},
                [
                    (0, 0, 10, 10, 0.7, 1),  # smoke
                    (20, 20, 30, 30, 0.6, 2),  # fire
                    (40, 40, 50, 50, 0.9, 7),  # car，不在 classes 中，丢弃
                ],
            )
        ]

    factory = CountingFactory({"fire_smoke.pt": responder})
    detector = Detector(_detector_cfg(), model_factory=factory)

    results = detector.detect_fire_smoke(FRAME)

    assert [label for label, _, _ in results] == ["smoke", "fire"]
    assert results[0][1] == (0.0, 0.0, 10.0, 10.0)
    assert abs(results[1][2] - 0.6) < 1e-6


def test_helmet_classes_map_to_helmet_and_no_helmet() -> None:
    def responder(_frame: Any) -> list[FakeResult]:
        return [
            FakeResult(
                {3: "helmet", 4: "no-helmet", 5: "head", 6: "hat"},
                [
                    (0, 0, 5, 5, 0.8, 3),  # helmet -> helmet
                    (10, 0, 15, 5, 0.7, 4),  # no-helmet -> no_helmet
                    (20, 0, 25, 5, 0.6, 5),  # head -> no_helmet
                    (30, 0, 35, 5, 0.5, 6),  # hat 未配置，丢弃
                ],
            )
        ]

    factory = CountingFactory({"helmet.pt": responder})
    detector = Detector(_detector_cfg(), model_factory=factory)

    results = detector.detect_helmets(FRAME)

    assert [kind for kind, _, _ in results] == ["helmet", "no_helmet", "no_helmet"]
    assert results[0][1] == (0.0, 0.0, 5.0, 5.0)


def test_no_detections_returns_empty_lists() -> None:
    factory = CountingFactory()
    detector = Detector(_detector_cfg(), model_factory=factory)

    assert detector.detect_persons(FRAME) == []
    assert detector.detect_fire_smoke(FRAME) == []
    assert detector.detect_helmets(FRAME) == []


def test_device_auto_and_imgsz_are_passed_to_predict() -> None:
    factory = CountingFactory()
    detector = Detector(_detector_cfg(imgsz=320), model_factory=factory)

    detector.detect_persons(FRAME)

    call = factory.models["person.pt"].calls[0]
    assert call["imgsz"] == 320
    assert call["device"] is None  # auto 交给 Ultralytics 自动选择
    assert call["verbose"] is False


def test_explicit_device_is_passed_to_predict() -> None:
    factory = CountingFactory()
    detector = Detector(_detector_cfg(device="cpu"), model_factory=factory)

    detector.detect_fire_smoke(FRAME)

    assert factory.models["fire_smoke.pt"].calls[0]["device"] == "cpu"


def test_predict_calls_are_serialized_across_threads() -> None:
    lock = threading.Lock()
    state = {"active": 0, "max_active": 0, "calls": 0}

    def slow_responder(_frame: Any) -> list[FakeResult]:
        with lock:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
            state["calls"] += 1
        time.sleep(0.05)
        with lock:
            state["active"] -= 1
        return [FakeResult({0: "person"}, [(0, 0, 10, 10, 0.9, 0)])]

    factory = CountingFactory({"person.pt": slow_responder})
    detector = Detector(_detector_cfg(), model_factory=factory)

    outputs: list[list[Any]] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            outputs.append(detector.detect_persons(FRAME))
        except BaseException as exc:  # pragma: no cover - 仅用于把线程异常带回主线程
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not errors
    assert state["calls"] == 4
    assert state["max_active"] == 1
    assert len(outputs) == 4
    assert all(len(out) == 1 for out in outputs)
