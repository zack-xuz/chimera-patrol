"""YOLO 权重共享加载与串行推理（任务 3.1，设计「线程与异步模型」要点 1）。

- person、smoke/fire、helmet 三份权重各加载一次。配置中路径相同的角色共享同一个模型对象。
- 所有 predict 调用经同一把 ``threading.Lock`` 串行化，因此多路 Camera 的 DetectWorker 可共用一个 Detector。
- 本模块不做跟踪（任务 3.2），也不按规则阈值过滤（烟火与安全帽阈值在 Rule_Engine 中应用）。

Ultralytics 8.4.x 使用的 API（已对照官方文档与已安装的 8.4.208 源码核对）：

- ``YOLO(weights_path)`` 加载模型。
- ``model.predict(source, imgsz=..., device=..., verbose=...)`` 返回 ``list[Results]``，
  其中 ``source`` 可为 BGR 格式的 numpy 数组（OpenCV 帧）。
- ``Results.boxes`` 提供 ``xyxy``、``conf``、``cls`` 三个张量；``Results.names`` 为 ``dict[int, str]`` 类别映射。
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

from chimera_patrol.config import DetectorSection

logger = logging.getLogger(__name__)

BBox = tuple[float, float, float, float]
HelmetKind = Literal["helmet", "no_helmet"]


class YoloModel(Protocol):
    """Detector 所需的最小模型接口（真实实现为 ``ultralytics.YOLO``）。"""

    def predict(self, source: Any, **kwargs: Any) -> Sequence[Any]: ...


ModelFactory = Callable[[str], YoloModel]


def default_model_factory(weights: str) -> YoloModel:
    """按权重路径加载 Ultralytics 模型。延迟导入，避免仅导入本模块时加载 ultralytics。"""
    from ultralytics import YOLO

    return YOLO(weights)


@dataclass(frozen=True)
class _RawDetection:
    name: str
    box: BBox
    conf: float


def _to_numpy(value: Any) -> np.ndarray:
    """把 torch 张量（可能位于 GPU）或数组统一转为 CPU numpy 数组。"""
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _extract_detections(results: Sequence[Any]) -> list[_RawDetection]:
    """从 predict 返回的 Results 列表中提取 (类别名, 框, 置信度)。"""
    out: list[_RawDetection] = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            continue
        names: dict[int, str] = result.names
        xyxy = _to_numpy(boxes.xyxy).reshape(-1, 4).tolist()
        conf = _to_numpy(boxes.conf).reshape(-1).tolist()
        cls = _to_numpy(boxes.cls).reshape(-1).tolist()
        for coords, score, class_id in zip(xyxy, conf, cls):
            index = int(class_id)
            x1, y1, x2, y2 = (float(v) for v in coords)
            out.append(_RawDetection(names.get(index, str(index)), (x1, y1, x2, y2), float(score)))
    return out


def _normalize_key(weights: Path | str) -> str:
    """同一文件的不同写法（相对/绝对路径）映射到同一个键，确保只加载一次。"""
    return os.path.normcase(os.path.abspath(str(weights)))


class Detector:
    """持有三份（可共享）YOLO 模型，对外提供按角色过滤后的检测接口。

    ``model_factory`` 默认加载 Ultralytics 模型；测试中可注入假模型，避免下载权重。
    """

    def __init__(
        self,
        cfg: DetectorSection,
        *,
        model_factory: ModelFactory | None = None,
    ) -> None:
        self._cfg = cfg
        self._factory: ModelFactory = model_factory or default_model_factory
        self._lock = threading.Lock()
        self._models: dict[str, YoloModel] = {}

        self._person_model = self._load(cfg.person.weights)
        self._fire_smoke_model = self._load(cfg.fire_smoke.weights)
        self._helmet_model = self._load(cfg.helmet.weights)

        self._person_classes = set(cfg.person.classes)
        self._fire_smoke_classes = set(cfg.fire_smoke.classes)
        self._helmet_kinds: dict[str, HelmetKind] = {}
        for name in cfg.helmet.no_helmet_classes:
            self._helmet_kinds[name] = "no_helmet"
        for name in cfg.helmet.helmet_classes:
            self._helmet_kinds[name] = "helmet"

    @property
    def loaded_weight_count(self) -> int:
        """实际加载过的权重文件数（路径相同只计一次）。"""
        return len(self._models)

    def _load(self, weights: Path | str) -> YoloModel:
        key = _normalize_key(weights)
        model = self._models.get(key)
        if model is None:
            logger.info("加载 YOLO 权重: %s", key)
            model = self._factory(key)
            self._models[key] = model
        return model

    def _predict(self, model: YoloModel, frame: np.ndarray, conf: float | None = None) -> list[_RawDetection]:
        kwargs: dict[str, Any] = {
            "imgsz": self._cfg.imgsz,
            "device": None if self._cfg.device == "auto" else self._cfg.device,
            "verbose": False,
        }
        if conf is not None:
            kwargs["conf"] = conf
        with self._lock:
            results = model.predict(frame, **kwargs)
        return _extract_detections(results)

    def detect_persons(self, frame: np.ndarray) -> list[tuple[BBox, float]]:
        """返回 ``(box, conf)`` 列表，仅保留 person 类且 conf 不低于 ``detector.person.conf``。"""
        threshold = self._cfg.person.conf
        return [
            (det.box, det.conf)
            for det in self._predict(self._person_model, frame, conf=threshold)
            if det.name in self._person_classes and det.conf >= threshold
        ]

    def detect_fire_smoke(self, frame: np.ndarray) -> list[tuple[str, BBox, float]]:
        """返回 ``(label, box, conf)`` 列表，仅保留 ``detector.fire_smoke.classes`` 中的类别。"""
        return [
            (det.name, det.box, det.conf)
            for det in self._predict(self._fire_smoke_model, frame)
            if det.name in self._fire_smoke_classes
        ]

    def detect_helmets(self, frame: np.ndarray) -> list[tuple[HelmetKind, BBox, float]]:
        """返回 ``(kind, box, conf)`` 列表。

        ``helmet_classes`` 映射为 ``"helmet"``，``no_helmet_classes`` 映射为 ``"no_helmet"``，
        其他类别丢弃。
        """
        out: list[tuple[HelmetKind, BBox, float]] = []
        for det in self._predict(self._helmet_model, frame):
            kind = self._helmet_kinds.get(det.name)
            if kind is not None:
                out.append((kind, det.box, det.conf))
        return out
