"""端到端自动化验证（任务 19.1）。

用 config.demo.yaml 与仓库样例视频驱动完整运行时：真实 YOLO 权重、MockAdapter 复核、
SQLite 落库。验证：
- 样例视频产生事件，事件经 Mock 复核后进入人工队列，证据文件落盘；
- 一路 Camera 故障（打开即失败）不影响另一路的事件产生（需求 12.4）；
- 通过仓储查询的事件与 SQL 统计、纯函数统计的计数一致。

需要真实权重（weights/ 与 yolov8n.pt，见 README「权重获取」）。权重缺失时整组测试跳过。
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import cv2
import pytest
import yaml

from chimera_patrol.api.stats import compute_stats
from chimera_patrol.config import parse_config
from chimera_patrol.runtime import PatrolRuntime
from chimera_patrol.store.repository import EventRepository
from chimera_patrol.store.stats_sql import compute_stats_sql

ROOT = Path(__file__).resolve().parents[2]
DEMO_CONFIG = ROOT / "config.demo.yaml"
SAMPLE_DIR = ROOT / "samples" / "videos"
BROKEN_CAMERA = "false_positive"
HEALTHY_CAMERA = "intrusion_real"

WEIGHTS_PRESENT = all(
    (ROOT / rel).is_file() for rel in ("weights/fire_smoke.pt", "weights/helmet.pt")
) and (ROOT / "yolov8n.pt").is_file()

pytestmark = pytest.mark.skipif(
    not WEIGHTS_PRESENT,
    reason="缺少 weights/ 下的烟火与安全帽权重或 yolov8n.pt，见 README「权重获取」",
)


class _BrokenCapture:
    """模拟损坏的视频源：打开失败，读帧始终失败。"""

    def isOpened(self) -> bool:  # noqa: N802 - 与 OpenCV 命名一致
        return False

    def read(self):
        return False, None

    def get(self, _prop: int) -> float:
        return 0.0

    def set(self, _prop: int, _value: float) -> bool:
        return False

    def release(self) -> None:
        return None


def _capture_factory(source: str):
    if BROKEN_CAMERA in Path(source).name:
        return _BrokenCapture()
    return cv2.VideoCapture(source)


def _config(tmp_path: Path):
    data = yaml.safe_load(DEMO_CONFIG.read_text(encoding="utf-8"))
    data["evidence"]["base_dir"] = str(tmp_path / "evidence")
    data["database"]["path"] = str(tmp_path / "chimera.db")
    data["logging"]["dir"] = str(tmp_path / "logs")
    data["video"]["source_dir"] = str(SAMPLE_DIR)
    return parse_config(data)


def _wait_for(predicate, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.2)
    return False


def test_end_to_end_with_isolated_camera_failure(tmp_path: Path, monkeypatch) -> None:
    from chimera_patrol.detect.detector import Detector
    from chimera_patrol.vlm.mock import MockAdapter

    # 配置校验要求复核开启时存在密钥；测试注入 MockAdapter，密钥不会被使用。
    monkeypatch.setenv("VLM_API_KEY", "placeholder-for-test-only")
    cfg = _config(tmp_path)
    repo = EventRepository(cfg.database.path)
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()

    runtime = PatrolRuntime(
        cfg,
        repo,
        Detector(cfg.detector),
        loop,
        capture_factory=_capture_factory,
        adapter=MockAdapter(default_verdict="uncertain"),
    )
    try:
        runtime.start()

        # 健康路在事件进入人工队列（Mock 复核为 uncertain）后才算完成。
        def healthy_reviewed() -> bool:
            events, _ = repo.list_events(camera_id=HEALTHY_CAMERA, limit=200)
            return any(e["status"] == "needs_human" and e["vlm_verdict"] == "uncertain" for e in events)

        assert _wait_for(healthy_reviewed, timeout_s=120), "健康路未产生已复核的事件"

        # 故障路：不产生事件，并被标记为离线。
        assert _wait_for(
            lambda: (repo.get_camera(BROKEN_CAMERA) or {}).get("status") == "offline",
            timeout_s=60,
        ), "故障路未被标记为离线"
        broken_events, _ = repo.list_events(camera_id=BROKEN_CAMERA)
        assert broken_events == []
    finally:
        runtime.stop()
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)

    # 证据落盘、事件与状态一致。
    events, total = repo.list_events(limit=200)
    assert total == len(events) >= 1
    healthy = [e for e in events if e["camera_id"] == HEALTHY_CAMERA]
    assert healthy, "no healthy-camera events"
    for event in healthy:
        assert event["status"] == "needs_human"
        assert event["vlm_description"], "VLM 描述应非空"
        assert (cfg.evidence.base_dir / event["keyframe_annotated_path"]).is_file()
        assert (cfg.evidence.base_dir / event["keyframe_raw_path"]).is_file()
        last_log = repo.list_status_log(event["id"])[-1]
        assert last_log["to_status"] == event["status"]

    # 统计计数：SQL 聚合与纯函数、事件列表一致。
    sql_stats = compute_stats_sql(repo)
    pure_stats = compute_stats(events, include_sop=False)
    assert sql_stats == pure_stats
    assert sql_stats["totals"]["total_alarms"] == len(events)
    assert sql_stats["totals"]["needs_human"] == sum(1 for e in events if e["status"] == "needs_human")
    repo.close()
