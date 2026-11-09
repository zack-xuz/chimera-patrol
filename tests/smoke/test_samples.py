"""样例视频冒烟测试：两段样例视频存在且时长位于 10~120 秒。

样例文件由 scripts/generate_samples.py 生成。文件缺失时跳过并给出提示，
以便在生成前冒烟套件仍保持通过。
"""
from pathlib import Path

import cv2
import pytest

ROOT = Path(__file__).resolve().parents[2]
SAMPLES_DIR = ROOT / "samples" / "videos"
CLIP_NAMES = ["intrusion_real.mp4", "false_positive.mp4"]


def _duration_s(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    try:
        assert cap.isOpened(), f"无法打开视频: {path}"
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    assert fps > 0, f"视频帧率无效: {path}"
    return frames / fps


@pytest.mark.parametrize("name", CLIP_NAMES)
def test_sample_clip_exists_and_duration_in_range(name: str) -> None:
    path = SAMPLES_DIR / name
    if not path.exists():
        pytest.skip(
            f"样例视频 {path} 不存在，请先运行 python scripts/generate_samples.py 生成"
        )
    duration = _duration_s(path)
    assert 10 <= duration <= 120, f"{name} 时长 {duration:.2f}s 不在 10~120 秒范围内"
