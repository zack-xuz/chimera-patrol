"""从真实素材剪辑出演示用样例视频，输出到 samples/videos/。

输入：samples/raw/ 中的原始素材（来源与许可见 README「样例视频」一节）。
输出：
- intrusion_real.mp4：工地/仓库通道中两名行人走动，用于区域闯入（ROI 见下）。
- false_positive.mp4：工业场景的蒸汽与烟囱画面，用于观察烟火模型的误报。

用法（需要 imageio-ffmpeg，已在 requirements.txt 中）：

    python scripts/generate_samples.py

脚本使用 imageio-ffmpeg 自带的 ffmpeg 二进制，重复运行会覆盖输出文件。
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "samples" / "raw"
OUT_DIR = ROOT / "samples" / "videos"

WIDTH, HEIGHT = 960, 540
CRF = "23"

# 闯入区多边形（归一化顶点，0~1）。与 config.demo.yaml 中 intrusion_real 的 ROI 一致。
# 依据：两名行人的脚点在整段视频中位于 x 约 0.33–0.59、y 约 0.97–0.99。
INTRUSION_POLYGON_NORM = [[0.30, 0.75], [0.70, 0.75], [0.70, 1.00], [0.30, 1.00]]


@dataclass(frozen=True)
class Clip:
    raw_name: str
    out_name: str
    start_s: float | None
    duration_s: float | None
    note: str


CLIPS: tuple[Clip, ...] = (
    Clip(
        raw_name="10817415-hd_1920_1080_30fps.mp4",
        out_name="intrusion_real.mp4",
        start_s=None,
        duration_s=None,
        note="仓库通道，两名行人行走（全长 14 秒，低于 20 秒的目标，见 README）",
    ),
    Clip(
        raw_name="13101854_3840_2160_30fps.mp4",
        out_name="false_positive.mp4",
        start_s=15.0,
        duration_s=30.0,
        note="工厂烟囱排放蒸汽，傍晚光影变化（取第 15 秒起 30 秒）",
    ),
)


def _ffmpeg_args(clip: Clip) -> list[str]:
    args = ["-y", "-loglevel", "error"]
    if clip.start_s is not None:
        args += ["-ss", str(clip.start_s)]
    args += ["-i", str(RAW_DIR / clip.raw_name)]
    if clip.duration_s is not None:
        args += ["-t", str(clip.duration_s)]
    args += [
        "-vf", f"scale={WIDTH}:{HEIGHT}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", CRF,
        "-an",
        str(OUT_DIR / clip.out_name),
    ]
    return args


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    for clip in CLIPS:
        src = RAW_DIR / clip.raw_name
        if not src.is_file():
            print(f"缺少原始素材: {src}", file=sys.stderr)
            return 1
        subprocess.run([ffmpeg, *_ffmpeg_args(clip)], check=True)
        print(f"{clip.out_name}: {clip.note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
