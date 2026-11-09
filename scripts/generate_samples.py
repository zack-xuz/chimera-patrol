"""生成演示用样例视频（合成数据）。

本脚本生成两段合成视频，输出到 samples/videos/：

- intrusion_real.mp4：一个矩形人形目标从左侧进入闯入区多边形，在区域内滞留约 12 秒后离开，
  用于触发区域闯入与滞留事件。
- false_positive.mp4：固定背景上的闪烁亮斑与缓慢移动的小斑点（模拟光影晃动、飞虫、雨纹），
  不含人形目标，用于触发 CV 误报。

重要说明：这两段视频是合成的替代素材，仅用于离线演示与冒烟测试，不代表真实场景的检测效果。
真实素材准备好后，直接替换 samples/videos/ 中的同名文件即可，无需修改代码。

编码说明：输出使用 OpenCV 的 mp4v 编码写入 .mp4 容器。这里选择 mp4v 是因为这些片段只由本服务读取，
不需要浏览器直接播放或与外部编码规范对齐；若需更高压缩比或兼容性，可改用其他编码。

确定性：所有随机量都来自 SEED 固定的 numpy 随机数生成器，并按固定顺序采样，
因此重复运行得到的帧内容一致。

样例 ROI（与 config.demo.yaml 的注释示例一致，归一化坐标，取值 0~1）：
    intrusion_real 的闯入区 INTRUSION_POLYGON_NORM = [[0.45, 0.40], [0.94, 0.40], [0.94, 0.94], [0.45, 0.94]]
    false_positive 无需 ROI。

用法：
    python scripts/generate_samples.py
"""


from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

SEED = 20240601
FPS = 10
DURATION_S = 30
WIDTH, HEIGHT = 640, 360
N_FRAMES = FPS * DURATION_S

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "samples" / "videos"
INTRUSION_NAME = "intrusion_real.mp4"
FALSE_POSITIVE_NAME = "false_positive.mp4"

# 闯入区多边形（归一化顶点）。与 config.demo.yaml 注释示例保持一致。
INTRUSION_POLYGON_NORM = [[0.45, 0.40], [0.94, 0.40], [0.94, 0.94], [0.45, 0.94]]

# 人形目标的脚点（bottom_center 参考点）纵坐标，单位像素。
# 脚点位于闯入区内（区域纵向范围约 144~338 px），目标在区域内滞留。
FEET_Y = 300
PERSON_HALF_W = 20
PERSON_H = 110
HEAD_R = 12
DWELL_X = 470  # 滞留时的脚点横坐标（位于区域内）
ENTER_START_X = -30
EXIT_END_X = 720

# 时间轴（帧索引）：空场 -> 进入 -> 滞留 -> 离开 -> 空场
ENTER_START_F, ENTER_END_F = 20, 60
LINGER_END_F = 180  # 滞留帧 [60, 180) 共 120 帧，约 12 秒
EXIT_END_F = 210


def _polygon_px() -> np.ndarray:
    pts = [[x * WIDTH, y * HEIGHT] for x, y in INTRUSION_POLYGON_NORM]
    return np.array(pts, dtype=np.int32)


def _background(rng: np.random.Generator) -> np.ndarray:
    """固定背景：竖向渐变加静态纹理，返回 float32 BGR 图像。"""
    yy = np.linspace(0.0, 1.0, HEIGHT, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, WIDTH, dtype=np.float32)[None, :]
    base = 110.0 + 25.0 * yy + 10.0 * xx
    texture = rng.normal(0.0, 3.0, size=(HEIGHT, WIDTH)).astype(np.float32)
    bg = np.repeat((base + texture)[:, :, None], 3, axis=2)
    return bg.astype(np.float32)


def _to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _person_x(frame_idx: int, rng: np.random.Generator) -> float | None:
    """返回第 frame_idx 帧人形脚点的横坐标；None 表示画面中无人。"""
    if ENTER_START_F <= frame_idx < ENTER_END_F:
        t = (frame_idx - ENTER_START_F) / (ENTER_END_F - ENTER_START_F)
        return _lerp(ENTER_START_X, DWELL_X, t)
    if ENTER_END_F <= frame_idx < LINGER_END_F:
        return DWELL_X + float(rng.uniform(-2.0, 2.0))
    if LINGER_END_F <= frame_idx < EXIT_END_F:
        t = (frame_idx - LINGER_END_F) / (EXIT_END_F - LINGER_END_F)
        return _lerp(DWELL_X, EXIT_END_X, t)
    return None


def _draw_person(frame: np.ndarray, cx: float) -> None:
    """绘制矩形躯干加圆形头部的人形目标（BGR 颜色）。"""
    x = int(round(cx))
    top = FEET_Y - PERSON_H
    cv2.rectangle(frame, (x - PERSON_HALF_W, top), (x + PERSON_HALF_W, FEET_Y), (40, 60, 200), -1)
    cv2.circle(frame, (x, top - HEAD_R), HEAD_R, (150, 170, 210), -1)


def _intrusion_frames() -> "list[np.ndarray]":
    rng = np.random.default_rng(SEED)
    bg = _background(rng)
    frames = []
    for i in range(N_FRAMES):
        frame = bg.copy()
        cx = _person_x(i, rng)
        if cx is not None:
            _draw_person(frame, cx)
        frame += rng.normal(0.0, 2.0, size=frame.shape).astype(np.float32)
        frames.append(_to_uint8(frame))
    return frames


def _false_positive_frames() -> "list[np.ndarray]":
    rng = np.random.default_rng(SEED + 1)
    bg = _background(rng)

    # 三处固定的软边椭圆亮斑（模拟墙面反光、灯光），每处有各自的闪烁相位与频率。
    blob_specs = [(120, 90, 45, 25), (330, 60, 60, 30), (520, 280, 50, 22)]
    masks = []
    for cx, cy, ax, ay in blob_specs:
        mask = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
        cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 1.0, -1)
        masks.append(cv2.GaussianBlur(mask, (0, 0), sigmaX=6))

    # 小斑点：模拟飞虫或雨纹，随机游走，尺寸 1 像素。
    n_speck = 30
    pos = rng.uniform([0.0, 0.0], [WIDTH, HEIGHT], size=(n_speck, 2))
    vel = rng.normal(0.0, 3.0, size=(n_speck, 2))

    frames = []
    for i in range(N_FRAMES):
        t = i / FPS
        frame = bg.copy()
        for k, mask in enumerate(masks):
            freq = 0.8 + 0.3 * k
            flicker = 35.0 * np.sin(2.0 * np.pi * freq * t + k) + float(rng.normal(0.0, 12.0))
            frame += mask[:, :, None] * flicker

        vel += rng.normal(0.0, 0.8, size=vel.shape)
        vel = np.clip(vel, -6.0, 6.0)
        pos = (pos + vel) % np.array([WIDTH, HEIGHT], dtype=np.float64)
        for px, py in pos:
            cv2.circle(frame, (int(px), int(py)), 1, (20.0, 20.0, 20.0), -1)

        frame += rng.normal(0.0, 2.0, size=frame.shape).astype(np.float32)
        frames.append(_to_uint8(frame))
    return frames


def write_clip(path: Path, frames: list[np.ndarray]) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"无法打开视频写入器: {path}")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def read_back(path: Path) -> dict:
    """回读视频，逐帧解码统计实际帧数，返回帧数、帧率与时长（秒）。"""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frames = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            frames += 1
    finally:
        cap.release()
    duration = frames / fps if fps > 0 else 0.0
    return {"frames": frames, "fps": fps, "duration_s": duration}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = [
        (OUT_DIR / INTRUSION_NAME, _intrusion_frames),
        (OUT_DIR / FALSE_POSITIVE_NAME, _false_positive_frames),
    ]
    for path, make_frames in targets:
        write_clip(path, make_frames())
        info = read_back(path)
        print(
            f"{path.name}: frames={info['frames']} fps={info['fps']:.2f} "
            f"duration={info['duration_s']:.2f}s"
        )
        if not 10 <= info["duration_s"] <= 120:
            raise SystemExit(f"{path.name} 时长 {info['duration_s']:.2f}s 不在 10~120 秒范围内")


if __name__ == "__main__":
    main()
