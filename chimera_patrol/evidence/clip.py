"""短视频片段编码 Evidence_Capturer（任务 5.4）。

- ``write_clip``：把解码后的 BGR 帧写成 H.264 yuv420p mp4，使用 imageio-ffmpeg 自带的 ffmpeg。
- ``clip_window``：从 RingBuffer 快照 ``[end_ms - pre_s*1000, end_ms + post_s*1000]`` 范围内的帧。
- ``save_clip_safely``：解码 JPEG、编码、原子改名为 clip.mp4；任何异常都转为 ``("failed", 错误信息)``，
  并记录带事件上下文的错误日志，不向调用方抛出（需求 4.3）。

像素约定：输入帧为 OpenCV 惯用的 BGR uint8 数组（HxWx3）。颜色转换只做一次，由 ffmpeg 通过
``pix_fmt_in="bgr24"`` 完成 BGR → yuv420p，Python 侧不再做颜色转换。
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

from chimera_patrol.config import EvidenceSection
from chimera_patrol.evidence.ring_buffer import Frame, RingBuffer

CLIP_FILE_NAME = "clip.mp4"
_PARTIAL_FILE_NAME = "clip.partial.mp4"
_log = logging.getLogger("chimera_patrol.evidence.clip")


def write_clip(frames: Iterable[np.ndarray], fps: float, out_path: str | Path) -> Path:
    """把 BGR 帧序列编码为 H.264 yuv420p mp4 并返回输出路径。

    ``frames`` 可以是列表或生成器，逐帧送入 ffmpeg，不会整体驻留内存。
    所有帧必须为同一尺寸的 uint8 HxWx3 数组。宽或高为奇数时，各裁掉最右列或最下行 1 个像素，
    以满足 yuv420p 对偶数尺寸的要求。
    ``macro_block_size=1`` 关闭 imageio-ffmpeg 默认的 16 倍数自动放大，保证输出尺寸只受奇偶裁剪影响。
    """
    if not fps or fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps!r}")

    iterator = iter(frames)
    try:
        first = next(iterator)
    except StopIteration:
        raise ValueError("write_clip requires at least one frame") from None

    if first.dtype != np.uint8 or first.ndim != 3 or first.shape[2] != 3:
        raise ValueError(
            f"frames must be uint8 HxWx3 BGR arrays, got dtype={first.dtype} shape={first.shape}"
        )
    height, width = first.shape[:2]
    enc_w = width - width % 2
    enc_h = height - height % 2
    if enc_w == 0 or enc_h == 0:
        raise ValueError(f"frame too small to encode: {width}x{height}")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    writer = imageio_ffmpeg.write_frames(
        str(out),
        (enc_w, enc_h),
        pix_fmt_in="bgr24",
        pix_fmt_out="yuv420p",
        fps=fps,
        codec="libx264",
        macro_block_size=1,
        output_params=["-movflags", "+faststart"],
    )
    writer.send(None)
    try:
        _send_frame(writer, first, first.shape, enc_w, enc_h)
        for frame in iterator:
            _send_frame(writer, frame, first.shape, enc_w, enc_h)
    finally:
        writer.close()

    if not out.is_file() or out.stat().st_size == 0:
        raise RuntimeError(f"encoder produced no output file at {out}")
    return out


def _send_frame(
    writer: object,
    frame: np.ndarray,
    expected_shape: tuple[int, ...],
    enc_w: int,
    enc_h: int,
) -> None:
    if frame.shape != expected_shape or frame.dtype != np.uint8:
        raise ValueError(
            f"frame shape/dtype {frame.shape}/{frame.dtype} differs from first frame "
            f"{expected_shape}/uint8"
        )
    writer.send(np.ascontiguousarray(frame[:enc_h, :enc_w]))  # type: ignore[attr-defined]


def clip_window(ring: RingBuffer, end_ms: int, pre_s: float, post_s: float) -> list[Frame]:
    """返回 RingBuffer 中 ``[end_ms - pre_s*1000, end_ms + post_s*1000]`` 闭区间内的帧记录（升序）。"""
    if pre_s < 0 or post_s < 0:
        raise ValueError(f"pre_s and post_s must be >= 0, got {pre_s}, {post_s}")
    start_ms = int(round(end_ms - pre_s * 1000))
    stop_ms = int(round(end_ms + post_s * 1000))
    return ring.snapshot(start_ms, stop_ms)


def _check_component(name: str, label: str) -> str:
    """拒绝可能逃逸出 base_dir 的路径分量。"""
    if not name or name in {".", ".."} or any(ch in name for ch in '/\\:*?"<>|') or name != name.strip():
        raise ValueError(f"{label} is not a safe path component: {name!r}")
    return name


def save_clip_safely(
    ring: RingBuffer,
    *,
    evidence: EvidenceSection,
    camera_id: str,
    event_id: str,
    end_ms: int,
) -> tuple[str, str]:
    """在事件结束时刻 ``end_ms`` 截取并编码片段，永不抛出异常。

    成功返回 ``("ready", rel_path)``，``rel_path`` 形如 ``{camera_id}/{event_id}/clip.mp4``，
    相对 ``evidence.base_dir``，与数据库 ``clip_path`` 的约定一致。
    失败返回 ``("failed", "异常类型: 信息")``，并记录带事件上下文的 ERROR 日志。
    """
    partial: Path | None = None
    try:
        camera = _check_component(camera_id, "camera_id")
        event = _check_component(event_id, "event_id")
        event_dir = Path(evidence.base_dir) / camera / event
        partial = event_dir / _PARTIAL_FILE_NAME
        final = event_dir / CLIP_FILE_NAME

        records = clip_window(ring, end_ms, evidence.clip_pre_s, evidence.clip_post_s)
        if not records:
            raise ValueError("no buffered frames inside the clip window")

        write_clip(
            (_decode_jpeg(ts_ms, data) for ts_ms, data in records),
            evidence.clip_fps,
            partial,
        )
        os.replace(partial, final)
        partial = None
        return "ready", f"{camera}/{event}/{CLIP_FILE_NAME}"
    except Exception as exc:
        if partial is not None:
            try:
                partial.unlink(missing_ok=True)
            except OSError:
                pass
        message = f"{type(exc).__name__}: {exc}"
        _log.error(
            "clip save failed event_id=%s camera_id=%s end_ms=%s error=%s",
            event_id,
            camera_id,
            end_ms,
            message,
            exc_info=True,
        )
        return "failed", message


def _decode_jpeg(ts_ms: int, data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"frame ts_ms={ts_ms} is not a decodable JPEG")
    return image
