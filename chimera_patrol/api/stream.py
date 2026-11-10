"""MJPEG 实时画面（任务 13.2；需求 7.2、2.4）。

- 每隔 ``1 / fps`` 秒读取 FrameSlot 的最新原始帧。帧的时间戳未变化时不重复编码。
- JPEG 编码在线程池中完成，画面输出不等待检测结果（需求 2.4）。
- ``overlay=true`` 时叠加 DetectWorker 最近一次发布的检测框，并绘制该路 ROI 多边形（只读）。
- 输出为 ``multipart/x-mixed-replace; boundary=frame``，``<img>`` 标签可直接播放。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from typing import Any

import cv2
import numpy as np

from chimera_patrol.config import RoiConfig
from chimera_patrol.ingest.capture import FrameSlot

BOUNDARY = "frame"
MEDIA_TYPE = f"multipart/x-mixed-replace; boundary={BOUNDARY}"
_BOX_COLOR = (0, 255, 0)
_ROI_COLOR = (255, 170, 0)
_TEXT_COLOR = (255, 255, 255)


def render_jpeg(
    image: np.ndarray,
    *,
    overlay: Sequence[Any] | None,
    rois: Sequence[RoiConfig],
    max_width: int,
    jpeg_quality: int,
) -> bytes:
    """在副本上绘制叠加项并编码为 JPEG。纯计算，在线程池中调用。

    ``overlay`` 中的项需有 ``box``、``label``、``conf`` 属性（OverlayItem）。
    ``rois`` 的顶点为归一化坐标，按当前帧尺寸换算。
    """
    canvas = image.copy()
    height, width = canvas.shape[:2]

    for region in rois:
        points = np.array(
            [[int(round(x * (width - 1))), int(round(y * (height - 1)))] for x, y in region.polygon],
            dtype=np.int32,
        )
        if len(points) >= 3:
            cv2.polylines(canvas, [points], isClosed=True, color=_ROI_COLOR, thickness=2)

    for item in overlay or ():
        x1, y1, x2, y2 = (int(round(v)) for v in item.box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), _BOX_COLOR, 2)
        cv2.putText(
            canvas,
            f"{item.label} {item.conf:.2f}",
            (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            _TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )

    if width > max_width:
        scale = max_width / width
        canvas = cv2.resize(canvas, (max_width, max(1, int(round(height * scale)))), interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return buf.tobytes()


def part_bytes(jpeg: bytes) -> bytes:
    """一个 multipart 分片。"""
    header = (
        f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n\r\n"
    ).encode("ascii")
    return header + jpeg + b"\r\n"


async def mjpeg_stream(
    slot: FrameSlot,
    *,
    rois: Sequence[RoiConfig],
    fps: int,
    max_width: int,
    jpeg_quality: int,
    overlay: bool,
) -> AsyncIterator[bytes]:
    """持续输出 multipart 分片。客户端断开时生成器被关闭，循环随之结束。"""
    interval = 1.0 / fps
    last_ts: int | None = None
    while True:
        latest = slot.get_latest()
        if latest is not None and latest.ts_ms != last_ts:
            last_ts = latest.ts_ms
            items = slot.overlay if overlay else None
            jpeg = await asyncio.to_thread(
                render_jpeg,
                latest.image,
                overlay=items,
                rois=rois if overlay else (),
                max_width=max_width,
                jpeg_quality=jpeg_quality,
            )
            yield part_bytes(jpeg)
        await asyncio.sleep(interval)
