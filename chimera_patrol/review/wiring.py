"""把 VlmReviewer 接入 EventPipeline 的 on_review 回调（任务 10.8）。

``build_on_review`` 返回一个 ``async def on_review(event_id)``，EventPipeline 在事件落库后调用它。
回调做三件事：

1. 经 ``asyncio.to_thread`` 从 EventRepository 读取事件。
2. 读取事件的标注版关键帧 ``base_dir / keyframe_annotated_path``。选用标注版是因为它画出了
   触发事件的检测框与 ROI 多边形，复核时更容易看清触发原因。原图版保留为证据，不送入 VLM。
   读取失败或文件不存在时传入 None，由 VlmReviewer 记为 ``keyframe_missing``。
3. 调用 ``VlmReviewer.review``。复核开关关闭时，VlmReviewer 不调用适配器，直接迁移为
   ``needs_human``（actor ``review_switch``）。

事件表只保存 ``camera_id``，本模块没有读取摄像头名称的接口。``camera_names`` 为可选映射，
未提供或缺少某个 ID 时以 ``camera_id`` 作为 ``camera_name``。
``cv_confidence`` 为 NULL 时（例如 sop_violation）以 0.0 传给复核器。

异常处理：本模块不捕获异常。VlmReviewer 自行处理复核失败，EventPipeline 的 ``_run_hook``
记录其余异常并吞掉，因此回调中的异常不会影响已落库的事件。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path

from chimera_patrol.review.reviewer import VlmReviewer
from chimera_patrol.store.repository import EventRepository

__all__ = ["build_on_review"]

_log = logging.getLogger("chimera_patrol.review")

OnReview = Callable[[str], Awaitable[None]]


def _read_image(path: Path) -> bytes | None:
    """在工作线程中读取关键帧。文件缺失或不可读时返回 None。"""
    try:
        return path.read_bytes()
    except OSError as exc:
        _log.warning("keyframe unreadable path=%s error=%s", path, exc)
        return None


def build_on_review(
    *,
    repository: EventRepository,
    reviewer: VlmReviewer,
    base_dir: Path,
    camera_names: Mapping[str, str] | None = None,
) -> OnReview:
    """构造供 ``EventPipeline(on_review=...)`` 使用的异步回调。

    Args:
        repository: 读取事件的 EventRepository，阻塞调用经 ``asyncio.to_thread`` 执行。
        reviewer: 已配置好 adapter、状态机与开关的 VlmReviewer。
        base_dir: 证据根目录（``evidence.base_dir``），与事件中的相对路径拼接。
        camera_names: 可选的 camera_id 到显示名称的映射。

    Returns:
        接收 ``event_id`` 的异步回调。事件不存在时记录 error 日志并返回。
    """
    names: dict[str, str] = dict(camera_names or {})

    async def on_review(event_id: str) -> None:
        event = await asyncio.to_thread(repository.get_event, event_id)
        if event is None:
            _log.error("review skipped, event not found event_id=%s", event_id)
            return

        camera_id = event["camera_id"]
        cv_confidence = event["cv_confidence"]
        image = await asyncio.to_thread(
            _read_image, base_dir / event["keyframe_annotated_path"]
        )

        await reviewer.review(
            event_id,
            event_type=event["event_type"],
            camera_name=names.get(camera_id, camera_id),
            roi_name=event["roi_id"],
            cv_confidence=0.0 if cv_confidence is None else float(cv_confidence),
            image_jpeg=image,
        )

    return on_review
