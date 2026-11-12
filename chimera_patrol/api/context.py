"""API 层共享的应用上下文，以及运行时回调到 WebSocket 推送的适配（API 与运行时的接缝）。

``AppContext`` 持有配置、仓储、推送中心、状态机与 SOP 仓储。运行时启动后，
它的 ``cameras``、``frame_slots``、``sop`` 才会被填充；启动前这些字段为空。

``ApiNotifier`` 实现 ``EventNotifier`` 与 ``SopNotifier`` 两个协议，把事件、摄像头、
SOP 的变化转换为 WebSocket 消息。它只在事件循环中运行。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from chimera_patrol.api.dto import load_camera_dtos, load_event_dto, sop_run_dto
from chimera_patrol.api.ws import WsHub
from chimera_patrol.config import AppConfig, RoiConfig
from chimera_patrol.control import CameraControl
from chimera_patrol.ingest.camera import Camera
from chimera_patrol.ingest.capture import FrameSlot
from chimera_patrol.review.state_machine import EventStateMachine
from chimera_patrol.store.repository import EventRepository
from chimera_patrol.store.sop_repo import SopRepository

if TYPE_CHECKING:
    from chimera_patrol.sop.checker import SopChecker

_log = logging.getLogger("chimera_patrol.api")


@dataclass
class AppContext:
    config: AppConfig
    repo: EventRepository
    hub: WsHub
    state: EventStateMachine = field(init=False)
    sop_repo: SopRepository = field(init=False)
    cameras: list[Camera] = field(default_factory=list)
    frame_slots: dict[str, FrameSlot] = field(default_factory=dict)
    sop: SopChecker | None = None
    control: CameraControl = field(default_factory=CameraControl)
    # 复检入口（运行时启动后设置）。未设置时复检接口拒绝执行，不改变事件状态。
    rereview: Callable[[str], Awaitable[None]] | None = None

    def __post_init__(self) -> None:
        self.state = EventStateMachine(self.repo)
        self.sop_repo = SopRepository(self.repo)

    def camera_names(self) -> dict[str, str]:
        return {cam.id: cam.name for cam in self.cameras}

    def camera_rois(self) -> dict[str, Sequence[RoiConfig]]:
        return {cam.id: cam.rois for cam in self.cameras}


class ApiNotifier:
    """把运行时与 SOP 检查的变化推送到 WebSocket。"""

    def __init__(self, ctx: AppContext) -> None:
        self._ctx = ctx

    def _event_dto(self, event_id: str) -> dict[str, Any] | None:
        return load_event_dto(
            self._ctx.repo,
            self._ctx.sop_repo,
            event_id,
            camera_names=self._ctx.camera_names(),
            include_detail=False,
        )

    async def _push_event(self, msg_type: str, event_id: str) -> None:
        dto = await asyncio.to_thread(self._event_dto, event_id)
        if dto is None:
            _log.error("推送跳过，事件不存在 event_id=%s type=%s", event_id, msg_type)
            return
        await self._ctx.hub.broadcast(msg_type, dto)

    # EventNotifier

    async def on_created(self, summary: dict[str, Any]) -> None:
        await self._push_event("event.created", summary["id"])

    async def on_clip_done(self, event_id: str, status: str) -> None:
        await self._push_event("event.updated", event_id)

    async def on_event_updated(self, event_id: str) -> None:
        await self._push_event("event.updated", event_id)

    async def on_camera_status(self, camera_id: str, status: str, ts_ms: int) -> None:
        await self._ctx.hub.broadcast(
            "camera.status",
            {"camera_id": camera_id, "status": status, "status_changed_ms": ts_ms},
        )

    # SopNotifier

    async def on_sop_run(self, run_id: str) -> None:
        record = await asyncio.to_thread(self._ctx.sop_repo.get_run, run_id)
        if record is None:
            return
        await self._ctx.hub.broadcast("sop.run.updated", sop_run_dto(record))


async def cameras_snapshot(ctx: AppContext) -> list[dict[str, Any]]:
    """在线程中读取 CameraDTO 列表（含每路的 paused 状态）。"""
    rois = ctx.camera_rois()
    paused = ctx.control.snapshot()
    return await asyncio.to_thread(load_camera_dtos, ctx.repo, rois, paused)
