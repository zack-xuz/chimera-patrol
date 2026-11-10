"""FastAPI 应用工厂与 REST 路由（任务 13.1、13.3、13.5、14.3、14.5、15.10；design HTTP API）。

``create_app(ctx, ...)`` 注册全部端点。生命周期由 ``on_startup`` / ``on_shutdown`` 回调决定，
__main__ 在其中启动与停止 PatrolRuntime。测试可以不传回调，直接用 TestClient 访问接口。

错误统一为 ``{"error": {"code", "message"}}``（event-contract 第 7 节）。
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Query, WebSocket
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from chimera_patrol.api.context import ApiNotifier, AppContext, cameras_snapshot
from chimera_patrol.api.dto import list_event_dtos, load_event_dto, sop_run_dto
from chimera_patrol.api.export import export_csv, export_json
from chimera_patrol.api.media import resolve_media_path
from chimera_patrol.api.routes_stats import get_stats
from chimera_patrol.api.stream import MEDIA_TYPE, mjpeg_stream
from chimera_patrol.review.state_machine import EventNotFound, InvalidTransition
from chimera_patrol.sop.checker import SopTriggerError

APP_VERSION = "0.1.0"
OPERATOR_MIN, OPERATOR_MAX = 1, 32
REASON_MAX = 200

_log = logging.getLogger("chimera_patrol.api")

_SOP_ERROR_STATUS: dict[str, int] = {
    "not_found": 404,
    "camera_offline": 409,
    "run_in_progress": 409,
    "sop_not_applicable": 422,
    "validation_error": 422,
}


class ApiError(Exception):
    """可直接映射为错误响应的异常。"""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message
        super().__init__(message)


class StatusChangeRequest(BaseModel):
    to_status: Literal["confirmed", "dismissed", "needs_human"]
    operator: str = Field(min_length=OPERATOR_MIN, max_length=OPERATOR_MAX)
    reason: str | None = Field(default=None, max_length=REASON_MAX)


class SopRunRequest(BaseModel):
    sop_id: str = Field(min_length=1)
    camera_id: str = Field(min_length=1)
    operator: str = Field(min_length=OPERATOR_MIN, max_length=OPERATOR_MAX)
    retry_of: str | None = None


def _error_body(code: str, message: str) -> dict[str, Any]:
    return {"error": {"code": code, "message": message}}


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


def create_app(
    ctx: AppContext,
    *,
    web_dir: Path | None = None,
    on_startup: Callable[[], Awaitable[None]] | None = None,
    on_shutdown: Callable[[], Awaitable[None]] | None = None,
) -> FastAPI:
    """构造应用。``web_dir`` 为前端静态目录（含 index.html、event.html）。"""

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if on_startup is not None:
            await on_startup()
        ctx.hub.start_heartbeat()
        try:
            yield
        finally:
            await ctx.hub.stop_heartbeat()
            if on_shutdown is not None:
                await on_shutdown()

    app = FastAPI(title="Chimera Patrol", version=APP_VERSION, lifespan=lifespan)
    app.state.ctx = ctx
    notifier = ApiNotifier(ctx)

    async def stats_data(
        camera_id: str | None,
        from_ms: int | None,
        to_ms: int | None,
        include_sop: bool,
    ) -> dict[str, Any]:
        try:
            data = await _run(get_stats, ctx.repo, camera_id, from_ms, to_ms, include_sop)
        except ValueError as exc:
            raise ApiError(422, "validation_error", str(exc)) from exc
        data["generated_ms"] = _now_ms()
        data["filters"] = {
            "camera_id": camera_id,
            "from_ms": from_ms,
            "to_ms": to_ms,
            "include_sop": include_sop,
        }
        return data

    # ------------------------------------------------------------ 错误处理

    @app.exception_handler(ApiError)
    async def _api_error(_: Any, exc: ApiError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=_error_body(exc.code, exc.message))

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Any, exc: RequestValidationError) -> JSONResponse:
        parts = []
        for err in exc.errors():
            loc = ".".join(str(p) for p in err.get("loc", ()) if p != "body")
            parts.append(f"{loc}: {err.get('msg')}" if loc else str(err.get("msg")))
        return JSONResponse(status_code=422, content=_error_body("validation_error", "; ".join(parts)))

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Any, exc: StarletteHTTPException) -> JSONResponse:
        if exc.status_code == 404:
            code = "not_found"
        elif exc.status_code == 422:
            code = "validation_error"
        else:
            code = "internal_error"
        return JSONResponse(status_code=exc.status_code, content=_error_body(code, str(exc.detail)))

    @app.exception_handler(Exception)
    async def _unexpected(_: Any, exc: Exception) -> JSONResponse:
        _log.exception("unhandled error: %s", type(exc).__name__)
        return JSONResponse(status_code=500, content=_error_body("internal_error", "internal server error"))

    # ------------------------------------------------------------ 基础

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "vlm_review_enabled": ctx.config.vlm.review_enabled,
            "version": APP_VERSION,
        }

    # ------------------------------------------------------------ 摄像头与 MJPEG

    @app.get("/api/cameras")
    async def list_cameras() -> dict[str, Any]:
        return {"items": await cameras_snapshot(ctx)}

    @app.get("/api/cameras/{camera_id}/stream.mjpg")
    async def stream(
        camera_id: str,
        overlay: bool = Query(True),
        fps: int | None = Query(None, ge=1, le=30),
    ) -> Response:
        slot = ctx.frame_slots.get(camera_id)
        if slot is None:
            raise ApiError(404, "not_found", f"camera {camera_id} not found")
        mjpeg = ctx.config.video.mjpeg
        generator = mjpeg_stream(
            slot,
            rois=ctx.camera_rois().get(camera_id, ()),
            fps=fps or mjpeg.fps,
            max_width=mjpeg.max_width,
            jpeg_quality=mjpeg.jpeg_quality,
            overlay=overlay,
        )
        return StreamingResponse(generator, media_type=MEDIA_TYPE)

    # ------------------------------------------------------------ 事件

    @app.get("/api/events")
    async def list_events(
        camera_id: str | None = None,
        event_type: str | None = None,
        status: str | None = None,
        from_ms: int | None = None,
        to_ms: int | None = None,
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
    ) -> dict[str, Any]:
        def work() -> tuple[list[dict[str, Any]], int]:
            rows, total = ctx.repo.list_events(
                camera_id=camera_id or None,
                event_type=event_type or None,
                status=status or None,
                from_ms=from_ms,
                to_ms=to_ms,
                limit=limit,
                offset=offset,
            )
            items = list_event_dtos(ctx.repo, ctx.sop_repo, rows, camera_names=ctx.camera_names())
            return items, total

        try:
            items, total = await _run(work)
        except ValueError as exc:
            raise ApiError(422, "validation_error", str(exc)) from exc
        return {"items": items, "total": total, "limit": limit, "offset": offset}

    @app.get("/api/events/{event_id}")
    async def get_event(event_id: str) -> dict[str, Any]:
        dto = await _run(
            load_event_dto,
            ctx.repo,
            ctx.sop_repo,
            event_id,
            camera_names=ctx.camera_names(),
            include_detail=True,
        )
        if dto is None:
            raise ApiError(404, "not_found", f"event {event_id} not found")
        return dto

    @app.post("/api/events/{event_id}/status")
    async def change_status(event_id: str, body: StatusChangeRequest) -> dict[str, Any]:
        try:
            await _run(
                ctx.state.override,
                event_id,
                body.to_status,
                body.operator,
                body.reason,
            )
        except EventNotFound as exc:
            raise ApiError(404, "not_found", str(exc)) from exc
        except InvalidTransition as exc:
            raise ApiError(409, "invalid_transition", str(exc)) from exc
        except ValueError as exc:
            raise ApiError(422, "validation_error", str(exc)) from exc

        dto = await _run(
            load_event_dto,
            ctx.repo,
            ctx.sop_repo,
            event_id,
            camera_names=ctx.camera_names(),
            include_detail=True,
        )
        if dto is None:
            raise ApiError(404, "not_found", f"event {event_id} not found")
        await notifier.on_event_updated(event_id)
        return dto

    # ------------------------------------------------------------ 统计与导出

    @app.get("/api/stats")
    async def stats(
        camera_id: str | None = None,
        from_ms: int | None = None,
        to_ms: int | None = None,
        include_sop: bool = False,
    ) -> dict[str, Any]:
        return await stats_data(camera_id, from_ms, to_ms, include_sop)

    @app.get("/api/stats/export")
    async def stats_export(
        camera_id: str | None = None,
        from_ms: int | None = None,
        to_ms: int | None = None,
        include_sop: bool = False,
        format: Literal["json", "csv"] = "json",
    ) -> Response:
        data = await stats_data(camera_id, from_ms, to_ms, include_sop)
        if format == "csv":
            body = export_csv(data)
            media_type = "text/csv; charset=utf-8"
            filename = "chimera_stats.csv"
        else:
            body = export_json(data)
            media_type = "application/json"
            filename = "chimera_stats.json"
        return Response(
            content=body,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    # ------------------------------------------------------------ SOP

    @app.get("/api/sops")
    async def list_sops() -> dict[str, Any]:
        if ctx.sop is None:
            return {"items": []}
        items = [
            {
                "id": sop.id,
                "name": sop.name,
                "camera_ids": list(sop.camera_ids),
                "steps": [
                    {
                        "id": step.id,
                        "description": step.description,
                        "criteria": step.criteria,
                        "requires_sequence": step.requires_sequence,
                    }
                    for step in sop.steps
                ],
            }
            for sop in ctx.sop.definitions
        ]
        return {"items": items}

    @app.post("/api/sop/runs", status_code=202)
    async def trigger_sop(body: SopRunRequest) -> dict[str, Any]:
        if ctx.sop is None:
            raise ApiError(500, "internal_error", "SOP checker is not running")
        try:
            run_id = await ctx.sop.trigger(
                sop_id=body.sop_id,
                camera_id=body.camera_id,
                operator=body.operator,
                retry_of=body.retry_of,
            )
        except SopTriggerError as exc:
            status = _SOP_ERROR_STATUS.get(exc.code, 500)
            raise ApiError(status, exc.code, exc.message) from exc
        record = await _run(ctx.sop_repo.get_run, run_id)
        if record is None:
            raise ApiError(500, "internal_error", "run record missing after trigger")
        return sop_run_dto(record)

    @app.get("/api/sop/runs")
    async def list_sop_runs(
        camera_id: str | None = None,
        status: str | None = None,
        limit: int = Query(20, ge=1, le=200),
    ) -> dict[str, Any]:
        def work() -> list[dict[str, Any]]:
            return [sop_run_dto(r) for r in ctx.sop_repo.list_runs(camera_id=camera_id, status=status, limit=limit)]

        try:
            items = await _run(work)
        except ValueError as exc:
            raise ApiError(422, "validation_error", str(exc)) from exc
        return {"items": items}

    @app.get("/api/sop/runs/{run_id}")
    async def get_sop_run(run_id: str) -> dict[str, Any]:
        record = await _run(ctx.sop_repo.get_run, run_id)
        if record is None:
            raise ApiError(404, "not_found", f"sop run {run_id} not found")
        return sop_run_dto(record)

    # ------------------------------------------------------------ 证据与前端

    @app.get("/media/{path:path}")
    async def media(path: str) -> FileResponse:
        resolved = resolve_media_path(Path(ctx.config.evidence.base_dir), path)
        if resolved is None:
            raise ApiError(404, "not_found", "media not found")
        return FileResponse(resolved)

    if web_dir is not None and web_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

        @app.get("/")
        async def index() -> FileResponse:
            return FileResponse(web_dir / "index.html")

    @app.websocket("/ws/events")
    async def ws_events(ws: WebSocket) -> None:
        await ctx.hub.connect(ws)
        await ctx.hub.handle_client(ws)

    return app


async def _run(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """在线程池中执行阻塞函数（数据库访问等）。"""
    return await asyncio.to_thread(func, *args, **kwargs)
