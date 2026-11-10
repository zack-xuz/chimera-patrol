"""WsHub：WebSocket 客户端集合、广播与心跳（任务 13.4；event-contract 第 6 节）。

- 所有消息使用统一信封 ``{type, seq, ts_ms, data}``，``seq`` 为进程内单调递增整数。
- 连接建立后立即发送 ``hello``；之后每 ``heartbeat_interval_s`` 秒发送一次 ``heartbeat``。
- 客户端只处理 ``ping``，回复 ``pong``，其他消息忽略。
- 单个客户端发送失败时只移除该客户端，不影响其他客户端。

``broadcast`` 与连接管理必须在事件循环中调用（非协程线程请用 ``run_coroutine_threadsafe``）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect, WebSocketState

_log = logging.getLogger("chimera_patrol.ws")


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


class WsHub:
    """管理所有 WebSocket 客户端并广播消息。"""

    def __init__(self, *, heartbeat_interval_s: int, vlm_review_enabled: bool) -> None:
        self._heartbeat_interval_s = heartbeat_interval_s
        self._vlm_review_enabled = vlm_review_enabled
        self._clients: set[WebSocket] = set()
        self._seq = 0
        self._heartbeat_task: asyncio.Task[None] | None = None

    @property
    def client_count(self) -> int:
        return len(self._clients)

    @property
    def seq(self) -> int:
        return self._seq

    def _envelope(self, msg_type: str, data: dict[str, Any]) -> str:
        self._seq += 1
        return json.dumps(
            {"type": msg_type, "seq": self._seq, "ts_ms": _now_ms(), "data": data},
            ensure_ascii=False,
        )

    async def connect(self, ws: WebSocket) -> None:
        """接受连接并发送 hello。"""
        await ws.accept()
        self._clients.add(ws)
        hello = self._envelope(
            "hello",
            {
                "server_time_ms": _now_ms(),
                "vlm_review_enabled": self._vlm_review_enabled,
                "heartbeat_interval_s": self._heartbeat_interval_s,
            },
        )
        try:
            await ws.send_text(hello)
        except Exception:
            self._clients.discard(ws)
            raise

    def disconnect(self, ws: WebSocket) -> None:
        self._clients.discard(ws)

    async def broadcast(self, msg_type: str, data: dict[str, Any]) -> None:
        """向所有客户端发送一条消息。发送失败的客户端会被移除。"""
        if not self._clients:
            # 仍然递增 seq，使 seq 在进程内连续，与是否有客户端无关。
            self._seq += 1
            return
        text = self._envelope(msg_type, data)
        await self._send_all(text)

    async def _send_all(self, text: str) -> None:
        dead: list[WebSocket] = []
        for ws in list(self._clients):
            try:
                if ws.client_state != WebSocketState.CONNECTED:
                    dead.append(ws)
                    continue
                await ws.send_text(text)
            except Exception:
                _log.info("websocket client dropped during send")
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)

    async def handle_client(self, ws: WebSocket) -> None:
        """处理一个已连接客户端的入站消息，直到连接断开。"""
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    message = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                if isinstance(message, dict) and message.get("type") == "ping":
                    await ws.send_text(self._envelope("pong", {}))
        except WebSocketDisconnect:
            return
        finally:
            self.disconnect(ws)

    def start_heartbeat(self) -> None:
        """启动心跳任务。需在事件循环中调用，重复调用无效。"""
        if self._heartbeat_task is None or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.ensure_future(self._heartbeat_loop())

    async def stop_heartbeat(self) -> None:
        task = self._heartbeat_task
        self._heartbeat_task = None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(self._heartbeat_interval_s)
            await self.broadcast("heartbeat", {})
