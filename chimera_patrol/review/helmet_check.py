"""安全帽 VLM 兜底的异步调用（任务 11.2）。

运行在 asyncio 事件循环中。DetectWorker 线程把头部裁剪图经 ``run_coroutine_threadsafe``
投递到这里，返回的 ``concurrent.futures.Future`` 由工作线程轮询。

- 与复核共用同一个 ``asyncio.Semaphore``（vlm.max_concurrency）限制在途请求数。
- ``asyncio.wait_for`` 以 ``vlm.timeout_s`` 限时。
- 超时、适配器异常、响应非法都只记录日志并返回 None，不产生事件（需求 3.7）。
- 日志只记录请求与响应摘要，不记录图片内容。
"""

from __future__ import annotations

import asyncio
import logging

from chimera_patrol.vlm.normalize import VLMResponseError
from chimera_patrol.vlm.protocol import HelmetCropRequest, HelmetCropResult, VLMAdapter

__all__ = ["HelmetCropChecker"]

_log = logging.getLogger("chimera_patrol.review.helmet")


class HelmetCropChecker:
    """对单张头部裁剪图执行 ``check_helmet_crop``，失败返回 None。"""

    def __init__(
        self,
        adapter: VLMAdapter,
        *,
        timeout_s: float,
        semaphore: asyncio.Semaphore,
    ) -> None:
        if isinstance(timeout_s, bool) or not timeout_s > 0:
            raise ValueError(f"timeout_s must be > 0, got {timeout_s!r}")
        self._adapter = adapter
        self._timeout_s = float(timeout_s)
        self._semaphore = semaphore

    async def check(self, req: HelmetCropRequest) -> HelmetCropResult | None:
        """返回适配器结果；失败时返回 None。``CancelledError`` 原样抛出。"""
        tag = f"camera={req.camera_id} track={req.track_id}"
        try:
            async with self._semaphore:
                _log.debug("helmet fallback request %s image_bytes=%d", tag, len(req.image_jpeg))
                result = await asyncio.wait_for(self._adapter.check_helmet_crop(req), timeout=self._timeout_s)
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            _log.warning("helmet fallback timeout %s timeout_s=%s", tag, self._timeout_s)
            return None
        except VLMResponseError as exc:
            _log.warning("helmet fallback response invalid %s error=%s", tag, exc)
            return None
        except Exception as exc:  # noqa: BLE001 - 兜底失败只记日志
            _log.warning("helmet fallback failed %s error=%s: %s", tag, type(exc).__name__, exc)
            return None
        _log.debug("helmet fallback response %s verdict=%s confidence=%.3f", tag, result.verdict, result.confidence)
        return result
