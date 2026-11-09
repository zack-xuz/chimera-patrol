"""FastMCP stdio 服务：把 VLM_Adapter 暴露为三个 MCP 工具（需求 5.4、13.4）。

工具：``review_event``、``check_sop``、``check_helmet_crop``。图片以 base64 字符串传入，
工具内转换为 ``chimera_patrol.vlm.protocol`` 的请求数据类，调用适配器后返回 JSON 可序列化的 dict。

API 依据：mcp Python SDK 1.x（requirements 锁定 mcp 1.30.0）的 ``mcp.server.fastmcp.FastMCP``：
``@mcp.tool()`` 注册工具，``mcp.run(transport="stdio")`` 启动 stdio 服务。

运行方式：``python -m chimera_patrol.mcp.vlm_server --config config.yaml``。
stdout 留给 MCP 协议；日志只写 stderr 与滚动日志文件。
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import os
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from chimera_patrol.config import AppConfig, ConfigError, load_config
from chimera_patrol.logging_setup import (
    log_vlm_request,
    log_vlm_response,
    register_secret,
    setup_logging,
)
from chimera_patrol.vlm.factory import build_adapter
from chimera_patrol.vlm.protocol import (
    HelmetCropRequest,
    ReviewRequest,
    SopCheckRequest,
    VLMAdapter,
)

SERVER_NAME = "chimera-vlm"
TOOL_REVIEW_EVENT = "review_event"
TOOL_CHECK_SOP = "check_sop"
TOOL_CHECK_HELMET_CROP = "check_helmet_crop"


class SopStepIn(BaseModel):
    step_id: str
    description: str
    criteria: str


class SopFrameIn(BaseModel):
    index: int
    ts_ms: int
    image_jpeg_base64: str = Field(description="JPEG 图片的 base64 编码")


def _decode_image(value: str, field: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"{field} 不是合法的 base64 字符串") from exc


async def _with_timeout(coro: Any, timeout_s: float) -> Any:
    return await asyncio.wait_for(coro, timeout=timeout_s)


def create_server(config: AppConfig, adapter: VLMAdapter) -> FastMCP:
    """创建注册了三个 VLM 工具的 FastMCP 实例。调用方负责传入已构造的适配器。"""
    mcp = FastMCP(SERVER_NAME)
    timeout_s = config.vlm.timeout_s
    default_max_chars = config.vlm.description_max_chars

    @mcp.tool(name=TOOL_REVIEW_EVENT)
    async def review_event(
        event_id: str,
        event_type: str,
        camera_name: str,
        cv_confidence: float,
        image_jpeg_base64: str,
        roi_name: str | None = None,
        description_max_chars: int | None = None,
    ) -> dict[str, Any]:
        """对单个候选事件的关键帧做 VLM 复核，返回 verdict、confidence、description、suggestion。"""
        image = _decode_image(image_jpeg_base64, "image_jpeg_base64")
        req = ReviewRequest(
            event_id=event_id,
            event_type=event_type,
            camera_name=camera_name,
            roi_name=roi_name,
            cv_confidence=cv_confidence,
            image_jpeg=image,
            description_max_chars=description_max_chars or default_max_chars,
        )
        log_vlm_request(event_id=event_id, image_bytes=image)
        started = time.perf_counter()
        verdict = await _with_timeout(adapter.review_event(req), timeout_s)
        result = asdict(verdict)
        log_vlm_response(
            event_id=event_id,
            verdict=result["verdict"],
            confidence=result["confidence"],
            duration_ms=int((time.perf_counter() - started) * 1000),
            summary=result["description"],
        )
        return result

    @mcp.tool(name=TOOL_CHECK_SOP)
    async def check_sop(
        run_id: str,
        sop_name: str,
        steps: list[SopStepIn],
        frames: list[SopFrameIn],
    ) -> dict[str, Any]:
        """按 SOP 步骤清单检查采样帧，每步返回 status（pass/fail/unknown）、evidence、evidence_frame。"""
        req = SopCheckRequest(
            run_id=run_id,
            sop_name=sop_name,
            steps=[(s.step_id, s.description, s.criteria) for s in steps],
            frames=[
                (f.index, f.ts_ms, _decode_image(f.image_jpeg_base64, f"frames[{f.index}].image_jpeg_base64"))
                for f in frames
            ],
        )
        result = await _with_timeout(adapter.check_sop(req), timeout_s)
        return {
            "steps": [
                {"step_id": sid, "status": status, "evidence": evidence, "evidence_frame": frame}
                for sid, status, evidence, frame in result.steps
            ]
        }

    @mcp.tool(name=TOOL_CHECK_HELMET_CROP)
    async def check_helmet_crop(
        track_id: str,
        camera_id: str,
        image_jpeg_base64: str,
    ) -> dict[str, Any]:
        """对人员头部裁剪图做安全帽兜底检查，返回 verdict（no_helmet/helmet/uncertain）与 confidence。"""
        req = HelmetCropRequest(
            track_id=track_id,
            camera_id=camera_id,
            image_jpeg=_decode_image(image_jpeg_base64, "image_jpeg_base64"),
        )
        result = await _with_timeout(adapter.check_helmet_crop(req), timeout_s)
        return asdict(result)

    return mcp


def build_runtime_adapter(config: AppConfig, env: Mapping[str, str] | None = None) -> VLMAdapter:
    """读取 ``vlm.openai_compatible.api_key_env`` 指定的环境变量，并交给 factory.build_adapter 创建适配器。"""
    source = os.environ if env is None else env
    api_key = source.get(config.vlm.openai_compatible.api_key_env) or None
    register_secret(api_key)
    return build_adapter(config.vlm, api_key)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="chimera_patrol.mcp.vlm_server", description="Chimera VLM MCP 服务（stdio）")
    parser.add_argument("--config", default="config.yaml", help="config.yaml 路径（默认 config.yaml）")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    setup_logging(
        level=config.logging.level,
        log_dir=config.logging.dir,
        max_bytes=config.logging.max_bytes,
        backup_count=config.logging.backup_count,
        console=True,
    )
    adapter = build_runtime_adapter(config)
    server = create_server(config, adapter)
    server.run(transport="stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
