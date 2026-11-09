"""VLM_Adapter 接口：provider 无关的视觉语言模型访问协议（需求 5.3）。

所有方法均为 async。应用进程内直接调用，MCP server 复用同一实现。
调用方统一用 ``asyncio.wait_for(..., vlm.timeout_s)`` 包裹调用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from chimera_patrol.vlm.normalize import VLMVerdict

SopStepStatus = Literal["pass", "fail", "unknown"]
HelmetVerdict = Literal["no_helmet", "helmet", "uncertain"]


@dataclass(frozen=True)
class ReviewRequest:
    """单个事件的 VLM 复核请求。"""

    event_id: str
    event_type: str
    camera_name: str
    roi_name: str | None
    cv_confidence: float
    image_jpeg: bytes  # 关键帧 JPEG
    description_max_chars: int


@dataclass(frozen=True)
class SopCheckRequest:
    """SOP 检查请求。``steps`` 与 ``frames`` 均按顺序排列。"""

    run_id: str
    sop_name: str
    steps: list[tuple[str, str, str]]  # (step_id, description, criteria)
    frames: list[tuple[int, int, bytes]]  # (index, ts_ms, jpeg bytes)


@dataclass(frozen=True)
class SopCheckResult:
    """SOP 检查结果。``steps`` 与请求中的步骤顺序一致。"""

    steps: list[tuple[str, SopStepStatus, str, int | None]]  # (step_id, status, evidence, evidence_frame)


@dataclass(frozen=True)
class HelmetCropRequest:
    """安全帽头部裁剪图的低频兜底检查请求。"""

    track_id: str
    camera_id: str
    image_jpeg: bytes


@dataclass(frozen=True)
class HelmetCropResult:
    """安全帽兜底检查结果。"""

    verdict: HelmetVerdict
    confidence: float  # 0..1


@runtime_checkable
class VLMAdapter(Protocol):
    """VLM 适配器接口。实现：OpenAI 兼容（8.4）与 MockAdapter（离线/演示）。"""

    async def review_event(self, req: ReviewRequest) -> VLMVerdict: ...

    async def check_sop(self, req: SopCheckRequest) -> SopCheckResult: ...

    async def check_helmet_crop(self, req: HelmetCropRequest) -> HelmetCropResult: ...
