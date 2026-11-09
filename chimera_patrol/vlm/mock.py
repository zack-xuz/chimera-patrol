"""MockAdapter：确定性的 VLM 替身，仅用于单元测试与离线演示。

注意：Mock 不调用任何外部服务，返回值不反映真实画面内容。
演示或测试时通过 ``vlm.provider = mock`` 启用，正式环境应使用真实 provider。

行为概要：
- ``review_event``：verdict 取 ``by_camera[camera_name]``，缺省取 ``default_verdict``；
  置信度 true/false 为 0.9，uncertain 为 0.5；description 为固定中文短句，按
  ``description_max_chars`` 截断；suggestion 为固定中文字符串。
- ``check_sop``：按请求顺序逐步返回；step_id 含 ``fail`` 的步骤判为 fail（测试钩子），其余为 pass。
- ``check_helmet_crop``：默认返回 uncertain、置信度 0.0。
"""

from __future__ import annotations

from collections.abc import Mapping

from chimera_patrol.vlm.normalize import VLMVerdict, VerdictLabel
from chimera_patrol.vlm.protocol import (
    HelmetCropRequest,
    HelmetCropResult,
    ReviewRequest,
    SopCheckRequest,
    SopCheckResult,
)

_CONFIDENCE: dict[str, float] = {
    "true_positive": 0.9,
    "false_positive": 0.9,
    "uncertain": 0.5,
}
_DESCRIPTIONS: dict[str, str] = {
    "true_positive": "模拟复核：确认为真实事件",
    "false_positive": "模拟复核：判定为误报",
    "uncertain": "模拟复核：无法确定是否为真实事件",
}
_SUGGESTION = "请人工核对现场画面后处置"
_FAIL_MARKER = "fail"


class MockAdapter:
    """确定性的 VLMAdapter 实现，满足 ``VLMAdapter`` Protocol。"""

    def __init__(self, default_verdict: VerdictLabel, by_camera: Mapping[str, VerdictLabel] | None = None) -> None:
        self._default_verdict = default_verdict
        self._by_camera: dict[str, VerdictLabel] = dict(by_camera or {})

    async def review_event(self, req: ReviewRequest) -> VLMVerdict:
        verdict = self._by_camera.get(req.camera_name, self._default_verdict)
        description = _DESCRIPTIONS[verdict][: req.description_max_chars]
        return VLMVerdict(
            verdict=verdict,
            confidence=_CONFIDENCE[verdict],
            description=description,
            suggestion=_SUGGESTION,
        )

    async def check_sop(self, req: SopCheckRequest) -> SopCheckResult:
        first_frame = req.frames[0][0] if req.frames else None
        steps: list[tuple[str, str, str, int | None]] = []
        for step_id, _description, _criteria in req.steps:
            if _FAIL_MARKER in step_id:
                steps.append((step_id, "fail", "模拟判定：未通过", first_frame))
            else:
                steps.append((step_id, "pass", "模拟判定：通过", None))
        return SopCheckResult(steps=steps)

    async def check_helmet_crop(self, req: HelmetCropRequest) -> HelmetCropResult:
        return HelmetCropResult(verdict="uncertain", confidence=0.0)
