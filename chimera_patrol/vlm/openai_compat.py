"""OpenAICompatibleAdapter：基于 httpx 调用 OpenAI 兼容 chat completions 接口（需求 5.2、5.3、12.6）。

请求形态：``POST {base_url}/chat/completions``，body 含 ``model``、``temperature`` 与 ``messages``。
图片作为 ``image_url`` 内容部件，值为 ``data:image/jpeg;base64,...`` 形式的 data URL。
响应取 ``choices[0].message.content`` 作为模型文本，要求其为单个 JSON 对象。

错误约定：
- 网络错误、非 2xx 状态码：抛出 ``VLMTransportError``。
- 响应体不是预期结构、内容不是合法 JSON 或字段不合规：抛出 ``VLMResponseError``。
- 重试仅发生在网络错误、HTTP 429 与 5xx 上，次数为 ``max_retries``（默认 0，即不重试）。
- 不捕获 ``asyncio.CancelledError``。超时由调用方用 ``asyncio.wait_for(..., vlm.timeout_s)`` 施加。

密钥由调用方从环境变量读取后传入，本模块不读取 ``os.environ``。
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
import time
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import httpx

from chimera_patrol.config import OpenAICompatibleSettings
from chimera_patrol.logging_setup import log_vlm_request, log_vlm_response, register_secret
from chimera_patrol.vlm.normalize import VLMResponseError, VLMVerdict, _strip_code_fence, normalize_vlm_response
from chimera_patrol.vlm.protocol import (
    HelmetCropRequest,
    HelmetCropResult,
    HelmetVerdict,
    ReviewRequest,
    SopCheckRequest,
    SopCheckResult,
    SopStepStatus,
)

_T = TypeVar("_T")

_HELMET_VERDICTS: tuple[str, ...] = ("no_helmet", "helmet", "uncertain")
_SOP_STATUSES: tuple[str, ...] = ("pass", "fail", "unknown")
_RETRYABLE_STATUS = 429
_MISSING_STEP_EVIDENCE = "模型未返回该步骤的判定结果"

_REVIEW_PROMPT = """你是安防巡检复核员。视频分析系统产生了一条候选事件，请结合附带的关键帧图片判断它是否为真实事件。

事件类型：{event_type}
摄像头：{camera_name}
区域（ROI）：{roi_name}
检测置信度（CV）：{cv_confidence:.2f}

verdict 含义：true_positive 表示确为真实事件；false_positive 表示误报；uncertain 表示无法确定。

只输出一个 JSON 对象，不要输出任何其他文字，也不要使用代码围栏。格式如下：
{{"verdict": "true_positive|false_positive|uncertain", "confidence": 0 到 1 之间的数字, "description": "一句中文描述，不超过{max_chars}个字", "suggestion": "处置建议"}}"""

_HELMET_PROMPT = """你是安全帽检查员。图片为某人员头部区域的裁剪图，请判断该人员是否佩戴安全帽。

只输出一个 JSON 对象，不要输出任何其他文字，也不要使用代码围栏。格式如下：
{"verdict": "no_helmet|helmet|uncertain", "confidence": 0 到 1 之间的数字}"""

_SOP_PROMPT_HEAD = """你是工业现场安全巡检员。下面按时间顺序给出一段时间窗口内的若干帧图片，每帧前标有帧序号与时间戳。
请逐一判断以下 SOP 步骤是否已在画面中完成：
{steps}

判定规则：画面可以确认完成记为 pass；可以确认未完成记为 fail；无法判断记为 unknown。

只输出一个 JSON 对象，不要输出任何其他文字，也不要使用代码围栏。格式如下：
{{"steps": [{{"step_id": "与上方一致", "status": "pass|fail|unknown", "evidence": "判定依据（一句中文）", "evidence_frame": 依据所在的帧序号或 null}}]}}"""


class VLMTransportError(Exception):
    """网络错误、非 2xx 状态码（429 与 5xx 在用尽重试后仍抛出）。"""


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _image_part(jpeg: bytes) -> dict[str, Any]:
    encoded = base64.b64encode(jpeg).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}


def _text_part(text: str) -> dict[str, str]:
    return {"type": "text", "text": text}


def _load_json_object(content: str) -> dict[str, Any]:
    try:
        payload = json.loads(_strip_code_fence(content))
    except json.JSONDecodeError as exc:
        raise VLMResponseError(f"response is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise VLMResponseError("response JSON must be an object")
    return payload


def _parse_helmet(content: str) -> HelmetCropResult:
    payload = _load_json_object(content)
    verdict = payload.get("verdict")
    if verdict not in _HELMET_VERDICTS:
        raise VLMResponseError(f"helmet verdict must be one of {_HELMET_VERDICTS}, got {verdict!r}")
    confidence = payload.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise VLMResponseError(f"confidence must be a number, got {confidence!r}")
    number = float(confidence)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise VLMResponseError(f"confidence must be within [0, 1], got {confidence!r}")
    helmet_verdict: HelmetVerdict = verdict  # type: ignore[assignment]
    return HelmetCropResult(verdict=helmet_verdict, confidence=number)


def _parse_sop(content: str, step_ids: Sequence[str]) -> SopCheckResult:
    payload = _load_json_object(content)
    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list):
        raise VLMResponseError("steps must be a list")

    parsed: dict[str, tuple[SopStepStatus, str, int | None]] = {}
    for item in raw_steps:
        if not isinstance(item, dict):
            raise VLMResponseError("each step must be an object")
        step_id = item.get("step_id")
        if not isinstance(step_id, str):
            raise VLMResponseError(f"step_id must be a string, got {step_id!r}")
        status = item.get("status")
        if status not in _SOP_STATUSES:
            raise VLMResponseError(f"step status must be one of {_SOP_STATUSES}, got {status!r}")
        evidence = item.get("evidence", "")
        if evidence is None:
            evidence = ""
        if not isinstance(evidence, str):
            raise VLMResponseError(f"evidence must be a string, got {evidence!r}")
        frame = item.get("evidence_frame")
        if frame is not None and (isinstance(frame, bool) or not isinstance(frame, int)):
            raise VLMResponseError(f"evidence_frame must be an integer or null, got {frame!r}")
        # 重复的 step_id 只取第一次出现的结果；请求中不存在的 step_id 忽略。
        if step_id not in parsed:
            status_value: SopStepStatus = status  # type: ignore[assignment]
            parsed[step_id] = (status_value, evidence.strip(), frame)

    steps: list[tuple[str, SopStepStatus, str, int | None]] = []
    for step_id in step_ids:
        if step_id in parsed:
            status, evidence, frame = parsed[step_id]
            steps.append((step_id, status, evidence, frame))
        else:
            steps.append((step_id, "unknown", _MISSING_STEP_EVIDENCE, None))
    return SopCheckResult(steps=steps)


class OpenAICompatibleAdapter:
    """VLMAdapter 的 OpenAI 兼容实现。满足 ``VLMAdapter`` Protocol。"""

    def __init__(
        self,
        settings: OpenAICompatibleSettings,
        api_key: str,
        *,
        client: httpx.AsyncClient | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> None:
        """
        Args:
            settings: ``vlm.openai_compatible`` 配置段。``base_url`` 与 ``model`` 必须已设置。
            api_key: 调用方从 ``settings.api_key_env`` 指定的环境变量读取的密钥。
            client: 可选的 httpx 客户端（测试时注入 MockTransport）。缺省时按 ``base_url`` 创建。
            semaphore: 限制同时在途请求数，应传入 ``Semaphore(vlm.max_concurrency)``。
                缺省时使用配置默认值 2。
        """
        if not settings.base_url:
            raise ValueError("vlm.openai_compatible.base_url is required")
        if not settings.model:
            raise ValueError("vlm.openai_compatible.model is required")
        if not api_key:
            raise ValueError("API key must be a non-empty string")

        register_secret(api_key)
        self._settings = settings
        self._base_url = settings.base_url.rstrip("/")
        self._api_key = api_key
        # 超时由调用方施加，这里关闭 httpx 默认的 5 秒超时。
        self._owns_client = client is None
        self._client = client if client is not None else httpx.AsyncClient(base_url=self._base_url, timeout=None)
        self._semaphore = semaphore if semaphore is not None else asyncio.Semaphore(2)

    async def aclose(self) -> None:
        """关闭自建的 httpx 客户端。注入的客户端由调用方负责关闭。"""
        if self._owns_client:
            await self._client.aclose()

    async def review_event(self, req: ReviewRequest) -> VLMVerdict:
        prompt = _REVIEW_PROMPT.format(
            event_type=req.event_type,
            camera_name=req.camera_name,
            roi_name=req.roi_name or "无",
            cv_confidence=req.cv_confidence,
            max_chars=req.description_max_chars,
        )
        parts = [_text_part(prompt), _image_part(req.image_jpeg)]

        def parse(content: str) -> VLMVerdict:
            return normalize_vlm_response(content, req.description_max_chars)

        async with self._semaphore:
            return await self._request(
                log_id=req.event_id,
                image_bytes=len(req.image_jpeg),
                parts=parts,
                parse=parse,
                describe=lambda result: (result.verdict, result.confidence),
            )

    async def check_sop(self, req: SopCheckRequest) -> SopCheckResult:
        steps_text = "\n".join(
            f"- step_id: {step_id}；内容：{description}；判定标准：{criteria}"
            for step_id, description, criteria in req.steps
        )
        parts: list[dict[str, Any]] = [_text_part(_SOP_PROMPT_HEAD.format(steps=steps_text))]
        for index, ts_ms, jpeg in req.frames:
            parts.append(_text_part(f"帧序号 {index}，时间戳 {ts_ms} ms"))
            parts.append(_image_part(jpeg))
        image_bytes = sum(len(jpeg) for _, _, jpeg in req.frames)
        step_ids = [step_id for step_id, _, _ in req.steps]

        def parse(content: str) -> SopCheckResult:
            return _parse_sop(content, step_ids)

        async with self._semaphore:
            return await self._request(
                log_id=req.run_id,
                image_bytes=image_bytes,
                parts=parts,
                parse=parse,
                describe=lambda result: (None, None),
            )

    async def check_helmet_crop(self, req: HelmetCropRequest) -> HelmetCropResult:
        parts = [_text_part(_HELMET_PROMPT), _image_part(req.image_jpeg)]
        async with self._semaphore:
            return await self._request(
                log_id=req.track_id,
                image_bytes=len(req.image_jpeg),
                parts=parts,
                parse=_parse_helmet,
                describe=lambda result: (result.verdict, result.confidence),
            )

    async def _request(
        self,
        *,
        log_id: str,
        image_bytes: int,
        parts: list[dict[str, Any]],
        parse: Callable[[str], _T],
        describe: Callable[[_T], tuple[str | None, float | None]],
    ) -> _T:
        """发送一次复核请求，解析结果，并记录响应摘要。"""
        started = time.perf_counter()
        try:
            content = await self._complete(log_id=log_id, image_bytes=image_bytes, parts=parts)
            result = parse(content)
        except (VLMResponseError, VLMTransportError) as exc:
            log_vlm_response(
                event_id=log_id,
                verdict=None,
                confidence=None,
                duration_ms=_elapsed_ms(started),
                summary=f"{type(exc).__name__}: {exc}",
            )
            raise

        verdict, confidence = describe(result)
        log_vlm_response(
            event_id=log_id,
            verdict=verdict,
            confidence=confidence,
            duration_ms=_elapsed_ms(started),
        )
        return result

    async def _complete(self, *, log_id: str, image_bytes: int, parts: list[dict[str, Any]]) -> str:
        """POST chat completions，返回 ``choices[0].message.content``。网络类错误按 max_retries 重试。"""
        body = {
            "model": self._settings.model,
            "temperature": self._settings.temperature,
            "messages": [{"role": "user", "content": parts}],
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}
        url = f"{self._base_url}/chat/completions"
        attempts = 1 + self._settings.max_retries

        log_vlm_request(event_id=log_id, image_bytes=image_bytes)
        for attempt in range(attempts):
            try:
                response = await self._client.post(url, json=body, headers=headers)
            except httpx.HTTPError as exc:
                error: VLMTransportError = VLMTransportError(f"network error: {type(exc).__name__}")
            else:
                status = response.status_code
                if 200 <= status < 300:
                    return _extract_content(response)
                error = VLMTransportError(f"HTTP {status}")
                if status != _RETRYABLE_STATUS and status < 500:
                    raise error
            if attempt == attempts - 1:
                raise error
        raise AssertionError("unreachable")  # pragma: no cover


def _extract_content(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError as exc:
        raise VLMResponseError("response body is not valid JSON") from exc
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise VLMResponseError("response body has no choices[0].message.content") from exc
    if not isinstance(content, str):
        raise VLMResponseError("message content must be a string")
    return content


__all__ = ["OpenAICompatibleAdapter", "VLMTransportError"]
