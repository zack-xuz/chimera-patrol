"""Unit tests for chimera_patrol.vlm.openai_compat and factory (task 8.4).

All HTTP traffic goes through httpx.MockTransport, so no network access is needed.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
import pytest

from chimera_patrol.config import OpenAICompatibleSettings, VlmSection
from chimera_patrol.vlm.factory import build_adapter
from chimera_patrol.vlm.mock import MockAdapter
from chimera_patrol.vlm.normalize import VLMResponseError, VLMVerdict
from chimera_patrol.vlm.openai_compat import OpenAICompatibleAdapter, VLMTransportError
from chimera_patrol.vlm.protocol import (
    HelmetCropRequest,
    HelmetCropResult,
    ReviewRequest,
    SopCheckRequest,
    VLMAdapter,
)

API_KEY = "sk-TESTKEY-0123456789abcdef"
JPEG = b"\xff\xd8\xff\xe0" + b"fake-jpeg-payload" + b"\xff\xd9"
SETTINGS = OpenAICompatibleSettings(
    base_url="https://vlm.test/v1",
    model="test-model",
    temperature=0,
    max_retries=0,
)
REVIEW_JSON = (
    '{"verdict": "true_positive", "confidence": 0.92, '
    '"description": "画面中有人员闯入禁区", "suggestion": "立即现场核查"}'
)

Handler = Callable[[httpx.Request], Awaitable[httpx.Response]]


def _body(content: str) -> dict[str, Any]:
    return {"choices": [{"message": {"content": content}}]}


def _review_req(**overrides: Any) -> ReviewRequest:
    fields: dict[str, Any] = {
        "event_id": "evt_000000000001",
        "event_type": "intrusion",
        "camera_name": "cam-yard",
        "roi_name": "zone-a",
        "cv_confidence": 0.87,
        "image_jpeg": JPEG,
        "description_max_chars": 50,
    }
    fields.update(overrides)
    return ReviewRequest(**fields)


def _adapter(
    handler: Handler,
    *,
    settings: OpenAICompatibleSettings = SETTINGS,
    semaphore: asyncio.Semaphore | None = None,
) -> OpenAICompatibleAdapter:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OpenAICompatibleAdapter(settings, API_KEY, client=client, semaphore=semaphore)


def _fixed_response(content: str, captured: list[httpx.Request] | None = None) -> Handler:
    async def handler(request: httpx.Request) -> httpx.Response:
        if captured is not None:
            captured.append(request)
        return httpx.Response(200, json=_body(content))

    return handler


# ----------------------------------------------------------------- review_event


def test_review_event_parses_valid_response() -> None:
    adapter = _adapter(_fixed_response(REVIEW_JSON))
    verdict = asyncio.run(adapter.review_event(_review_req()))
    assert verdict == VLMVerdict(
        verdict="true_positive",
        confidence=0.92,
        description="画面中有人员闯入禁区",
        suggestion="立即现场核查",
    )


def test_review_request_carries_data_url_prompt_and_auth_header() -> None:
    captured: list[httpx.Request] = []
    adapter = _adapter(_fixed_response(REVIEW_JSON, captured))
    asyncio.run(adapter.review_event(_review_req()))

    assert len(captured) == 1
    request = captured[0]
    assert request.method == "POST"
    assert str(request.url) == "https://vlm.test/v1/chat/completions"
    assert request.headers["Authorization"] == f"Bearer {API_KEY}"

    body = json.loads(request.content)
    assert body["model"] == "test-model"
    assert body["temperature"] == 0
    parts = body["messages"][0]["content"]
    assert [p["type"] for p in parts] == ["text", "image_url"]

    prompt = parts[0]["text"]
    assert "intrusion" in prompt
    assert "cam-yard" in prompt
    assert "zone-a" in prompt
    assert "0.87" in prompt
    assert "只输出一个 JSON 对象" in prompt

    expected_url = "data:image/jpeg;base64," + base64.b64encode(JPEG).decode("ascii")
    assert parts[1]["image_url"]["url"] == expected_url


def test_fenced_json_is_accepted() -> None:
    fenced = "```json\n" + REVIEW_JSON + "\n```"
    adapter = _adapter(_fixed_response(fenced))
    verdict = asyncio.run(adapter.review_event(_review_req()))
    assert verdict.verdict == "true_positive"
    assert verdict.confidence == 0.92


def test_description_is_truncated_to_max_chars() -> None:
    adapter = _adapter(_fixed_response(REVIEW_JSON))
    verdict = asyncio.run(adapter.review_event(_review_req(description_max_chars=4)))
    assert verdict.description == "画面中有"


def test_http_500_raises_transport_error_without_retry() -> None:
    calls: list[int] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(500, text="internal error")

    adapter = _adapter(handler)
    with pytest.raises(VLMTransportError, match="HTTP 500"):
        asyncio.run(adapter.review_event(_review_req()))
    assert len(calls) == 1


def test_http_401_raises_transport_error_without_retry() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "bad key"})

    adapter = _adapter(handler)
    with pytest.raises(VLMTransportError, match="HTTP 401"):
        asyncio.run(adapter.review_event(_review_req()))


def test_retry_happens_only_up_to_max_retries() -> None:
    attempts: list[int] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) == 1:
            return httpx.Response(503)
        return httpx.Response(200, json=_body(REVIEW_JSON))

    settings = SETTINGS.model_copy(update={"max_retries": 1})
    adapter = _adapter(handler, settings=settings)
    verdict = asyncio.run(adapter.review_event(_review_req()))
    assert verdict.verdict == "true_positive"
    assert len(attempts) == 2


def test_network_error_raises_transport_error() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    adapter = _adapter(handler)
    with pytest.raises(VLMTransportError, match="ConnectError"):
        asyncio.run(adapter.review_event(_review_req()))


@pytest.mark.parametrize(
    "response",
    [
        pytest.param(httpx.Response(200, text="not json at all"), id="body-not-json"),
        pytest.param(httpx.Response(200, json={"id": "x"}), id="no-choices"),
        pytest.param(httpx.Response(200, json=_body("this is not json")), id="content-not-json"),
        pytest.param(httpx.Response(200, json=_body('{"verdict": "maybe"}')), id="invalid-verdict"),
        pytest.param(httpx.Response(200, json=_body("[1, 2, 3]")), id="content-not-object"),
    ],
)
def test_malformed_response_raises_response_error(response: httpx.Response) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return response

    adapter = _adapter(handler)
    with pytest.raises(VLMResponseError):
        asyncio.run(adapter.review_event(_review_req()))


def test_logs_contain_summaries_but_not_image_or_key(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="chimera_patrol")
    adapter = _adapter(_fixed_response(REVIEW_JSON))
    asyncio.run(adapter.review_event(_review_req()))

    messages = [record.getMessage() for record in caplog.records if record.name.startswith("chimera_patrol")]
    joined = "\n".join(messages)
    assert f'vlm request id="evt_000000000001" image_bytes={len(JPEG)}' in joined
    assert 'vlm response id="evt_000000000001" verdict="true_positive"' in joined
    assert "duration_ms=" in joined
    assert API_KEY not in joined
    assert base64.b64encode(JPEG).decode("ascii") not in joined
    assert "data:image" not in joined


def test_error_path_logs_summary_without_key(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="chimera_patrol")
    adapter = _adapter(_fixed_response("not json"))
    with pytest.raises(VLMResponseError):
        asyncio.run(adapter.review_event(_review_req()))

    joined = "\n".join(record.getMessage() for record in caplog.records)
    assert 'vlm response id="evt_000000000001" verdict=null' in joined
    assert API_KEY not in joined


def test_cancelled_error_is_not_swallowed() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise asyncio.CancelledError()

    adapter = _adapter(handler)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(adapter.review_event(_review_req()))


def test_caller_timeout_applies_to_slow_transport() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(1.0)
        return httpx.Response(200, json=_body(REVIEW_JSON))

    adapter = _adapter(handler)

    async def run() -> None:
        await asyncio.wait_for(adapter.review_event(_review_req()), timeout=0.05)

    with pytest.raises(TimeoutError):
        asyncio.run(run())


# ---------------------------------------------------------------- concurrency


@pytest.mark.parametrize("max_concurrency", [1, 2])
def test_semaphore_limits_in_flight_requests(max_concurrency: int) -> None:
    state = {"current": 0, "peak": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        state["current"] += 1
        state["peak"] = max(state["peak"], state["current"])
        await asyncio.sleep(0.05)
        state["current"] -= 1
        return httpx.Response(200, json=_body(REVIEW_JSON))

    adapter = _adapter(handler, semaphore=asyncio.Semaphore(max_concurrency))

    async def run() -> list[VLMVerdict]:
        return await asyncio.gather(*(adapter.review_event(_review_req()) for _ in range(4)))

    results = asyncio.run(run())
    assert len(results) == 4
    assert state["peak"] == max_concurrency


# ----------------------------------------------------------------- check_sop


def _sop_request() -> SopCheckRequest:
    return SopCheckRequest(
        run_id="run_1",
        sop_name="entry",
        steps=[
            ("wear_helmet", "佩戴安全帽", "头盔清晰可见"),
            ("wear_vest", "穿反光衣", "反光条可见"),
            ("show_badge", "佩戴工牌", "工牌在胸前"),
        ],
        frames=[(0, 1000, JPEG), (1, 2000, JPEG)],
    )


def test_sop_request_sends_text_and_one_image_per_frame() -> None:
    captured: list[httpx.Request] = []
    content = json.dumps(
        {
            "steps": [
                {"step_id": "wear_helmet", "status": "pass", "evidence": "头盔可见", "evidence_frame": 0},
                {"step_id": "wear_vest", "status": "fail", "evidence": "未穿反光衣", "evidence_frame": 1},
                {"step_id": "show_badge", "status": "unknown", "evidence": "遮挡", "evidence_frame": None},
            ]
        },
        ensure_ascii=False,
    )
    adapter = _adapter(_fixed_response(content, captured))
    result = asyncio.run(adapter.check_sop(_sop_request()))

    parts = json.loads(captured[0].content)["messages"][0]["content"]
    assert [p["type"] for p in parts] == ["text", "text", "image_url", "text", "image_url"]
    assert [s[1] for s in result.steps] == ["pass", "fail", "unknown"]
    assert result.steps[1][2] == "未穿反光衣"
    assert result.steps[1][3] == 1


def test_sop_missing_step_becomes_unknown() -> None:
    content = json.dumps(
        {
            "steps": [
                {"step_id": "wear_vest", "status": "pass", "evidence": "反光衣可见", "evidence_frame": 1},
                {"step_id": "wear_helmet", "status": "pass", "evidence": "头盔可见", "evidence_frame": 0},
            ]
        },
        ensure_ascii=False,
    )
    adapter = _adapter(_fixed_response(content))
    result = asyncio.run(adapter.check_sop(_sop_request()))

    assert [s[0] for s in result.steps] == ["wear_helmet", "wear_vest", "show_badge"]
    assert result.steps[2][1] == "unknown"
    assert result.steps[2][3] is None
    assert result.steps[2][2] != ""


def test_sop_ignores_unknown_step_ids() -> None:
    content = json.dumps(
        {"steps": [{"step_id": "not_in_request", "status": "fail", "evidence": "x", "evidence_frame": 0}]}
    )
    adapter = _adapter(_fixed_response(content))
    result = asyncio.run(adapter.check_sop(_sop_request()))
    assert [s[1] for s in result.steps] == ["unknown", "unknown", "unknown"]


@pytest.mark.parametrize(
    "steps_payload",
    [
        pytest.param([{"step_id": "wear_helmet", "status": "maybe", "evidence": "x"}], id="invalid-status"),
        pytest.param("not-a-list", id="steps-not-list"),
        pytest.param([{"status": "pass", "evidence": "x"}], id="missing-step-id"),
    ],
)
def test_sop_invalid_structure_raises(steps_payload: Any) -> None:
    adapter = _adapter(_fixed_response(json.dumps({"steps": steps_payload})))
    with pytest.raises(VLMResponseError):
        asyncio.run(adapter.check_sop(_sop_request()))


# ------------------------------------------------------------ check_helmet_crop


def _helmet_req() -> HelmetCropRequest:
    return HelmetCropRequest(track_id="trk_7", camera_id="cam-yard", image_jpeg=JPEG)


def test_helmet_crop_valid_response() -> None:
    adapter = _adapter(_fixed_response('```json\n{"verdict": "no_helmet", "confidence": 0.8}\n```'))
    result = asyncio.run(adapter.check_helmet_crop(_helmet_req()))
    assert result == HelmetCropResult(verdict="no_helmet", confidence=0.8)


@pytest.mark.parametrize(
    "content",
    [
        pytest.param('{"verdict": "maybe", "confidence": 0.5}', id="bad-verdict"),
        pytest.param('{"verdict": "helmet", "confidence": 1.5}', id="confidence-above-1"),
        pytest.param('{"verdict": "helmet", "confidence": true}', id="confidence-bool"),
        pytest.param('{"verdict": "helmet"}', id="missing-confidence"),
    ],
)
def test_helmet_crop_invalid_response_raises(content: str) -> None:
    adapter = _adapter(_fixed_response(content))
    with pytest.raises(VLMResponseError):
        asyncio.run(adapter.check_helmet_crop(_helmet_req()))


# ------------------------------------------------------- shared review contract


def _mock_adapter() -> VLMAdapter:
    return MockAdapter("true_positive")


def _openai_adapter() -> VLMAdapter:
    return _adapter(_fixed_response(REVIEW_JSON))


@pytest.mark.parametrize("make_adapter", [_mock_adapter, _openai_adapter], ids=["mock", "openai_compatible"])
def test_review_contract_holds_for_each_adapter(make_adapter: Callable[[], VLMAdapter]) -> None:
    adapter = make_adapter()
    assert isinstance(adapter, VLMAdapter)
    verdict = asyncio.run(adapter.review_event(_review_req(description_max_chars=8)))
    assert verdict.verdict in ("true_positive", "false_positive", "uncertain")
    assert 0.0 <= verdict.confidence <= 1.0
    assert 1 <= len(verdict.description) <= 8
    assert isinstance(verdict.suggestion, str)


# ------------------------------------------------------------------- factory


def test_build_adapter_mock_uses_mock_settings() -> None:
    section = VlmSection(provider="mock", mock={"default_verdict": "false_positive", "by_camera": {"cam-yard": "uncertain"}})
    adapter = build_adapter(section, api_key=None)
    assert isinstance(adapter, MockAdapter)
    verdict = asyncio.run(adapter.review_event(_review_req(camera_name="cam-yard")))
    assert verdict.verdict == "uncertain"


def test_build_adapter_openai_compatible_requires_key() -> None:
    section = VlmSection(provider="openai_compatible", openai_compatible=SETTINGS)
    with pytest.raises(ValueError, match="VLM_API_KEY"):
        build_adapter(section, api_key="")


def test_build_adapter_openai_compatible_returns_adapter() -> None:
    section = VlmSection(provider="openai_compatible", openai_compatible=SETTINGS, max_concurrency=3)
    adapter = build_adapter(section, api_key=API_KEY)
    assert isinstance(adapter, OpenAICompatibleAdapter)
    assert isinstance(adapter, VLMAdapter)


