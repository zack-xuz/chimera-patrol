"""Unit tests for chimera_patrol.vlm.normalize (task 8.1)."""

from __future__ import annotations

import json

import pytest

from chimera_patrol.vlm.normalize import VLMResponseError, VLMVerdict, normalize_vlm_response

VALID = {
    "verdict": "true_positive",
    "confidence": 0.9,
    "description": "有人翻越围栏",
    "suggestion": "立即通知值班人员",
}


def _json(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False)


def test_plain_json_is_parsed_into_verdict() -> None:
    result = normalize_vlm_response(_json(VALID), 50)
    assert result == VLMVerdict(
        verdict="true_positive",
        confidence=0.9,
        description="有人翻越围栏",
        suggestion="立即通知值班人员",
    )


def test_fenced_json_equals_unfenced_json() -> None:
    plain = normalize_vlm_response(_json(VALID), 50)
    fenced_json = normalize_vlm_response(f"```json\n{_json(VALID)}\n```", 50)
    fenced_bare = normalize_vlm_response(f"```\n{_json(VALID)}\n```", 50)
    fenced_inline = normalize_vlm_response(f"```json {_json(VALID)}```", 50)
    assert plain == fenced_json == fenced_bare == fenced_inline


def test_surrounding_whitespace_around_fence_is_ignored() -> None:
    result = normalize_vlm_response(f"\n  ```json\n{_json(VALID)}\n```\n", 50)
    assert result.verdict == "true_positive"


def test_description_is_stripped_and_truncated_to_max_chars() -> None:
    raw = _json({**VALID, "description": "   abcdefghij   "})
    result = normalize_vlm_response(raw, 5)
    assert result.description == "abcde"


def test_description_at_exact_limit_is_kept() -> None:
    raw = _json({**VALID, "description": "abcde"})
    assert normalize_vlm_response(raw, 5).description == "abcde"


def test_missing_suggestion_becomes_empty_string() -> None:
    obj = {k: v for k, v in VALID.items() if k != "suggestion"}
    assert normalize_vlm_response(_json(obj), 50).suggestion == ""


def test_suggestion_is_stripped() -> None:
    raw = _json({**VALID, "suggestion": "  复核  "})
    assert normalize_vlm_response(raw, 50).suggestion == "复核"


@pytest.mark.parametrize("verdict", ["TRUE_POSITIVE", "suspicious", "", None, 1])
def test_invalid_verdict_raises(verdict: object) -> None:
    with pytest.raises(VLMResponseError):
        normalize_vlm_response(_json({**VALID, "verdict": verdict}), 50)


@pytest.mark.parametrize("confidence", [-0.1, 1.01, 2, True, False, "0.5", None])
def test_invalid_confidence_raises(confidence: object) -> None:
    with pytest.raises(VLMResponseError):
        normalize_vlm_response(_json({**VALID, "confidence": confidence}), 50)


@pytest.mark.parametrize("confidence", [0, 1, 0.0, 1.0])
def test_confidence_boundaries_are_accepted(confidence: float) -> None:
    result = normalize_vlm_response(_json({**VALID, "confidence": confidence}), 50)
    assert result.confidence == float(confidence)


@pytest.mark.parametrize("description", ["", "   ", "\n\t", 42])
def test_empty_or_non_string_description_raises(description: object) -> None:
    with pytest.raises(VLMResponseError):
        normalize_vlm_response(_json({**VALID, "description": description}), 50)


@pytest.mark.parametrize("suggestion", [None, 3, ["a"]])
def test_non_string_suggestion_raises(suggestion: object) -> None:
    with pytest.raises(VLMResponseError):
        normalize_vlm_response(_json({**VALID, "suggestion": suggestion}), 50)


@pytest.mark.parametrize("raw", ["not json", "", "```json\nnot json\n```", "[1, 2]", "42"])
def test_non_object_or_invalid_json_raises(raw: str) -> None:
    with pytest.raises(VLMResponseError):
        normalize_vlm_response(raw, 50)


@pytest.mark.parametrize("missing", ["verdict", "confidence", "description"])
def test_missing_required_field_raises(missing: str) -> None:
    obj = {k: v for k, v in VALID.items() if k != missing}
    with pytest.raises(VLMResponseError):
        normalize_vlm_response(_json(obj), 50)


@pytest.mark.parametrize("max_chars", [0, -1])
def test_max_chars_below_one_raises_value_error(max_chars: int) -> None:
    with pytest.raises(ValueError):
        normalize_vlm_response(_json(VALID), max_chars)
