"""Property tests for VLM response normalization.

**Validates: Requirements 5.2**

Property 12: for any text response, normalize_vlm_response either returns a
VLMVerdict satisfying the verdict contract or raises VLMResponseError. A valid
JSON object wrapped in a markdown code fence yields the same result as the
unfenced JSON.
"""

from __future__ import annotations

import json

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from chimera_patrol.vlm.normalize import (
    VLMResponseError,
    VLMVerdict,
    normalize_vlm_response,
)

PBT_SETTINGS = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

VALID_VERDICTS = ["true_positive", "false_positive", "uncertain"]
max_chars_strategy = st.integers(min_value=1, max_value=60)


def _assert_contract(result: VLMVerdict, max_chars: int) -> None:
    assert isinstance(result, VLMVerdict)
    assert result.verdict in {"true_positive", "false_positive", "uncertain"}
    assert isinstance(result.confidence, float)
    assert 0.0 <= result.confidence <= 1.0
    assert 1 <= len(result.description) <= max_chars
    assert isinstance(result.suggestion, str)


@st.composite
def valid_payloads(draw: st.DrawFn) -> dict:
    description = draw(st.text(min_size=1, max_size=200).filter(lambda s: s.strip() != ""))
    suggestion = draw(st.text(max_size=100))
    confidence = draw(
        st.one_of(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            st.integers(min_value=0, max_value=1),
        )
    )
    return {
        "verdict": draw(st.sampled_from(VALID_VERDICTS)),
        "confidence": confidence,
        "description": description,
        "suggestion": suggestion,
    }


@st.composite
def invalid_payloads(draw: st.DrawFn) -> dict:
    """Payloads that violate exactly one part of the verdict contract."""
    payload = draw(valid_payloads())
    kind = draw(st.sampled_from(["verdict", "confidence", "description", "suggestion"]))
    if kind == "verdict":
        payload["verdict"] = draw(
            st.text(max_size=20).filter(lambda s: s not in VALID_VERDICTS)
        )
    elif kind == "confidence":
        payload["confidence"] = draw(
            st.one_of(
                st.floats(allow_nan=False, allow_infinity=False).filter(
                    lambda f: not 0.0 <= f <= 1.0
                ),
                st.booleans(),
                st.text(max_size=10),
            )
        )
    elif kind == "description":
        payload["description"] = draw(
            st.text(alphabet=st.sampled_from(" \t\n\r\x0b\x0c"), max_size=10)
        )
    else:
        payload["suggestion"] = draw(
            st.one_of(st.none(), st.integers(), st.booleans(), st.lists(st.integers(), max_size=3))
        )
    return payload


@PBT_SETTINGS
@given(raw=st.text(), max_chars=max_chars_strategy)
def test_arbitrary_text_yields_verdict_or_response_error(raw: str, max_chars: int) -> None:
    """Property 12: arbitrary text either normalizes to a valid verdict or raises VLMResponseError."""
    try:
        result = normalize_vlm_response(raw, max_chars)
    except VLMResponseError:
        return
    _assert_contract(result, max_chars)


@PBT_SETTINGS
@given(payload=valid_payloads(), max_chars=max_chars_strategy)
def test_valid_object_round_trips_with_constraints(payload: dict, max_chars: int) -> None:
    """Property 12: a valid object yields a verdict whose description is the stripped text truncated."""
    raw = json.dumps(payload, ensure_ascii=False)
    result = normalize_vlm_response(raw, max_chars)

    _assert_contract(result, max_chars)
    assert result.verdict == payload["verdict"]
    assert result.confidence == float(payload["confidence"])
    assert result.description == payload["description"].strip()[:max_chars]
    assert result.suggestion == payload["suggestion"].strip()


@PBT_SETTINGS
@given(payload=valid_payloads(), max_chars=max_chars_strategy)
def test_code_fence_is_equivalent_to_unfenced_json(payload: dict, max_chars: int) -> None:
    """Property 12: fenced and unfenced valid JSON produce identical results."""
    body = json.dumps(payload, ensure_ascii=False)
    plain = normalize_vlm_response(body, max_chars)
    json_fenced = normalize_vlm_response(f"```json\n{body}\n```", max_chars)
    bare_fenced = normalize_vlm_response(f"```\n{body}\n```", max_chars)

    assert json_fenced == plain
    assert bare_fenced == plain


@PBT_SETTINGS
@given(payload=invalid_payloads(), max_chars=max_chars_strategy)
def test_invalid_object_raises_response_error(payload: dict, max_chars: int) -> None:
    """Property 12: objects violating the verdict contract raise VLMResponseError."""
    raw = json.dumps(payload, ensure_ascii=False)
    with pytest.raises(VLMResponseError):
        normalize_vlm_response(raw, max_chars)
