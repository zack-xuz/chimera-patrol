"""Normalization of raw VLM text responses into VLMVerdict (requirement 5.2).

The VLM is asked to answer with a single JSON object. Providers often wrap it
in a markdown code fence, so the fence is stripped before parsing. Anything
that does not satisfy the verdict contract raises VLMResponseError.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Literal

VerdictLabel = Literal["true_positive", "false_positive", "uncertain"]

_VALID_VERDICTS: tuple[str, ...] = ("true_positive", "false_positive", "uncertain")

# Matches a whole-response code fence: ```json\n{...}\n``` or ```\n{...}\n```.
_FENCE_RE = re.compile(
    r"^\s*```[A-Za-z0-9_+-]*[ \t]*\r?\n?(?P<body>.*?)\r?\n?[ \t]*```\s*$",
    re.DOTALL,
)


class VLMResponseError(Exception):
    """Raised when a VLM response does not satisfy the verdict contract."""


@dataclass(frozen=True)
class VLMVerdict:
    verdict: VerdictLabel
    confidence: float  # 0..1
    description: str  # 1..description_max_chars
    suggestion: str


def _strip_code_fence(raw_text: str) -> str:
    match = _FENCE_RE.match(raw_text)
    if match is None:
        return raw_text
    return match.group("body")


def _parse_verdict(value: Any) -> VerdictLabel:
    if not isinstance(value, str) or value not in _VALID_VERDICTS:
        raise VLMResponseError(f"verdict must be one of {_VALID_VERDICTS}, got {value!r}")
    return value  # type: ignore[return-value]


def _parse_confidence(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VLMResponseError(f"confidence must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise VLMResponseError(f"confidence must be within [0, 1], got {value!r}")
    return number


def _parse_description(value: Any, max_chars: int) -> str:
    if not isinstance(value, str):
        raise VLMResponseError(f"description must be a string, got {value!r}")
    text = value.strip()
    if not text:
        raise VLMResponseError("description must not be empty")
    return text[:max_chars]


def _parse_suggestion(value: Any) -> str:
    if value is None:
        raise VLMResponseError("suggestion must be a string, got null")
    if not isinstance(value, str):
        raise VLMResponseError(f"suggestion must be a string, got {value!r}")
    return value.strip()


def normalize_vlm_response(raw_text: str, max_chars: int) -> VLMVerdict:
    """Parse a raw VLM text response into a VLMVerdict.

    Args:
        raw_text: Response text, optionally wrapped in a markdown code fence.
        max_chars: Upper bound for description length (vlm.description_max_chars).
            Longer descriptions are truncated, not rejected.

    Raises:
        ValueError: If max_chars is not an integer >= 1.
        VLMResponseError: If the response is not a valid verdict object.
    """
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 1:
        raise ValueError(f"max_chars must be an integer >= 1, got {max_chars!r}")

    if not isinstance(raw_text, str):
        raise VLMResponseError("response must be text")

    body = _strip_code_fence(raw_text)
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise VLMResponseError(f"response is not valid JSON: {exc.msg}") from exc

    if not isinstance(payload, dict):
        raise VLMResponseError("response JSON must be an object")

    if "verdict" not in payload:
        raise VLMResponseError("verdict is missing")
    if "confidence" not in payload:
        raise VLMResponseError("confidence is missing")
    if "description" not in payload:
        raise VLMResponseError("description is missing")

    return VLMVerdict(
        verdict=_parse_verdict(payload["verdict"]),
        confidence=_parse_confidence(payload["confidence"]),
        description=_parse_description(payload["description"], max_chars),
        suggestion=_parse_suggestion(payload.get("suggestion", "")),
    )
