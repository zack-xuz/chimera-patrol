"""Unit tests for chimera_patrol.review.decision.decide_status (task 10.1)."""

from __future__ import annotations

import pytest

from chimera_patrol.review.decision import decide_status

THRESHOLD = 0.7


@pytest.mark.parametrize(
    ("verdict", "confidence", "expected"),
    [
        ("false_positive", 0.95, "dismissed"),
        ("true_positive", 0.95, "confirmed"),
        ("uncertain", 0.99, "needs_human"),
        ("false_positive", 0.10, "needs_human"),
        ("true_positive", 0.10, "needs_human"),
        ("uncertain", 0.0, "needs_human"),
    ],
)
def test_example_decisions(verdict: str, confidence: float, expected: str) -> None:
    assert decide_status(verdict, confidence, THRESHOLD) == expected


def test_confidence_equal_to_threshold_is_auto_decision() -> None:
    assert decide_status("false_positive", 0.7, 0.7) == "dismissed"
    assert decide_status("true_positive", 0.7, 0.7) == "confirmed"


def test_confidence_just_below_threshold_needs_human() -> None:
    assert decide_status("false_positive", 0.6999, 0.7) == "needs_human"
    assert decide_status("true_positive", 0.6999, 0.7) == "needs_human"


def test_uncertain_at_high_confidence_needs_human() -> None:
    assert decide_status("uncertain", 1.0, 0.7) == "needs_human"


def test_threshold_boundaries_zero_and_one() -> None:
    assert decide_status("true_positive", 0.0, 0.0) == "confirmed"
    assert decide_status("false_positive", 1.0, 1.0) == "dismissed"
    assert decide_status("false_positive", 0.99, 1.0) == "needs_human"


def test_integer_confidence_and_threshold_are_accepted() -> None:
    assert decide_status("true_positive", 1, 1) == "confirmed"
    assert decide_status("false_positive", 0, 0) == "dismissed"


@pytest.mark.parametrize("verdict", ["", "TRUE_POSITIVE", "maybe", None, 1])
def test_invalid_verdict_raises(verdict: object) -> None:
    with pytest.raises(ValueError):
        decide_status(verdict, 0.9, THRESHOLD)  # type: ignore[arg-type]


@pytest.mark.parametrize("confidence", [-0.01, 1.01, float("nan"), float("inf"), "0.9", None, True])
def test_invalid_confidence_raises(confidence: object) -> None:
    with pytest.raises(ValueError):
        decide_status("true_positive", confidence, THRESHOLD)  # type: ignore[arg-type]


@pytest.mark.parametrize("threshold", [-0.1, 1.5, float("nan"), "0.7", None, False])
def test_invalid_threshold_raises(threshold: object) -> None:
    with pytest.raises(ValueError):
        decide_status("true_positive", 0.9, threshold)  # type: ignore[arg-type]
