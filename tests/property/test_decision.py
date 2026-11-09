"""Property tests for the VLM review decision (decide_status).

**Validates: Requirements 3.8, 6.2, 6.3, 6.4**

Property 14: for any verdict, confidence in [0, 1] and threshold theta:
false_positive with confidence >= theta -> "dismissed"; true_positive with
confidence >= theta -> "confirmed"; everything else (uncertain, or confidence
below theta) -> "needs_human". The result does not depend on detection_source.
"""

from __future__ import annotations

import inspect
import math

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from chimera_patrol.review.decision import decide_status

PBT_SETTINGS = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

VERDICTS = ["true_positive", "false_positive", "uncertain"]
DETECTION_SOURCES = ["yolo", "vlm_fallback"]


def _reference_status(verdict: str, confidence: float, threshold: float) -> str:
    """Independent restatement of Property 14, written from the spec text."""
    if verdict == "false_positive" and confidence >= threshold:
        return "dismissed"
    if verdict == "true_positive" and confidence >= threshold:
        return "confirmed"
    return "needs_human"


unit_floats = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


@st.composite
def thresholds_and_confidences(draw: st.DrawFn) -> tuple[float, float]:
    """Draw a threshold theta and a confidence, biased toward theta itself.

    Confidences are sampled from theta and its immediate float neighbours so the
    >= boundary is exercised exactly, plus free draws from [0, 1].
    """
    theta = draw(unit_floats)
    confidence = draw(
        st.one_of(
            st.just(theta),
            st.just(math.nextafter(theta, -math.inf)),
            st.just(math.nextafter(theta, math.inf)),
            unit_floats,
        )
    )
    # Keep the confidence inside [0, 1] after the neighbour step.
    confidence = min(max(confidence, 0.0), 1.0)
    return theta, confidence


@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    verdict=st.sampled_from(VERDICTS),
    pair=thresholds_and_confidences(),
)
def test_decide_status_matches_reference(verdict: str, pair: tuple[float, float]) -> None:
    """Property 14: decide_status agrees with the independent reference rule."""
    threshold, confidence = pair
    assert decide_status(verdict, confidence, threshold) == _reference_status(
        verdict, confidence, threshold
    )


@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    verdict=st.sampled_from(VERDICTS),
    pair=thresholds_and_confidences(),
)
def test_decide_status_at_threshold_boundary(verdict: str, pair: tuple[float, float]) -> None:
    """Property 14: confidence == theta counts as meeting the threshold."""
    threshold, _ = pair
    status = decide_status(verdict, threshold, threshold)
    if verdict == "false_positive":
        assert status == "dismissed"
    elif verdict == "true_positive":
        assert status == "confirmed"
    else:
        assert status == "needs_human"


@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    verdict=st.sampled_from(VERDICTS),
    pair=thresholds_and_confidences(),
)
def test_decide_status_independent_of_detection_source(
    verdict: str, pair: tuple[float, float]
) -> None:
    """Property 14: outcomes for yolo and vlm_fallback events are identical.

    decide_status takes no source argument, so the same inputs must give the
    same status regardless of which detection_source the event carries.
    """
    threshold, confidence = pair
    outcomes = {
        source: decide_status(verdict, confidence, threshold) for source in DETECTION_SOURCES
    }
    assert len(set(outcomes.values())) == 1
    assert outcomes["yolo"] == _reference_status(verdict, confidence, threshold)


def test_decide_status_signature_has_no_detection_source() -> None:
    """Property 14: the decision function does not accept detection_source."""
    params = list(inspect.signature(decide_status).parameters)
    assert params == ["verdict", "confidence", "threshold"]
    assert "detection_source" not in params
