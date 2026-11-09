"""Map a VLM verdict to an event status (design: 事件状态机 / VLM 复核层, A12).

The function is pure: it takes the threshold as an argument and never reads
configuration or ``detection_source``. Callers pass ``vlm.auto_decision_threshold``.
"""

from __future__ import annotations

import math
import numbers

_VALID_VERDICTS = frozenset({"true_positive", "false_positive", "uncertain"})


def _check_unit_interval(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a real number in [0, 1], got {value!r}")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be a real number in [0, 1], got {value!r}")
    return number


def decide_status(verdict: str, confidence: float, threshold: float) -> str:
    """Return the event status implied by a VLM verdict.

    Rules:
      - ``false_positive`` with ``confidence >= threshold`` -> ``"dismissed"``
      - ``true_positive`` with ``confidence >= threshold`` -> ``"confirmed"``
      - anything else (``uncertain``, or confidence below threshold) -> ``"needs_human"``

    Raises:
        ValueError: if ``verdict`` is not one of the three allowed values, or if
            ``confidence`` or ``threshold`` is not a real number in [0, 1]
            (``bool`` is rejected).
    """
    if not isinstance(verdict, str) or verdict not in _VALID_VERDICTS:
        raise ValueError(
            f"verdict must be one of {sorted(_VALID_VERDICTS)}, got {verdict!r}"
        )
    conf = _check_unit_interval(confidence, "confidence")
    limit = _check_unit_interval(threshold, "threshold")

    if conf < limit:
        return "needs_human"
    if verdict == "false_positive":
        return "dismissed"
    if verdict == "true_positive":
        return "confirmed"
    return "needs_human"
