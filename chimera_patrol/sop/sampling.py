"""SOP sampling instants (assumption A2).

Given a trigger time ``T`` and a window ``[T - before_s, T + after_s]``,
``sample_times`` returns ``n`` UTC millisecond timestamps:

* ``inclusive=True``: both window endpoints are sampled.
  ``t_i = T - before + i * (before + after) / (n - 1)`` for ``i = 0..n-1``.
* ``inclusive=False``: the midpoints of ``n`` equal sub-intervals.
  ``t_i = T - before + (i + 0.5) * (before + after) / n``.

For ``n == 1`` the function returns ``[T]`` in both modes.

Arithmetic is exact (``fractions.Fraction``); each instant is rounded to the
nearest millisecond with halves rounded up.
"""

from __future__ import annotations

import math
from fractions import Fraction

__all__ = ["sample_times"]

_MS_PER_S = 1000
_HALF = Fraction(1, 2)


def _as_seconds(name: str, value: object) -> Fraction:
    """Convert a non-negative finite int/float number of seconds to an exact Fraction."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {type(value).__name__}")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")
        # str() gives the shortest decimal repr, so 0.1 becomes exactly 1/10.
        return Fraction(str(value))
    return Fraction(value)


def _round_half_up(value: Fraction) -> int:
    """Round to the nearest integer, with exact halves going up."""
    return math.floor(value + _HALF)


def sample_times(
    T_ms: int,
    before_s: float,
    after_s: float,
    n: int,
    inclusive: bool,
) -> list[int]:
    """Return ``n`` sampling instants (UTC ms) inside ``[T - before, T + after]``.

    Args:
        T_ms: Trigger time in UTC milliseconds.
        before_s: Window length before ``T`` in seconds, ``>= 0``.
        after_s: Window length after ``T`` in seconds, ``>= 0``.
        n: Number of sampling instants, ``>= 1``.
        inclusive: ``True`` to include both window endpoints, ``False`` to use
            the midpoints of ``n`` equal sub-intervals.

    Returns:
        ``n`` integer millisecond timestamps in ascending order.

    Raises:
        ValueError: If any argument is invalid. This covers a non-int ``T_ms``
            or ``n``, a boolean where a number is expected, negative or
            non-finite windows, a zero-length window when ``n != 1``, and
            ``n < 1``.
    """
    if isinstance(T_ms, bool) or not isinstance(T_ms, int):
        raise ValueError(f"T_ms must be an int, got {type(T_ms).__name__}")
    if isinstance(n, bool) or not isinstance(n, int):
        raise ValueError(f"n must be an int, got {type(n).__name__}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if not isinstance(inclusive, bool):
        raise ValueError(f"inclusive must be a bool, got {type(inclusive).__name__}")

    before = _as_seconds("before_s", before_s)
    after = _as_seconds("after_s", after_s)
    if before < 0:
        raise ValueError(f"before_s must be >= 0, got {before_s!r}")
    if after < 0:
        raise ValueError(f"after_s must be >= 0, got {after_s!r}")
    if n == 1:
        return [T_ms]
    if before + after <= 0:
        raise ValueError("before_s + after_s must be > 0 when n > 1")

    start = Fraction(T_ms) - before * _MS_PER_S
    window = (before + after) * _MS_PER_S

    if inclusive:
        step = window / (n - 1)
        times = [_round_half_up(start + i * step) for i in range(n)]
    else:
        step = window / n
        times = [_round_half_up(start + (i + _HALF) * step) for i in range(n)]

    return sorted(times)
