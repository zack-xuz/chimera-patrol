"""Unit tests for chimera_patrol.sop.sampling.sample_times (task 15.3)."""

from __future__ import annotations

import pytest

from chimera_patrol.sop.sampling import sample_times

T = 100_000


def test_default_design_example_inclusive() -> None:
    times = sample_times(T, 5, 5, 8, True)

    assert len(times) == 8
    assert times[0] == T - 5000
    assert times[-1] == T + 5000
    # Step is 10000/7 ms = 1428.571...; each point is rounded half up.
    assert times == [95000, 96429, 97857, 99286, 100714, 102143, 103571, 105000]
    assert times == sorted(times)


def test_inclusive_two_points_are_window_endpoints() -> None:
    assert sample_times(T, 5, 5, 2, True) == [95000, 105000]


def test_exclusive_uses_midpoints_of_equal_subintervals() -> None:
    # Window 10000 ms split into 4 sub-intervals of 2500 ms.
    assert sample_times(T, 5, 5, 4, False) == [96250, 98750, 101250, 103750]


def test_exclusive_rounds_midpoints_half_up() -> None:
    # Window 10000 ms split into 3 parts: midpoints 96666.67, 100000, 103333.33.
    assert sample_times(T, 5, 5, 3, False) == [96667, 100000, 103333]


def test_inclusive_rounds_exact_half_up() -> None:
    # Window is 0.5 ms; the second endpoint 0.5 ms rounds up to 1.
    assert sample_times(0, 0, 0.0005, 2, True) == [0, 1]


def test_n_one_returns_trigger_time_in_both_modes() -> None:
    assert sample_times(T, 5, 5, 1, True) == [T]
    assert sample_times(T, 5, 5, 1, False) == [T]
    assert sample_times(T, 0, 0, 1, True) == [T]


def test_zero_before_and_after_allowed_for_single_sample() -> None:
    assert sample_times(T, 0, 0, 1, False) == [T]


def test_asymmetric_window_inclusive() -> None:
    # Window [T-2s, T+8s] (10 s) sampled at its two endpoints and its centre.
    assert sample_times(T, 2, 8, 3, True) == [98000, 103000, 108000]


@pytest.mark.parametrize(
    ("before", "after", "n", "inclusive"),
    [
        (5, 5, 0, True),  # n below 1
        (5, 5, -3, False),  # negative n
        (5, 5, True, True),  # bool is not an int count
        (5, 5, 2.0, True),  # float count
        (-1, 5, 4, True),  # negative before
        (5, -1, 4, False),  # negative after
        (0, 0, 2, True),  # zero-length window with n > 1
        (0, 0, 4, False),
        (float("nan"), 5, 4, True),  # non-finite window
        (5, float("inf"), 4, True),
        (True, 5, 4, True),  # bool is not a number
    ],
)
def test_invalid_inputs_raise_value_error(
    before: float, after: float, n: int, inclusive: bool
) -> None:
    with pytest.raises(ValueError):
        sample_times(T, before, after, n, inclusive)


def test_non_int_trigger_time_raises_value_error() -> None:
    with pytest.raises(ValueError):
        sample_times(1.5, 5, 5, 8, True)  # type: ignore[arg-type]
