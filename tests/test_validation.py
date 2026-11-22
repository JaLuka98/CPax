"""Contracts for the shared argument-validation helpers."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cpax._validation import (
    as_array,
    concrete_nonnegative_real,
    enum_code,
    positive_integer,
    positive_real,
    require_callable,
    time_grid,
)
from cpax.results import SolutionStatus


@pytest.mark.parametrize("value", [1e-3, 2, np.float32(0.5), jnp.asarray(3.0)])
def test_positive_real_accepts_positive_scalars(value):
    positive_real("dt0", value)


@pytest.mark.parametrize(
    ("value", "exception"),
    [
        (0.0, ValueError),
        (-1.0, ValueError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        (True, ValueError),
        (jnp.asarray([1.0]), ValueError),
        ([1.0, [2.0]], TypeError),
    ],
)
def test_positive_real_rejects_invalid_scalars(value, exception):
    with pytest.raises(exception, match="dt0 must be a positive finite real scalar"):
        positive_real("dt0", value)


def test_positive_real_defers_traced_values():
    def check(value):
        positive_real("dt0", value)
        return value

    assert float(jax.jit(check)(-1.0)) == -1.0


def test_concrete_nonnegative_real_accepts_zero_and_returns_float():
    assert concrete_nonnegative_real("transient_time", 0) == 0.0
    assert isinstance(concrete_nonnegative_real("transient_time", jnp.asarray(2)), float)


@pytest.mark.parametrize(
    ("value", "exception"),
    [
        (-1e-12, ValueError),
        (float("nan"), ValueError),
        ("0.1", ValueError),
        ([1.0, [2.0]], TypeError),
    ],
)
def test_concrete_nonnegative_real_rejects_invalid_scalars(value, exception):
    with pytest.raises(exception, match="transient_time"):
        concrete_nonnegative_real("transient_time", value)


def test_concrete_nonnegative_real_rejects_traced_values():
    with pytest.raises(TypeError, match="concrete"):
        jax.jit(lambda value: concrete_nonnegative_real("transient_time", value))(1.0)


def test_positive_integer_accepts_numpy_integers():
    assert positive_integer("num_segments", np.int64(3)) == 3


@pytest.mark.parametrize("value", [0, -1, True, 2.0, jnp.asarray(2)])
def test_positive_integer_rejects_non_positive_or_non_integral_values(value):
    with pytest.raises(ValueError, match="num_segments must be a positive integer"):
        positive_integer("num_segments", value)


def test_enum_code_returns_a_scalar_array():
    code = enum_code("status", SolutionStatus.FAILED, SolutionStatus)
    assert code.shape == ()
    assert int(code) == SolutionStatus.FAILED


@pytest.mark.parametrize(
    ("value", "exception", "match"),
    [
        (True, TypeError, "scalar integer"),
        (0.0, TypeError, "scalar integer"),
        (jnp.asarray([0]), TypeError, "scalar integer"),
        (5, ValueError, "SolutionStatus outcome code"),
    ],
)
def test_enum_code_rejects_invalid_codes(value, exception, match):
    with pytest.raises(exception, match=match):
        enum_code("status", value, SolutionStatus)


def test_enum_code_defers_membership_for_traced_values():
    assert int(jax.jit(lambda value: enum_code("status", value, SolutionStatus))(5)) == 5


def test_time_grid_promotes_integer_times():
    times = time_grid("ts", jnp.asarray([0, 1, 2]))
    assert times.dtype.kind == "f"
    np.testing.assert_array_equal(times, [0.0, 1.0, 2.0])


@pytest.mark.parametrize("value", [jnp.zeros(0), jnp.zeros((2, 2)), jnp.asarray([1j]), jnp.asarray([True])])
def test_time_grid_rejects_invalid_grids(value):
    with pytest.raises(ValueError, match="ts must be a nonempty one-dimensional array"):
        time_grid("ts", value)


def test_as_array_reports_conversion_failures_as_type_errors():
    with pytest.raises(TypeError, match="y0 leaf"):
        as_array([1.0, [2.0]], "y0 leaf")


@pytest.mark.parametrize(
    ("value", "float_scalars", "kind"),
    [
        (2, True, "f"),
        (np.int64(2), True, "f"),
        (True, True, "b"),
        (jnp.asarray([1, 2]), True, "i"),
        (2, False, "i"),
    ],
)
def test_as_array_promotes_only_python_numeric_scalars(value, float_scalars, kind):
    assert as_array(value, "unused", float_scalars=float_scalars).dtype.kind == kind


def test_require_callable():
    require_callable("potential", abs)
    require_callable("force_fn", None, optional=True)
    with pytest.raises(TypeError, match=r"^potential must be callable\.$"):
        require_callable("potential", None)
    with pytest.raises(TypeError, match=r"^force_fn must be callable or None\.$"):
        require_callable("force_fn", 1.0, optional=True)
