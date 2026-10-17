"""Declarative request for CPax's documented IVP solver default."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Auto:
    """Request automatic solver selection without specifying an algorithm.

    This is equivalent to ``solver="auto"``. The selection policy is
    deliberately conservative and does not analyse physical properties.
    """
