"""Shared continuous-time dynamical-system interfaces.

These systems specify equations of motion, not numerical solution methods.
They deliberately do not depend on physical domains or integration backends.
"""

from abc import abstractmethod
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax

type Time = float | jax.Array
type State = Any  # Array-valued PyTree; concrete state types are model-specific.
type Args = Any


class DynamicalSystem(eqx.Module):
    """Base interface for continuous-time evolution.

    Subclasses implement ``vector_field(t, state, args=None)`` and return
    derivatives with the same PyTree structure and leaf shapes as ``state``.
    Time is the first argument, including for autonomous systems.
    """

    @abstractmethod
    def vector_field(
        self,
        t: Time,
        state: State,
        args: Args = None,
    ) -> State:
        """Evaluate the instantaneous derivative of ``state`` at ``t``."""
        raise NotImplementedError


class VectorFieldSystem(DynamicalSystem):
    """Wrap a user-supplied vector field as a continuous-time system.

    ``rhs`` has signature ``rhs(t, state, args)``. It may be a plain
    Python function or a callable Equinox module with differentiable array
    parameters. It is intentionally not marked as a static field.
    """

    rhs: Callable[[Time, State, Args], State]

    def vector_field(
        self,
        t: Time,
        state: State,
        args: Args = None,
    ) -> State:
        """Evaluate the wrapped vector field without modifying the state."""
        return self.rhs(t, state, args)
