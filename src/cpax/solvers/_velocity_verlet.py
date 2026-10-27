"""Velocity Verlet integration for autonomous separable mechanics."""

from typing import Any, ClassVar

import diffrax

from cpax.mechanics import PhaseSpaceState


class VelocityVerlet(diffrax.AbstractSolver):
    """Second-order kick-drift-kick integrator for separable Hamiltonians.

    The state is a PhaseSpaceState, and terms is a pair of ODETerm instances
    for qdot(p) and pdot(q). Both vector fields must be autonomous.

    Use constant time steps; no local error estimate is provided.
    Interpolation between steps is not symplectic.

    The force at the end of a step is cached for the next step.
    """

    term_structure: ClassVar = (diffrax.ODETerm, diffrax.ODETerm)
    interpolation_cls: ClassVar = diffrax.LocalLinearInterpolation

    def order(self, terms: Any) -> int:
        del terms
        return 2

    def init(self, terms: Any, t0: Any, t1: Any, y0: PhaseSpaceState, args: Any) -> Any:
        del t1
        _, p_term = terms
        return p_term.vf(t0, y0.q, args)

    def step(
        self,
        terms: Any,
        t0: Any,
        t1: Any,
        y0: PhaseSpaceState,
        args: Any,
        solver_state: Any,
        made_jump: Any,
    ) -> tuple[PhaseSpaceState, None, dict[str, PhaseSpaceState], Any, Any]:
        del made_jump
        q_term, p_term = terms
        half_dt = (t1 - t0) / 2

        # Kick, drift, kick. Cache the final force for the next step.
        p_half = y0.p + half_dt * solver_state
        q1 = y0.q + (t1 - t0) * q_term.vf(t0, p_half, args)
        force1 = p_term.vf(t1, q1, args)
        p1 = p_half + half_dt * force1

        y1 = PhaseSpaceState(q=q1, p=p1)
        return y1, None, {"y0": y0, "y1": y1}, force1, diffrax.RESULTS.successful

    def func(self, terms: Any, t0: Any, y0: PhaseSpaceState, args: Any) -> PhaseSpaceState:
        q_term, p_term = terms
        return PhaseSpaceState(q=q_term.vf(t0, y0.p, args), p=p_term.vf(t0, y0.q, args))
