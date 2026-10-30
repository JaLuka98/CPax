Classical mechanics examples
===========================

These executable examples use physical potentials, generalised forces and
canonical momenta with the public CPax workflow. They do not require a
dedicated Python class for each mechanical system. For the minimal harmonic
oscillator and conservative double-well Duffing oscillator, see
:doc:`simple`; for gravitational N-body dynamics and an orbital Velocity
Verlet demonstration, see :doc:`advanced`.

Run any example from the repository root, for instance::

   uv run python examples/simple_pendulum.py

Each example calls `main()` only when executed directly. Its `simulate()`
function can be imported for programmatic reuse and numerical testing.

Conservative nonlinear pendulum
-------------------------------

A point mass `m` on a massless rigid rod of length `L` uses the angle
`theta` in radians and the conjugate angular momentum
`p_theta = I * angular_velocity` with `I = m * L**2`. Its gravitational
potential is `V = m * g * L * (1 - cos(theta))`. For small angles the
angular frequency is `sqrt(g / L)`; at finite amplitude, the true period
is longer. The example compares the small-angle solution against the full
nonlinear trajectory and monitors energy deviations.

`NewtonianSystem` with fixed-step `VelocityVerlet` is appropriate because
the Hamiltonian is autonomous and separable. The nominal saved grid is
aligned with the integration step; off-step interpolation does not itself
preserve symplectic structure. Parameters are given in SI units, and JAX
default floating-point precision is used for the illustrative plot.

.. literalinclude:: ../../../examples/simple_pendulum.py
   :language: python

Coupled harmonic oscillators
----------------------------

Two identical masses connected to wall springs `k` and a coupling spring
`k_c` have the potential

.. math::

   V(q_1,q_2) = \frac{k}{2}(q_1^2+q_2^2)
                + \frac{k_c}{2}(q_1-q_2)^2.

The in-phase and out-of-phase modes have frequencies
`omega_in = sqrt(k / m)` and
`omega_out = sqrt((k + 2 * k_c) / m)`, respectively. An initial displacement
of one mass excites both modes. This shows how a single vector-valued
potential automatically produces coupled equations, without manually
coding the forces. The example uses SI-consistent values and fixed-step
Velocity Verlet.

.. literalinclude:: ../../../examples/coupled_oscillators.py
   :language: python

Driven and damped Duffing oscillator
------------------------------------

The dimensionless double-well model uses
`V(q) = -alpha*q**2/2 + beta*q**4/4` and applied force
`F(t, q, v) = -damping*v + drive*cos(frequency*t)`.
The nonlinear motion is integrated using `ForcedNewtonianSystem` and
adaptive Tsit5. The phase-space trajectory illustrates forced, dissipative
behaviour; it is not, by itself, evidence of chaos.

With external forces, the instantaneous mechanical energy is not conserved.
It obeys `dE/dt = v * F_applied`, so viscous drag removes energy and the
driving term can add or remove it. The example's displayed energy history
should be interpreted accordingly.

.. literalinclude:: ../../../examples/forced_duffing_oscillator.py
   :language: python

Driven and damped pendulum
-------------------------

For the physical pendulum, `ForcedNewtonianSystem` receives the same
gravitational potential as the conservative example, plus viscous damping
and a periodic *torque*. Its applied-force callback receives the angular
velocity, not angular momentum. The system uses moment of inertia as its
generalised mass. The plot shows angle, mechanical energy and applied power
with SI units. The default driving frequency of 3 rad/s is near the
small-amplitude natural frequency, but a nonlinear periodically driven
pendulum need not obey the linear resonant-response approximation.

.. literalinclude:: ../../../examples/forced_damped_pendulum.py
   :language: python

Numerical validation
--------------------

The example tests compare against the small-angle pendulum approximation,
the exact coupled-oscillator normal modes, conservative energy behaviour,
and non-conservative work-energy balance. Separate numerical reference
checks use tighter solver tolerances in a scoped float64 context. The
scripts themselves use JAX's default precision (typically float32), with
float32-appropriate solver tolerances in the adaptive examples.

Neither sampling density nor solver success alone establishes accuracy:
for precision-sensitive research, check step-size or tolerance refinement
and meaningful physical diagnostics. Use `VelocityVerlet` only for its
supported autonomous separable conservative systems; a forced or damped
trajectory is not symplectic merely because it is expressed in `(q, p)`
coordinates. See :doc:`../concepts/solvers` for numerical limitations.
