"""JAX-based tools for defining and solving continuous-time dynamical systems.

The public API separates physical systems, initial-value problems, numerical
solvers, and structured solutions, with built-in support for conservative and driven
classical mechanics.
"""

from importlib.metadata import version as _distribution_version

__version__ = _distribution_version("CPax")

from . import dynamics, mechanics, ode
from .problems import IVP
from .solvers import SaveAt, solve
