# cpax/__init__.py

from importlib.metadata import version as _distribution_version

__version__ = _distribution_version("CPax")

# Keep the legacy ODE interface available while introducing physics-first APIs.
from . import dynamics, mechanics, ode
from .problems import IVP
from .solvers import SaveAt, solve
