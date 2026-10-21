"""CPax public physics-first API."""

from importlib.metadata import version as _distribution_version

__version__ = _distribution_version("CPax")

from . import dynamics, mechanics, ode
from .problems import IVP
from .solvers import SaveAt, solve
