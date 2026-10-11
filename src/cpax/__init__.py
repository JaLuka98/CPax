# cpax/__init__.py

__version__ = "2025.05.0"

# Keep the legacy ODE interface available while introducing physics-first APIs.
from . import dynamics, ode
from .problems import IVP
