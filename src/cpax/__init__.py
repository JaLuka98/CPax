# cpax/__init__.py

from importlib.metadata import version
__version__ = version("cpax")

# Import submodules but don’t expose specific functions
from . import ode
