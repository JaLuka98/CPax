"""Public numerical solver interface for supported CPax problems."""

from ._auto import Auto
from ._saveat import SaveAt
from ._solve import solve

__all__ = ["Auto", "SaveAt", "solve"]
