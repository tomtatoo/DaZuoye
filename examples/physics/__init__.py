"""Simulation backend implementations for ManiDreams."""

from .maniskill_base import ManiSkillBaseBackend
from .push_backend_sim import PushBackend
from .pick_backend import PickBackend
from .catch_backend import CatchBackend
from .push_backend_learned.backend import DiffusionBackend
from .maniskill_default_tasks import DRISBackend, DRISMixin, make_dris_env

__all__ = [
    "ManiSkillBaseBackend",
    "PushBackend",
    "PickBackend",
    "CatchBackend",
    "DiffusionBackend",
    "DRISBackend",
    "DRISMixin",
    "make_dris_env",
    "NewtonBackend",
]


def __getattr__(name):
    """Load Newton only when its optional dependencies are requested."""
    if name == "NewtonBackend":
        from .newton_backend import NewtonBackend

        return NewtonBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
