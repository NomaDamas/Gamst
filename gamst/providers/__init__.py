"""Optional adapters. Importing this package never downloads or loads a model."""

from .multimodal_decision import MultimodalDecisionProvider
from .rotation import RotationPlugin, RotationProvider

__all__ = ["MultimodalDecisionProvider", "RotationPlugin", "RotationProvider"]
