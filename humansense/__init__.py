"""HumanSense: local, evidence-grounded human interaction intelligence."""
from .engine import HumanSense, PerceptionPlugin
from .schema import SCHEMA_VERSION, validate_output
from .feedback import CalibrationArtifact, LabeledOutcome, fit_calibration

Observer = HumanSense
__version__ = "0.2.0"
__all__ = ["HumanSense", "Observer", "PerceptionPlugin", "SCHEMA_VERSION", "validate_output", "CalibrationArtifact", "LabeledOutcome", "fit_calibration"]
