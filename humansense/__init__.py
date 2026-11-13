"""HumanSense: local, evidence-grounded human interaction intelligence."""
from .engine import HumanSense, PerceptionPlugin
from .schema import SCHEMA_VERSION, validate_output
from .feedback import CalibrationArtifact, LabeledOutcome, fit_calibration
from .capabilities import CapabilitySpec, CapabilityRegistry, OutputField, SensorMetadata
from .streams import SensorStreams
from .process_provider import ProcessProvider

Observer = HumanSense
__version__ = "0.3.1"
__all__ = ["HumanSense", "Observer", "PerceptionPlugin", "SCHEMA_VERSION", "validate_output", "CalibrationArtifact", "LabeledOutcome", "fit_calibration"]
__all__ += ['CapabilitySpec', 'CapabilityRegistry', 'OutputField', 'SensorMetadata']
__all__ += ['SensorStreams']
__all__ += ['ProcessProvider']

from .transcripts import transcript_capability
__all__ += ['transcript_capability']

from .object_appearance import ObjectAppearancePipeline
__all__ += ['ObjectAppearancePipeline']
