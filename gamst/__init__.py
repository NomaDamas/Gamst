"""Gamst: local, evidence-grounded human interaction intelligence."""
from .engine import Gamst, PerceptionPlugin
from .schema import SCHEMA_VERSION, validate_output
from .feedback import CalibrationArtifact, LabeledOutcome, fit_calibration
from .capabilities import CapabilitySpec, CapabilityRegistry, OutputField, SensorMetadata
from .streams import SensorStreams
from .process_provider import ProcessProvider

Observer = Gamst
__version__ = "0.3.1"
__all__ = ["Gamst", "Observer", "PerceptionPlugin", "SCHEMA_VERSION", "validate_output", "CalibrationArtifact", "LabeledOutcome", "fit_calibration"]
__all__ += ['CapabilitySpec', 'CapabilityRegistry', 'OutputField', 'SensorMetadata']
__all__ += ['SensorStreams']
__all__ += ['ProcessProvider']

from .providers.multimodal_decision import MultimodalDecisionProvider
__all__ += ['MultimodalDecisionProvider']

from .transcripts import transcript_capability
__all__ += ['transcript_capability']

from .contradictions import (contradiction_candidate, detect_contradictions,
                             detect_contradiction_candidates)
__all__ += ['contradiction_candidate', 'detect_contradictions',
            'detect_contradiction_candidates']

from .object_appearance import ObjectAppearancePipeline
__all__ += ['ObjectAppearancePipeline']

from .mask_tracking import MaskTrackingPipeline
__all__ += ['MaskTrackingPipeline']

from .audiovisual import TalkNetWindows, decode_talknet_window, talknet_stream_frame
__all__ += ['TalkNetWindows', 'decode_talknet_window', 'talknet_stream_frame']

from .openeb_process import OpenEBProcess
__all__ += ['OpenEBProcess']

from .smolvlm_call import SmolVLMCalls, smolvlm_frame_sha256
__all__ += ['SmolVLMCalls', 'smolvlm_frame_sha256']

from .frame_sentinel import FreshFrameSentinel
__all__ += ['FreshFrameSentinel']

from .text_context_call import TextContextCalls
__all__ += ['TextContextCalls']

from .visual_context_call import VisualContextCalls
__all__ += ['VisualContextCalls']

from .orbslam_process import ORBSLAMProcess
__all__ += ['ORBSLAMProcess']

from .fill_conflict import FillConflictRuntime
__all__ += ['FillConflictRuntime']

from .evaluation_snapshot import (SourceContextView, EvaluationSnapshotPublisher,
                                 evaluation_snapshot_implementation_sha256)
__all__ += ['SourceContextView', 'EvaluationSnapshotPublisher', 'evaluation_snapshot_implementation_sha256']

from .openai_responses import OpenAIResponsesClient
from .responses_evidence import ResponsesEvidenceSession
from .responses_evidence_reader import read_responses_evidence
__all__ += ['OpenAIResponsesClient', 'ResponsesEvidenceSession', 'read_responses_evidence']

from .temporal_context import TemporalContextEngine
from .context_memory import CheckpointStore, ContextMemory, ContextMemoryEngine, MemoryContext
__all__ += ['TemporalContextEngine', 'CheckpointStore', 'ContextMemory', 'ContextMemoryEngine', 'MemoryContext']
