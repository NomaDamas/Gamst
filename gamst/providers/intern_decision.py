"""Optional local Intern-Decision adapter for the multimodal decision contract.

Importing this module is dependency-light and never loads or downloads weights.
Real inference requires an explicit, pinned local checkpoint through
``InternDecisionProvider.from_local`` or ``decision_provider``.
"""
from __future__ import annotations

import hashlib
import importlib.util
import math
import queue
import stat
import sys
import threading
from pathlib import Path

from ..capabilities import CapabilitySpec, SensorMetadata, bounded_json, identifier
from .multimodal_decision import (
    CAPABILITY_ID,
    OUTPUT_NAME,
    MultimodalDecisionProvider,
    select_decision_evidence,
)


MODEL_PROFILES = {
    "intern-decision-2b": {
        "model_id": "internlm/Intern-Decision-2B",
        "revision": "85c5657708a09884ff9fd8d7f57789b410a97b33",
        "inference_sha256": "2918862f34770c6db57cf3f86666cbd7cff1139b24f300fef01d9146584d4fd8",
        "config_sha256": "ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4",
    },
    "intern-decision-4b": {
        "model_id": "internlm/Intern-Decision-4B",
        "revision": "38c9dafbbef52dd04bae943c0c6ac918419adfd5",
        "inference_sha256": "c904e2c67ca0775621a22375ee373d2ba30b52117cda870c6c9ef74143b29863",
        "config_sha256": "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670",
    },
}

FIXED_DECISIONS = (
    "fuse_multimodal_context",
    "escalate_vlm",
    "request_fresh_sensor",
    "request_more_audio",
    "request_more_video",
    "insufficient_evidence",
)

_QUESTION = {
    "type": "choice",
    "instructions": (
        "Choose the safest routing action for the bounded, fresh multimodal observations. "
        "Use insufficient_evidence when the observations do not support another action."
    ),
    "criteria": {
        "fuse_multimodal_context": "Fuse the selected observations into multimodal context.",
        "escalate_vlm": "Escalate to a VLM for additional semantic inspection.",
        "request_fresh_sensor": "Request a newer observation from an existing sensor.",
        "request_more_audio": "Request audio evidence to complement the current observations.",
        "request_more_video": "Request video evidence to complement the current observations.",
        "insufficient_evidence": "Abstain because the selected evidence is insufficient.",
    },
}

_RUNTIME = {"torch": "2.9.1", "torchvision": "0.24.1", "transformers": "5.14.1"}


def _check_runtime():
    if sys.version_info < (3, 12):
        raise RuntimeError("Intern-Decision requires Python 3.12+ in a dedicated existing environment")
    from importlib.metadata import PackageNotFoundError, version
    try:
        for name, expected in _RUNTIME.items():
            if version(name).split("+")[0] != expected:
                raise RuntimeError(f"Intern-Decision requires {name}=={expected}; no automatic installation")
        pillow = version("Pillow").split("+")[0]
        if not 10 <= int(pillow.split(".", 1)[0]) < 13:
            raise RuntimeError("Intern-Decision requires Pillow>=10,<13; no automatic installation")
    except (PackageNotFoundError, ValueError) as exc:
        raise RuntimeError(
            "Intern-Decision runtime unavailable; install the 'decision' extra in a dedicated Python 3.12+ environment"
        ) from exc


def _verified_file(path: Path, expected_sha256: str) -> Path:
    try:
        info = path.lstat()
    except OSError as exc:
        raise ValueError(f"pinned local checkpoint file missing: {path.name}") from exc
    if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= 1024 * 1024:
        raise ValueError(f"pinned local checkpoint entry must be a bounded regular file: {path.name}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    after = path.lstat()
    identity_before = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    if digest != expected_sha256 or identity_after != identity_before:
        raise ValueError(f"local {path.name} does not match the pinned Intern-Decision revision")
    return path


def _load_engine(model_path: str, kind: str, device: str, dtype: str):
    """Load only an explicitly supplied checkpoint matching the pinned release."""
    _check_runtime()
    if type(model_path) is not str or not model_path or "\x00" in model_path:
        raise ValueError("model_path must be an explicit existing local directory")
    path = Path(model_path)
    if not path.is_absolute():
        raise ValueError("model_path must be an absolute local directory")
    try:
        path = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError("model_path must be an explicit existing local directory") from exc
    if not path.is_dir():
        raise ValueError("model_path must be an explicit existing local directory")
    profile = MODEL_PROFILES[kind]
    inference_path = _verified_file(path / "inference.py", profile["inference_sha256"])
    _verified_file(path / "config.json", profile["config_sha256"])

    # Execute only the hash-verified inference wrapper shipped by the selected
    # checkpoint. Its HF loader uses local_files_only=True for every model asset.
    module_name = "_gamst_intern_decision_" + profile["revision"]
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, inference_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("Intern-Decision inference runtime could not be loaded")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            sys.modules.pop(module_name, None)
            raise RuntimeError(
                "Intern-Decision inference runtime unavailable or incompatible; no fallback was selected"
            ) from exc
    if device == "cuda" and not module.torch.cuda.is_available():
        raise RuntimeError("Intern-Decision CUDA runtime requested but unavailable")
    try:
        engine = module.DecisionEngine(
            checkpoint=str(path), device=device, dtype=dtype,
            attn_implementation="sdpa", temperature=1.0,
        )
    except BaseException as exc:
        raise RuntimeError(
            f"could not load pinned {profile['model_id']} from the explicit local path; no fallback was selected"
        ) from exc
    return engine.predict


class InternDecisionProvider:
    """Fail-closed learned decisions over host-selected bounded observations.

    ``model`` is an injected callable implementing the official
    ``DecisionEngine.predict(request)`` interface. It is useful for tests and
    for an explicitly provisioned runtime; no runtime import occurs here.
    """

    def __init__(self, model, *, kind: str = "intern-decision-2b",
                 deadline_ms: int = 250, min_confidence: float = 0.6,
                 max_evidence: int = 16, device: str = "cpu"):
        if not callable(model):
            raise ValueError("model must implement the DecisionEngine.predict callable interface")
        if kind not in MODEL_PROFILES:
            raise ValueError("kind must be intern-decision-2b or intern-decision-4b")
        if type(deadline_ms) is not int or not 1 <= deadline_ms <= 60000:
            raise ValueError("deadline_ms must be an integer from 1 to 60000")
        if isinstance(min_confidence, bool) or not isinstance(min_confidence, (int, float)):
            raise ValueError("min_confidence must be a finite number from 0 to 1")
        if not math.isfinite(min_confidence) or not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be a finite number from 0 to 1")
        if type(max_evidence) is not int or not 1 <= max_evidence <= 32:
            raise ValueError("max_evidence must be an integer from 1 to 32")
        if device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        self.model = model
        self.kind = kind
        self.deadline_ms = deadline_ms
        self.min_confidence = float(min_confidence)
        self.max_evidence = max_evidence
        self.device = device
        self.spec = self.capability(kind=kind, device=device)
        self._sequence = 0
        self._jobs = queue.Queue()
        self._state_lock = threading.Lock()
        self._running = False
        self._closed = False
        self._worker = threading.Thread(
            target=self._run, name=f"gamst-{kind}", daemon=True)
        self._worker.start()

    @staticmethod
    def capability(*, kind: str = "intern-decision-2b", device: str = "cpu") -> CapabilitySpec:
        if kind not in MODEL_PROFILES:
            raise ValueError("kind must be intern-decision-2b or intern-decision-4b")
        if device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        profile = MODEL_PROFILES[kind]
        return CapabilitySpec(
            CAPABILITY_ID,
            OUTPUT_NAME,
            "events",
            "gamst.multimodal_decision." + kind.replace("-", "_"),
            profile["revision"],
            MultimodalDecisionProvider.capability().fields,
            "frame",
            freshness_ms=5000,
            stateful=False,
            interpretation="estimate",
            devices=(device,),
            sensor_requirements=("retained-registered-observations",),
            license="Apache-2.0;Qwen-LICENSE-QWEN-notices",
        )

    @classmethod
    def from_local(cls, model_path: str, *, kind: str = "intern-decision-2b",
                   device: str = "cuda", dtype: str = "bfloat16",
                   deadline_ms: int = 250, min_confidence: float = 0.6,
                   max_evidence: int = 16):
        """Explicitly load a pinned local checkpoint; never download or fallback."""
        if kind not in MODEL_PROFILES:
            raise ValueError("kind must be intern-decision-2b or intern-decision-4b")
        if device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if dtype not in ("bfloat16", "float16", "float32"):
            raise ValueError("dtype must be bfloat16, float16 or float32")
        model = _load_engine(model_path, kind, device, dtype)
        return cls(model, kind=kind, device=device, deadline_ms=deadline_ms,
                   min_confidence=min_confidence, max_evidence=max_evidence)

    @property
    def descriptor(self) -> dict:
        profile = MODEL_PROFILES[self.kind]
        return {
            "capability": self.spec.describe(),
            "provider_kind": "internal_perception_module",
            "model_status": "learned_local",
            "model_id": profile["model_id"],
            "revision": profile["revision"],
            "raw_media": False,
            "calibrated": False,
            "confidence": "uncalibrated_candidate_confidence",
            "deadline_ms": self.deadline_ms,
            "min_confidence": self.min_confidence,
            "max_evidence": self.max_evidence,
        }

    def _run(self):
        while True:
            job = self._jobs.get()
            if job is None:
                self.model = None
                return
            request, response = job
            try:
                response.put((True, self.model(request)))
            except BaseException:
                # Model exceptions and their text never enter context.
                response.put((False, None))
            finally:
                with self._state_lock:
                    self._running = False

    def close(self):
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        self._jobs.put(None)

    @staticmethod
    def _request(selection: dict) -> dict:
        observations = []
        for item, age in zip(selection["selected"], selection["evidence_ages_ms"]):
            record = item[4]
            observations.append({
                "capability": item[2],
                "modality": item[3],
                "age_ms": age,
                "value": record["observation"].get("value", {}),
            })
        request = {
            "state": {
                "observations": observations,
                "input_window_ms": selection["input_window_ms"],
                "requested_capabilities": selection["requested_capabilities"],
            },
            "questions": {"decision": _QUESTION},
        }
        # This also deep-copies values, rejects bytes/non-JSON/raw fields, and
        # caps total serialized input. No images, record IDs or media are added.
        return bounded_json(request, forbid_raw=True)

    @staticmethod
    def _answer(response) -> tuple[str, float]:
        if type(response) is not dict or type(response.get("answers")) is not dict:
            raise ValueError("invalid Intern-Decision response")
        answers = response["answers"]
        if set(answers) != {"decision"} or type(answers["decision"]) is not dict:
            raise ValueError("invalid Intern-Decision answers")
        answer = answers["decision"]
        if answer.get("type") != "choice":
            raise ValueError("Intern-Decision answer must be a choice")
        choice = answer.get("choice")
        confidence = answer.get("confidence")
        probabilities = answer.get("probabilities")
        if choice not in FIXED_DECISIONS:
            raise ValueError("Intern-Decision selected an unknown action")
        if (isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or
                not math.isfinite(confidence) or not 0 <= confidence <= 1):
            raise ValueError("invalid Intern-Decision confidence")
        if type(probabilities) is not dict or set(probabilities) != set(FIXED_DECISIONS):
            raise ValueError("invalid Intern-Decision candidate distribution")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or
               not math.isfinite(value) or not 0 <= value <= 1
               for value in probabilities.values()):
            raise ValueError("invalid Intern-Decision candidate probability")
        if abs(sum(probabilities.values()) - 1.0) > 0.001:
            raise ValueError("Intern-Decision probabilities must sum to one")
        best = min(FIXED_DECISIONS, key=lambda label: (-probabilities[label], label))
        if choice != best or abs(confidence - probabilities[choice]) > 0.000001:
            raise ValueError("inconsistent Intern-Decision choice or confidence")
        return choice, float(confidence)

    def _value(self, selection: dict, decision: str, reason: str,
               confidence: float, abstained: bool) -> dict:
        value = {
            "decision": decision,
            "selected_capabilities": selection["selected_capabilities"],
            "evidence_ids": selection["evidence_ids"],
            "reason": reason,
            "confidence": round(confidence, 4),
            "abstained": abstained,
            "modalities": selection["modalities"],
            "input_watermark": selection["input_watermark"],
            "input_window_ms": selection["input_window_ms"],
            "evidence_ages_ms": selection["evidence_ages_ms"],
            "requested_capabilities": selection["requested_capabilities"],
        }
        return self.spec.validate(bounded_json(value, forbid_raw=True))

    def _abstain(self, selection: dict, reason: str, confidence: float = 0.0) -> dict:
        return self._value(selection, "insufficient_evidence", reason, confidence, True)

    def decide(self, records: list[dict], *, now_ms: int | None = None,
               requested_capabilities: list[str] | None = None,
               input_window_ms: int = 0) -> dict:
        """Return the baseline envelope with host-owned IDs and bounded abstention."""
        selection = select_decision_evidence(
            records, now_ms=now_ms, requested_capabilities=requested_capabilities,
            input_window_ms=input_window_ms, max_evidence=self.max_evidence)
        if not selection["selected"]:
            reason = "stale_evidence" if selection["stale_evidence"] else "insufficient_evidence"
            return self._abstain(selection, reason)
        try:
            request = self._request(selection)
        except (TypeError, ValueError):
            return self._abstain(selection, "invalid_evidence")

        response = queue.Queue(maxsize=1)
        with self._state_lock:
            if self._closed:
                raise RuntimeError("InternDecisionProvider is closed")
            if self._running:
                return self._abstain(selection, "deadline_exceeded")
            self._running = True
            self._jobs.put((request, response))
        try:
            ok, result = response.get(timeout=self.deadline_ms / 1000)
        except queue.Empty:
            # The daemon worker may finish later; its result is never reused and
            # no second inference is allowed to overlap it.
            return self._abstain(selection, "deadline_exceeded")
        if not ok:
            return self._abstain(selection, "model_inference_failed")
        try:
            decision, confidence = self._answer(result)
        except (TypeError, ValueError):
            return self._abstain(selection, "invalid_model_output")
        if confidence < self.min_confidence:
            return self._abstain(selection, "below_min_confidence", confidence)
        if decision == "insufficient_evidence":
            return self._abstain(selection, "model_abstained", confidence)
        return self._value(selection, decision, "model_selected_" + decision, confidence, False)

    def next_sensor(self, *, source_id: str = "multimodal", sensor_id: str = "decision") -> SensorMetadata:
        identifier(source_id)
        identifier(sensor_id)
        sensor = SensorMetadata(source_id, sensor_id, self._sequence)
        self._sequence += 1
        return sensor


def decision_provider(kind: str = "rules", **options):
    """Construct the deterministic default or an explicit pinned learned provider."""
    if kind == "rules":
        return MultimodalDecisionProvider(**options)
    if kind in MODEL_PROFILES:
        return InternDecisionProvider.from_local(kind=kind, **options)
    raise ValueError("kind must be rules, intern-decision-2b or intern-decision-4b")
