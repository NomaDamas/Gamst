"""Role registry: every model GAMST can install, with its packages and pinned weights.

This is the only place in GAMST that names a weight URL and its SHA256. ``gamst setup``,
the MCP ``plan_setup``/``apply_setup`` tools, ``gamst recipe setup`` and
``gamst demo rotation --fetch-models`` all resolve files from here and download them only
through :func:`gamst.crew.setup.download_weight`.

Resource numbers are planning reservations, not host acceptance claims; each names its
evidence or says "estimated".
"""
from __future__ import annotations
import json
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from .. import __version__


@dataclass(frozen=True)
class ProviderProfile:
    name: str
    runtime: str
    python: str
    accelerator: str
    provider: str
    model_revision: str
    license: str
    lock_sha256: str
    artifact_sha256: str | None = None
    lockfile: str | None = None
    artifact: str | None = None
    arch: str | None = None

    def verify_artifacts(self, root=None):
        """Verify declared files when a profile ships them; metadata alone is not evidence."""
        base = Path(root) if root is not None else Path.cwd()
        checks = {}
        for field, expected in (("lockfile", self.lock_sha256), ("artifact", self.artifact_sha256)):
            path = getattr(self, field)
            if not path:
                checks[field] = "unmeasured"
                continue
            candidate = (base / path).resolve()
            if base.resolve() not in candidate.parents and candidate != base.resolve():
                raise ValueError(f"{field} path escapes profile root")
            if not candidate.is_file():
                checks[field] = "missing"
                continue
            digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
            checks[field] = "verified" if digest == expected else "mismatch"
            if digest != expected:
                raise ValueError(f"{field} sha256 mismatch")
        return checks

    def validate_for(self, host: dict):
        if host.get("os") not in self.runtime and self.runtime != "any":
            raise RuntimeError(f"{self.name} is not compatible with {host.get('os')}")
        if self.accelerator not in ("any", host.get("accelerator")) and not (
            self.accelerator == "cuda" and host.get("accelerator") == "cuda-possible"
        ):
            raise RuntimeError(f"{self.name} requires accelerator {self.accelerator}")
        if self.arch not in (None, "any"):
            host_arch = str(host.get("arch", "")).lower()
            allowed = {part.strip().lower() for part in str(self.arch).split(",") if part.strip()}
            if host_arch not in allowed:
                raise RuntimeError(f"{self.name} requires architecture {self.arch}")
        match = re.fullmatch(r"(?:python)?\s*(\d+)\.(\d+)(?:\.(\d+))?", self.python)
        if not match:
            raise ValueError(f"{self.name} has invalid python version metadata")
        host_python = host.get("python")
        if host_python is not None:
            host_match = re.fullmatch(
                r"(?:python)?\s*(\d+)\.(\d+)(?:\.(\d+))?", str(host_python)
            )
            if host_match is None:
                raise ValueError("host has invalid python version metadata")
            required = (int(match.group(1)), int(match.group(2)))
            actual = (int(host_match.group(1)), int(host_match.group(2)))
            if actual != required:
                raise RuntimeError(
                    f"{self.name} requires Python {required[0]}.{required[1]}, "
                    f"got {actual[0]}.{actual[1]}"
                )
        return True


def read_profiles(path):
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, list) or not payload or len(payload) > 128:
        raise ValueError("profiles must be a bounded non-empty list")
    profiles = []
    names = set()
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("profile must be an object")
        required = (
            "name",
            "runtime",
            "python",
            "accelerator",
            "provider",
            "model_revision",
            "license",
            "lock_sha256",
        )
        if any(not isinstance(item.get(key), str) or not item[key] for key in required):
            raise ValueError("profile metadata is incomplete")
        if len(item["lock_sha256"]) != 64 or any(
            c not in "0123456789abcdef" for c in item["lock_sha256"]
        ):
            raise ValueError("invalid lock sha256")
        if item["name"] in names:
            raise ValueError("duplicate provider profile")
        names.add(item["name"])
        profiles.append(
            ProviderProfile(
                **{
                    key: item[key]
                    for key in (*required, "artifact_sha256", "lockfile", "artifact", "arch")
                    if key in item
                }
            )
        )
    return profiles


# Agentic setup catalog. Resource numbers are planning reservations, not host
# acceptance claims. Measured values name their evidence; everything else is
# explicitly estimated. URLs are null where the source record pins a digest but
# does not provide a downloadable location.
MIB = 1024 ** 2
GIB = 1024 * MIB
TIERS = ("cpu-only", "apple-mps", "cuda-lt24gb", "cuda-ge24gb", "cuda-ge80gb")


@dataclass(frozen=True)
class Weight:
    name: str
    url: str | None
    sha256: str | None
    size_bytes: int
    size_source: str = "estimated"

    def as_dict(self):
        return {
            "name": self.name,
            "url": self.url,
            "sha256": self.sha256,
            "status": "pinned" if self.sha256 else "unpinned",
            "size_bytes": self.size_bytes,
            "size_source": self.size_source,
        }


@dataclass(frozen=True)
class RoleCandidate:
    id: str
    provider_profile: str | None
    packages: tuple[str, ...]
    devices: tuple[str, ...]
    ram_bytes: int
    vram_bytes: int
    disk_bytes: int
    resource_source: dict[str, str]
    license: str
    verification: str
    weights: tuple[Weight, ...] = ()
    config: dict = None
    environment: str | None = None
    python: str = "3.10"
    min_vram_bytes: int = 0
    automatic: bool = True
    source: dict | None = None
    # "module:Class" for a candidate that must run inside the caller's Gamst runtime (it reads
    # committed records); None means an isolated ProcessProvider worker (``provider_profile``).
    in_process: str | None = None

    def as_dict(self):
        return {
            "id": self.id,
            "provider_profile": self.provider_profile,
            "packages": list(self.packages),
            "environment": self.environment or self.id,
            "python": self.python,
            "devices": list(self.devices),
            "resources": {
                "ram_bytes": self.ram_bytes,
                "vram_bytes": self.vram_bytes,
                "disk_bytes": self.disk_bytes,
                "source": dict(self.resource_source),
            },
            "minimum_vram_bytes": self.min_vram_bytes,
            "license": self.license,
            "verification": self.verification,
            "weights": [item.as_dict() for item in self.weights],
            "config": dict(self.config or {}),
            "automatic": self.automatic,
            "source": dict(self.source) if self.source else None,
            "in_process": self.in_process,
        }


@dataclass(frozen=True)
class Role:
    id: str
    description: str
    candidates: dict[str, RoleCandidate]
    defaults: dict[str, str]
    dependencies: tuple[str, ...] = ()
    capture_fps: int | None = None
    kind: str = "module"  # "input": packages an input source needs; it runs no model


def _weight(name, url, sha=None, size=0, source="estimated"):
    return Weight(name, url, sha, size, source)


def _candidate(model, profile, packages, devices, ram, vram, disk, resource_source,
               license_name, verification, weights=(), **kwargs):
    return RoleCandidate(
        model, profile, tuple(packages), tuple(devices), ram * MIB, vram * MIB,
        disk * MIB, dict(resource_source), license_name, verification, tuple(weights), **kwargs
    )


ESTIMATED = {"ram": "estimated", "vram": "estimated", "disk": "estimated"}
CPU_MEASURED = {
    "ram": "estimated",
    "vram": "VERIFY.md measured 0 MiB GPU on RTX PRO 6000 host (CPU worker)",
    "disk": "estimated; pinned artifact sizes are exact where supplied",
}
# A worker environment speaks the caller's provider wire, so it gets the caller's GAMST release.
BASE = ("gamst-sdk==" + __version__,)
TORCH = ("torch==2.8.0", "torchvision==0.23.0")
MEDIAPIPE = ("mediapipe==0.10.35", "numpy>=1.24")
RF_NANO_SHA = "d8d6b9ee57d4d0ed2b1f305163624712a0532cb7bce0c747317984fc5457440d"
RF_LARGE_SHA = "0f4e20e19a99c0f8a62b5685f57f6c8b5c371c59081feda6752a0561a79ccf38"
MOBILECLIP_SHA = "35d7f213e4d75f38514e4656ad3cb91158bd33e3805d8ac349f23b186f66982f"
HF = "https://huggingface.co"


def _rfdetr(variant, sha, url, verification, vram=0, source=ESTIMATED):
    return _candidate(
        "rfdetr." + variant, "rfdetr." + variant,
        ("rfdetr==1.12.1",), ("cpu", "mps", "cuda"),
        2048 if variant == "nano" else 4096, vram, 3000, source,
        "Apache-2.0",
        verification,
        (_weight("rf-detr-" + variant + ".pth", url, sha,
                 110 * MIB if variant == "nano" else 500 * MIB),),
        environment="rfdetr-" + variant,
        config={"device": "cpu"},
    )


def _mediapipe(kind, filename, url, sha, size, verification, ram=768, size_source="pinned source bytes"):
    return _candidate(
        "mediapipe." + kind, "mediapipe." + ("body" if kind.startswith("pose") else kind) + ".video",
        MEDIAPIPE, ("cpu",), ram, 0, 300, CPU_MEASURED, "Apache-2.0",
        verification, (_weight(filename, url, sha, size, size_source),),
        environment="mediapipe",
    )


DETECTION = {
    "rfdetr.nano": _rfdetr(
        "nano", RF_NANO_SHA,
        "https://repo.roboflow.com/rfdetr/nano_coco/checkpoint_best_regular.pth",
        "GAMST vision-detection.md: real-image CPU/MPS replay; no representative p95 benchmark",
    ),
    "rfdetr.large": _rfdetr(
        "large", RF_LARGE_SHA, "https://repo.roboflow.com/rfdetr/rf-detr-large-2026.pth",
        "VERIFY.md: real-video success, 7.9 ms compute, 9.9 ms p95, 910 MiB GPU",
        910, {"ram": "estimated", "vram": "VERIFY.md measured 910 MiB", "disk": "estimated"},
    ),
    "yoloe.26s": _candidate(
        "yoloe.26s", "yoloe.local",
        ("ultralytics==8.4.212", *TORCH, "numpy>=1.24", "Pillow>=10",
         "opencv-python>=4.8,<5",
         "git+https://github.com/ultralytics/CLIP.git@7ffa84b3bfa40c42ecc1c77147a855e69cb2dd40"),
        ("cpu", "mps", "cuda"), 2048, 1024, 4500, ESTIMATED,
        "Ultralytics AGPL-3.0; MobileCLIP/weights terms require review",
        "GAMST #357: real-clip p50 45.7 ms on Apple MPS, 157.1 ms on CPU; trained target-host acceptance not established",
        (_weight("yoloe-26s-seg.pt", "https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26s-seg.pt", "48f24206bc8680d60cbbfa296b0140da849669b9515058b72f5a945142df0654", 50 * MIB),
         _weight("mobileclip2_b.ts", "https://github.com/ultralytics/assets/releases/download/v8.4.0/mobileclip2_b.ts", MOBILECLIP_SHA, 350 * MIB)),
        environment="yoloe", config={"mode": "text", "labels": ["object"], "encoder_family": "mobileclip2:b", "device": "cpu"},
    ),
    "yoloe.26x": _candidate(
        "yoloe.26x", "yoloe.local",
        ("ultralytics==8.4.212", *TORCH, "numpy>=1.24", "Pillow>=10",
         "opencv-python>=4.8,<5",
         "git+https://github.com/ultralytics/CLIP.git@7ffa84b3bfa40c42ecc1c77147a855e69cb2dd40"),
        ("cpu", "mps", "cuda"), 4096, 1710, 5000,
        {"ram": "estimated", "vram": "VERIFY.md measured 1710 MiB", "disk": "estimated"},
        "Ultralytics AGPL-3.0; MobileCLIP/weights terms require review",
        "VERIFY.md: real-video success, 12.0 ms compute, 16.9 ms p95",
        (_weight("yoloe-26x-seg.pt", "https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26x-seg.pt", "d08d390a08f98195f7c87807839fe4ff93a5491645fef1bc3bf0700efafdd639", 250 * MIB),
         _weight("mobileclip2_b.ts", "https://github.com/ultralytics/assets/releases/download/v8.4.0/mobileclip2_b.ts", MOBILECLIP_SHA, 350 * MIB)),
        environment="yoloe", config={"mode": "text", "labels": ["object"], "encoder_family": "mobileclip2:b", "device": "cpu"},
    ),
    "custom": _candidate(
        "custom", "rfdetr.nano", ("rfdetr==1.12.1",), ("cpu", "mps", "cuda"),
        2048, 0, 3000, ESTIMATED, "user checkpoint license",
        "user-supplied RF-DETR Nano-compatible fine-tune; quality unverified",
        environment="rfdetr-custom", config={"device": "cpu"},
    ),
}


def _defaults(model, cuda24=None, cuda80=None):
    return {
        "cpu-only": model, "apple-mps": model, "cuda-lt24gb": model,
        "cuda-ge24gb": cuda24 or model, "cuda-ge80gb": cuda80 or cuda24 or model,
    }


POSE_LITE_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
POSE_FULL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
# Versioned (float16/1), not "latest": a moving URL cannot keep a pinned digest.
POSE_HEAVY_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
EFFICIENTTAM_URL = HF + "/yunyangx/efficient-track-anything/resolve/9bdd8ab585b19ef95f9c9ed847ac9478301890b4/efficienttam_ti_512x512.pt"
EFFICIENTTAM_SOURCE = {"url": "https://github.com/yformer/EfficientTAM", "revision": "abcd061ebd3cc6e7527d152d75b890126aaa53f6"}
OPENCV = ("opencv-python>=4.8,<5", "numpy>=1.24")
HAND_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
FACE_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
SILERO_URL = "https://raw.githubusercontent.com/snakers4/silero-vad/066409f3d9db45a30d21dfcbf9147826ffdd7052/src/silero_vad/data/silero_vad.jit"
WHISPER_REV = "536b0662742c02347bc0e980a01041f333bce120"
WHISPER_FILES = {
    "config.json": (2370, "b55496ac7940a7ae47d2c01eab40edfd8701feec1229d9cce3b40014383fb828"),
    "model.bin": (483546902, "3e305921506d8872816023e4c273e75d2419fb89b24da97b4fe7bce14170d671"),
    "tokenizer.json": (2203239, "fb7b63191e9bb045082c79fd742a3106a12c99513ab30df4a0d47fa6cb6fd0ab"),
    "vocabulary.txt": (459861, "34ce3fe1c5041027b3f8d42912270993f986dbc4bb34cf27f951e34a1e453913"),
}
DIAR_REV = "52a432672732425e4c19765b9fec9b5810e6f4ba"


ROLES = {
    "vision.detect": Role("vision.detect", "Object detection", DETECTION, _defaults("rfdetr.nano", "rfdetr.large")),
    "vision.track": Role("vision.track", "OpenCV motion/track continuity", {
        "opencv.flow": _candidate("opencv.flow", "opencv.flow", ("opencv-python>=4.8,<5", "numpy>=1.24"), ("cpu",), 256, 0, 150, ESTIMATED, "Apache-2.0;OpenCV:Apache-2.0", "GAMST deterministic adapter; target-video quality unverified", environment="opencv", config={"method": "lucas_kanade"}),
    }, _defaults("opencv.flow"), ("vision.detect",)),
    "vision.rotation": Role("vision.rotation", "Rotation counting (in-plane turns and flips) from detections, an ROI or motion", {
        "gamst.rotation": _candidate("gamst.rotation", "vision.rotation", OPENCV, ("cpu",), 256, 0, 150, ESTIMATED, "Apache-2.0;OpenCV:Apache-2.0", "GAMST per-frame tracker + spin/angular counters; no learned model; count accuracy unvalidated", environment="opencv", in_process="gamst.providers.rotation:RotationPlugin"),
    }, _defaults("gamst.rotation"), ("vision.detect",), 160),
    "vision.video_segmentation": Role("vision.video_segmentation", "Promptable video object segmentation and tracking", {
        "efficienttam.ti_512": _candidate("efficienttam.ti_512", "efficienttam.tiny512", ("torch==2.8.0", "torchvision==0.23.0", "hydra-core==1.3.2", "omegaconf==2.3.0", "numpy>=1.24", "Pillow>=10", "tqdm>=4.66"), ("cpu", "mps", "cuda"), 2048, 0, 1500, ESTIMATED, "Apache-2.0 upstream code; checkpoint license requires review", "GAMST efficienttam.tiny512 loader; coin-flip refine 6 of 7 tosses within one flip of a human count (docs/halo-coin.md); trained-checkpoint quality otherwise unverified", (_weight("efficienttam_ti_512x512.pt", EFFICIENTTAM_URL, "7d4d652a465f0081391050932f45d8a66768ccc99c8ea393ce8a5927e83f3b9b", 40 * MIB),), environment="efficienttam", source=EFFICIENTTAM_SOURCE, config={"device": "cpu"}),
    }, _defaults("efficienttam.ti_512"), ("vision.detect",)),
    "input.video": Role("input.video", "Camera, video file and recorded-take decoding (no model)", {
        "opencv": _candidate("opencv", None, OPENCV, ("cpu",), 128, 0, 150, ESTIMATED, "Apache-2.0", "decoder only; no model", environment="opencv"),
    }, _defaults("opencv"), kind="input"),
    "input.microphone": Role("input.microphone", "Live microphone capture (no model)", {
        "sounddevice": _candidate("sounddevice", None, ("sounddevice>=0.4,<1", "numpy>=1.24"), ("cpu",), 64, 0, 20, ESTIMATED, "MIT (python-sounddevice); PortAudio MIT", "capture only; no model", environment="audio-input"),
    }, _defaults("sounddevice"), kind="input"),
    "vision.pose": Role("vision.pose", "Body pose landmarks", {
        "mediapipe.pose_lite": _mediapipe("pose_lite", "pose_landmarker_lite.task", POSE_LITE_URL, "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a", 5777746, "scripts/local_models.py pinned; real target-host acceptance unmeasured"),
        "mediapipe.pose_full": _mediapipe("pose_full", "pose_landmarker_full.task", POSE_FULL_URL, "5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1", 9398198, "pinned float16/1 bytes (size and SHA256 measured on a downloaded copy); target-host acceptance unmeasured"),
        "mediapipe.pose_heavy": _mediapipe("pose_heavy", "pose_landmarker_heavy.task", POSE_HEAVY_URL, "64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b", 30664242, "VERIFY.md: real-video CPU success, 32.3 ms compute, 91.6 ms p95"),
    }, _defaults("mediapipe.pose_lite")),
    "vision.hand": Role("vision.hand", "Hand landmarks", {
        "mediapipe.hand": _mediapipe("hand", "hand_landmarker.task", HAND_URL, "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1", 8 * MIB, "VERIFY.md: real-video CPU success, 9.8 ms compute, 15.6 ms p95", size_source="estimated"),
    }, _defaults("mediapipe.hand")),
    "vision.face": Role("vision.face", "Face landmarks", {
        "mediapipe.face": _mediapipe("face", "face_landmarker.task", FACE_URL, "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff", 3758596, "VERIFY.md: real-video CPU success, 5.7 ms compute, 10.6 ms p95"),
    }, _defaults("mediapipe.face")),
    "vision.depth": Role("vision.depth", "Relative or temporal depth", {
        "depth_anything_v2.small": _candidate("depth_anything_v2.small", "depth_anything_v2.small", ("transformers==4.57.6", "torch==2.8.0", "huggingface-hub>=0.34,<1", "safetensors>=0.4.3,<1", "numpy>=1.24", "Pillow>=10"), ("cpu",), 2048, 0, 3000, ESTIMATED, "Apache-2.0", "GAMST adapter contract; real-weight acceptance pending", (_weight("model.safetensors", HF + "/depth-anything/Depth-Anything-V2-Small-hf/resolve/5426e4f0f36572d16453bbda7a8389317b1bef99/model.safetensors", None, 99200000, "HF page size; SHA256 unpinned"),), environment="depth-small"),
        "video_depth.large": _candidate("video_depth.large", "video_depth_anything.large", ("torch==2.8.0", "torchvision==0.23.0", "numpy==1.24.0", "einops==0.4.1", "opencv-python"), ("cuda",), 8192, 7858, 5000, {"ram": "estimated", "vram": "VERIFY.md measured 7858 MiB", "disk": "estimated"}, "CC-BY-NC-4.0 (research/non-commercial)", "VERIFY.md: run completed with 53.5% drop; not full-rate real time", (_weight("video_depth_anything_vitl.pth", HF + "/depth-anything/Video-Depth-Anything-Large/resolve/7aafbcb5c6af0bac741aad2b6471894fb4761afa/video_depth_anything_vitl.pth", "43df27c6b396042ba34ff7b798ab279f64d204d2e86d7a373968f8fa36d0e6fa", 1500 * MIB),), environment="video-depth", min_vram_bytes=12 * GIB, source={"url": "https://github.com/DepthAnything/Video-Depth-Anything.git", "revision": "4f5ae23172ba60fd7bc11ef671cca678842c7072"}),
    }, _defaults("depth_anything_v2.small", "depth_anything_v2.small")),
    "vision.semantic_scene": Role("vision.semantic_scene", "Low-fps semantic scene state", {
        "smolvlm.500m": _candidate("smolvlm.500m", None, ("torch==2.8.0", "torchvision==0.23.0", "transformers==4.57.6", "huggingface-hub==0.36.0", "safetensors==0.7.0", "num2words==0.5.14", "numpy>=1.24", "Pillow>=10"), ("cpu",), 4096, 0, 4000, ESTIMATED, "Apache-2.0", "GAMST direct SmolVLM adapter; real trained acceptance pending; no ProcessProvider profile", (_weight("SmolVLM2-500M-Video-Instruct", HF + "/HuggingFaceTB/SmolVLM2-500M-Video-Instruct/resolve/4cde9d208e19519c36917aa170fffc7e381aa0ad", None, 1500 * MIB, "estimated bundle; per-file SHA256 unpinned"),), environment="smolvlm", config={"adapter": "gamst.providers.smolvlm_context.SmolVLMContext"}),
        "qwen3.5.9b": _candidate("qwen3.5.9b", "streamingvlm.local", ("vllm==0.34.0",), ("cuda",), 24576, 18432, 30000, {"ram": "estimated", "vram": "PLAN.md reports about 18 GB; not a VERIFY.md process measurement", "disk": "estimated"}, "Qwen model license requires review", "VERIFY.md: JSON output success; 412 ms/frame compute and 4.86 s p95; endpoint server required", (_weight("Qwen3.5-9B", HF + "/Qwen/Qwen3.5-9B/resolve/c202236235762e1c871ad0ccb60c8ee5ba337b9a", None, 18000 * MIB, "revision-pinned snapshot; per-file SHA256 unpinned"),), environment="qwen35-vllm", python="3.12", min_vram_bytes=24 * GIB, automatic=False, config={"endpoint": "http://127.0.0.1:8000/v1", "model": "Qwen/Qwen3.5-9B", "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a", "device": "cuda"}),
        "qwen3.5.endpoint": _candidate("qwen3.5.endpoint", "streamingvlm.local", (), ("cpu", "mps", "cuda"), 128, 0, 0, ESTIMATED, "endpoint operator must verify model license", "Remote serving is explicit; endpoint/model acceptance is not inferred", environment="semantic-endpoint", config={"endpoint": "http://127.0.0.1:8000/v1", "model": "Qwen/Qwen3.5-9B", "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a"}),
    }, _defaults("smolvlm.500m", "qwen3.5.9b")),
    "vision.physical_concepts": Role("vision.physical_concepts", "PG-InstructBLIP physical property estimates", {
        "pg.instructblip": _candidate("pg.instructblip", "pg.physical.local", ("transformers==4.33.2", "timm==0.4.12", "fairscale==0.4.4", "torch==2.8.0", "torchvision==0.23.0", "omegaconf==2.3.0", "sentencepiece", "accelerate", "safetensors==0.7.0", "numpy>=1.24", "Pillow>=10"), ("cuda",), 49152, 40960, 60000, ESTIMATED, "PG/assets/canonical export: operator must verify", "GAMST adapter implemented; real canonical export/reference parity pending", (_weight("PG-InstructBLIP", HF + "/bidiptas/PG-InstructBLIP/resolve/7caa1683fbc3f6889868accf25c2074aee83324c", "6b3923683eb83405ca12d81e10a564e7523efe72a1bdf6efa9b493545984bd1e", 24999831084, "GAMST pg_sources.py pinned aggregate"),), environment="pg-instructblip", python="3.11", min_vram_bytes=40 * GIB, automatic=False, config={"device": "cuda"}),
    }, _defaults("pg.instructblip"), ("vision.detect",)),
    "audio.vad": Role("audio.vad", "Speech activity detection", {
        "silero.vad": _candidate("silero.vad", "silero.activity", ("torch>=2.2,<3",), ("cpu",), 512, 0, 1200, ESTIMATED, "MIT", "scripts/local_models.py pinned; target-host real-audio acceptance unmeasured", (_weight("silero_vad.jit", SILERO_URL, "e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720", 2272526, "pinned source bytes"),), environment="silero-vad"),
    }, _defaults("silero.vad")),
    "audio.asr": Role("audio.asr", "Speech transcription", {
        "faster_whisper.small": _candidate("faster_whisper.small", None, ("faster-whisper>=1.0,<2",), ("cpu", "cuda"), 2048, 1024, 1500, ESTIMATED, "MIT code; Whisper model license requires review", "GAMST direct FasterWhisperPlugin; pinned converted bundle; target acceptance unmeasured", tuple(_weight(name, HF + "/Systran/faster-whisper-small/resolve/" + WHISPER_REV + "/" + name, sha, size, "pinned source bytes") for name, (size, sha) in WHISPER_FILES.items()), environment="faster-whisper", config={"plugin": "gamst.plugins.FasterWhisperPlugin", "device": "cpu"}),
        "qwen3.asr_1_7b": _candidate("qwen3.asr_1_7b", "qwen3.asr", ("vllm==0.14.0", "transformers==5.23.0", "torch==2.8.0", "safetensors==0.8.0", "numpy>=1.24"), ("cuda",), 16384, 7844, 12000, {"ram": "estimated", "vram": "VERIFY.md measured 7844 MiB", "disk": "estimated"}, "Qwen model license requires review", "VERIFY.md: real-audio success, 34.4 ms/20 ms block, 133.9 ms p95", (_weight("Qwen3-ASR-1.7B", HF + "/Qwen/Qwen3-ASR-1.7B/resolve/7278e1e70fe206f11671096ffdd38061171dd6e5", None, 4000 * MIB, "revision-pinned snapshot; per-file SHA256 unpinned"),), environment="qwen-asr", python="3.11", min_vram_bytes=12 * GIB, automatic=False, config={"gpu_memory_utilization": 0.07, "max_model_len": 8192}),
        "moonshine.tiny": _candidate("moonshine.tiny", "moonshine_voice.native", ("moonshine-voice==0.0.45", "numpy>=1.24"), ("cpu",), 512, 0, 500, ESTIMATED, "MIT", "GAMST adapter implemented; real-weight acceptance pending", (_weight("moonshine-v2-tiny-streaming-en", "https://github.com/moonshine-ai/moonshine-v2/tree/49bc3af5bb0d767d5156fb579fa5f9853b559bf3", None, 200 * MIB),), environment="moonshine", automatic=False),
    }, _defaults("faster_whisper.small", "qwen3.asr_1_7b"), ("audio.vad",)),
    "audio.events": Role("audio.events", "Environmental audio events", {
        "yamnet": _candidate("yamnet", "yamnet.events", MEDIAPIPE, ("cpu",), 512, 0, 300, ESTIMATED, "Apache-2.0", "GAMST pinned adapter; target real-audio acceptance unmeasured", (_weight("yamnet.tflite", "https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite", "4d8b4a53282dc83ef04e3e7dbc4fbc98082e34e44ed798e16c3a0cdd4c584faf", 4 * MIB),), environment="mediapipe"),
        "ced.base": _candidate("ced.base", "ced.onnx", ("onnxruntime==1.23.2", "torch==2.8.0", "torchaudio==2.8.0", "numpy>=1.24"), ("cpu",), 2048, 0, 2500, CPU_MEASURED, "GPL-3.0 upstream CED code; weights/data/runtime terms separate", "VERIFY.md: real-audio success; top-2 labels matched PyTorch reference; pinned 527-label JSON still required", (_weight("model.onnx", HF + "/mispeech/ced-base/resolve/db3e14a8db4c21b56b165261c39649741a900e7f/model.onnx", "1cb33c4300b6c52ae099a5af72058982e673ec79862855961b8b8c10eeaba74c", 400 * MIB),), environment="ced-base", automatic=False),
    }, _defaults("yamnet")),
    "audio.diarization": Role("audio.diarization", "Streaming speaker activity slots, not identity", {
        "nemotron3": _candidate("nemotron3", "nemotron3.diarization", ("transformers==5.23.0", "torch==2.8.0", "safetensors==0.8.0", "numpy>=1.24"), ("cpu", "cuda"), 4096, 1100, 3500, {"ram": "estimated", "vram": "VERIFY.md measured 1100 MiB", "disk": "estimated"}, "OpenMDW-1.1 declared; redistribution review required", "VERIFY.md: real-audio success, 1.5 ms/80 ms block, 13.8 ms p95", tuple(_weight(name, HF + "/nvidia/Nemotron-3-Diarization/resolve/" + DIAR_REV + "/" + name, sha, size, "GAMST diarization.py pinned bytes") for name, size, sha in (("config.json", 1458, "dd795ec07ffa09acf18d231aa3fa8642aa5c521760953754e02559a7413b6873"), ("processor_config.json", 623, "91726746788a024a2f0f2f699a6c3667f0d74e6a0e9288fc44fe3018e23af68e"), ("model.safetensors", 396954592, "c074d86335b3b794f8fa5edc25594558f128bdb3914d27806a3a5a2e44963cb6"))), environment="nemotron3", python="3.11"),
    }, _defaults("nemotron3"), ("audio.vad",)),
    "audio.prosody": Role("audio.prosody", "Pitch/energy/rate DSP features", {
        "dsp": _candidate("dsp", "audio.prosody", ("numpy>=1.24",), ("cpu",), 128, 0, 50, ESTIMATED, "Apache-2.0", "Deterministic GAMST DSP; not emotion recognition", environment="audio-dsp"),
    }, _defaults("dsp")),
    "audio.vocal_expression": Role("audio.vocal_expression", "Vocal-expression classification estimates", {
        "emotion2vec.seed": _candidate("emotion2vec.seed", "emotion2vec.seed.onnx", ("onnxruntime==1.23.2", "numpy>=1.24"), ("cpu",), 4096, 0, 1500, ESTIMATED, "other/model-license; redistribution/commercial terms unverified", "GAMST export and ORT adapter; trained export/reference parity pending", (_weight("model.pt", HF + "/emotion2vec/emotion2vec_plus_seed/resolve/7be9a30d52bdc793cc24e6f7d25358d6648927c1/model.pt", "327e861049c7105a3f8589edfd16543a7cdc0c1926f1b64de975d98e0650cdd3", 1118245358, "GAMST vocal-expression.md pinned bytes"),), environment="emotion2vec", automatic=False),
    }, _defaults("emotion2vec.seed"), ("audio.vad",)),
    "physio.rppg": Role("physio.rppg", "Camera pulse estimate; not a medical measurement", {
        "factorizephys": _candidate("factorizephys", "factorizephys.local", ("torch==2.8.0", "numpy>=1.24", "opencv-python>=4.8,<5", "scipy>=1.10"), ("cpu", "mps", "cuda"), 4096, 1104, 3500, {"ram": "estimated", "vram": "VERIFY.md measured 1104 MiB", "disk": "estimated"}, "rPPG-Toolbox Responsible-AI-License", "VERIFY.md: checkpoint/graph loaded; pulse output not confirmed", (_weight("UBFC-rPPG_FactorizePhys_FSAM_Res.pth", "https://github.com/ubicomplab/rPPG-Toolbox/raw/b7500b848f84ad7f86e277b4612563b69f4f88f9/final_model_release/UBFC-rPPG_FactorizePhys_FSAM_Res.pth", "03cb8590e0b1f0fe669534d9708520fed0b20e1fe28e1b57a56a2504b4a246f6", 100 * MIB),), environment="factorizephys", source={"url": "https://github.com/ubicomplab/rPPG-Toolbox.git", "revision": "b7500b848f84ad7f86e277b4612563b69f4f88f9"}),
    }, _defaults("factorizephys"), ("vision.face",)),
}


GOALS = {
    "rotation": ("vision.detect", "vision.rotation"),
    "speech": ("audio.vad", "audio.asr"),
    "audio": ("audio.vad", "audio.asr", "audio.events", "audio.prosody"),
    "scene": ("vision.detect", "vision.semantic_scene"),
    "depth": ("vision.detect", "vision.depth"),
    "body": ("vision.pose",),
    "pose": ("vision.pose", "vision.hand", "vision.face"),
    "physical": ("vision.detect", "vision.physical_concepts"),
    "physio": ("vision.face", "physio.rppg"),
    "all": tuple(ROLES),
}


def candidate(role_id, model):
    """The registry entry for ``model`` in ``role_id``; ValueError names the choices."""
    role = ROLES.get(role_id)
    if role is None:
        raise ValueError(f"unknown role {role_id!r}")
    if model not in role.candidates:
        raise ValueError(f"unknown model {model!r} for {role_id}; choose from {', '.join(role.candidates)}")
    return role.candidates[model]


def weight_assets(role_id, model):
    """``{file name: (url, sha256)}`` of one candidate's pinned weights."""
    return {weight.name: (weight.url, weight.sha256) for weight in candidate(role_id, model).weights}


def role_catalog():
    """Return a JSON-safe copy; callers cannot mutate the process catalog."""
    return {
        role_id: {
            "description": role.description,
            "dependencies": list(role.dependencies),
            "capture_fps": role.capture_fps,
            "kind": role.kind,
            "defaults": dict(role.defaults),
            "candidates": {name: candidate.as_dict() for name, candidate in role.candidates.items()},
        }
        for role_id, role in ROLES.items()
    }
