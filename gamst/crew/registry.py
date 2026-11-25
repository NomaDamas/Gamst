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
    # The loader reads (and hashes) its whole directory, so it must hold this model's files only.
    bundle: bool = False
    # Why its worker cannot read plain camera frames or microphone chunks, and what adapts them; None: it can.
    adapter: str | None = None

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
    # "module": a model role. "input" / "view": packages an input source or a view needs in the
    # Pipeline process itself; it runs no model.
    kind: str = "module"

    @property
    def modality(self):
        """What a model role reads; the worker's own capability is checked against it when it opens."""
        return "audio" if self.id.startswith("audio.") else "video"


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
TORCH = ("torch==2.8.0", "torchvision==0.23.0")
# Distributions that install the same ``cv2`` files: at most one of them belongs in an environment.
CV2_DISTRIBUTIONS = ("opencv-python", "opencv-contrib-python", "opencv-python-headless", "opencv-contrib-python-headless")
# What a declared package pulls in by itself (from its own metadata).
CV2_DEPENDENCY = {"mediapipe": "opencv-contrib-python", "ultralytics": "opencv-python"}
# YOLOE and EfficientTAM run in one process for the after-take count (gamst.takes refine), so they share
# one environment; both pin the same torch.
REFINE_ENVIRONMENT = "yoloe-efficienttam"
MEDIAPIPE = ("mediapipe==0.10.35", "numpy>=1.24")
RF_NANO_SHA = "d8d6b9ee57d4d0ed2b1f305163624712a0532cb7bce0c747317984fc5457440d"
RF_LARGE_SHA = "0f4e20e19a99c0f8a62b5685f57f6c8b5c371c59081feda6752a0561a79ccf38"
MOBILECLIP_SHA = "35d7f213e4d75f38514e4656ad3cb91158bd33e3805d8ac349f23b186f66982f"
HF = "https://huggingface.co"
VLM_ADAPTER = "a VLM interprets sampled segments, not every frame: use the semantic-scene part"


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
        environment="mediapipe", config={"allow_skips": True},   # a skipped frame must not rebuild the detector
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
        (_weight("yoloe-26s-seg.pt", "https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26s-seg.pt", "48f24206bc8680d60cbbfa296b0140da849669b9515058b72f5a945142df0654", 31072171, "pinned source bytes"),
         _weight("mobileclip2_b.ts", "https://github.com/ultralytics/assets/releases/download/v8.4.0/mobileclip2_b.ts", MOBILECLIP_SHA, 253794476, "pinned source bytes")),
        environment=REFINE_ENVIRONMENT, config={"mode": "text", "labels": ["object"], "encoder_family": "mobileclip2:b", "device": "cpu"},
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
         _weight("mobileclip2_b.ts", "https://github.com/ultralytics/assets/releases/download/v8.4.0/mobileclip2_b.ts", MOBILECLIP_SHA, 253794476, "pinned source bytes")),
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
WHISPER_TURBO_REV = "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf"
WHISPER_TURBO_FILES = {
    "config.json": (2263, "b0253ea6c0d3bea6b1e19e91a02acfd3b53f4467362efcb5a3e6b16c9b3a9b7e"),
    "model.bin": (1617884929, "e76620f83d5f5b69efd3d87e3dc180c1bd21df9fbebacfd4335e5e1efcc018da"),
    "preprocessor_config.json": (340, "7ccc62c6f2765af1f3b46c00c9b5894426835a05021c8b9c01eecb6dfb542711"),
    "tokenizer.json": (2710337, "297b13372ac43916285644fb9687add3cc62ee2a1adb60da3dc25cc94c1871fd"),
    "vocabulary.json": (1068114, "c69260f2ab26d659b7c398f9a2b2b48ed0df16c3b47d7326782fd9cba71690c1"),
}
DIAR_REV = "52a432672732425e4c19765b9fec9b5810e6f4ba"


# Enrolled-person modules. ``automatic=False`` marks what apply must not install by itself: licences
# the owner has to accept, unpinned files, or multi-file snapshots. A recipe item then names its local
# files in ``"config"``. Every sha256 here was read from the publisher's own metadata (GitHub release
# digest, Hugging Face LFS oid, ModelScope file API); "downloaded" says the bytes were also hashed here.
LVFACE = HF + "/bytedance-research/LVFace/resolve/b12702ab1f5c721748e054a66dc90e1edd1f0724/"
SHERPA_SV = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/"
WESPEAKER = "wespeaker @ git+https://github.com/wenet-e2e/wespeaker.git@9fecd6cb4f47475d01761d87c826298dff4ef18c"
W2VBERT_SV = "https://www.modelscope.cn/models/shangguanqituan/wespeaker-w2v-bert2/resolve/master/"
W2VBERT_BACKBONE = _weight("w2v-bert-2.0.safetensors", HF + "/facebook/w2v-bert-2.0/resolve/da985ba0987f70aaeb84a80f2851cfac8c697a7b/model.safetensors", "eb890c9660ed6e3414b6812e27257b8ce5454365d5490d3ad581ea60b93be043", 2322063736, "publisher metadata")
SAPIENS2_LICENSE = "Sapiens2 License: forbids biometric processing and re-identification; owner must accept"


def _lvface(size, sha, count, verification):
    return _candidate(
        "lvface." + size.lower(), "face.embedding", ("onnxruntime>=1.18", "opencv-python>=4.8,<5", "numpy>=1.24"), ("cpu", "cuda"),
        2048, 0, 200, ESTIMATED, "MIT (code and weights); Glint360K training-data terms require review", verification,
        (_weight(f"LVFace-{size}_Glint360K.onnx", f"{LVFACE}LVFace-{size}_Glint360K/LVFace-{size}_Glint360K.onnx", sha, count, "publisher metadata"),),
        environment="lvface", config={})


def _sapiens2(size, revision, sha, count):
    return _candidate(
        "sapiens2." + size, "sapiens2.pose", ("transformers>=5.10.3", "torch>=2.7", "torchvision", "timm", "safetensors", "numpy>=1.24"),
        ("cuda",), 16384, count // MIB, 2000, ESTIMATED, SAPIENS2_LICENSE,
        "GAMST adapter written from the transformers port; no weights run; snapshot directories provisioned by hand",
        (_weight(f"sapiens2_{size}_pose.safetensors", f"{HF}/facebook/sapiens2-pose-{size}/resolve/{revision}/sapiens2_{size}_pose.safetensors", sha, count, "publisher metadata"),
         _weight("detr-resnet-101-dc5.safetensors", HF + "/facebook/detr-resnet-101-dc5/resolve/96317ca979e231bd960cb3cac31328e0165a3e94/model.safetensors", "707de69ca32dee6b518798a866219809eb3e76c190f70acda1c1f67c9f7de6c7", 242797616, "publisher metadata")),
        environment="sapiens2", python="3.11", automatic=False, config={})


def _vlm_endpoint(name, model, revision, license_name, verification, *, automatic=True, **config):
    return _candidate(
        name, "streamingvlm.local", (), ("cpu", "mps", "cuda"), 128, 0, 0, ESTIMATED, license_name, verification,
        environment="semantic-endpoint", automatic=automatic, adapter=VLM_ADAPTER,
        config={"endpoint": "http://127.0.0.1:8000/v1", "model": model, "revision": revision, "window_frames": 1, **config})


ROLES = {
    "vision.detect": Role("vision.detect", "Object detection", DETECTION, _defaults("rfdetr.nano", "rfdetr.large")),
    "vision.track": Role("vision.track", "OpenCV motion/track continuity", {
        "opencv.flow": _candidate("opencv.flow", "opencv.flow", ("opencv-python>=4.8,<5", "numpy>=1.24"), ("cpu",), 256, 0, 150, ESTIMATED, "Apache-2.0;OpenCV:Apache-2.0", "GAMST deterministic adapter; target-video quality unverified", environment="opencv", config={"method": "lucas_kanade"}),
    }, _defaults("opencv.flow"), ("vision.detect",)),
    "vision.rotation": Role("vision.rotation", "Rotation counting (in-plane turns and flips) from detections, an ROI or motion", {
        "gamst.rotation": _candidate("gamst.rotation", "vision.rotation", OPENCV, ("cpu",), 256, 0, 150, ESTIMATED, "Apache-2.0;OpenCV:Apache-2.0", "GAMST per-frame tracker + spin/angular counters; no learned model; count accuracy unvalidated", environment="opencv", in_process="gamst.providers.rotation:RotationPlugin"),
    }, _defaults("gamst.rotation"), ("vision.detect",), 160),
    "vision.video_segmentation": Role("vision.video_segmentation", "Promptable video object segmentation and tracking", {
        "efficienttam.ti_512": _candidate("efficienttam.ti_512", "efficienttam.tiny512", ("torch==2.8.0", "torchvision==0.23.0", "hydra-core==1.3.2", "omegaconf==2.3.0", "numpy>=1.24", "Pillow>=10", "tqdm>=4.66"), ("cpu", "mps", "cuda"), 2048, 0, 1500, ESTIMATED, "Apache-2.0 upstream code; checkpoint license requires review", "GAMST efficienttam.tiny512 loader; coin-flip refine 6 of 7 tosses within one flip of a human count (docs/halo-coin.md); trained-checkpoint quality otherwise unverified", (_weight("efficienttam_ti_512x512.pt", EFFICIENTTAM_URL, "7d4d652a465f0081391050932f45d8a66768ccc99c8ea393ce8a5927e83f3b9b", 71620052, "pinned source bytes"),), environment=REFINE_ENVIRONMENT, source=EFFICIENTTAM_SOURCE, config={"device": "cpu"}, adapter="needs a seed box of the exact capture: a stage of the flip part, or gamst.mask_tracking"),
    }, _defaults("efficienttam.ti_512"), ("vision.detect",)),
    "input.video": Role("input.video", "Camera, video file and recorded-take decoding (no model)", {
        "opencv": _candidate("opencv", None, OPENCV, ("cpu",), 128, 0, 150, ESTIMATED, "Apache-2.0", "decoder only; no model", environment="opencv"),
    }, _defaults("opencv"), kind="input"),
    "input.microphone": Role("input.microphone", "Live microphone capture (no model)", {
        "sounddevice": _candidate("sounddevice", None, ("sounddevice>=0.4,<1", "numpy>=1.24"), ("cpu",), 64, 0, 20, ESTIMATED, "MIT (python-sounddevice); PortAudio MIT", "capture only; no model", environment="audio-input"),
    }, _defaults("sounddevice"), kind="input"),
    "view.hud": Role("view.hud", "HUD drawing and take rendering (no model)", {
        "pillow": _candidate("pillow", None, (*OPENCV, "Pillow>=10"), ("cpu",), 64, 0, 170, ESTIMATED, "Apache-2.0 (OpenCV); HPND (Pillow)", "drawing only; no model", environment="opencv"),
    }, _defaults("pillow"), kind="view"),
    "vision.pose": Role("vision.pose", "Body pose landmarks", {
        "mediapipe.pose_lite": _mediapipe("pose_lite", "pose_landmarker_lite.task", POSE_LITE_URL, "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a", 5777746, "scripts/local_models.py pinned; real target-host acceptance unmeasured"),
        "mediapipe.pose_full": _mediapipe("pose_full", "pose_landmarker_full.task", POSE_FULL_URL, "5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1", 9398198, "pinned float16/1 bytes (size and SHA256 measured on a downloaded copy); target-host acceptance unmeasured"),
        "mediapipe.pose_heavy": _mediapipe("pose_heavy", "pose_landmarker_heavy.task", POSE_HEAVY_URL, "64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b", 30664242, "VERIFY.md: real-video CPU success, 32.3 ms compute, 91.6 ms p95"),
        "sapiens2.1b": _sapiens2("1b", "f5fed8b97b99698d5eea1d14ff0855d0b4c3f000", "2dab7014a17e99e460c18817325a71dd7a81ce48d87027f01c2ee7d7b3af969f", 6079194752),
        "sapiens2.5b": _sapiens2("5b", "ada1f29aa1fd454ca28665c700923a0101b6b24f", "b4848da8691c72e14d3ff71319f077363107129bf4128019eb39d072129b2a52", 20480899148),
    }, _defaults("mediapipe.pose_lite")),
    "vision.mediapipe_graph": Role("vision.mediapipe_graph", "MediaPipe Framework C++ graph (pose, hands, face) in its own process", {
        "graph.holistic_cpu": _candidate(
            "graph.holistic_cpu", None, ("numpy>=1.24",), ("cpu",), 1024, 0, 10000, ESTIMATED,
            "MediaPipe code Apache-2.0; bundled model files: see MediaPipe model cards (not reviewed here)",
            "not compiled or run yet; build gamst/mediapipe_graph/build.sh (source pin: pin.json), then set config.manifest to its manifest.json (SHA256 of runner, graphs and models)",
            automatic=False, in_process="gamst.mediapipe_graph:holistic_plugin"),
        "graph.holistic_gpu": _candidate(
            "graph.holistic_gpu", None, ("numpy>=1.24",), ("cuda",), 1024, 1024, 10000, ESTIMATED,
            "MediaPipe code Apache-2.0; bundled model files: see MediaPipe model cards (not reviewed here)",
            "not compiled or run yet; build.sh --gpu on Linux with EGL, then set config.manifest to its manifest.json",
            automatic=False, in_process="gamst.mediapipe_graph:holistic_plugin"),
    }, _defaults("graph.holistic_cpu")),
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
        "smolvlm.500m": _candidate("smolvlm.500m", None, ("torch==2.8.0", "torchvision==0.23.0", "transformers==4.57.6", "huggingface-hub==0.36.0", "safetensors==0.7.0", "num2words==0.5.14", "numpy>=1.24", "Pillow>=10"), ("cpu",), 4096, 0, 4000, ESTIMATED, "Apache-2.0", "GAMST direct SmolVLM adapter; real trained acceptance pending; no ProcessProvider profile", (_weight("SmolVLM2-500M-Video-Instruct", HF + "/HuggingFaceTB/SmolVLM2-500M-Video-Instruct/resolve/4cde9d208e19519c36917aa170fffc7e381aa0ad", None, 1500 * MIB, "estimated bundle; per-file SHA256 unpinned"),), environment="smolvlm", config={"adapter": "gamst.providers.smolvlm_context.SmolVLMContext"}, adapter="runs in the caller's process (gamst.providers.smolvlm_context); it has no worker"),
        "qwen3.5.9b": _candidate("qwen3.5.9b", "streamingvlm.local", ("vllm==0.34.0",), ("cuda",), 24576, 18432, 30000, {"ram": "estimated", "vram": "PLAN.md reports about 18 GB; not a VERIFY.md process measurement", "disk": "estimated"}, "Qwen model license requires review", "VERIFY.md: JSON output success; 412 ms/frame compute and 4.86 s p95; endpoint server required", (_weight("Qwen3.5-9B", HF + "/Qwen/Qwen3.5-9B/resolve/c202236235762e1c871ad0ccb60c8ee5ba337b9a", None, 18000 * MIB, "revision-pinned snapshot; per-file SHA256 unpinned"),), environment="qwen35-vllm", python="3.12", min_vram_bytes=24 * GIB, automatic=False, config={"endpoint": "http://127.0.0.1:8000/v1", "model": "Qwen/Qwen3.5-9B", "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a", "device": "cuda"}, adapter=VLM_ADAPTER),
        "qwen3.5.endpoint": _candidate("qwen3.5.endpoint", "streamingvlm.local", (), ("cpu", "mps", "cuda"), 128, 0, 0, ESTIMATED, "endpoint operator must verify model license", "Remote serving is explicit; endpoint/model acceptance is not inferred", environment="semantic-endpoint", config={"endpoint": "http://127.0.0.1:8000/v1", "model": "Qwen/Qwen3.5-9B", "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a"}, adapter=VLM_ADAPTER),
        "qwen3.8-flash.hosted": _vlm_endpoint("qwen3.8-flash.hosted", "qwen3.8-flash", "hosted-unversioned", "Alibaba Model Studio terms", "Not run. Frames leave the machine and it costs money: quote first. The endpoint and key are regional", automatic=False, endpoint="https://ap-southeast-1.maas.qwencloudapi.com/compatible-mode/v1", api_key_env="DASHSCOPE_API_KEY", allow_remote=True, request_options={"enable_thinking": False, "response_format": {"type": "json_object"}}),
        "qwen3.8-flash-next.endpoint": _vlm_endpoint("qwen3.8-flash-next.endpoint", "Qwen/Qwen3.8-Flash-Next", "812d96e10a0dd82b69d749e68693698cf6c36daa", "qwen-community-1.0 (gated); operator must accept", "Not run. Operator hosts it (360.0 GB BF16 safetensors, 185.5 GB FP8); quote the hosting cost first"),
        "qwen3.6-35b-a3b.endpoint": _vlm_endpoint("qwen3.6-35b-a3b.endpoint", "Qwen/Qwen3.6-35B-A3B", "345d467b117d1fd5eda9b879a5988de9ae5f47a7", "Apache-2.0", "Not run. Operator hosts it (71.90 GB BF16 safetensors)"),
        "qwen3.5-4b.endpoint": _vlm_endpoint("qwen3.5-4b.endpoint", "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", "Apache-2.0", "Not run. Operator hosts it (9.32 GB BF16 safetensors)"),
        "qwen3.5-2b.endpoint": _vlm_endpoint("qwen3.5-2b.endpoint", "Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc", "Apache-2.0", "Not run. Operator hosts it (4.55 GB BF16 safetensors)"),
    }, _defaults("smolvlm.500m", "qwen3.5.9b")),
    "vision.physical_concepts": Role("vision.physical_concepts", "PG-InstructBLIP physical property estimates", {
        "pg.instructblip": _candidate("pg.instructblip", "pg.physical.local", ("transformers==4.33.2", "timm==0.4.12", "fairscale==0.4.4", "torch==2.8.0", "torchvision==0.23.0", "omegaconf==2.3.0", "sentencepiece", "accelerate", "safetensors==0.7.0", "numpy>=1.24", "Pillow>=10"), ("cuda",), 49152, 40960, 60000, ESTIMATED, "PG/assets/canonical export: operator must verify", "GAMST adapter implemented; real canonical export/reference parity pending", (_weight("PG-InstructBLIP", HF + "/bidiptas/PG-InstructBLIP/resolve/7caa1683fbc3f6889868accf25c2074aee83324c", "6b3923683eb83405ca12d81e10a564e7523efe72a1bdf6efa9b493545984bd1e", 24999831084, "GAMST pg_sources.py pinned aggregate"),), environment="pg-instructblip", python="3.11", min_vram_bytes=40 * GIB, automatic=False, config={"device": "cuda"}, adapter="reads one RGB object crop and a named concept per request (docs/pg-physical-context.md), not camera frames"),
    }, _defaults("pg.instructblip"), ("vision.detect",)),
    "audio.vad": Role("audio.vad", "Speech activity detection", {
        "silero.vad": _candidate("silero.vad", "silero.activity", ("torch>=2.2,<3",), ("cpu",), 512, 0, 1200, ESTIMATED, "MIT", "scripts/local_models.py pinned; target-host real-audio acceptance unmeasured", (_weight("silero_vad.jit", SILERO_URL, "e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720", 2272526, "pinned source bytes"),), environment="silero-vad"),
    }, _defaults("silero.vad")),
    "audio.asr": Role("audio.asr", "Speech transcription", {
        "faster_whisper.small": _candidate("faster_whisper.small", None, ("faster-whisper>=1.0,<2",), ("cpu", "cuda"), 2048, 1024, 1500, ESTIMATED, "MIT code; Whisper model license requires review", "GAMST direct FasterWhisperPlugin; pinned converted bundle; target acceptance unmeasured", tuple(_weight(name, HF + "/Systran/faster-whisper-small/resolve/" + WHISPER_REV + "/" + name, sha, size, "pinned source bytes") for name, (size, sha) in WHISPER_FILES.items()), environment="faster-whisper", bundle=True, config={"plugin": "gamst.plugins.FasterWhisperPlugin", "device": "cpu"}, adapter="use the whisper-words part"),
        "faster_whisper.large_v3_turbo": _candidate("faster_whisper.large_v3_turbo", None, ("faster-whisper>=1.0,<2",), ("cpu", "cuda"), 4096, 3072, 1800, ESTIMATED, "MIT code; converted weights: license not checked", "Run on one 54 s Korean clip, 2026-11-26; accuracy not measured", tuple(_weight(name, HF + "/dropbox-dash/faster-whisper-large-v3-turbo/resolve/" + WHISPER_TURBO_REV + "/" + name, sha, size, "pinned source bytes") for name, (size, sha) in WHISPER_TURBO_FILES.items()), environment="faster-whisper", bundle=True, automatic=False, config={"plugin": "gamst.plugins.FasterWhisperPlugin", "device": "cpu"}, adapter="use the whisper-words part"),
        "qwen3.asr_1_7b": _candidate("qwen3.asr_1_7b", "qwen3.asr", ("qwen-asr==0.0.6", "vllm==0.14.0", "transformers==4.57.6", "numpy>=1.24"), ("cuda",), 16384, 7844, 12000, {"ram": "estimated", "vram": "VERIFY.md measured 7844 MiB", "disk": "estimated"}, "Qwen model license requires review", "not run with these pins (VERIFY.md's real-audio result predates them)", (_weight("Qwen3-ASR-1.7B", HF + "/Qwen/Qwen3-ASR-1.7B/resolve/7278e1e70fe206f11671096ffdd38061171dd6e5", None, 4000 * MIB, "revision-pinned snapshot; per-file SHA256 unpinned"),), environment="qwen-asr", python="3.11", min_vram_bytes=12 * GIB, automatic=False, config={"gpu_memory_utilization": 0.07, "max_model_len": 8192}, adapter="reads exact hops of one channel: use the qwen-words part"),
        "moonshine.tiny": _candidate("moonshine.tiny", "moonshine_voice.native", ("moonshine-voice==0.0.45", "numpy>=1.24"), ("cpu",), 512, 0, 500, ESTIMATED, "MIT", "GAMST adapter implemented; real-weight acceptance pending", (_weight("moonshine-v2-tiny-streaming-en", "https://github.com/moonshine-ai/moonshine-v2/tree/49bc3af5bb0d767d5156fb579fa5f9853b559bf3", None, 200 * MIB),), environment="moonshine", automatic=False, adapter="reads exact hops and has no worker profile: gamst.providers.moonshine_asr"),
    }, _defaults("faster_whisper.small", "qwen3.asr_1_7b"), ("audio.vad",)),
    "audio.events": Role("audio.events", "Environmental audio events", {
        "yamnet": _candidate("yamnet", "yamnet.events", MEDIAPIPE, ("cpu",), 512, 0, 300, ESTIMATED, "Apache-2.0", "GAMST pinned adapter; target real-audio acceptance unmeasured", (_weight("yamnet.tflite", "https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite", "4d8b4a53282dc83ef04e3e7dbc4fbc98082e34e44ed798e16c3a0cdd4c584faf", 4 * MIB),), environment="mediapipe"),
        "ced.base": _candidate("ced.base", "ced.onnx", ("onnxruntime==1.23.2", "torch==2.8.0", "torchaudio==2.8.0", "numpy>=1.24"), ("cpu",), 2048, 0, 2500, CPU_MEASURED, "GPL-3.0 upstream CED code; weights/data/runtime terms separate", "VERIFY.md: real-audio success; top-2 labels matched PyTorch reference; pinned 527-label JSON still required", (_weight("model.onnx", HF + "/mispeech/ced-base/resolve/db3e14a8db4c21b56b165261c39649741a900e7f/model.onnx", "1cb33c4300b6c52ae099a5af72058982e673ec79862855961b8b8c10eeaba74c", 400 * MIB),), environment="ced-base", automatic=False),
    }, _defaults("yamnet")),
    "audio.diarization": Role("audio.diarization", "Streaming speaker activity slots, not identity", {
        "nemotron3": _candidate("nemotron3", "nemotron3.diarization", ("transformers==5.23.0", "torch==2.8.0", "safetensors==0.8.0", "numpy>=1.24"), ("cpu", "cuda"), 4096, 1100, 3500, {"ram": "estimated", "vram": "VERIFY.md measured 1100 MiB", "disk": "estimated"}, "OpenMDW-1.1 declared; redistribution review required", "VERIFY.md: real-audio success, 1.5 ms/80 ms block, 13.8 ms p95", tuple(_weight(name, HF + "/nvidia/Nemotron-3-Diarization/resolve/" + DIAR_REV + "/" + name, sha, size, "GAMST diarization.py pinned bytes") for name, size, sha in (("config.json", 1458, "dd795ec07ffa09acf18d231aa3fa8642aa5c521760953754e02559a7413b6873"), ("processor_config.json", 623, "91726746788a024a2f0f2f699a6c3667f0d74e6a0e9288fc44fe3018e23af68e"), ("model.safetensors", 396954592, "c074d86335b3b794f8fa5edc25594558f128bdb3914d27806a3a5a2e44963cb6"))), environment="nemotron3", python="3.11", bundle=True),
    }, _defaults("nemotron3"), ("audio.vad",)),
    "audio.prosody": Role("audio.prosody", "Pitch/energy/rate DSP features", {
        "dsp": _candidate("dsp", "audio.prosody", ("numpy>=1.24",), ("cpu",), 128, 0, 50, ESTIMATED, "Apache-2.0", "Deterministic GAMST DSP; not emotion recognition", environment="audio-dsp"),
    }, _defaults("dsp")),
    "audio.vocal_expression": Role("audio.vocal_expression", "Vocal-expression classification estimates", {
        "emotion2vec.seed": _candidate("emotion2vec.seed", "emotion2vec.seed.onnx", ("onnxruntime==1.23.2", "numpy>=1.24"), ("cpu",), 4096, 0, 1500, ESTIMATED, "other/model-license; redistribution/commercial terms unverified", "GAMST export and ORT adapter; trained export/reference parity pending", (_weight("model.pt", HF + "/emotion2vec/emotion2vec_plus_seed/resolve/7be9a30d52bdc793cc24e6f7d25358d6648927c1/model.pt", "327e861049c7105a3f8589edfd16543a7cdc0c1926f1b64de975d98e0650cdd3", 1118245358, "GAMST vocal-expression.md pinned bytes"),), environment="emotion2vec", automatic=False),
    }, _defaults("emotion2vec.seed"), ("audio.vad",)),
    "audio.speaker_verify": Role("audio.speaker_verify", "Is this speech the enrolled person's voice (needs a consent record)", {
        "sherpa.campplus_zh_en": _candidate("sherpa.campplus_zh_en", "speaker.embedding", ("sherpa-onnx==2.0.3", "numpy>=1.24"), ("cpu",), 512, 0, 60, ESTIMATED, "Apache-2.0 code (3D-Speaker, sherpa-onnx); weight licence on ModelScope not checked", "Run on macOS Korean TTS voices only (tests/test_speaker_verify.py, 2026-11-25); not measured on people", (_weight("3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx", SHERPA_SV + "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx", "53cab20ba8f2eb2f6e2d89ecc18e88ef3222169b61b4419cbc13666744a4574b", 28281165, "pinned source bytes"),), environment="sherpa-onnx", config={"backend": "sherpa-onnx"}),
        "wespeaker.w2vbert2_mfa_vox": _candidate("wespeaker.w2vbert2_mfa_vox", "speaker.embedding", (WESPEAKER, "transformers", "peft", "torch", "torchaudio"), ("cpu", "mps", "cuda"), 8192, 4096, 6000, ESTIMATED, "Apache-2.0 code; VoxCeleb weights CC BY 4.0 per WeSpeaker; backbone MIT", "Adapter written from source; not run. Save the .pt as avg_model.pt with config_LM.yaml as config.yaml, and point w2vbert_args.model_name at the local backbone", (_weight("w2v_bert2_voxceleb_reproduced_LM.pt", W2VBERT_SV + "w2v_bert2_voxceleb_reproduced_LM.pt", "26d3ac4a724af599d89e6f67d27fc773596e7693bce1055fb4ef9e19607527ca", 2353130955, "publisher metadata"), W2VBERT_BACKBONE), environment="wespeaker", automatic=False, config={"backend": "wespeaker"}),
        "wespeaker.w2vbert2_mfa_voxblink2": _candidate("wespeaker.w2vbert2_mfa_voxblink2", "speaker.embedding", (WESPEAKER, "transformers", "peft", "torch", "torchaudio"), ("cpu", "mps", "cuda"), 8192, 4096, 6000, ESTIMATED, "OPT-IN: VoxBlink2 data licence not confirmed, treat as non-commercial", "Adapter written from source; not run. Prepare like w2vbert2_mfa_vox", (_weight("w2v_bert2_voxblink_official_LM.pth", W2VBERT_SV + "w2v_bert2_voxblink_official_LM.pth", "4c9ace77c99cc9998f66ce85f8d20bfd2bbdd8eba075b1a2e17456cdad4d1e43", 2346932921, "publisher metadata"), W2VBERT_BACKBONE), environment="wespeaker", automatic=False, config={"backend": "wespeaker"}),
    }, _defaults("sherpa.campplus_zh_en")),
    "audio.target_speech": Role("audio.target_speech", "One enrolled person's voice separated from a mixture (needs a consent record)", {
        "wesep.bsrnn_ecapa": _candidate("wesep.bsrnn_ecapa", "target_speech.local", ("wesep @ git+https://github.com/wenet-e2e/wesep.git@4c899c49c300a83f9cfd0d06a4acde7894abae0c",), ("cpu", "cuda"), 4096, 0, 1000, ESTIMATED, "Apache-2.0 code; weight licence not stated", "Adapter written from source; not run. Extract the tar into a model directory", (_weight("bsrnn_ecapa_vox1.tar.gz", "https://www.modelscope.cn/datasets/wenet/wesep_pretrained_models/resolve/master/bsrnn_ecapa_vox1.tar.gz", None, 261570027, "Content-Length; sha256 not published as such"),), environment="wesep", automatic=False, config={"backend": "wesep"}),
        "usef.tfgridnet": _candidate("usef.tfgridnet", "target_speech.local", ("torch", "torchaudio", "librosa", "hyperpyyaml", "python_speech_features"), ("cpu", "cuda"), 4096, 0, 1000, ESTIMATED, "OPT-IN: CC BY-NC 4.0, non-commercial only (allow_noncommercial)", "Adapter written from source; not run. 8 kHz model", (_weight("temp_best.pth.tar", HF + "/ZBang/USEF-TSE/resolve/6e2bd14dd3a2d4fcb6abd246cce5f91538c08dbf/chkpt/USEF-TFGridNet/wham!/temp_best.pth.tar", "74737c8e1fd41aaadd05393442328b909bb1891bc475c999ec1e7e1c973e0eb7", 188051307, "publisher metadata"),), environment="usef-tse", automatic=False, source={"url": "https://github.com/ZBang/USEF-TSE", "revision": "bdaeb3609735b3a9dbedd1b0625343826f2cb5fe"}, config={"backend": "usef"}),
        "sam_audio.span": _candidate("sam_audio.span", "target_speech.local", (), ("cpu", "cuda"), 8192, 0, 0, ESTIMATED, "SAM License; gated weights", "Reuses the existing SAM-Audio loader with speaker-match spans; not run", environment="sam-audio", automatic=False, config={"backend": "sam-span"}),
    }, _defaults("wesep.bsrnn_ecapa"), ("audio.speaker_verify",)),
    "audio.word_alignment": Role("audio.word_alignment", "Word times for final transcripts", {
        "qwen3.forced_aligner_0_6b": _candidate("qwen3.forced_aligner_0_6b", "qwen3.forced_aligner", ("qwen-asr==0.0.6",), ("cuda",), 8192, 4096, 4000, ESTIMATED, "Apache-2.0", "Adapter written from source; not run. Needs the whole snapshot c7cbfc2048c462b0d63a45797104fc9db3ad62b7", (_weight("model.safetensors", HF + "/Qwen/Qwen3-ForcedAligner-0.6B/resolve/c7cbfc2048c462b0d63a45797104fc9db3ad62b7/model.safetensors", "47831d0e82f96b20e9034dba01a075ee06436654719f6a68289e49f1b65ce0e7", 1835544544, "publisher metadata"),), environment="qwen-aligner", automatic=False, config={}),
    }, _defaults("qwen3.forced_aligner_0_6b"), ("audio.asr",)),
    "vision.face_identity": Role("vision.face_identity", "Is this visible face the enrolled person (needs a consent record)", {
        "lvface.t": _lvface("T", "bf8da0e1e93c432d9a1d874a9ba0990f5859f970e8864b3990f2f33d11f9cdb3", 76653813, "Not run"),
        "lvface.b": _lvface("B", "9d834ed8e927fd35b9123b2bf97c40aad05785b1f9ecfb1c4c1f6242d38d1382", 455533594, "Not run; same adapter as lvface.t"),
    }, _defaults("lvface.b")),
    "vision.body_identity": Role("vision.body_identity", "Short-horizon body match of the enrolled person (needs a consent record)", {
        "solider.swin_b_msmt17": _candidate("solider.swin_b_msmt17", "body.embedding", ("torch>=2.2", "torchvision", "yacs", "opencv-python>=4.8,<5", "numpy>=1.24"), ("cuda",), 4096, 2048, 2000, ESTIMATED, "MIT (SOLIDER-REID), Apache-2.0 (SOLIDER); MSMT17 data terms require review", "Adapter written from source; not run. Checkpoint is on Google Drive only, no published digest", (_weight("transformer_120.pth", "https://drive.google.com/file/d/1Y-RFAYdT56vnMjwxH1Ym3DVhZzZuMQZs/view", None, 0, "size unknown"),), environment="solider", automatic=False, source={"url": "https://github.com/tinyvision/SOLIDER-REID", "revision": "8c08e1c3255e8e1e51e006bf189e52cc57b009ed"}, config={}),
    }, _defaults("solider.swin_b_msmt17"), ("vision.pose",)),
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


def runs_in_host(role_id, model):
    """True when the Pipeline process itself imports this entry's packages.

    Input and view roles, and ``in_process`` candidates; every other candidate is an isolated
    worker in its own environment.
    """
    return ROLES[role_id].kind != "module" or bool(candidate(role_id, model).in_process)


def cv2_distributions(packages):
    """The OpenCV distributions a package list installs, declared or pulled in by a declared package."""
    found = set()
    for package in packages:
        name = re.match(r"[A-Za-z0-9_.-]+", package).group(0).lower().replace("_", "-")
        if name in CV2_DISTRIBUTIONS:
            found.add(name)
        if name in CV2_DEPENDENCY:
            found.add(CV2_DEPENDENCY[name])
    return found


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
