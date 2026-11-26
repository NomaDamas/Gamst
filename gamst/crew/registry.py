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
QWEN_ASR_REV = "7278e1e70fe206f11671096ffdd38061171dd6e5"
QWEN_ASR_FILES = {   # the whole snapshot (the transformers loader reads the directory); sizes and digests of these bytes
    "chat_template.json": (1161, "75a8cfca24f00de72d796fbfed6858fc9614ef3dabd8696684cc3bc03a9c58ff"),
    "config.json": (6194, "2e74a751548b8ad7d7526d29365ad8144c345d8b412b1152d25dc6698452712f"),
    "generation_config.json": (142, "1da527824d81e07118facff437e03f2e24a23311e3bdeb2368973fe77e5f275c"),
    "merges.txt": (1671853, "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"),
    "model-00001-of-00002.safetensors": (4220320824, "a4cd1f1a04d90b757dc7f7dd26254e69a013b19e80efe590a83c6a3bde8608d6"),
    "model-00002-of-00002.safetensors": (478200688, "6e0b9d9e09e2e0238e7ef3cc8a484ab387e91b90f1900bedf88bc92d7929ccfc"),
    "model.safetensors.index.json": (64821, "f994739fe38e5210b9e3e8ce6c6307315e2ceac3cb630e7b7414d69dce520f60"),
    "preprocessor_config.json": (330, "45e120a4eda2c20c5d7f2ea9354e63536bf35e27aa573fb7cdf78017b378770d"),
    "tokenizer_config.json": (12487, "4942d005604266809309cabc9f4e9cb89ce855d59b14681fdc0e1cc62ea26c4c"),
    "vocab.json": (2776833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
}
QWEN_ALIGNER_REV = "c7cbfc2048c462b0d63a45797104fc9db3ad62b7"
QWEN_ALIGNER_FILES = {
    "chat_template.json": (1161, "75a8cfca24f00de72d796fbfed6858fc9614ef3dabd8696684cc3bc03a9c58ff"),
    "config.json": (5982, "d616c65d46c4b90bdc651b0a0963ea932732241140f337f9bb6b0335a9c8ef09"),
    "generation_config.json": (115, "948d089b23bca1d214e768d59c4438365665f52ec6d33678f4062206b3fbbb8c"),
    "merges.txt": (1671853, "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"),
    "model.safetensors": (1835544544, "47831d0e82f96b20e9034dba01a075ee06436654719f6a68289e49f1b65ce0e7"),
    "preprocessor_config.json": (330, "45e120a4eda2c20c5d7f2ea9354e63536bf35e27aa573fb7cdf78017b378770d"),
    "tokenizer_config.json": (12666, "3ab80063f8511deb9566e6ad438d17b7a6277fcffd52d92854112f19d36bd81c"),
    "vocab.json": (2776833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
}
# qwen-asr's transformers backend (no vLLM): ASR and aligner share one environment so one worker holds both.
QWEN_TRANSFORMERS = ("qwen-asr==0.0.6", "transformers==4.57.6", "accelerate==1.12.0", "torch==2.8.0", "numpy>=1.24")
QWEN_TRANSFORMERS_ENVIRONMENT = "qwen-asr-transformers"
SHERPA_KO_STREAMING_REV = "84fd200583460a3adb7f68b8dd6a740359a775ff"
SHERPA_KO_STREAMING_FILES = {   # int8 encoder and joiner, fp32 decoder (sherpa-onnx's usual mix); sizes and digests of these bytes
    "encoder-epoch-99-avg-1.int8.onnx": (126968852, "8d0b1aa24fbedd4e3948564ab7facd151b8ce9b0c48fc987c541de2de3af5697"),
    "decoder-epoch-99-avg-1.onnx": (11309084, "b29cfb4575141e50a30a22b2c4579934f3d4f45b83c9c8c08c3aef5a3fa7abfc"),
    "joiner-epoch-99-avg-1.int8.onnx": (2581421, "128b80a66a1f718488af8560f9d15895109b99ff3e573f0a0130e03774ef1ced"),
    "tokens.txt": (60246, "016bdf0965029263b7ad01b742366ee542ef0bef38261510e8176ff6f2e9e668"),
}
VOXTRAL_MLX_REV = "fdebf7b2af834a1db4b8a3c99ab7480b333adf9e"
VOXTRAL_MLX_FILES = {   # mlx-community 4-bit conversion of mistralai/Voxtral-Mini-4B-Realtime-2602 (Apache-2.0)
    "config.json": (1513, "02060864a4f33df5e4944684fc17f3026af4011830cac4def6e9e025315b10c5"),
    "model.safetensors": (3133798126, "6f59b425d8a1ceb2de795454558be63937cf75b59f9c9bc77accd85aaf32af05"),
    "tekken.json": (14910348, "8434af1d39eba99f0ef46cf1450bf1a63fa941a26933a1ef5dbbf4adf0d00e44"),
}
STREAMING_HOSTED = ("websockets>=13,<18", "numpy>=1.24")
STREAMING_ADAPTER = "streaming words with capture-time word times: use the stream-words part (or --asr CANDIDATE)"
HOSTED_ASR_ADAPTER = "use the hosted-words part (phrases; audio leaves the device)"
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
PIXIT_PACKAGES = ("pyannote.audio==3.3.2", "asteroid==0.7.0", "pyannote.core==5.0.0", "pyannote.database==5.1.3", "pyannote.metrics==3.2.1", "pyannote.pipeline==3.0.1", "speechbrain==1.1.2", "huggingface-hub==0.36.0", "transformers==4.57.6", "torch==2.8.0", "torchaudio==2.8.0", "numpy<2.3", "setuptools<81", "matplotlib")
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


LIGHT_ASD_SOURCE = {"url": "https://github.com/Junhua-Liao/Light-ASD", "revision": "ed38c232de5efe0261dbd68627c0ade7cdfe14eb"}


def _light_asd(name, filename, sha, license_name, verification, **kwargs):
    return _candidate(
        name, "active_speaker.lightasd", ("torch==2.8.0", "numpy>=1.24", "scipy>=1.10"), ("cpu", "cuda"),   # MPS lacks an operator it uses
        1024, 0, 400, ESTIMATED, license_name, verification,
        (_weight(filename, f"https://raw.githubusercontent.com/Junhua-Liao/Light-ASD/{LIGHT_ASD_SOURCE['revision']}/weight/{filename}",
                 sha, 4175289, "pinned source bytes"),),
        environment="lightasd", source=LIGHT_ASD_SOURCE, config={"device": "cpu"}, **kwargs)


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
        "qwen3.asr_1_7b.transformers": _candidate("qwen3.asr_1_7b.transformers", "qwen3.asr.words", QWEN_TRANSFORMERS, ("cpu", "mps"), 12288, 0, 4700, {"ram": "estimated", "vram": "none (CPU/MPS)", "disk": "pinned snapshot bytes"}, "Apache-2.0", "Run on one 54 s Korean clip on Apple M2 Pro, 2026-11-26 (docs/measurements/sota-asr-2026-11-26.md); needs the audio.word_alignment candidate qwen3.forced_aligner_0_6b in the same environment", tuple(_weight(name, HF + "/Qwen/Qwen3-ASR-1.7B/resolve/" + QWEN_ASR_REV + "/" + name, sha, size, "pinned source bytes") for name, (size, sha) in QWEN_ASR_FILES.items()), environment=QWEN_TRANSFORMERS_ENVIRONMENT, python="3.11", bundle=True, config={"device": "mps", "dtype": "bfloat16"}, adapter="phrases with aligned word times: use the qwen-words part"),
        "alibaba.qwen_audio_3_1_asr_flash.hosted": _candidate("alibaba.qwen_audio_3_1_asr_flash.hosted", "hosted.asr.words", ("numpy>=1.24",), ("cpu", "mps", "cuda"), 256, 0, 0, ESTIMATED, "Alibaba Model Studio terms", "Not run with a key. Adapter checked against a stand-in server only. Audio leaves the machine and it costs money ($0.15 / 1M input tokens, Singapore list price, 2026-11-26)", environment="asr-endpoint", automatic=False, config={"api": "dashscope", "endpoint": "https://{WorkspaceId}.ap-southeast-1.maas.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation", "model": "qwen-audio-3.1-asr-flash", "revision": "hosted-unversioned", "api_key_env": "DASHSCOPE_API_KEY", "allow_remote": False, "price": "$0.15 per 1M input tokens + $0.47 per 1M output tokens (Singapore list price, 2026-11-26)"}, adapter=HOSTED_ASR_ADAPTER),
        "elevenlabs.scribe_v2.hosted": _candidate("elevenlabs.scribe_v2.hosted", "hosted.asr.words", ("numpy>=1.24",), ("cpu", "mps", "cuda"), 256, 0, 0, ESTIMATED, "ElevenLabs terms", "Not run with a key. Adapter checked against a stand-in server only. Audio leaves the machine and it costs money ($0.20 per hour of audio, keyterms +20-25%, 2026-11-26)", environment="asr-endpoint", automatic=False, config={"api": "elevenlabs", "endpoint": "https://api.elevenlabs.io/v1/speech-to-text", "model": "scribe_v2", "revision": "hosted-unversioned", "api_key_env": "ELEVENLABS_API_KEY", "allow_remote": False, "price": "$0.20 per hour of audio sent; keyterms add 20-25% (2026-11-26)"}, adapter=HOSTED_ASR_ADAPTER),
        "openai.gpt_live_transcribe.hosted": _candidate("openai.gpt_live_transcribe.hosted", None, STREAMING_HOSTED, ("cpu", "mps", "cuda"), 256, 0, 0, ESTIMATED, "OpenAI API terms", "Streaming (Realtime WebSocket, transcription intent). The API returns no word times (official guide): the text is gpt-live-transcribe's, word times come from the audio.word_alignment candidate qwen3.forced_aligner_0_6b (default) on each committed turn. Run on one 54 s Korean shop clip, 2026-11-26 (docs/measurements/streaming-asr-2026-11-26.md). Audio leaves the machine; $0.017 per minute of audio (model page, 2026-11-26)", environment="host", automatic=False, in_process="gamst.providers.streaming_asr:StreamingWords", config={"api": "openai", "endpoint": "wss://api.openai.com/v1/realtime?intent=transcription", "model": "gpt-live-transcribe", "revision": "hosted-unversioned", "api_key_env": "OPENAI_API_KEY", "allow_remote": False, "usd_per_hour": 1.02, "price": "$0.017 per minute of audio sent (2026-11-26)"}, adapter=STREAMING_ADAPTER),
        "elevenlabs.scribe_v2_realtime.hosted": _candidate("elevenlabs.scribe_v2_realtime.hosted", None, STREAMING_HOSTED, ("cpu", "mps", "cuda"), 256, 0, 0, ESTIMATED, "ElevenLabs terms", "Streaming (WebSocket) with word times in its committed transcripts. Run on one 54 s Korean shop clip, 2026-11-26 (docs/measurements/streaming-asr-2026-11-26.md). Audio leaves the machine; $0.22 per hour, keyterms +$0.05 per hour (pricing page, 2026-11-26)", environment="host", automatic=False, in_process="gamst.providers.streaming_asr:StreamingWords", config={"api": "elevenlabs", "endpoint": "wss://api.elevenlabs.io/v1/speech-to-text/realtime", "model": "scribe_v2_realtime", "revision": "hosted-unversioned", "api_key_env": "ELEVENLABS_API_KEY", "allow_remote": False, "commit_strategy": "vad", "usd_per_hour": 0.22, "usd_per_hour_keyterms": 0.05, "price": "$0.22 per hour of audio sent; keyterms +$0.05 per hour (2026-11-26)"}, adapter=STREAMING_ADAPTER),
        "elevenlabs.scribe_v3_live.hosted": _candidate("elevenlabs.scribe_v3_live.hosted", None, STREAMING_HOSTED, ("cpu", "mps", "cuda"), 256, 0, 0, ESTIMATED, "ElevenLabs terms", "Scribe v3 Live (successor of v2 Realtime, same WebSocket and price). Run on one 54 s Korean shop clip, 2026-11-26 (docs/measurements/streaming-asr-2026-11-26.md). Audio leaves the machine; $0.22 per hour, keyterms +$0.05 per hour (pricing page, 2026-11-26)", environment="host", automatic=False, in_process="gamst.providers.streaming_asr:StreamingWords", config={"api": "elevenlabs", "endpoint": "wss://api.elevenlabs.io/v1/speech-to-text/realtime", "model": "scribe_v3_live", "revision": "hosted-unversioned", "api_key_env": "ELEVENLABS_API_KEY", "allow_remote": False, "commit_strategy": "vad", "usd_per_hour": 0.22, "usd_per_hour_keyterms": 0.05, "price": "$0.22 per hour of audio sent; keyterms +$0.05 per hour (2026-11-26)"}, adapter=STREAMING_ADAPTER),
        "sherpa.zipformer_ko_streaming": _candidate("sherpa.zipformer_ko_streaming", "streaming.asr.words", ("sherpa-onnx==2.0.3", "numpy>=1.24"), ("cpu",), 1024, 0, 150, {"ram": "estimated", "vram": "none (CPU)", "disk": "pinned file bytes"}, "Apache-2.0 (k2-fsa model card; trained on KsponSpeech)", "Run on one 54 s Korean shop clip on Apple M2 Pro, 2026-11-26 (docs/measurements/streaming-asr-2026-11-26.md); 54 s decoded in about 2 s of CPU", tuple(_weight(name, HF + "/k2-fsa/sherpa-onnx-streaming-zipformer-korean-2024-06-16/resolve/" + SHERPA_KO_STREAMING_REV + "/" + name, sha, size, "pinned source bytes") for name, (size, sha) in SHERPA_KO_STREAMING_FILES.items()), environment="sherpa-onnx", bundle=True, config={"backend": "sherpa-zipformer", "encoder": "encoder-epoch-99-avg-1.int8.onnx", "decoder": "decoder-epoch-99-avg-1.onnx", "joiner": "joiner-epoch-99-avg-1.int8.onnx"}, adapter=STREAMING_ADAPTER),
        "voxtral.mini_4b_realtime_2602.mlx": _candidate("voxtral.mini_4b_realtime_2602.mlx", "streaming.asr.words", ("mlx-audio==0.6.8", "numpy>=1.24"), ("mps",), 6144, 0, 3200, {"ram": "estimated", "vram": "unified memory (Apple silicon)", "disk": "pinned file bytes"}, "Apache-2.0 (Mistral model card; mlx-community 4-bit conversion)", "Run on one 54 s Korean shop clip on Apple M2 Pro, 2026-11-26 (docs/measurements/streaming-asr-2026-11-26.md). Published: FLEURS Korean WER 15.74 % at 480 ms delay (model card)", tuple(_weight(name, HF + "/mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit/resolve/" + VOXTRAL_MLX_REV + "/" + name, sha, size, "pinned source bytes") for name, (size, sha) in VOXTRAL_MLX_FILES.items()), environment="voxtral-mlx", python="3.11", bundle=True, automatic=False, config={"backend": "voxtral-mlx", "delay_ms": 480}, adapter=STREAMING_ADAPTER),
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
    "audio.speaker_sources": Role("audio.speaker_sources", "Overlapping voices separated into one source per speaker (no identity; the voice score decides who)", {
        "pyannote.pixit_ami": _candidate("pyannote.pixit_ami", "pixit.sources", PIXIT_PACKAGES, ("cpu", "mps", "cuda"), 6144, 0, 1400, ESTIMATED, "MIT (pyannote.audio, PixIT model cards); SpeechBrain ECAPA (clustering embedding named by the config) Apache-2.0; gated: the owner accepts the conditions of pyannote/speech-separation-ami-1.0 and pyannote/separation-ami-1.0", "Run on one 54 s Korean shop clip, Mac M2 Pro (2026-11-26): 8 s blocks + 4 s context took 101.5 s on CPU and 84.8 s on MPS for 54.3 s of sound (not real time). Word scores: docs/measurements/overlap-2026-11-26.md", (_weight("config.yaml", HF + "/pyannote/speech-separation-ami-1.0/resolve/9486b106945ae0cc0784041a08bfcdba5edadfb9/config.yaml", "141e143ae24f691e4e9d0b8c4dc5f57e25a9d4333995d8c57c50095c50dcfa63", 597, "downloaded"), _weight("pytorch_model.bin", HF + "/pyannote/separation-ami-1.0/resolve/4d38e95cfd067c894b8b60b00761831fb01e4a8c/pytorch_model.bin", "3c94804b873ecc6c972dd095f1c33a4f9a5c9802dabf0275067ecbdcf76bf024", 1277290690, "publisher metadata"), *(_weight("ecapa/" + name, HF + "/speechbrain/spkrec-ecapa-voxceleb/resolve/5c0be3875fda05e81f3c004ed8c7c06be308de1e/" + name, sha, size, "downloaded") for name, sha, size in (("hyperparams.yaml", "ecd11c44202b32edb72709dd1013a16f2f060ebee3438ae8a9f9fecb0666ecd2", 1920), ("embedding_model.ckpt", "0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2", 83316686), ("mean_var_norm_emb.ckpt", "cd70225b05b37be64fc5a95e24395d804231d43f74b2e1e5a513db7b69b34c33", 1921), ("classifier.ckpt", "fd9e3634fe68bd0a427c95e354c0c677374f62b3f434e45b78599950d860d535", 5534328), ("label_encoder.txt", "e13c3a167bb4112685670ee896d20e2b565af16b3a4ceeaa8689fa4d22adb8b9", 128619))), _weight("wavlm-large-config.json", HF + "/microsoft/wavlm-large/resolve/c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c/config.json", "a3d8fe831aaf63d725b54a8ac36f3549cd4365c5086774b2c89cabbc6f9e129d", 2222, "downloaded")), environment="pyannote-separation", python="3.11", automatic=False, config={"backend": "pixit", "device": "cpu"}),
        "speechbrain.sepformer_whamr16k": _candidate("speechbrain.sepformer_whamr16k", "sepformer.sources", ("speechbrain==1.1.2", "torch==2.8.0", "torchaudio==2.8.0", "numpy<2.3"), ("cpu", "mps", "cuda"), 2048, 0, 120, ESTIMATED, "Apache-2.0 (SpeechBrain, model card); trained on WHAMR! (English)", "Measured on one 54 s Korean shop clip (2026-11-26): CAM++ word scores on its two sources were lower than on the mixture for the person's words; not better than no separation. Kept as a candidate, never a default", tuple(_weight(name, HF + "/speechbrain/sepformer-whamr16k/resolve/21a5b500c6f52fddc387c5d9e5fb13ffd6f039c5/" + name, sha, size, "publisher metadata; downloaded") for name, sha, size in (("hyperparams.yaml", "eeef8d1450e1e06417eda07aa87ded71c6d9f3394852a92805e6aaf5229a50c6", 1522), ("encoder.ckpt", "31d23d395a408b887b8f6ac01e477f00bcba27f95785426359fb52d52f1dc6ed", 17272), ("decoder.ckpt", "2d959cb46de5b15f008bf5476cf7c19bc680b310c7e10883e3eddfdbde533cb8", 17272), ("masknet.ckpt", "e5fb8c690668e5d1bbbc9a8256974577093a09cd08e845e9f30024fdd33472ce", 113112646))), environment="sepformer", python="3.11", automatic=False, bundle=True, config={"backend": "sepformer", "device": "cpu"}),
    }, _defaults("pyannote.pixit_ami")),
    "audio.speaker_turns": Role("audio.speaker_turns", "Who speaks when, matched to an enrolled voice (needs a consent record)", {
        "pyannoteai.precision2": _candidate("pyannoteai.precision2", None, (), ("cpu",), 128, 0, 0, ESTIMATED, "pyannoteAI terms (hosted service); audio leaves the device", "Hosted: the audio leaves the device and each job is billed. Prices read on pyannote.ai/pricing 2026-11-26 (Developer plan list prices): voiceprint 0.015 EUR each; 0.112 EUR/h is the Precision-3 diarization price, Precision-2 is not listed there; /identify is billed by audio duration with a 20 s minimum per job", environment="pyannoteai-hosted", automatic=False, in_process="gamst.providers.pyannoteai:HostedIdentify", config={"endpoint": "https://api.pyannote.ai", "api_key_env": "PYANNOTEAI_API_KEY", "model": "precision-2", "eur_per_hour": 0.112, "eur_per_voiceprint": 0.015, "allow_remote": False}),
    }, _defaults("pyannoteai.precision2")),
    "audio.word_alignment": Role("audio.word_alignment", "Word times for final transcripts", {
        "qwen3.forced_aligner_0_6b": _candidate("qwen3.forced_aligner_0_6b", "qwen3.forced_aligner", QWEN_TRANSFORMERS, ("cpu", "mps", "cuda"), 4096, 2048, 1900, {"ram": "estimated", "vram": "estimated", "disk": "pinned snapshot bytes"}, "Apache-2.0", "Run inside the qwen3.asr_1_7b.transformers worker on Apple M2 Pro, 2026-11-26; the CUDA streaming path (qwen-words with its own config) not run", tuple(_weight(name, HF + "/Qwen/Qwen3-ForcedAligner-0.6B/resolve/" + QWEN_ALIGNER_REV + "/" + name, sha, size, "pinned source bytes") for name, (size, sha) in QWEN_ALIGNER_FILES.items()), environment=QWEN_TRANSFORMERS_ENVIRONMENT, python="3.11", bundle=True, config={}),
        "faster_whisper.align_large_v3_turbo": _candidate("faster_whisper.align_large_v3_turbo", "faster_whisper.align", ("faster-whisper>=1.0,<2",), ("cpu", "cuda"), 4096, 3072, 1800, ESTIMATED, "MIT code; converted weights: license not checked", "Non-default baseline (owner: Whisper is not the product path). Word times for a given transcript (ctranslate2 Whisper alignment, the same weights as faster_whisper.large_v3_turbo; nothing is transcribed). Run on one 54 s Korean shop clip, 2026-11-26 (docs/measurements/streaming-asr-2026-11-26.md)", tuple(_weight(name, HF + "/dropbox-dash/faster-whisper-large-v3-turbo/resolve/" + WHISPER_TURBO_REV + "/" + name, sha, size, "pinned source bytes") for name, (size, sha) in WHISPER_TURBO_FILES.items()), environment="faster-whisper", bundle=True, automatic=False, config={"device": "cpu"}),
    }, _defaults("qwen3.forced_aligner_0_6b"), ("audio.asr",)),
    "vision.face_identity": Role("vision.face_identity", "Is this visible face the enrolled person (needs a consent record)", {
        "lvface.t": _lvface("T", "bf8da0e1e93c432d9a1d874a9ba0990f5859f970e8864b3990f2f33d11f9cdb3", 76653813, "Not run"),
        "lvface.b": _lvface("B", "9d834ed8e927fd35b9123b2bf97c40aad05785b1f9ecfb1c4c1f6242d38d1382", 455533594, "Not run; same adapter as lvface.t"),
    }, _defaults("lvface.b")),
    "vision.active_speaker": Role("vision.active_speaker", "Which visible face is speaking: lips that move with the sound (with vision.face and, for the enrolled person, vision.face_identity)", {
        "lightasd.ava": _light_asd("lightasd.ava", "pretrain_AVA_CVPR.model", "d44bc3ea7baa8e0946fa3921311714a630ed8b90a1928fab0dbe30d918909317",
                                   "MIT (code and weights, Junhua-Liao/Light-ASD); trained on AVA-ActiveSpeaker (its terms not checked here)",
                                   "Upstream: 94.06 mAP on AVA-ActiveSpeaker val. GAMST: one 54 s shop clip, 2026-11-26 (docs/measurements/av-speaker-2026-11-26.md)"),
        "lightasd.talkset": _light_asd("lightasd.talkset", "finetuning_TalkSet.model", "efc375833887eefa9d209dc92810e18519b04c3c73ea35a549f2a7f40b7d94d5",
                                       "OPT-IN: MIT code; fine-tuned on TalkSet (VoxCeleb2 + LRS3, LRS3 non-commercial): allow_noncommercial",
                                       "Upstream: 95.5 F1 on Columbia ASD. GAMST: same clip as lightasd.ava",
                                       automatic=False),
    }, _defaults("lightasd.ava"), ("vision.face",)),
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
