"""Optional, local perception backends. No model downloads occur implicitly."""
from __future__ import annotations

from dataclasses import dataclass
import asyncio
import hashlib
import importlib.metadata
import math
from pathlib import Path
import struct
import threading
from typing import Any
from .capabilities import SensorMetadata


@dataclass(frozen=True)
class InputFrame:
    data: Any
    modality: str
    timestamp_ms: int
    subject_id: str = "local-user"
    sample_rate: int | None = None
    sensor: SensorMetadata | None = None


class PerceptionPlugin:
    """Implement process(InputFrame) returning measurable signal dictionaries."""
    name = "custom"
    modalities = ("video", "audio")

    async def process(self, frame: InputFrame) -> list[dict]:
        raise NotImplementedError


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _signal(frame: InputFrame, name: str, value: Any, model: str,
            version: str, quality: str = "sufficient") -> dict:
    return dict(name=name, value=value, modality=frame.modality,
                timestamp_ms=frame.timestamp_ms, subject_id=frame.subject_id,
                model_id=model, model_version=version, quality=quality)


def _pcm_floats(data: Any) -> list[float]:
    """Accept mono normalized floats or little-endian signed PCM16 bytes."""
    if isinstance(data, (bytes, bytearray, memoryview)):
        if len(data) % 2:
            raise ValueError("PCM16 byte count must be even")
        values = [p[0] / 32768 for p in struct.iter_unpack("<h", data)]
    else:
        if getattr(data, "ndim", 1) > 1:
            if len(data.shape) != 2 or data.shape[1] != 1:
                raise ValueError("Audio must be mono; mix channels explicitly before submitting")
        if hasattr(data, "reshape"):
            data = data.reshape(-1)
        values = [float(v) for v in data]
    if len(values) > 480_000:
        raise ValueError("Audio blocks may contain at most 480000 mono samples")
    if any(not math.isfinite(v) or abs(v) > 1 for v in values):
        raise ValueError("audio samples must be finite normalized floats in [-1, 1]")
    return values


class AudioLevelPlugin(PerceptionPlugin):
    """RMS/dBFS for normalized mono PCM. Energy is not proof of speech."""
    name = "audio_level"
    modalities = ("audio",)

    def __init__(self, threshold: float = 0.02):
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be in (0, 1]")
        self.threshold = threshold

    async def process(self, frame: InputFrame) -> list[dict]:
        values = _pcm_floats(frame.data)
        if not values:
            return []
        rms = math.sqrt(sum(v * v for v in values) / len(values))
        return [_signal(frame, "audio.rms", rms, self.name, "0.1"),
                _signal(frame, "audio.dbfs", 20 * math.log10(max(rms, 1e-12)), self.name, "0.1"),
                _signal(frame, "audio.energy_active", rms >= self.threshold, self.name, "0.1")]


class OpenCVPersonPlugin(PerceptionPlugin):
    """HOG person detections. Boxes do not establish identity or speaker ownership."""
    name = "opencv_hog_person"
    modalities = ("video",)

    def __init__(self):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError('Install humansense-sdk[vision] for OpenCV') from exc
        from .tracking import PersonTracker
        self.tracker = PersonTracker()
        self._generation = 0
        self.cv2 = cv2
        self.detector = cv2.HOGDescriptor()
        self.detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    async def process(self, frame: InputFrame) -> list[dict]:
        import asyncio
        generation = self._generation
        def detect():
            height, width = frame.data.shape[:2]
            if height < 128 or width < 64:
                return []
            scale = min(1.0, 640 / width)
            image = self.cv2.resize(frame.data, (int(width * scale), int(height * scale)))
            if image.shape[0] < 128 or image.shape[1] < 64:
                return []
            rectangles, _ = self.detector.detectMultiScale(image, winStride=(8, 8), padding=(8, 8), scale=1.05)
            return [{"x": float(x / image.shape[1]), "y": float(y / image.shape[0]),
                     "width": float(w / image.shape[1]), "height": float(h / image.shape[0])}
                    for x, y, w, h in rectangles]
        boxes = await asyncio.to_thread(detect)
        if generation != self._generation:
            return []
        # Clip native boxes before normalized stream-local association.
        boxes = [{"x": max(0., min(1., b["x"])), "y": max(0., min(1., b["y"])),
                  "width": min(b["width"], 1-max(0., b["x"])),
                  "height": min(b["height"], 1-max(0., b["y"]))} for b in boxes]
        boxes = [b for b in boxes if b["width"] > 0 and b["height"] > 0]
        tracks = self.tracker.update(boxes, frame.timestamp_ms, stream_id=frame.subject_id)
        return [_signal(frame, "scene.person_count", len(boxes), self.name, self.cv2.__version__),
                _signal(frame, "scene.person_boxes", boxes, self.name, self.cv2.__version__),
                _signal(frame, "scene.person_tracks", tracks, self.name + ":iou-tracker", "0.1")]

    def reset(self):
        self._generation += 1
        self.tracker.reset()

    def close(self):
        self.reset()


class MediaPipeFacePlugin(PerceptionPlugin):
    """Face landmarks/blendshapes from an explicitly supplied local Tasks model.

    Blendshape values are expression measurements, never emotion probabilities.
    Faces are frame-local indexes, not identities or linked audio speakers.
    """
    name = "mediapipe_face_landmarker"
    modalities = ("video",)

    def __init__(self, model_path: str, max_faces: int = 4):
        path = Path(model_path)
        if not path.is_file():
            raise ValueError("Supply a local licensed MediaPipe face_landmarker.task file")
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            raise RuntimeError('Install humansense-sdk[mediapipe]') from exc
        self.mp = mp
        self.model_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        self._native_lock = threading.Lock()
        self._closed = False
        self.detector = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.IMAGE, num_faces=max_faces,
            output_face_blendshapes=True))

    async def process(self, frame: InputFrame) -> list[dict]:
        import asyncio
        # Input is OpenCV BGR; the Tasks API consumes RGB.
        rgb = frame.data[:, :, ::-1].copy()
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        def detect():
            with self._native_lock:
                if self._closed:
                    raise RuntimeError("MediaPipeFacePlugin is closed")
                return self.detector.detect(image)
        result = await asyncio.to_thread(detect)
        faces = []
        for i, landmarks in enumerate(result.face_landmarks):
            blendshapes = result.face_blendshapes[i] if i < len(result.face_blendshapes) else []
            faces.append({"frame_face_index": i,
                          "landmarks": [{"x": p.x, "y": p.y, "z": p.z} for p in landmarks],
                          "blendshapes": {p.category_name: float(p.score) for p in blendshapes}})
        return [_signal(frame, "scene.faces", faces, self.name,
                        f"{_version('mediapipe')}:sha256:{self.model_hash}")]

    async def close(self):
        def close_native():
            with self._native_lock:
                if not self._closed:
                    self._closed = True
                    self.detector.close()
        await asyncio.to_thread(close_native)


class FasterWhisperPlugin(PerceptionPlugin):
    """Bounded audio-window ASR using local weights; no network download.

    Emits a measured transcript signal; callers can route it to push_text.
    Windowed ASR is not a calibrated emotion or intent model.
    """
    name = "faster_whisper"
    modalities = ("audio",)

    def __init__(self, model_path: str, window_seconds: float = 3.0,
                 language: str | None = None, device: str = "cpu"):
        if not Path(model_path).is_dir():
            raise ValueError("Supply an existing local faster-whisper model directory")
        if not math.isfinite(window_seconds) or window_seconds < .032 or window_seconds > 30:
            raise ValueError("window_seconds must be in [0.032, 30]")
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError('Install humansense-sdk[asr]') from exc
        self.model = WhisperModel(model_path, device=device, compute_type="int8" if device == "cpu" else "float16", local_files_only=True)
        artifact_hash = hashlib.sha256()
        for artifact in sorted(Path(model_path).glob("*")):
            if artifact.is_file():
                artifact_hash.update(artifact.name.encode())
                with artifact.open("rb") as handle:
                    while block := handle.read(1024 * 1024):
                        artifact_hash.update(block)
        self.model_hash = artifact_hash.hexdigest()
        self.window_seconds = window_seconds
        self.language = language
        self.buffers: dict[str, list[float]] = {}
        self._native_lock = threading.Lock()
        self._process_lock = asyncio.Lock()
        self._generation = 0

    async def process(self, frame: InputFrame) -> list[dict]:
        if frame.sample_rate != 16000:
            raise ValueError("FasterWhisperPlugin requires 16kHz mono input")
        values = _pcm_floats(frame.data)
        generation = self._generation
        async with self._process_lock:
            if generation != self._generation:
                return []
            buf = self.buffers.setdefault(frame.subject_id, [])
            buf.extend(values)
            size = int(self.window_seconds * 16000)
            output = []
            while len(buf) >= size:
                chunk = buf[:size]
                del buf[:size]
                end_ms = max(0, frame.timestamp_ms - round(len(buf) / 16))
                def transcribe():
                    import numpy as np
                    with self._native_lock:
                        if generation != self._generation:
                            return ""
                        segments, _ = self.model.transcribe(np.asarray(chunk, dtype=np.float32),
                                                           language=self.language, vad_filter=True,
                                                           beam_size=1, best_of=1, temperature=0,
                                                           condition_on_previous_text=False)
                        # Short live windows must abstain on weak/no-speech output.
                        return " ".join(segment.text.strip() for segment in segments
                                        if getattr(segment, 'avg_logprob', 0) >= -1.2
                                        and getattr(segment, 'no_speech_prob', 0) <= .6
                                        and getattr(segment, 'compression_ratio', 0) <= 2.4).strip()
                try:
                    text = await asyncio.to_thread(transcribe)
                except asyncio.CancelledError:
                    self.reset()
                    raise
                if generation != self._generation:
                    return []
                if text:
                    window = InputFrame(None, "audio", end_ms, frame.subject_id, 16000)
                    version = f"{_version('faster-whisper')}:sha256:{self.model_hash}"
                    output.append(_signal(window, "text.transcript", text, self.name, version))
                    words = len(text.split())
                    output.append(_signal(window, "speech.transcribed_word_count", words, self.name, version))
                    output.append(_signal(window, "speech.transcribed_words_per_minute", words*60/self.window_seconds,
                                          self.name, version))
            return output

    def reset(self):
        self._generation += 1
        self.buffers.clear()


class SileroVADPlugin(PerceptionPlugin):
    """Local Silero JIT VAD, bound to one stream to isolate recurrent state.

    The score is an uncalibrated model output for speech activity, not a
    probability about emotion/intent. Thresholding uses configurable hysteresis.
    The caller supplies licensed model weights; no download or torch.hub code runs.
    """
    name = "silero_vad"
    modalities = ("audio",)

    def __init__(self, model_path: str, subject_id: str = "local-user",
                 start_threshold: float = .6, stop_threshold: float = .35):
        path = Path(model_path)
        if not path.is_file():
            raise ValueError("Supply a local Silero VAD .jit model file")
        if not 0 <= stop_threshold < start_threshold <= 1:
            raise ValueError("Require 0 <= stop_threshold < start_threshold <= 1")
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError('Install humansense-sdk[vad]') from exc
        self.torch = torch
        self.model_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        self.model = torch.jit.load(str(path), map_location="cpu")
        self.model.eval()
        self.subject_id = subject_id
        self.start_threshold, self.stop_threshold = start_threshold, stop_threshold
        self.buffer: list[float] = []
        self.active = False
        self._native_lock = threading.Lock()
        self._generation = 0

    async def process(self, frame: InputFrame) -> list[dict]:
        if frame.subject_id != self.subject_id:
            raise ValueError("SileroVADPlugin is bound to one subject stream; use a separate observer for another stream")
        if frame.sample_rate != 16000:
            raise ValueError("SileroVADPlugin requires 16kHz mono input")
        values = _pcm_floats(frame.data)
        generation = self._generation
        def process_native():
            with self._native_lock:
                if generation != self._generation:
                    return []
                return self._process_windows(frame, values)
        try:
            result = await asyncio.to_thread(process_native)
        except asyncio.CancelledError:
            await self.reset()
            raise
        return result if generation == self._generation else []

    def _process_windows(self, frame, values):
        self.buffer.extend(values)
        if len(self.buffer) < 512:
            return []
        # A single input may contain several 32ms model windows. Return each
        # decision, preserving transitions that happen inside an input block.
        output = []
        chunks = len(self.buffer) // 512
        for index in range(chunks):
            chunk, self.buffer = self.buffer[:512], self.buffer[512:]
            with self.torch.no_grad():
                score = float(self.model(self.torch.tensor(chunk, dtype=self.torch.float32), 16000).item())
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Silero model returned invalid speech activity score")
            self.active = score >= (self.stop_threshold if self.active else self.start_threshold)
            # Timestamp denotes the end of this model window, relative to the
            # caller-provided capture-block end timestamp.
            ts = max(0, frame.timestamp_ms - (chunks - index - 1) * 32)
            window = InputFrame(None, "audio", ts, frame.subject_id, 16000)
            version = f"{_version('torch')}:sha256:{self.model_hash}"
            output.append(_signal(window, "speech.vad_model_score", score, self.name, version))
            output.append(_signal(window, "speech.activity", self.active, self.name, version))
        return output

    async def reset(self):
        self._generation += 1
        def reset_native():
            with self._native_lock:
                self.buffer.clear()
                self.active = False
                self.model.reset_states()
        await asyncio.to_thread(reset_native)
