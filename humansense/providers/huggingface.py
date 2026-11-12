"""Adapt an explicitly constructed Transformers classification pipeline."""
import asyncio
import threading
import math
from ..extensions import PluginManifest, classification, classification_name
from ..plugins import _signal, InputFrame, _pcm_floats


class HuggingFaceClassifier:
    def __init__(self, pipeline, *, modality, name, model_id, revision, exclusive=False, window_seconds=0, hop_seconds=1):
        if modality not in ('video','audio'):
            raise ValueError('modality must be video or audio')
        if not callable(pipeline) or not model_id or not revision:
            raise ValueError('An explicit pipeline, model ID and revision are required')
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in (window_seconds,hop_seconds)) or not 0 <= window_seconds <= 10 or not .1 <= hop_seconds <= 10:
            raise ValueError('window_seconds must be 0..10 and hop_seconds .1..10')
        if window_seconds and (modality != 'audio' or hop_seconds > window_seconds):
            raise ValueError('Rolling windows require audio and hop <= window')
        self.window_seconds,self.hop_seconds = window_seconds,hop_seconds
        self._audio_buffer=[]
        self._audio_owner=None
        self._audio_since=0
        self._audio_emitted=False
        self.pipeline, self.modality = pipeline, modality
        self.modalities, self.name = (modality,), classification_name(name)
        self.model_id, self.version, self.exclusive = model_id, revision, exclusive
        self.manifest = PluginManifest(self.name,revision,self.modalities,(self.name,),
                                       'Transformers image/audio classification pipeline')
        self._generation = 0
        self._lock = asyncio.Lock()
        self._inference_lock = threading.Lock()

    async def process(self, frame):
        generation = self._generation
        async with self._lock:
            if generation != self._generation:
                return []
            if self.window_seconds:
                if type(frame.sample_rate) is not int or not 8000 <= frame.sample_rate <= 96000:
                    raise ValueError('Rolling audio requires a valid sample rate')
                owner=(frame.subject_id,frame.sample_rate)
                if owner != self._audio_owner:
                    self._audio_buffer=[];self._audio_since=0;self._audio_emitted=False
                    self._audio_owner=owner
                block=_pcm_floats(frame.data)
                count=int(self.window_seconds*frame.sample_rate)
                self._audio_buffer=(self._audio_buffer+block)[-count:]
                self._audio_since+=len(block)
                if len(self._audio_buffer)<count or (self._audio_emitted and self._audio_since<int(self.hop_seconds*frame.sample_rate)):
                    return []
                self._audio_since=0;self._audio_emitted=True
                frame=InputFrame(list(self._audio_buffer),'audio',frame.timestamp_ms,frame.subject_id,frame.sample_rate)
            def run_pipeline(data):
                with self._inference_lock:
                    if generation != self._generation:
                        return []
                    return self.pipeline(data, top_k=32)
            def infer():
                if self.modality == 'video':
                    from PIL import Image
                    # Core camera frames use OpenCV BGR; Transformers uses RGB.
                    data = Image.fromarray(frame.data[:,:,::-1])
                else:
                    import numpy as np
                    data = {'array':np.asarray(frame.data,dtype=np.float32),'sampling_rate':frame.sample_rate}
                return run_pipeline(data)
            output = await asyncio.to_thread(infer)
        if generation != self._generation:
            return []
        labels = {row['label']:float(row['score']) for row in sorted(output,key=lambda row:float(row['score']),reverse=True)[:32]}
        value = classification(labels,exclusive=self.exclusive)
        return [_signal(frame,self.name,value,self.model_id,self.version)]

    def reset(self):
        self._generation += 1
        self._audio_buffer=[]
        self._audio_owner=None
        self._audio_since=0
        self._audio_emitted=False

    @classmethod
    def from_local(cls, *, path, modality, name, revision, exclusive=False, window_seconds=0, hop_seconds=1):
        """Load only an existing model directory; no Hub download or remote code."""
        from pathlib import Path
        model_path = Path(path).expanduser().resolve()
        if not model_path.is_dir():
            raise ValueError('Supply an existing local Hugging Face model directory')
        from transformers import (pipeline, AutoModelForImageClassification, AutoImageProcessor,
                                  AutoModelForAudioClassification, AutoFeatureExtractor)
        options = dict(local_files_only=True, trust_remote_code=False)
        if modality == 'video':
            model = AutoModelForImageClassification.from_pretrained(str(model_path), **options)
            processor = AutoImageProcessor.from_pretrained(str(model_path), **options)
            backend = pipeline('image-classification', model=model, image_processor=processor, device='cpu')
        elif modality == 'audio':
            model = AutoModelForAudioClassification.from_pretrained(str(model_path), **options)
            processor = AutoFeatureExtractor.from_pretrained(str(model_path), **options)
            backend = pipeline('audio-classification', model=model, feature_extractor=processor, device='cpu')
        else:
            raise ValueError('modality must be video or audio')
        return cls(backend,modality=modality,name=name,model_id=model_path.name,revision=revision,exclusive=exclusive,window_seconds=window_seconds,hop_seconds=hop_seconds)
