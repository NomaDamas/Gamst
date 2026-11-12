"""Adapt an explicitly constructed Transformers classification pipeline."""
import asyncio
import threading
from ..extensions import PluginManifest, classification, classification_name
from ..plugins import _signal


class HuggingFaceClassifier:
    def __init__(self, pipeline, *, modality, name, model_id, revision, exclusive=False):
        if modality not in ('video','audio'):
            raise ValueError('modality must be video or audio')
        if not callable(pipeline) or not model_id or not revision:
            raise ValueError('An explicit pipeline, model ID and revision are required')
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

    @classmethod
    def from_local(cls, *, path, modality, name, revision, exclusive=False):
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
        return cls(backend,modality=modality,name=name,model_id=model_path.name,revision=revision,exclusive=exclusive)
