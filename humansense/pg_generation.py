"""Fixed PG generation policy and authenticated original function; no auto-download."""
import ast
import hashlib
import math

from .pg_artifact import manifest_digest
from .sam_audio_artifact import _keys,_json,_encoded

PROFILE='pg.physical.local'
HEADER_LIMIT=65536
PIXEL_LIMIT=8388608
PAYLOAD_LIMIT=PIXEL_LIMIT*3
PACKET_LIMIT=32*1024**2
TRANSPORT_ALLOWANCE=128*1024**2
ANSWERS={
    'mass':('light','medium','heavy'),
    'fragility':('fragile','not fragile'),
    'deformability':('deformable','not deformable'),
    'material':('wood','metal','plastic','glass','ceramic','paper','fabric','rubber','stone'),
    'transparency':('transparent','translucent','opaque'),
    'contents':('empty','solid','liquid','gas'),
    'can_contain_liquid':('yes','no'),
    'sealed':('yes','no'),
}
QUESTIONS={
    'mass':'Estimate this object as light, medium, or heavy?',
    'fragility':'Classify this object as fragile or not fragile?',
    'deformability':'Classify this object as deformable or not deformable?',
    'material':'Classify the visible material as wood, metal, plastic, glass, ceramic, paper, fabric, rubber, or stone?',
    'transparency':'Classify this object as transparent, translucent, or opaque?',
    'contents':'Classify the apparent contents as empty, solid, liquid, or gas?',
    'can_contain_liquid':'Can this object contain liquid? Respond yes or no.',
    'sealed':'Does this object appear sealed? Respond yes or no.',
}
PROMPTS={k:'Question: '+v+' Respond unknown if you are not sure. Short answer:' for k,v in QUESTIONS.items()}
PROMPT_SHA=manifest_digest(dict(version='sdk-physical-prompts-v1',prompts=PROMPTS,answers={k:list(v) for k,v in ANSWERS.items()}))


def policy(value,execution):
    value=_json(_encoded(value),HEADER_LIMIT)
    _keys(value,('schema_version','prompt_sha256','execution','transport_allowance_bytes','generation_workspace_bytes'),'PG generation policy')
    if value['schema_version']!=1 or type(value['schema_version']) is not int or value['prompt_sha256']!=PROMPT_SHA or value['execution']!=execution:
        raise ValueError('fixed generation policy mismatch')
    if execution['length_penalty']!=0 or execution['repetition_penalty']!=1:
        raise ValueError('PG generation profile requires reference example penalties')
    if type(value['transport_allowance_bytes']) is not int or value['transport_allowance_bytes']!=TRANSPORT_ALLOWANCE:
        raise ValueError('explicit128MiB transport allowance required')
    n=value['generation_workspace_bytes']
    if type(n) is not int or not 1024**3<=n<=512*1024**3:raise ValueError('explicit bounded generation workspace required')
    return value


def layout(shape,crop):
    if (type(shape) is not list or len(shape)!=3 or any(type(x) is not int for x in shape) or
        shape[2]!=3 or not all(1<=x<=4096 for x in shape[:2]) or shape[0]*shape[1]>PIXEL_LIMIT):
        raise ValueError('bounded RGB image shape required')
    if (type(crop) is not list or len(crop)!=4 or any(type(x) is not int for x in crop) or
        not 0<=crop[0]<crop[2]<=shape[1] or not 0<=crop[1]<crop[3]<=shape[0]):
        raise ValueError('half-open integer object crop required')
    return math.prod(shape)


def original_generate(manifest):
    from .pg_graph import _read_entry
    from pathlib import Path
    import torch
    from .pg_sources import SOURCE_HASHES
    source=manifest['sources']['pg']['files']['generate.py']
    if source['sha256']!=SOURCE_HASHES['pg']['generate.py']:raise ValueError('original PG source pin required')
    for name in ('lavis/processors/blip_processors.py','lavis/configs/models/blip2/blip2_instruct_flant5xxl.yaml'):
        entry=manifest['sources']['lavis_candidate']['files'][name]
        if entry['sha256']!=SOURCE_HASHES['lavis_candidate'][name]:raise ValueError('original eval source/config pin required')
        _read_entry(Path(manifest['bundle_root']),entry,1024**2)
    raw=_read_entry(Path(manifest['bundle_root']),source,1024**2)
    tree=ast.parse(raw)
    if (len(tree.body)!=2 or not isinstance(tree.body[0],ast.Import) or
        [a.name for a in tree.body[0].names]!=['torch'] or not isinstance(tree.body[1],ast.FunctionDef) or tree.body[1].name!='generate'):
        raise ValueError('unexpected authenticated PG function projection')
    namespace={'torch':torch}
    exec(compile(ast.Module(body=[tree.body[1]],type_ignores=[]),'authenticated-pg-generate','exec'),namespace)
    return namespace['generate']


def preprocess(payload,shape,crop):
    """Exact original eval recipe, with caller's explicit crop first."""
    layout(shape,crop)
    if type(payload) is not bytes or len(payload)!=math.prod(shape):raise ValueError('owned RGB bytes required')
    from PIL import Image
    from torchvision import transforms
    from torchvision.transforms.functional import InterpolationMode
    picture=Image.frombytes('RGB',(shape[1],shape[0]),payload).crop(tuple(crop))
    transform=transforms.Compose([transforms.Resize((224,224),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),transforms.Normalize((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711))])
    return transform(picture).unsqueeze(0)


def crop_digest(payload,shape,crop):
    layout(shape,crop);digest=hashlib.sha256();stride=shape[1]*3
    for y in range(crop[1],crop[3]):digest.update(payload[y*stride+crop[0]*3:y*stride+crop[2]*3])
    return digest.hexdigest()


def validate_result(result,concept,execution):
    result=_json(_encoded(result),HEADER_LIMIT)
    _keys(result,('alternatives','input_tokens','output_tokens','inference_ms'),'PG generated result')
    alternatives=result['alternatives']
    if type(alternatives) is not list or len(alternatives)!=execution['num_captions']:raise ValueError('bounded PG alternatives required')
    for item in alternatives:
        _keys(item,('text','beam_score'),'PG alternative')
        if type(item['text']) is not str or len(item['text'].encode())>4096:raise ValueError('bounded generated text required')
        score=item['beam_score']
        if type(score) not in (float,int) or not math.isfinite(score):raise ValueError('finite raw beam score required')
    n=result['input_tokens']
    if type(n) is not int or not 1<=n<=execution['prompt_tokens'] or result['output_tokens'] is not None:
        raise ValueError('exact input and explicitly unknown output accounting required')
    elapsed=result['inference_ms']
    if type(elapsed) not in (float,int) or not math.isfinite(elapsed) or not 0<=elapsed<=600000:raise ValueError('bounded inference duration required')
    choices={row['text'].strip().lower().removesuffix('.') for row in alternatives}
    answer=next(iter(choices)) if len(choices)==1 and choices<=set(ANSWERS[concept]) else None
    return result,answer


def infer(model,generate,manifest,generation,header,payload):
    import time
    import torch
    execution=generation['execution'];concept=header['concept'];prompt=PROMPTS[concept]
    tokens=model.t5_tokenizer([prompt],padding='longest',return_tensors='pt')
    count=int(tokens.input_ids.shape[-1])
    if not 1<=count<=execution['prompt_tokens']:raise ValueError('original tokenizer prompt limit exceeded')
    pixels=preprocess(payload,header['shape'],header['crop']).to(model.device)
    started=time.monotonic()
    with torch.no_grad():
        texts,scores=generate(model,{'image':pixels,'prompt':prompt},use_nucleus_sampling=False,
            num_beams=execution['num_beams'],max_length=execution['max_new_tokens'],min_length=1,
            repetition_penalty=1,length_penalty=0,num_captions=execution['num_captions'])
    if not isinstance(texts,list) or len(texts)!=execution['num_captions'] or tuple(scores.shape)!=(len(texts),):
        raise ValueError('original PG output shape mismatch')
    result=dict(alternatives=[dict(text=text,beam_score=float(score)) for text,score in zip(texts,scores.detach().cpu().tolist())],
        input_tokens=count,output_tokens=None,inference_ms=round((time.monotonic()-started)*1000,3))
    return validate_result(result,concept,execution)[0]


def pack_capture(pixels,crop,concept,*,track_provenance):
    """Immutable RGB packet for SensorStreams; capture metadata stays in InputFrame."""
    import numpy as np
    import struct
    from .capabilities import identifier
    if type(pixels) is not np.ndarray or pixels.dtype!=np.uint8:raise ValueError('plain uint8 RGB array required')
    shape=list(pixels.shape);layout(shape,crop)
    if type(concept) is not str or concept not in ANSWERS:raise ValueError('supported physical concept required')
    identifier(track_provenance)
    payload=pixels.tobytes(order='C')
    if len(payload)!=math.prod(shape):raise ValueError('capture changed layout during copy')
    head=_encoded(dict(shape=shape,crop=list(crop),concept=concept,track_provenance=track_provenance))
    if len(head)>HEADER_LIMIT:raise ValueError('bounded capture metadata required')
    return b'PGC1'+struct.pack('!I',len(head))+head+payload


def unpack_capture(packet):
    import struct
    if type(packet) is not bytes or not 8<=len(packet)<=PACKET_LIMIT or packet[:4]!=b'PGC1':raise ValueError('immutable PG capture packet required')
    n=struct.unpack('!I',packet[4:8])[0]
    if not 1<=n<=HEADER_LIMIT or len(packet)<8+n:raise ValueError('bounded PG capture header required')
    head=_json(packet[8:8+n],HEADER_LIMIT)
    _keys(head,('shape','crop','concept','track_provenance'),'PG capture packet')
    size=layout(head['shape'],head['crop'])
    if len(packet)!=8+n+size:raise ValueError('PG pixel payload size mismatch')
    return head,packet[8+n:]
