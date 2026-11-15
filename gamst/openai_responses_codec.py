"""Versioned direct Responses text/single-PNG projection; no network or money."""
import base64
import hashlib
import json
import re
import struct
from .native_entry import encoded, digest, _id, _hash

PROJECTION='gamst.openai.responses.text-image.v1'
MAX_RESPONSE=1024*1024


def _owned_request(request):
    """Snapshot only the bounded concrete SDK shape, never arbitrary caller graphs."""
    if type(request) is not dict or set(request) not in ({'provider','model_revision','messages'},{'provider','model_revision','messages','images'}):raise ValueError('exact SDK request required')
    result={k:request[k] for k in ('provider','model_revision')}
    for value in result.values():_id(value)
    messages=request['messages']
    if type(messages) is not list or len(messages)!=2:raise ValueError('exact system/user messages required')
    result['messages']=[]
    for index in range(2):
        item=messages[index]
        if type(item) is not dict or set(item)!={'role','content'}:raise ValueError('exact message required')
        role,content=item['role'],item['content']
        if type(role) is not str or role not in ('system','user') or type(content) is not str or len(content)>12*1024*1024:raise ValueError('bounded plain message required')
        result['messages'].append(dict(role=role,content=content))
    if 'images' in request:
        images=request['images']
        if type(images) is not list or len(images)!=1:raise ValueError('one prepared PNG required')
        item=images[0]
        keys={'mime_type','data_base64','sha256','source_frame_sha256','capture_ms','shape'}
        if type(item) is not dict or set(item)!=keys:raise ValueError('prepared PNG metadata required')
        owned={k:item[k] for k in keys if k!='shape'}
        for key in ('mime_type','data_base64','sha256','source_frame_sha256'):
            value=owned[key]
            if type(value) is not str or len(value)>(12*1024*1024 if key=='data_base64' else 256):raise ValueError('bounded plain image field required')
        if type(owned['capture_ms']) is not int or not 0<=owned['capture_ms']<=2**53-1:raise ValueError('plain capture timestamp required')
        shape=item['shape']
        if type(shape) is not list or len(shape)!=3:raise ValueError('fixed shape required')
        owned['shape']=[shape[0],shape[1],shape[2]]
        if any(type(x) is not int or not 1<=x<=8192 for x in owned['shape']):raise ValueError('plain shape integers required')
        result['images']=[owned]
    return result


def prepare_body(request,provider,model,max_output_tokens):
    request=_owned_request(request)
    if type(request) is not dict or set(request) not in ({'provider','model_revision','messages'},{'provider','model_revision','messages','images'}):raise ValueError('exact SDK request required')
    if (request['provider'],request['model_revision'])!=(provider,model):raise ValueError('selected provider/revision mismatch')
    messages=request['messages']
    if type(messages) is not list or len(messages)!=2:raise ValueError('exact system/user messages required')
    for message,role in zip(messages,('system','user')):
        if type(message) is not dict or set(message)!={'role','content'} or message['role']!=role or type(message['content']) is not str or not message['content']:raise ValueError('exact nonempty system/user message required')
    sdk=encoded(request)
    if len(sdk)>12*1024*1024:raise ValueError('SDK request exceeds12MiB')
    content=[dict(type='input_text',text=messages[1]['content'])]
    if 'images' in request:
        images=request['images']
        if type(images) is not list or len(images)!=1:raise ValueError('one prepared PNG required')
        item=images[0]
        if type(item) is not dict or set(item)!={'mime_type','data_base64','sha256','source_frame_sha256','capture_ms','shape'} or item['mime_type']!='image/png':raise ValueError('prepared PNG metadata required')
        _hash(item['sha256']);_hash(item['source_frame_sha256'])
        if type(item['data_base64']) is not str or len(item['data_base64'])>12*1024*1024:raise ValueError('bounded PNG base64 required')
        try:png=base64.b64decode(item['data_base64'],validate=True)
        except Exception:raise ValueError('invalid PNG base64') from None
        shape=item['shape']
        if (type(shape) is not list or len(shape)!=3 or any(type(x) is not int for x in shape) or
            not 1<=shape[0]<=8192 or not 1<=shape[1]<=8192 or shape[2]!=3 or
            type(item['capture_ms']) is not int or not 0<=item['capture_ms']<=2**53-1):raise ValueError('bounded PNG capture geometry required')
        if len(png)>8*1024*1024 or len(png)<33 or png[:8]!=b'\x89PNG\r\n\x1a\n' or png[12:16]!=b'IHDR' or struct.unpack('>II',png[16:24])!=(shape[1],shape[0]) or hashlib.sha256(png).hexdigest()!=item['sha256']:raise ValueError('prepared PNG identity differs')
        content.append(dict(type='input_image',image_url='data:image/png;base64,'+item['data_base64'],detail='auto'))
    body=encoded(dict(model=model,instructions=messages[0]['content'],input=[dict(role='user',content=content)],max_output_tokens=max_output_tokens,stream=False,background=False,store=False))
    if len(body)>16*1024*1024:raise ValueError('wire body exceeds16MiB')
    return body,hashlib.sha256(sdk).hexdigest()


def strict_json(raw):
    if type(raw) is not bytes or len(raw)>MAX_RESPONSE:raise ValueError('bounded original response required')
    text=raw.decode('utf8');depth=nodes=0;quoted=escaped=primitive=False
    for c in text:
        if quoted:
            if escaped:escaped=False
            elif c=='\\':escaped=True
            elif c=='"':quoted=False
        elif c=='"':quoted=True;nodes+=1;primitive=False
        elif c in '[{':depth+=1;nodes+=1
        elif c in ']}':depth-=1;primitive=False
        elif c in ',: \t\r\n':primitive=False
        elif not primitive:nodes+=1;primitive=True
        if depth>32 or nodes>32768:raise ValueError('response JSON structure exceeds bounds')
    def pairs(items):
        result={}
        for k,v in items:
            if k in result:raise ValueError('duplicate response key')
            result[k]=v
        return result
    def number(value):
        if len(value)>32:raise ValueError('response number exceeds bound')
        exponent=re.search('[eE]([+-]?[0-9]+)$',value)
        if exponent and abs(int(exponent.group(1)))>12:raise ValueError('response exponent exceeds bound')
        result=float(value) if any(x in value for x in '.eE') else int(value)
        return result
    return json.loads(text,object_pairs_hook=pairs,parse_int=number,parse_float=number,
        parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite response number')))


def response_projection(raw,model):
    value=strict_json(raw)
    if type(value) is not dict:raise ValueError('response object required')
    metadata=dict(response_id=None,returned_model=None,response_status=None,created_at=None,usage_raw_locator=None,normalized_usage=None)
    for key,target in (('id','response_id'),('model','returned_model'),('status','response_status')):
        if value.get(key) is not None:_id(value[key]);metadata[target]=value[key]
    created=value.get('created_at')
    if type(created) is int and 0<=created<=2**53-1:metadata['created_at']=created
    if 'usage' in value:metadata['usage_raw_locator']='/usage'
    return value,metadata


def normalize(value,model):
    if value.get('object')!='response' or value.get('model')!=model or value.get('status')!='completed' or value.get('error') is not None:raise ValueError('unsupported or incomplete Responses result')
    _id(value.get('id'));output=value.get('output')
    if type(output) is not list or not 1<=len(output)<=64:raise ValueError('bounded output list required')
    parts=[];size=0
    for item in output:
        if type(item) is not dict:raise ValueError('output item required')
        if item.get('type')=='reasoning':continue
        if item.get('type')!='message' or item.get('role')!='assistant' or item.get('status') not in (None,'completed'):raise ValueError('unsupported output item')
        blocks=item.get('content')
        if type(blocks) is not list or not 1<=len(blocks)<=64:raise ValueError('bounded text content required')
        for block in blocks:
            if type(block) is not dict or block.get('type')!='output_text' or type(block.get('text')) is not str:raise ValueError('only output_text is supported')
            text=block['text'];size+=len(text.encode())
            if size>8192:raise ValueError('normalized answer exceeds8192bytes')
            parts.append(text)
    text=''.join(parts)
    if not text.strip():raise ValueError('nonempty output text required')
    usage=None;original=value.get('usage')
    if type(original) is dict:
        details=original.get('input_tokens_details')
        values=[original.get('input_tokens'),original.get('output_tokens'),details.get('cached_tokens') if type(details) is dict else None]
        if all(type(x) is int and 0<=x<=10**12 for x in values) and values[2]<=values[0]:
            usage=dict(zip(('input_tokens','output_tokens','cached_input_tokens'),values))
    return dict(text=text,usage=usage)
