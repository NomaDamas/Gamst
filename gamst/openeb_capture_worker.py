"""Owned source protocol. Raw events go only to the parent's private pipe."""
import os
import sys

from .openeb_process import configuration,header,prefix,PROTOCOL
from .provider_worker import _read_exact
from .provider_wire import envelope,encode_frame


def _construct(config):
    from .providers.openeb_capture import OpenEBCapture
    config=configuration(config);kind=config.pop('kind');source=config.pop('source')
    return getattr(OpenEBCapture,kind)(source,**config)


def main():
    output=os.fdopen(os.dup(sys.stdout.fileno()),'wb',buffering=0)
    os.dup2(sys.stderr.fileno(),sys.stdout.fileno())
    source=sys.stdin.buffer;capture=None;previous=0
    try:
        while True:
            size,count=prefix(_read_exact(source,8))
            if count:raise ValueError('capture commands cannot carry media')
            request=header(_read_exact(source,size));number=request.get('id') if type(request) is dict else None
            if type(number) is not int or number!=previous+1 or not 1<=number<=2**53-1:raise ValueError('invalid capture command ID')
            previous=number
            try:
                operation=request.get('operation');payload=b''
                if operation=='start' and capture is None and set(request)=={'id','operation','config'}:
                    capture=_construct(request['config'])
                    result=dict(protocol=PROTOCOL,config=capture.config)
                elif operation=='next' and capture is not None and set(request)=={'id','operation'}:
                    try:frame=capture.next_frame()
                    except StopIteration:
                        output.write(envelope(dict(id=number,ok=True,result={'eof':True})))
                        return
                    encoded,payload=encode_frame(frame);result={'frame':encoded}
                elif operation=='close' and capture is not None and set(request)=={'id','operation'}:
                    capture.close();capture=None
                    output.write(envelope(dict(id=number,ok=True,result=None)))
                    return
                else:raise ValueError('invalid capture operation/state')
                output.write(envelope(dict(id=number,ok=True,result=result),payload))
            except Exception as exc:
                output.write(envelope(dict(id=number,ok=False,error_type=type(exc).__name__[:128])))
                return
    except (EOFError,BrokenPipeError):pass
    finally:
        if capture is not None:capture.close()
        output.close()


if __name__=='__main__':main()
