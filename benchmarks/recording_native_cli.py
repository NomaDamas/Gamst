"""Portable strict native4 artifact audit; no execution or numeric acceptance."""
import argparse
import hashlib
import os
import stat

from .recording_native_schema import decode,encoded,MAX_DECLARATIONS,sha
from .recording_native_validation import validate_recording_native_inventory


def read_declarations(path):
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        value=os.fstat(fd)
        if not stat.S_ISREG(value.st_mode) or value.st_size>MAX_DECLARATIONS:
            raise ValueError('bounded regular declaration snapshot required')
        chunks=[];size=0
        while True:
            part=os.read(fd,65536)
            if not part:break
            size+=len(part)
            if size>MAX_DECLARATIONS:raise ValueError('declaration snapshot exceeds16MiB')
            chunks.append(part)
        return decode(b''.join(chunks))
    finally:os.close(fd)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declarations',required=True)
    parser.add_argument('--artifacts',required=True)
    parser.add_argument('--terminal-sha256',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    d=read_declarations(args.declarations)
    index=validate_recording_native_inventory(d,args.artifacts,sha(args.terminal_sha256),source_inventory=d['source_manifest'])
    report=dict(schema_version=1,kind='recording_native_structural_audit',context=index.terminal['context'],
        terminal_sha256=args.terminal_sha256,state=index.terminal['state'],entry_count=index.count,
        chunk_count=len(index.chunk_refs),unknown_liability_count=index.count,
        actual_acceptance='incomplete',receipt_authenticity_verified=False,worker_backend_coverage_verified=False)
    data=encoded(report)+b'\n'
    # Validation precedes exclusive publication: refusal creates no output artifact.
    fd=os.open(args.output,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    try:
        position=0
        while position<len(data):
            written=os.write(fd,data[position:])
            if written<=0:raise OSError('native audit publication failed')
            position+=written
        os.fsync(fd)
    finally:os.close(fd)
    return report


if __name__=='__main__':main()
