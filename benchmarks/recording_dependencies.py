"""Owned bounded installed metadata/file verification, without package imports."""
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat

from .recorded_source import digest, encoded, _keys
from .recording_runtime_contract import sha, rows
from gamst.capabilities import identifier

MAX_FILES = 32768
MAX_READ_BYTES = 2*1024**3
MAX_METADATA = 1024**2
MAX_RECORD = 8*1024**2
VERIFICATION_WORKSPACE_BYTES = 192*1024**2
_ISSUER = object()


def _roots(values):
    if type(values) is not tuple or not values:raise ValueError('explicit absolute installation roots required')
    result=[]
    for value in values:
        path=Path(value)
        if not path.is_absolute():raise ValueError('absolute installation root required')
        result.append(Path(os.path.abspath(path)))
    return tuple(result)


def _directory(path):
    fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
    try:
        for part in Path(path).parts[1:]:
            child=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
            os.close(fd);fd=child
        result=fd;fd=None;return result
    finally:
        if fd is not None:os.close(fd)


def _read(path,authorized,remaining,*,maximum=None,content=False):
    path=Path(os.path.abspath(path))
    if not any(path==root or root in path.parents for root in authorized):raise ValueError('distribution file outside authorized roots')
    directory=_directory(path.parent)
    try:fd=os.open(path.name,os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW,dir_fd=directory)
    finally:os.close(directory)
    try:
        before=os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size>remaining or maximum is not None and before.st_size>maximum:
            raise ValueError('distribution file type/joint read bound exceeded')
        remaining-=before.st_size;count=0;hashed=hashlib.sha256();parts=[] if content else None
        while True:
            block=os.read(fd,min(65536,before.st_size+1-count))
            if not block:break
            count+=len(block)
            if count>before.st_size:raise ValueError('distribution file grew during verification')
            hashed.update(block)
            if content:parts.append(block)
        after=os.fstat(fd)
        if count!=before.st_size or (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(
                after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):raise ValueError('distribution file changed during verification')
        return (b''.join(parts) if content else dict(bytes=count,sha256=hashed.hexdigest())),remaining
    finally:os.close(fd)


def _normalized(name):return re.sub(r'[-_.]+','-',name).lower()


def _metadata_directory(distribution,metadata_roots,authorized,budget):
    found=[]
    for root in _roots(metadata_roots):
        if not any(root==a or a in root.parents for a in authorized):raise ValueError('metadata root outside authorized roots')
        fd=_directory(root)
        try:
            with os.scandir(fd) as entries:
                for entry in entries:
                    budget[2]-=1
                    if budget[2]<0:raise ValueError('joint metadata directory scan exceeds32768 entries')
                    name=entry.name
                    if len(name.encode())>4096:raise ValueError('bounded metadata directory name required')
                    if not name.endswith('.dist-info') or '-' not in name[:-10]:continue
                    package,_=name[:-10].rsplit('-',1)
                    if _normalized(package)==_normalized(distribution):
                        # No ZIP/custom provider/egg metadata or followed directory symlink.
                        child=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(child)
                        found.append(root/name)
        finally:os.close(fd)
    if len(found)!=1:raise ValueError('one explicit regular dist-info installation required')
    return found[0]


def distribution_descriptor(distribution,*,allowed_roots,metadata_roots,_budget=None):
    """Hash all unique RECORD-listed bytes after bounded owned metadata admission."""
    identifier(distribution);authorized=_roots(allowed_roots)
    budget=[MAX_FILES,MAX_READ_BYTES,32768] if _budget is None else _budget
    directory=_metadata_directory(distribution,metadata_roots,authorized,budget)
    metadata,budget[1]=_read(directory/'METADATA',authorized,budget[1],maximum=MAX_METADATA,content=True)
    if metadata.count(b'\n')+metadata.count(b'\r')-metadata.count(b'\r\n')>8192:
        raise ValueError('bounded METADATA line count required')
    lines=metadata.splitlines()
    if len(lines)>8192 or any(len(line)>8192 for line in lines):raise ValueError('bounded METADATA header/body required')
    fields={}
    for line in lines:
        if not line:break
        for key in (b'Name',b'Version'):
            if line.startswith(key+b':'):
                if key in fields:raise ValueError('duplicate installed identity metadata')
                fields[key]=line[len(key)+1:].strip().decode('ascii')
    if set(fields)!={b'Name',b'Version'} or _normalized(fields[b'Name'])!=_normalized(distribution):raise ValueError('actual installed distribution identity required')
    version=identifier(fields[b'Version'])
    record,budget[1]=_read(directory/'RECORD',authorized,budget[1],maximum=MAX_RECORD,content=True)
    if record.count(b'\n')+record.count(b'\r')-record.count(b'\r\n')>MAX_FILES or any(len(line)>12288 for line in record.splitlines()):raise ValueError('bounded RECORD rows required')
    names={}
    for row in csv.reader(io.StringIO(record.decode('utf-8')),strict=True):
        if len(row)!=3:raise ValueError('exact installed RECORD path/hash/size rows required')
        name=row[0]
        if not 1<=len(name.encode())<=4096 or '\x00' in name or Path(name).is_absolute():raise ValueError('bounded relative installed path required')
        # RECORD is inventoried itself; its asserted hash/size is not billing or
        # installation authority. Repeated exact path strings select one actual
        # file, whose bytes we hash. Differently spelled actual aliases reject.
        if name in names:
            continue
        budget[0]-=1
        if budget[0]<0:raise ValueError('joint installed file inventory exceeds32768')
        names[name]=row
    if not names or not {directory.name+'/METADATA',directory.name+'/RECORD'}<=set(names):raise ValueError('METADATA and RECORD must be inventoried')
    result=[];actual_paths=set()
    for name in sorted(names):
        path=Path(os.path.abspath(directory.parent/name))
        if path in actual_paths:raise ValueError('installed file path aliases unsupported')
        actual_paths.add(path)
        info,budget[1]=_read(path,authorized,budget[1])
        bootstrap = metadata if name==directory.name+'/METADATA' else record if name==directory.name+'/RECORD' else None
        if bootstrap is not None and info['sha256']!=hashlib.sha256(bootstrap).hexdigest():
            raise ValueError('bootstrap metadata changed before inventory completion')
        result.append(dict(path=name,**info))
    return dict(distribution=distribution,version=version,files=result)


@dataclass(frozen=True,slots=True,init=False)
class VerifiedDependencyInventory:
    _canonical:bytes
    def __init__(self,value,*,_issuer=None):
        if _issuer is not _ISSUER:raise TypeError('dependency receipt requires actual installed verification')
        object.__setattr__(self,'_canonical',encoded(value))
    @property
    def sha256(self):return digest(json.loads(self._canonical))
    def require(self,distribution,version=None):
        entry=next((r for r in json.loads(self._canonical) if r['distribution']==distribution),None)
        if entry is None or version is not None and entry['version']!=version:raise ValueError('actual dependency differs from frozen inventory')
        return dict(entry)


def verify_dependencies(expected,*,allowed_roots,metadata_roots):
    expected=rows(expected,256,1);names=set();budget=[MAX_FILES,MAX_READ_BYTES,32768];checked=[]
    for entry in expected:
        _keys(entry,('distribution','version','installation_sha256'))
        identifier(entry['distribution']);identifier(entry['version']);sha(entry['installation_sha256'])
        if _normalized(entry['distribution']) in names:raise ValueError('duplicate installed dependency declaration')
        names.add(_normalized(entry['distribution']))
        descriptor=distribution_descriptor(entry['distribution'],allowed_roots=allowed_roots,metadata_roots=metadata_roots,_budget=budget)
        if descriptor['version']!=entry['version'] or digest(descriptor)!=entry['installation_sha256']:raise ValueError('actual installed distribution differs from frozen identity')
        checked.append(dict(entry))
    return VerifiedDependencyInventory(checked,_issuer=_ISSUER)
