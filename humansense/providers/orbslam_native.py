"""Explicit ctypes binding to gamst's pinned ORB-SLAM3 native bridge.

Not a complete context provider yet; use only inside an owned worker process.
"""
import ctypes
import hashlib
import math
import os
from pathlib import Path
import re
import threading

SOURCE_REVISION='4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4'


def _asset(path,digest,limit):
    if type(path) is not str or not Path(path).is_absolute() or not Path(path).is_file():
        raise ValueError('absolute existing local asset required')
    if type(digest) is not str or not re.fullmatch('[0-9a-f]{64}',digest):
        raise ValueError('lowercase SHA256 required')
    h=hashlib.sha256();size=0
    with open(path,'rb') as source:
        while True:
            chunk=source.read(1024*1024)
            if not chunk:break
            size+=len(chunk)
            if size>limit:raise ValueError('native asset exceeds bound')
            h.update(chunk)
    if not size or h.hexdigest()!=digest:raise ValueError('native asset hash mismatch')
    return path


class ORBSLAMNative:
    """One monocular session per owned process; arbitrary-scale Tcw, no metrics."""
    def __init__(self,library_path,vocabulary_path,settings_path,*,library_sha256,
                 vocabulary_sha256,settings_sha256,max_frames=300):
        if os.environ.get('GAMST_PROVIDER_WORKER')!='1':
            raise RuntimeError('ORB-SLAM3 requires an explicitly owned provider worker')
        if type(max_frames) is not int or not 1<=max_frames<=3000:
            raise ValueError('max_frames must be1..3000 across all resets')
        library_path=_asset(library_path,library_sha256,256*1024*1024)
        vocabulary_path=_asset(vocabulary_path,vocabulary_sha256,256*1024*1024)
        settings_path=_asset(settings_path,settings_sha256,65536)
        # Deliberately accept only a flat numeric pinhole settings subset.
        # No free-form strings, atlas paths, XML or YAML aliases/extensions.
        text=Path(settings_path).read_text(encoding='utf-8')
        numeric={'Camera1.'+key for key in ('fx','fy','cx','cy','k1','k2','p1','p2','k3')}
        numeric.update('Camera.'+key for key in ('fps','RGB','width','height'))
        numeric.update('ORBextractor.'+key for key in ('nFeatures','scaleFactor','nLevels','iniThFAST','minThFAST'))
        numeric.update('Viewer.'+key for key in ('KeyFrameSize','KeyFrameLineWidth','GraphLineWidth','PointSize','CameraSize','CameraLineWidth','ViewpointX','ViewpointY','ViewpointZ','ViewpointF'))
        integer_bounds={'Camera.width':(32,1920),'Camera.height':(32,1080),
                        'Camera.fps':(1,240),'Camera.RGB':(0,1),
                        'ORBextractor.nFeatures':(1,5000),'ORBextractor.nLevels':(1,8),
                        'ORBextractor.iniThFAST':(1,255),'ORBextractor.minThFAST':(1,255)}
        distortion={'Camera1.'+key for key in ('k1','k2','p1','p2','k3')}
        required=(numeric-distortion)|{'File.version','Camera.type'}
        seen=set();values={}
        for line in text.splitlines():
            line=line.split('#',1)[0].strip()
            if not line or line=='%YAML:1.0':continue
            key,sep,value=line.partition(':');value=value.strip()
            if not sep or key in seen:raise ValueError('flat unique camera settings required')
            seen.add(key)
            values[key]=value
            if key in ('File.version','Camera.type'):
                if value not in ({'"1.0"'} if key=='File.version' else {'"PinHole"'}):
                    raise ValueError('version1 pinhole settings required')
            elif key not in numeric or not re.fullmatch(r'-?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?',value):
                raise ValueError('unsupported camera settings field')
            elif not math.isfinite(float(value)) or abs(float(value))>1000000:
                raise ValueError('bounded finite camera settings required')
        if not required<=seen:
            raise ValueError('complete version1 pinhole/ORB/viewer settings required')
        present=seen & distortion
        if present and present not in (distortion,distortion-{'Camera1.k3'}):
            raise ValueError('distortion requires k1/k2/p1/p2 and optional k3')
        # OpenCV FileStorage types matter: upstream exits on int/real mismatch.
        for key in seen & numeric:
            value=values[key]
            if key in integer_bounds:
                low,high=integer_bounds[key]
                if not re.fullmatch(r'0|[1-9][0-9]*',value) or not low<=int(value)<=high:
                    raise ValueError('bounded integer camera/ORB field required')
            elif '.' not in value or not re.fullmatch(r'-?(?:[0-9]+\.[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?',value):
                raise ValueError('explicit decimal real camera/ORB/viewer field required')
        if (not 1.01<=float(values['ORBextractor.scaleFactor'])<=2.0 or
                int(values['ORBextractor.minThFAST'])>int(values['ORBextractor.iniThFAST'])):
            raise ValueError('invalid bounded ORB pyramid/threshold settings')

        width=float(values['Camera.width']);height=float(values['Camera.height'])
        if (not width.is_integer() or not height.is_integer() or not 32<=width<=1920 or not 32<=height<=1080
                or not 0<float(values['Camera1.fx'])<=1000000 or not 0<float(values['Camera1.fy'])<=1000000
                or not 0<=float(values['Camera1.cx'])<width or not 0<=float(values['Camera1.cy'])<height
                or not 0<float(values['Camera.fps'])<=240):
            raise ValueError('invalid bounded pinhole calibration')
        self._shape=(int(height),int(width))
        self._lock=threading.RLock();self._handle=None
        lib=ctypes.CDLL(library_path);self._lib=lib
        lib.gamst_orb_abi.argtypes=[];lib.gamst_orb_abi.restype=ctypes.c_int
        lib.gamst_orb_source.argtypes=[];lib.gamst_orb_source.restype=ctypes.c_char_p
        if lib.gamst_orb_abi()!=1 or lib.gamst_orb_source()!=SOURCE_REVISION.encode():
            raise ValueError('native bridge ABI/source mismatch')
        lib.gamst_orb_open.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_ulong];lib.gamst_orb_open.restype=ctypes.c_void_p
        lib.gamst_orb_track.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_double,
                                    ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int)]
        lib.gamst_orb_track.restype=ctypes.c_int
        for name in ('gamst_orb_reset','gamst_orb_close'):
            fn=getattr(lib,name);fn.argtypes=[ctypes.c_void_p];fn.restype=ctypes.c_int
        self._handle=lib.gamst_orb_open(os.fsencode(vocabulary_path),os.fsencode(settings_path),max_frames)
        if not self._handle:raise RuntimeError('native construction failed; restart owned worker')

    def track(self,gray,seconds):
        import numpy as np
        with self._lock:
            if self._handle is None:raise RuntimeError('native session closed')
            if (type(gray) is not np.ndarray or gray.dtype!=np.uint8 or gray.ndim!=2 or
                    gray.shape!=self._shape or
                    type(seconds) not in (int,float) or not np.isfinite(seconds) or not 0<=seconds<=10**12):
                raise ValueError('bounded uint8 grayscale and finite nonnegative seconds required')
            pixels=np.ascontiguousarray(gray);matrix=(ctypes.c_float*16)();state=ctypes.c_int();changed=ctypes.c_int()
            code=self._lib.gamst_orb_track(self._handle,pixels.ctypes.data,*pixels.shape,float(seconds),matrix,ctypes.byref(state),ctypes.byref(changed))
            if code==-2:raise RuntimeError('native lifetime frame budget exhausted; restart owned worker')
            if code!=0:raise RuntimeError('native tracking failed')
            if state.value not in (-1,0,1,2,3,4,5) or changed.value not in (0,1):
                raise ValueError('invalid native tracking state')
            pose=None
            if state.value==2:
                pose=np.array(list(matrix),dtype=float).reshape(4,4)
                rotation=pose[:3,:3]
                if (not np.isfinite(pose).all() or np.max(np.abs(pose))>1000000 or
                        not np.allclose(pose[3],[0,0,0,1],rtol=0,atol=1e-5) or
                        not np.allclose(rotation@rotation.T,np.eye(3),rtol=0,atol=1e-3) or
                        abs(np.linalg.det(rotation)-1)>1e-3):
                    raise ValueError('native pose is not a bounded rigid transform')
                pose=pose.tolist()
            return dict(state=state.value,map_changed=bool(changed.value),world_to_camera=pose)

    def reset(self):
        with self._lock:
            if self._handle is None:raise RuntimeError('native session closed')
            if self._lib.gamst_orb_reset(self._handle)!=0:raise RuntimeError('native reset failed')

    def close(self):
        with self._lock:
            handle=self._handle;self._handle=None
            if handle is not None and self._lib.gamst_orb_close(handle)!=0:
                raise RuntimeError('native shutdown failed; reap owned worker')
