"""Compile the real ABI bridge against a test double, NOT actual SLAM inference."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
try:
    import numpy as np
except ImportError:np=None
from humansense.providers.orbslam_native import ORBSLAMNative

ROOT=Path(__file__).resolve().parents[1]
HEADER='''#pragma once
#include <stdexcept>
#define CV_8UC1 0
namespace cv { struct Mat { unsigned char* data; Mat(int,int,int,void* p):data((unsigned char*)p){} }; }
struct Matrix { double t; float operator()(int r,int c) const { return r==c?1.f:(r==0&&c==3?t:0.f); } };
struct Pose { double t; Matrix matrix()const{return Matrix{t};} };
namespace ORB_SLAM3 { class System { int state=0;bool changed=false; public:
enum eSensor {MONOCULAR};System(const char*,const char*,eSensor,bool){}
Pose TrackMonocular(cv::Mat im,double t){if(im.data[0]==254)throw std::runtime_error("test");state=im.data[0]?2:1;changed=im.data[0]==2;return Pose{t};}
int GetTrackingState(){return state;}bool MapChanged(){return changed;}void Reset(){state=0;}void Shutdown(){}
}; }
'''
SETTINGS='''%YAML:1.0
File.version: "1.0"
Camera.type: "PinHole"
Camera1.fx: 100.0
Camera1.fy: 100.0
Camera1.cx: 16.0
Camera1.cy: 16.0
Camera.width: 96
Camera.height: 96
Camera.fps: 30
Camera.RGB: 0
ORBextractor.nFeatures: 1000
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 1
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0
'''


@unittest.skipIf(np is None or shutil.which('c++') is None,'requires numpy and C++ compiler')
class Native(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT);self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name);(self.path/'System.h').write_text(HEADER)
        self.lib=self.path/'bridge.so'
        subprocess.run(['c++','-std=c++14','-shared','-fPIC','-pthread','-I'+str(self.path),str(ROOT/'native/orbslam3/bridge.cpp'),'-o',str(self.lib)],check=True,capture_output=True)
        self.vocab=self.path/'vocabulary.txt';self.vocab.write_text('test double, not a real vocabulary')
        self.settings=self.path/'camera.yaml';self.settings.write_text(SETTINGS)
        self.env=patch.dict(os.environ,{'GAMST_PROVIDER_WORKER':'1'});self.env.start();self.addCleanup(self.env.stop)

    def args(self):
        digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        return dict(library_path=str(self.lib),vocabulary_path=str(self.vocab),settings_path=str(self.settings),
                    library_sha256=digest(self.lib),vocabulary_sha256=digest(self.vocab),settings_sha256=digest(self.settings))

    def open(self,**kwargs):
        n=ORBSLAMNative(**dict(self.args(),**kwargs));self.addCleanup(n.close);return n

    def test_abi_tracking_lost_reset_lifetime_bound_and_single_session(self):
        n=self.open(max_frames=3);image=np.zeros((96,96),dtype=np.uint8)
        self.assertIsNone(n.track(image,0)['world_to_camera'])
        image[:]=2;result=n.track(image,1)
        self.assertTrue(result['map_changed']);self.assertEqual(result['world_to_camera'][0][3],1)
        n.reset();self.assertEqual(n.track(image,0)['state'],2)
        with self.assertRaisesRegex(RuntimeError,'budget exhausted'):n.track(image,2)
        n.close();n.close()
        with self.assertRaises(RuntimeError):n.track(image,3)
        with self.assertRaises(RuntimeError):self.open()

    def test_hash_guard_worker_boundary_and_settings_rejection(self):
        args=self.args();args['library_sha256']='0'*64
        with self.assertRaises(ValueError):ORBSLAMNative(**args)
        with patch.dict(os.environ,{'GAMST_PROVIDER_WORKER':'0'}):
            with self.assertRaises(RuntimeError):ORBSLAMNative(**self.args())
        for extra in ('System.SaveAtlasFromFile: "outside"','"System.SaveAtlasFromFile": "outside"',
                      'Camera.width: 40','Camera1.k1: .nan','Camera1.k1: 1e999','garbage: 1'):
            self.settings.write_text(SETTINGS+extra+'\n')
            with self.assertRaises(ValueError):ORBSLAMNative(**self.args())
        self.settings.write_text(SETTINGS);self.open()

    def test_required_types_ranges_and_distortion_are_checked_before_native_open(self):
        replacements=(('Camera.RGB: 0\n',''),('Camera.fps: 30','Camera.fps: 30.0'),
                      ('Camera1.fx: 100.0','Camera1.fx: 100'),
                      ('Camera.width: 96','Camera.width: 096'),
                      ('Camera.width: 96','Camera.width: 32'),
                      ('Camera.height: 96','Camera.height: 400'),
                      ('ORBextractor.nLevels: 1','ORBextractor.nLevels: 8'),
                      ('ORBextractor.nLevels: 1','ORBextractor.nLevels: 9'),
                      ('ORBextractor.scaleFactor: 1.2','ORBextractor.scaleFactor: 1.0'),
                      ('ORBextractor.minThFAST: 7','ORBextractor.minThFAST: 21'),
                      ('Viewer.ViewpointF: 500.0\n',''))
        for old,new in replacements:
            with self.subTest(old=old):
                self.settings.write_text(SETTINGS.replace(old,new))
                with self.assertRaises(ValueError):ORBSLAMNative(**self.args())
        self.settings.write_text(SETTINGS+'Camera1.k1: 0.0\n')
        with self.assertRaises(ValueError):ORBSLAMNative(**self.args())
        self.settings.write_text(SETTINGS+'Camera1.k1: 0.0\nCamera1.k2: 0.0\nCamera1.p1: 0.0\nCamera1.p2: 0.0\n')
        self.open()

    def test_malformed_and_native_error_are_not_pose(self):
        n=self.open();image=np.ones((96,96),dtype=np.uint8)
        for bad in (image.astype(float),image[:1],image[:,:,None]):
            with self.assertRaises(ValueError):n.track(bad,0)
        n.track(image[:,::-1],0)
        with self.assertRaises(RuntimeError):n.track(image,0)
        with self.assertRaises(ValueError):n.track(image,float('nan'))
        image[:]=254
        with self.assertRaises(RuntimeError):n.track(image,1)
        with self.assertRaises(RuntimeError):n.reset()
