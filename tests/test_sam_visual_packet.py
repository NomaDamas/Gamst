import hashlib
import json
import struct
import unittest
from dataclasses import replace

from humansense.capabilities import SensorMetadata
from humansense.plugins import InputFrame
from humansense.sam_visual_packet import visual_frame, unpack_visual_packet, MAGIC

try:
    import numpy as np
except ImportError:
    np=None


@unittest.skipIf(np is None,'optional numpy missing')
class VisualPacketTests(unittest.TestCase):
    def setUp(self):
        self.audio=InputFrame(data=np.full(48000,.125,dtype=np.float32),modality='audio',
            timestamp_ms=2000,subject_id='subject',sample_rate=48000,
            sensor=SensorMetadata('microphone','mic0',1,channels=('mono',),stream_epoch='a1'))
        self.rgb=np.arange(2*3*16*16,dtype=np.uint8).reshape(2,3,16,16)
        self.masks=np.zeros((2,1,16,16),dtype=np.uint8)
        self.masks[:,:,:,8:]=1
        self.kw=dict(visual_sensor=SensorMetadata('camera','cam0',0,channels=('R','G','B'),
            stream_epoch='v1',coordinate_frame='optical',calibration_id='rig1',capture_sequence=10),
            intervals_us=[[990000,1000000],[1090000,1100000]],sequences=[10,11],
            track_id='track7',segment_id='segment1',mask_origin={'producer_id':'manual','revision':'1'},
            clock=dict(audio_clock_id='a',camera_clock_id='v',alignment_id='cal1',
                       camera_to_audio_offset_us=0,clock_error_us=0,max_age_us=150000))

    def packet(self):
        return visual_frame(self.audio,self.rgb,self.masks,**self.kw)

    def rewrite(self,data,change):
        n,=struct.unpack('>I',data[8:12])
        h=json.loads(data[12:12+n]); change(h)
        raw=json.dumps(h,separators=(',',':')).encode()
        return MAGIC+struct.pack('>I',len(raw))+raw+data[12+n:]

    def test_owned_roundtrip_preserves_pcm_rgb_mask_and_capture(self):
        packet=self.packet()
        h,pcm,rgb,masks=unpack_visual_packet(packet.data)
        self.assertEqual(packet.timestamp_ms,self.audio.timestamp_ms)
        self.assertEqual(pcm,self.audio.data.tobytes())
        self.assertEqual(rgb,self.rgb.tobytes())
        self.assertEqual(masks,self.masks.tobytes())
        self.assertEqual(h['frames'][1]['capture_sequence'],11)
        self.assertEqual(h['audio_sha256'],hashlib.sha256(pcm).hexdigest())
        self.rgb[:]=0; self.masks[:]=1; self.audio.data[:]=0
        self.assertEqual(unpack_visual_packet(packet.data),(h,pcm,rgb,masks))

    def test_hash_and_length_tampering_rejected(self):
        original=self.packet().data
        for value in (original[:-1],original+b'x',original[:-1]+bytes([original[-1]^1]),
                      self.rewrite(original,lambda h:h.update(audio_sha256='0'*64))):
            with self.assertRaises(ValueError): unpack_visual_packet(value)

    def test_closed_fields_and_no_independent_audio_clock(self):
        original=self.packet().data
        changes=[lambda h:h.update(audio_end_us=2000001),
                 lambda h:h['audio']['metadata']['sensor'].update(extra='value'),
                 lambda h:h['frames'][0].update(status='occluded'),
                 lambda h:h['frames'][1].update(capture_sequence=10),
                 lambda h:h['clock'].update(clock_error_us=True),
                 lambda h:h.update(width=True)]
        for change in changes:
            with self.subTest(change=change),self.assertRaises(ValueError):
                unpack_visual_packet(self.rewrite(original,change))

    def test_duplicate_nested_key_rejected(self):
        original=self.packet().data
        n,=struct.unpack('>I',original[8:12]); raw=original[12:12+n]
        raw=raw.replace(b'"clock_error_us":0',b'"clock_error_us":0,"clock_error_us":0')
        with self.assertRaises(ValueError):
            unpack_visual_packet(MAGIC+struct.pack('>I',len(raw))+raw+original[12+n:])

    def test_masks_and_channel_order_fail_closed(self):
        for value in (2,1):
            self.masks[:]=value
            with self.assertRaises(ValueError): self.packet()
        self.masks[:]=0
        self.kw['visual_sensor']=replace(self.kw['visual_sensor'],channels=('B','G','R'))
        with self.assertRaises(ValueError): self.packet()

    def test_plain_dtype_and_geometry_only(self):
        for rgb in (self.rgb.astype(np.float32),self.rgb.transpose(0,2,3,1),
                    np.ma.array(self.rgb)):
            with self.assertRaises(ValueError): visual_frame(self.audio,rgb,self.masks,**self.kw)
        with self.assertRaises(ValueError):
            visual_frame(self.audio,self.rgb,self.masks.astype(bool),**self.kw)

    def test_parent_binding_exact_policy_outer_capture_and_runtime_epoch(self):
        from humansense.provider_wire import encode_frame
        from humansense.sam_visual_packet import visual_policy,visual_request_binding
        packet=self.packet(); metadata,*_=unpack_visual_packet(packet.data)
        policy=visual_policy(metadata)
        runtime=replace(packet,sensor=replace(packet.sensor,sequence=999,
                        capture_sequence=packet.sensor.sequence,stream_epoch='runtime2'))
        outer,payload=encode_frame(runtime)
        binding=visual_request_binding(outer,payload,policy=policy,generation=5,config_id='cfg')
        self.assertEqual(binding['input_sha256'],hashlib.sha256(payload).hexdigest())
        self.assertEqual(binding['audio_input_sha256'],metadata['audio_sha256'])
        self.assertEqual(binding['original_audio_epoch'],'a1')
        self.assertEqual(binding['sensor']['stream_epoch'],'runtime2')
        for changed in (replace(runtime,timestamp_ms=2001),
                        replace(runtime,subject_id='different'),
                        replace(runtime,sensor=replace(runtime.sensor,capture_sequence=2))):
            with self.assertRaises(ValueError):
                visual_request_binding(*encode_frame(changed),policy=policy,generation=5,config_id='cfg')
        policy['clock']['camera_to_audio_offset_us']=1
        with self.assertRaises(ValueError):
            visual_request_binding(outer,payload,policy=policy,generation=5,config_id='cfg')
