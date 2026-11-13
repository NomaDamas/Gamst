import asyncio
import dataclasses
import json
import math
import struct
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

from humansense import HumanSense, ProcessProvider, SensorMetadata, SensorStreams
from humansense.mcp import MCPServer
from humansense.plugins import InputFrame
from humansense.providers.radar import HEADER, MAGIC, TIRadar, TIRadarPackets, parse_ti_radar

VERSION = 0x03060000  # Synthetic declared version; no hardware run is claimed.
CONFIG = dict(sdk_version=VERSION, coordinate_frame='radar_local', calibration_id='cal1')


def packet(tlvs=(), *, number=0, detected=0, version=VERSION, platform=0xa6843, subframe=0):
    body = b''.join(struct.pack('<II', kind,len(value))+value for kind,value in tlvs)
    length = (40+len(body)+31)//32*32
    return HEADER.pack(MAGIC,version,length,platform,number,number,detected,len(tlvs),subframe)+body+b'\xa5'*(length-40-len(body))


def target(tid=3):
    return struct.pack('<I27f',tid,3,4,0,0,-2,0,0,0,0,*([0.0]*16),1,.8)


def frame(i=0, *, data=None, number=None, ts=None):
    return InputFrame(packet([(1010,target())],number=i if number is None else number) if data is None else data,
        'radar',1000+i*50 if ts is None else ts,
        sensor=SensorMetadata('radar1','radar',i,coordinate_frame='radar_local',calibration_id='cal1',stream_epoch='e1'))


class Radar(unittest.TestCase):
    def test_cartesian_target_height_presence_and_missing_are_distinct(self):
        raw = packet([(1,struct.pack('<4f',3,4,0,-2)),(7,struct.pack('<hh',125,43)),
            (1010,target()),(1012,struct.pack('<I2f',3,1.8,.1)),(1021,struct.pack('<I',1)),
            (1011,bytes([3,253,254,255])),(999,b'extension')],detected=1)
        result = parse_ti_radar(raw,sdk_version=VERSION)
        point = result['point_cloud']['nearest_points'][0]
        self.assertEqual(point['range_m'],5)
        self.assertAlmostEqual(point['azimuth_rad'],math.atan2(3,4))
        self.assertEqual(point['snr_db'],12.5)
        track = result['tracks'][0]
        self.assertEqual(track['position_m'],[3,4,0]); self.assertEqual(track['speed_m_s'],2)
        self.assertEqual(track['range_to_frame_origin_m'],5)
        self.assertAlmostEqual(result['target_heights'][0]['max_z_m'],1.8,places=6)
        self.assertEqual(result['firmware_presence'],1)
        self.assertEqual(result['target_index_summary']['point_count'],4)  # Not current count1.
        self.assertFalse(result['target_index_summary']['joined_to_current_points'])
        self.assertEqual(result['ignored_tlv_types'],[999])
        missing = parse_ti_radar(packet(),sdk_version=VERSION)
        self.assertIsNone(missing['tracks']); self.assertIsNone(missing['firmware_presence'])
        self.assertFalse(missing['point_cloud']['decoded'])
        empty = parse_ti_radar(packet([(1,b''),(1010,b'')]),sdk_version=VERSION)
        self.assertEqual(empty['tracks'],[]); self.assertIsNone(empty['point_cloud']['minimum_range_m'])

    def test_compressed_signedness_scales_and_nearest_selection(self):
        units = struct.pack('<5f',.01,.01,.1,.01,.1)
        # uint16 values above32767 must remain positive, signed Doppler negative.
        data = units+struct.pack('<bbhHH',-10,20,-32000,40000,50000)
        p = parse_ti_radar(packet([(1020,data)],detected=1),sdk_version=VERSION)['point_cloud']['nearest_points'][0]
        self.assertAlmostEqual(p['range_m'],400,places=4)
        self.assertAlmostEqual(p['radial_velocity_m_s'],-3200,places=3)
        self.assertAlmostEqual(p['snr_db'],5000,places=3)
        data = b''.join(struct.pack('<4f',0,i,0,-i) for i in range(40,0,-1))
        cloud = parse_ti_radar(packet([(1,data)],detected=40),sdk_version=VERSION)['point_cloud']
        self.assertEqual([p['range_m'] for p in cloud['nearest_points']],list(range(1,17)))
        self.assertEqual(cloud['point_count'],40); self.assertEqual(cloud['maximum_range_m'],40)
        self.assertEqual(cloud['mean_radial_velocity_m_s'],-20.5)

    def test_bad_lengths_counts_encodings_and_values(self):
        good = packet([(1010,target())])
        bad = [b'',good[:-1],good+bytes(32),packet(version=1),packet(platform=0xa1642),packet(subframe=1),
            packet([(1010,target()+b'x')]),packet([(1010,target(250))]),packet([(1010,target()*2)]),
            packet([(1021,b'')]),packet([(1021,struct.pack('<I',2))]),packet([(1011,b'\xfa')]),
            packet([(1012,struct.pack('<I2f',1,0,1))]),packet([(1,b'')],detected=1),
            packet([(1,struct.pack('<4f',float('nan'),0,0,0))],detected=1),
            packet([(1020,struct.pack('<5f',1,1,0,1,1))]),packet([(7,b'')]),
            packet([(1,b''),(1020,struct.pack('<5f',1,1,1,1,1))]),
            packet([(1010,b''),(1010,b'')]),packet(detected=513),packet([(i,b'') for i in range(33)])]
        corrupt = bytearray(good); struct.pack_into('<I',corrupt,44,65536); bad.append(bytes(corrupt))
        for raw in bad:
            with self.assertRaises(ValueError,msg=repr(raw[:64])): parse_ti_radar(raw,sdk_version=VERSION)

    def test_incremental_framing_every_split_noise_resync_and_bounds(self):
        raw = packet([(1010,target())])
        for split in range(len(raw)+1):
            f = TIRadarPackets()
            self.assertEqual(f.feed(raw[:split])+f.feed(raw[split:]),[raw])
            self.assertEqual(f.buffered_bytes,0)
        f = TIRadarPackets(); invalid = bytearray(raw); struct.pack_into('<I',invalid,12,0xffffffff)
        out = f.feed(b'log\r\n'+bytes(invalid)+raw+raw[:17])
        self.assertEqual(out,[raw]); self.assertEqual(f.buffered_bytes,17)
        self.assertEqual(f.feed(raw[17:]),[raw]); self.assertGreater(f.discarded_bytes,0)
        for _ in range(100): f.feed(b'x'*4096); self.assertLessEqual(f.buffered_bytes,7)
        with self.assertRaises(ValueError): f.feed(b'x'*4097)
        f.reset(); self.assertEqual(f.buffered_bytes,0); self.assertEqual(f.discarded_bytes,0)
        for _ in range(64): self.assertEqual(len(f.feed(packet()*64)),64)

    def test_continuity_wrap_reset_and_errors_clear_sessions(self):
        p = TIRadar(**CONFIG)
        a = p.process(frame(number=0xffffffff))['scene']
        b = p.process(frame(1,number=0))['scene']
        self.assertTrue(b['continuity']); self.assertEqual(a['session_id'],b['session_id'])
        c = p.process(frame(2,number=2))['scene']
        self.assertFalse(c['continuity']); self.assertNotEqual(b['session_id'],c['session_id'])
        with self.assertRaises(ValueError): p.process(frame(3,number=2))
        d = p.process(frame(4))['scene']; self.assertFalse(d['continuity'])
        e = p.process(frame(5,ts=9999))['scene']; self.assertFalse(e['continuity'])
        f = p.process(dataclasses.replace(frame(6,ts=10000),subject_id='other'))['scene']
        self.assertFalse(f['continuity'])
        p.reset(); self.assertFalse(p.process(frame(0))['scene']['continuity'])
        p.close(); p.reset()
        with self.assertRaises(RuntimeError): p.process(frame(1))

    def test_metadata_clock_and_dependency_light_discovery(self):
        p = TIRadar(**CONFIG)
        for f in (dataclasses.replace(frame(),timestamp_ms=True),dataclasses.replace(frame(),modality='csi'),
                  dataclasses.replace(frame(),sensor=dataclasses.replace(frame().sensor,calibration_id='wrong'))):
            with self.assertRaises(ValueError): p.process(f)
        p.process(frame(ts=2**53-2))
        self.assertTrue(p.process(frame(1,ts=2**53-1))['scene']['continuity'])
        for changes in ({'sdk_version':True},{'max_gap_ms':0},{'coordinate_frame':''}):
            with self.assertRaises(ValueError): TIRadar(**dict(CONFIG,**changes))
        code = "import sys;from humansense import ProcessProvider;ProcessProvider.capability('ti.radar',"+repr(CONFIG)+");assert not {'numpy','torch','serial'} & sys.modules.keys()"
        result = subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_maximum_context_is_bounded_and_has_no_inferred_identity(self):
        points = b''.join(struct.pack('<4f',i,0,0,-i) for i in range(512))
        data = packet([(1,points),(1010,b''.join(target(i) for i in range(32)))],detected=512)
        scene = TIRadar(**CONFIG).process(frame(data=data))['scene']
        self.assertLess(len(json.dumps(scene)),65536)
        self.assertEqual(len(scene['tracks']),32)
        self.assertEqual(scene['point_cloud']['retained_point_count'],16)
        for key in ('hardware_verified','calibration_verified','clock_mapping_verified','identity_verified','through_wall_verified'):
            self.assertIs(scene[key],False)


class Managed(unittest.IsolatedAsyncioTestCase):
    async def test_actual_parser_in_owned_worker_to_mcp(self):
        children = []; real_spawn = asyncio.create_subprocess_exec
        async def spawn(*args,**kwargs):
            child = await real_spawn(*args,**kwargs); children.append(child); return child
        try:
            with patch('asyncio.create_subprocess_exec',spawn):
                async with HumanSense(consent=True) as human:
                    streams = SensorStreams(human)
                    spec = ProcessProvider.capability('ti.radar',CONFIG)
                    async def create(): return await ProcessProvider.open(sys.executable,'ti.radar',CONFIG)
                    streams.register(spec,create)
                    await streams.open('radar1','radar',[spec.id])
                    start = int(time.time()*1000)-200
                    for i in range(3):
                        f = frame(i,ts=start+i*50)
                        f = dataclasses.replace(f,sensor=dataclasses.replace(f.sensor,stream_epoch=None))
                        await streams.submit(f); await streams.drain()
                    output = await MCPServer(human).call_tool('get_observations',{'capability_id':spec.id})
                    scenes = [v['observation']['value']['scene'] for v in output['items']]
                    self.assertTrue(scenes)
                    self.assertTrue(any(s['continuity'] for s in scenes))
                    self.assertTrue(all(s['tracks'][0]['position_m']==[3,4,0] for s in scenes))
                    await streams.finish('radar1','radar')
            self.assertTrue(children and all(c.returncode is not None for c in children))
        finally:
            for child in children:
                if child.returncode is None: child.kill(); await child.wait()


if __name__ == '__main__': unittest.main()
