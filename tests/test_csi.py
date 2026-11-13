import asyncio
import csv
import dataclasses
import io
import json
import math
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

from humansense import HumanSense, ProcessProvider, SensorMetadata, SensorStreams
from humansense.mcp import MCPServer
from humansense.plugins import InputFrame
from humansense.providers.csi import COLUMNS, ESP32CSI, parse_esp32_csi

MAC='1a:00:00:00:00:00'
CONFIG=dict(transmitter_mac=MAC,clock_calibration_id='clock1')


def row(index=0, clock=0, *, iq=None, **updates):
    values=dict(zip(COLUMNS,['CSI_DATA',index,MAC,-45,12,0,0,0,0,1,0,0,0,0,-95,0,11,0,clock,0,100,0,128,0,None]))
    values.update(updates)
    if iq is None: iq=[3,4]*64
    values['data']=json.dumps(iq,separators=(',',':'))
    out=io.StringIO();csv.writer(out,lineterminator='\n').writerow([values[k] for k in COLUMNS]);return out.getvalue().encode()


def frame(index=0, *, clock=None, ts=None, data=None, **kwargs):
    return InputFrame(data if data is not None else row(index,index*20000 if clock is None else clock,**kwargs),
        'csi',1000+index*20 if ts is None else ts,
        sensor=SensorMetadata('receiver','csi',index,calibration_id='clock1',stream_epoch='e1'))


class CSI(unittest.TestCase):
    def test_parser_complex_order_carrier_mask_and_signed_wrap(self):
        fields,indices,amplitudes,phases=parse_esp32_csi(row(-1,-2))
        self.assertEqual(fields['id'],2**32-1);self.assertEqual(fields['local_timestamp'],2**32-2)
        self.assertEqual(len(indices),48)
        self.assertEqual(set(indices),set(range(-26,27))-{0,-21,-7,7,21})
        self.assertTrue(all(v==5 for v in amplitudes))
        self.assertTrue(all(abs(v-math.atan2(3,4))<1e-12 for v in phases))
        _,masked,_,_=parse_esp32_csi(row(first_word=1))
        self.assertEqual(len(masked),47);self.assertNotIn(1,masked)
        _,ht,amps,_=parse_esp32_csi(row(sig_mode=1,rx_format=1,len=256,iq=[3,4]*64+[100,100]*64))
        self.assertEqual(ht,indices);self.assertEqual(amps,amplitudes)

    def test_reject_wrong_chips_modes_samples_and_malformed_rows(self):
        bad=[b'',b'x'*4097,row()+row(),b'log '+row(),row().replace(b'CSI_DATA',b'type'),
             row(bandwidth=1),row(stbc=1),row(secondary_channel=1),row(rx_format=1),
             row(iq=[True]*128),row(iq=[128]*128),row(iq=[0]*126),row(iq=[[1]]*128),
             row(mac='not-a-mac'),row(channel=15),row(local_timestamp=2**32),
             row().replace(b'CSI_DATA,',b'CSI_DATA,1,'),b'CSI_DATA,"unterminated']
        for value in bad:
            with self.assertRaises(ValueError,msg=repr(value[:80])):parse_esp32_csi(value)

    def test_normalized_gain_invariance_variation_and_no_presence_claim(self):
        provider=ESP32CSI(**CONFIG)
        for i in range(61):
            out=provider.process(frame(i,iq=([3,4] if i%2 else [6,8])*64))['channel_response']
        self.assertTrue(out['ready']);self.assertEqual(out['normalized_amplitude_variation_rms'],0)
        self.assertFalse(out['threshold_exceeded']);self.assertIsNone(out['presence'])
        self.assertFalse(out['through_wall_object_verified']);self.assertFalse(out['hardware_verified'])
        provider.reset()
        for i in range(61):
            iq=[3,4]*64
            if i%2:
                for k in range(2,26):iq[k*2:k*2+2]=[12,16]
            out=provider.process(frame(i,iq=iq))['channel_response']
        self.assertTrue(out['ready']);self.assertTrue(out['threshold_exceeded'])
        self.assertGreater(out['normalized_amplitude_variation_rms'],.05)
        self.assertLessEqual(out['sample_count'],256)

    def test_uint32_rollover_preserves_continuity(self):
        provider=ESP32CSI(**CONFIG)
        for i in range(61):
            rawid=(0xfffffffd+i)&0xffffffff;rawclock=(0xffff0000+i*20000)&0xffffffff
            signed=lambda v:v-2**32 if v>=2**31 else v
            value=provider.process(frame(i,data=row(signed(rawid),signed(rawclock))))['channel_response']
            self.assertIsNone(value['reset_reason'])
        self.assertTrue(value['ready'])

    def test_gap_owner_mask_clock_and_zero_energy_reset(self):
        def warmed():
            p=ESP32CSI(**CONFIG)
            p.process(frame(0));p.process(frame(1));return p
        cases=[(frame(3),'capture_gap'),(frame(2,clock=10000),'device_or_transmitter_restarted'),
               (frame(2,ts=1100),'capture_clock_disagreement'),(frame(2,channel=10),'ownership_or_radio_layout_changed'),
               (frame(2,first_word=1),'ownership_or_radio_layout_changed'),
               (frame(2,iq=[0]*128),'zero_channel_energy')]
        for value,reason in cases:
            p=warmed();out=p.process(value)['channel_response']
            self.assertEqual(out['reset_reason'],reason);self.assertFalse(out['ready'])
        p=warmed()
        for bad in (frame(1),frame(2,data=row(1,40000)),frame(2,clock=20000),
                    frame(2,mac='aa:00:00:00:00:00'),dataclasses.replace(frame(2),timestamp_ms=True)):
            with self.assertRaises(ValueError):p.process(bad)
            self.assertEqual(len(p._history),0)
            p=warmed()
        p.close();p.reset()
        with self.assertRaises(RuntimeError):p.process(frame(0))

    def test_sample_bound_and_timestamp_precision(self):
        p=ESP32CSI(**CONFIG)
        for i in range(300):
            out=p.process(frame(i,clock=i*1000,ts=2**53-1000+i))['channel_response']
        self.assertEqual(out['sample_count'],256)
        self.assertTrue(out['sample_limit_ever_hit']);self.assertFalse(out['ready'])
        self.assertEqual(out['effective_span_ms'],255.)
        self.assertLess(len(json.dumps(out)),16000)
        with self.assertRaises(ValueError):p.process(frame(301,ts=2**53))

    def test_accumulated_clock_drift_cannot_appear_ready(self):
        p=ESP32CSI(**dict(CONFIG,window_ms=100,max_gap_ms=50))
        reasons=[]
        for i in range(130):
            out=p.process(frame(i,clock=i*1000,ts=1000))['channel_response']
            self.assertFalse(out['ready'])
            reasons.append(out['reset_reason'])
        self.assertIn('capture_clock_disagreement',reasons)

    def test_profile_stdlib_and_config(self):
        code="import sys; from humansense import ProcessProvider; ProcessProvider.capability('esp32.csi',{'transmitter_mac':'1a:00:00:00:00:00','clock_calibration_id':'c'}); assert not {'numpy','torch','serial'} & sys.modules.keys()"
        result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        for changes in ({'window_ms':True},{'variation_threshold':float('nan')},{'max_gap_ms':2000}):
            config=dict(CONFIG);config.update(changes)
            with self.assertRaises(ValueError):ESP32CSI(**config)


class Managed(unittest.IsolatedAsyncioTestCase):
    async def test_real_decoder_dsp_owned_worker_to_mcp_and_shutdown(self):
        children=[];real_spawn=asyncio.create_subprocess_exec
        async def spawn(*args,**kwargs):
            child=await real_spawn(*args,**kwargs);children.append(child);return child
        config=dict(CONFIG,window_ms=100,max_gap_ms=50)
        try:
            with patch('asyncio.create_subprocess_exec',spawn):
                async with HumanSense(consent=True) as human:
                    streams=SensorStreams(human)
                    spec=ProcessProvider.capability('esp32.csi',config)
                    async def create():return await ProcessProvider.open(sys.executable,'esp32.csi',config)
                    streams.register(spec,create)
                    await streams.open('receiver','csi',[spec.id])
                    start=int(time.time()*1000)-300
                    for i in range(12):
                        f=frame(i,clock=i*10000,ts=start+i*10)
                        f=dataclasses.replace(f,sensor=dataclasses.replace(f.sensor,stream_epoch=None))
                        await streams.submit(f);await streams.drain()
                    output=await MCPServer(human).call_tool('get_observations',{'capability_id':spec.id})
                    values=[v['observation']['value']['channel_response'] for v in output['items']]
                    self.assertTrue(any(v['ready'] for v in values))
                    self.assertTrue(all(v['motion_verified'] is False for v in values))
                    await streams.finish('receiver','csi')
            self.assertTrue(children and all(c.returncode is not None for c in children))
        finally:
            for child in children:
                if child.returncode is None:child.kill();await child.wait()


if __name__=='__main__':unittest.main()
