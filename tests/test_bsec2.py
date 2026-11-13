import asyncio
from dataclasses import replace
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

from humansense import HumanSense, ProcessProvider, SensorMetadata, SensorStreams
from humansense.plugins import InputFrame
from humansense.providers.bsec2 import HEADER,ROW,MAGIC,BoschBSEC2,BSEC2Packets,decode_bsec2

CONFIG=dict(config_sha256='a'*64,bsec_version='2.6.1.0',calibration_id='clock1',sensor_origin_ns='0',
            capture_origin_ms=1000,classes=['clean','coffee'])


def packet(i=0,rows=None,number=None):
    ns=i*3000000000
    if rows is None:rows=[(ns,25.,6,0,1),(ns,100000.,7,0,1),(ns,50.,8,0,1),(ns,12000.,9,0,1),
                         (ns,.25,22,3,1),(ns,.75,23,3,1)]
    value=HEADER.pack(MAGIC,2,6,1,0,len(rows),i if number is None else number,bytes.fromhex('a'*64))
    value+=b''.join(ROW.pack(*row) for row in rows)
    return value+struct.pack('<I',zlib.crc32(value))


def frame(i=0,**kwargs):
    return InputFrame(packet(i,**kwargs),'environment',1000+i*3000,
        sensor=SensorMetadata('env','environment',i,calibration_id='clock1',stream_epoch='e1'))


class BSEC(unittest.TestCase):
    def test_units_estimates_and_complete_classification(self):
        p=BoschBSEC2(**CONFIG);value=p.process(frame())
        self.assertEqual([r['unit'] for r in value['measurements']],['degC','Pa','percent','ohm'])
        self.assertEqual(value['classification']['status'],'prediction_available')
        self.assertEqual(value['classification']['candidates'],[dict(label='clean',score=.25),dict(label='coffee',score=.75)])
        self.assertFalse(value['summary']['configuration_attested'])
        value=p.process(frame(1,rows=[(3000000000,800.,3,3,1)]))
        self.assertEqual(value['measurements'],[]);self.assertEqual(value['estimates'][0]['unit'],'ppm_equivalent')
        self.assertEqual(value['classification']['candidates'],[])
        self.assertEqual(value['classification']['status'],'incomplete')

    def test_no_stale_mixed_unmapped_or_collecting_class_scores(self):
        cases=[([(0,.25,22,0,1),(0,.75,23,3,1)],'collecting_features'),
               ([(0,.25,22,3,1),(1,.75,23,3,1)],'stale_or_mixed_timestamps'),
               ([(0,.25,22,3,1)],'incomplete'),
               ([(0,.25,22,3,1),(0,.75,23,3,1),(0,0.,24,3,1)],'unmapped_channels')]
        for rows,status in cases:
            result=BoschBSEC2(**CONFIG).process(frame(rows=rows))
            self.assertEqual(result['classification']['status'],status)
            self.assertEqual(result['classification']['candidates'],[])
        p=BoschBSEC2(**dict(CONFIG,classes=[]))
        self.assertEqual(p.process(frame())['classification']['status'],'unconfigured')

    def test_sequence_wrap_gap_owner_and_clock_checks(self):
        p=BoschBSEC2(**CONFIG);a=p.process(frame(number=0xffffffff))
        b=p.process(frame(1,number=0));self.assertTrue(b['summary']['continuous'])
        self.assertEqual(a['summary']['state_segment'],b['summary']['state_segment'])
        c=p.process(frame(3,number=2));self.assertFalse(c['summary']['continuous'])
        self.assertNotEqual(b['summary']['state_segment'],c['summary']['state_segment'])
        with self.assertRaises(ValueError):p.process(frame(3,number=2))
        self.assertIsNone(p.last)
        with self.assertRaises(ValueError):p.process(replace(frame(),timestamp_ms=999))
        with self.assertRaises(ValueError):p.process(replace(frame(),sensor=replace(frame().sensor,calibration_id='other')))
        p.close();p.close()
        with self.assertRaises(RuntimeError):p.process(frame())
        with self.assertRaises(RuntimeError):p.reset()

    def test_strict_decode_crc_and_stream_fragment_bounds(self):
        p=packet();changed=bytearray(p);changed[60]^=1
        for value in (b'',p[:-1],p+b'x',bytes(changed),packet(rows=[(0,float('nan'),6,0,1)]),
                      packet(rows=[(0,.2,22,2,1)]),packet(rows=[(0,1.,27,3,1)]),
                      packet(rows=[(0,1.,6,0,1),(0,2.,6,0,1)])):
            with self.assertRaises(ValueError):decode_bsec2(value)
        stream=BSEC2Packets();result=[]
        source=b'junk'+bytes(changed)+p+packet(1)
        for byte in source:
            result+=stream.feed(bytes([byte]));self.assertLessEqual(len(stream.buffer),397)
        self.assertEqual(result,[p,packet(1)])
        stream.feed(b'GBSC001');stream.reset();self.assertEqual(stream.buffer,b'')
        with self.assertRaises(ValueError):stream.feed(b'x'*4097)

    def test_integer_ns_and_discovery_without_optional_packages(self):
        ns=2**53+123
        config=dict(CONFIG,sensor_origin_ns=str(ns))
        value=BoschBSEC2(**config).process(frame(rows=[(ns,25.,6,0,1)]))
        self.assertEqual(value['summary']['sensor_timestamp_ns'],str(ns))
        for change in ({'sensor_origin_ns':'-1'},{'capture_origin_ms':True},{'classes':['a']},{'bsec_version':'3.0.0.0'}):
            with self.assertRaises(ValueError):ProcessProvider.capability('bosch.bsec2',dict(CONFIG,**change))
        code='import sys;from humansense import ProcessProvider;ProcessProvider.capability("bosch.bsec2",'+repr(CONFIG)+');assert not {"numpy","torch","serial"}&sys.modules.keys()'
        self.assertEqual(subprocess.run([sys.executable,'-c',code],capture_output=True).returncode,0)

    def test_cpp_callback_encoder_matches_python_transport(self):
        compiler=shutil.which('c++')
        if not compiler:self.skipTest('C++ compiler unavailable')
        header=Path(__file__).resolve().parents[1]/'examples/firmware/gamst_bsec2_packet.h'
        source='''#include "gamst_bsec2_packet.h"
#include <stdio.h>
struct Row {int64_t time_stamp;float signal;uint8_t signal_dimensions,sensor_id,accuracy;};
struct Outputs {Row output[23];uint8_t nOutputs;};
struct Version {uint8_t major,minor,major_bugfix,minor_bugfix;};
int main(){Outputs data={};data.nOutputs=2;data.output[0]={3000000000LL,100000.f,1,7,0};
data.output[1]={3000000000LL,.25f,1,22,3};Version version={2,6,1,0};uint8_t hash[32];memset(hash,0xaa,32);
uint8_t out[gamst::bsec2_packet_capacity];
if(gamst::encode_bsec2(out,1,data,version,7,hash)!=0)return 2;
size_t size=gamst::encode_bsec2(out,sizeof(out),data,version,7,hash);return fwrite(out,1,size,stdout)==size?0:3;}
'''
        with tempfile.TemporaryDirectory(dir='.') as root:
            src=Path(root)/'fixture.cpp';binary=Path(root)/'fixture';src.write_text(source)
            run=subprocess.run([compiler,'-std=c++11','-Wall','-Wextra','-Werror','-I',str(header.parent),str(src),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            output=subprocess.run([str(binary.resolve())],capture_output=True,check=True).stdout
            self.assertEqual(output,packet(1,number=7,rows=[(3000000000,100000.,7,0,1),(3000000000,.25,22,3,1)]))


class Managed(unittest.IsolatedAsyncioTestCase):
    async def test_real_worker_and_managed_mcp_context(self):
        from humansense.mcp import MCPServer
        from humansense.schema import timestamp
        config=dict(CONFIG,capture_origin_ms=timestamp()-10)
        async with HumanSense(consent=True) as human:
            streams=SensorStreams(human);spec=ProcessProvider.capability('bosch.bsec2',config)
            async def factory():return await ProcessProvider.open(sys.executable,'bosch.bsec2',config)
            streams.register(spec,factory);await streams.open('env','environment',[spec.id])
            await streams.submit(replace(frame(),timestamp_ms=config['capture_origin_ms'],sensor=replace(frame().sensor,stream_epoch=None)))
            await streams.drain()
            record=human.store.latest_signals('local-user',0,timestamp())[0]
            self.assertEqual(record['observation']['value']['classification']['status'],'prediction_available')
            result=await MCPServer(human).call_tool('get_observations',{'capability_id':spec.id})
            self.assertIn('coffee',str(result))
            await streams.clear()


if __name__=='__main__':unittest.main()
