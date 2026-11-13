import dataclasses
import json
import struct
import subprocess
import sys
import unittest

from humansense.audiovisual import TalkNetWindows, decode_talknet_window
from humansense.capabilities import SensorMetadata
from humansense.plugins import InputFrame
from humansense.provider_wire import encode_frame, decode_frame

try:
    import numpy as np
except ImportError:
    np = None


def audio(seq, start, duration=40, *, epoch='e1', clock=0, source='mic', capture=False):
    values = (np.arange(start*16, (start+duration)*16, dtype=np.int64) % 65536-32768).astype(np.int16)
    sensor = SensorMetadata(source, 'input', seq, channels=('mono',), stream_epoch=epoch,
                            capture_sequence=seq if capture else None)
    return InputFrame(values, 'audio', start+duration+clock, sample_rate=16000, sensor=sensor)


def video(seq, start, *, epoch='e1', clock=0):
    return InputFrame(np.full((112, 112), seq % 256, np.uint8), 'video', start+40+clock,
                      sensor=SensorMetadata('camera', 'face', seq, stream_epoch=epoch))


class Discovery(unittest.TestCase):
    def test_dependency_light_import(self):
        result = subprocess.run([sys.executable, '-c',
            "import sys; import humansense.audiovisual; assert not {'numpy','torch','cv2'} & sys.modules.keys()"],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for packet in (None, b'', b'GTAV001\n'+struct.pack('!I', 1000000)):
            with self.assertRaises(ValueError):
                decode_talknet_window(packet)


@unittest.skipIf(np is None, 'optional NumPy absent')
class Alignment(unittest.TestCase):
    def test_exact_windows_all_lengths_and_lane_arrival_orders(self):
        for n in (25, 50, 75, 100, 125, 150):
            for order in ('audio-first', 'video-first'):
                join = TalkNetWindows(clock_id='capture', window_frames=n, hop_frames=5)
                output = []
                for i in range(n+16):
                    a, v = audio(i, i*40), video(i, i*40)
                    calls = [(join.push_audio, a, {}), (join.push_face, v, {'track_id': 'face-1'})]
                    if order == 'video-first':
                        calls.reverse()
                    for call, frame, kwargs in calls:
                        output.extend(call(frame, clock_id='capture', **kwargs))
                    a.data[:] = 0
                    v.data[:] = 0
                self.assertEqual(len(output), 4)
                for index, frame in enumerate(output):
                    meta, pcm, faces = decode_talknet_window(frame.data)
                    start = index*200
                    self.assertEqual((meta['start_ms'], meta['video_end_ms'], meta['audio_end_ms']),
                                     (start, start+n*40, start+n*40+15))
                    expected = (np.arange(start*16, (start+n*40+15)*16, dtype=np.int64)%65536-32768).astype(np.int16)
                    np.testing.assert_array_equal(pcm, expected)
                    np.testing.assert_array_equal(faces[:, 0, 0], np.arange(index*5, index*5+n)%256)
                    self.assertEqual(frame.timestamp_ms, meta['audio_end_ms'])
                    self.assertEqual(frame.sensor.sequence, index)
                    self.assertEqual(frame.sensor.stream_epoch, meta['session_id'])
                    self.assertFalse(meta['identity_verified'])
                    self.assertFalse(meta['synchronization_verified'])
                    header, payload = encode_frame(frame)
                    copied = decode_frame(header, payload)
                    self.assertEqual(copied, frame)
                self.assertLessEqual(join.status()['buffered_audio_ms'], n*40+1000)
                self.assertLessEqual(join.status()['buffered_face_frames'], n+25)

    def test_hop_extremes_and_unsupported_video_prefix(self):
        for hop in (1, 25):
            join = TalkNetWindows(clock_id='capture', hop_frames=hop)
            result = []
            for i in range(52):
                result += join.push_audio(audio(i, i*40), clock_id='capture')
                result += join.push_face(video(i, i*40), track_id='f', clock_id='capture')
            starts = [decode_talknet_window(v.data)[0]['start_ms'] for v in result]
            self.assertEqual(starts, list(range(0, 27*40, hop*40)))
        join = TalkNetWindows(clock_id='capture')
        join.push_face(video(0, 0), track_id='f', clock_id='capture')
        join.push_audio(audio(0, 40), clock_id='capture')
        self.assertEqual(join.status()['unsupported_prefix_face_frames'], 1)
        self.assertEqual(join.status()['buffered_face_frames'], 0)
        for config in ({'window_frames': True}, {'window_frames': 24},
                       {'hop_frames': 0}, {'hop_frames': 26}):
            with self.assertRaises(ValueError):
                TalkNetWindows(clock_id='capture', **config)

    def test_staggered_start_discards_irrecoverable_prefix_without_overflow(self):
        for delayed in ('audio', 'video'):
            join = TalkNetWindows(clock_id='capture')
            output = []
            if delayed == 'audio':
                join.push_audio(audio(0, 1000, 1000), clock_id='capture')
                for i in range(50):
                    output += join.push_face(video(i, i*40), track_id='f', clock_id='capture')
                output += join.push_audio(audio(1, 2000, 40), clock_id='capture')
                self.assertEqual(join.status()['unsupported_prefix_face_frames'], 25)
            else:
                for i in range(25):
                    join.push_face(video(i, 1000+i*40), track_id='f', clock_id='capture')
                for i in range(51):
                    output += join.push_audio(audio(i, i*40), clock_id='capture')
            self.assertEqual(len(output), 1)
            self.assertEqual(decode_talknet_window(output[0].data)[0]['start_ms'], 1000)
            self.assertEqual(join.status()['resets'], 0)

    def test_irregular_pcm_blocks_wait_for_lookahead_without_padding(self):
        join = TalkNetWindows(clock_id='capture')
        for i in range(25):
            self.assertEqual(join.push_face(video(i, i*40), track_id='f', clock_id='capture'), [])
        self.assertEqual(join.push_audio(audio(0, 0, 1000), clock_id='capture'), [])
        self.assertEqual(join.push_audio(audio(1, 1000, 10), clock_id='capture'), [])
        output = join.push_audio(dataclasses.replace(audio(2, 1010, 10),
            data=audio(2, 1010, 10).data.astype('<i2').tobytes()), clock_id='capture')
        self.assertEqual(len(output), 1)
        meta, pcm, faces = decode_talknet_window(output[0].data)
        self.assertEqual(len(pcm), 16240)
        self.assertEqual(meta['audio_sequence'], [0, 2])
        self.assertEqual(meta['video_sequence'], [0, 24])
        self.assertEqual(faces.shape, (25, 112, 112))
        self.assertEqual(meta['audio_end_ms'], 1015)
        # The last5ms arrived but are not part of this feature window.
        self.assertEqual(pcm[-1], audio(0, 0, 1000).data[-1]+240)

    def test_gaps_ownership_track_and_face_absence_clear_both_lanes(self):
        def warm():
            join = TalkNetWindows(clock_id='capture')
            join.push_audio(audio(0, 0), clock_id='capture')
            join.push_face(video(0, 0), track_id='f', clock_id='capture')
            return join
        cases = [
            ('capture_gap', lambda j:j.push_audio(audio(2, 40), clock_id='capture')),
            ('capture_gap', lambda j:j.push_audio(audio(1, 50), clock_id='capture')),
            ('source_changed', lambda j:j.push_audio(audio(1, 40, epoch='e2'), clock_id='capture')),
            ('source_changed', lambda j:j.push_audio(audio(1, 40, capture=True), clock_id='capture')),
            ('track_changed', lambda j:j.push_face(video(1, 40), track_id='other', clock_id='capture')),
            ('face_absent', lambda j:j.push_face(dataclasses.replace(video(1, 40), data=None), track_id='f', clock_id='capture'))]
        for reason, call in cases:
            join = warm()
            before = join.status()['session_id']
            self.assertEqual(call(join), [])
            status = join.status()
            self.assertNotEqual(before, status['session_id'])
            self.assertEqual(status['last_reset_reason'], reason)
            self.assertTrue(status['buffered_audio_ms'] == 0 or status['buffered_face_frames'] == 0)
        join = warm()
        with self.assertRaises(ValueError):
            join.push_audio(audio(0, 0), clock_id='capture')
        self.assertEqual(join.status()['buffered_audio_ms'], 0)
        self.assertEqual(join.status()['buffered_face_frames'], 0)
        join.close()
        with self.assertRaises(RuntimeError):
            join.push_audio(audio(1, 40), clock_id='capture')
        join.reset()
        with self.assertRaises(RuntimeError):
            join.push_face(video(1, 40), track_id='f', clock_id='capture')

    def test_clock_skew_invalid_shapes_and_missing_lane_are_bounded(self):
        join = TalkNetWindows(clock_id='capture')
        join.push_audio(audio(0, 0, clock=1), clock_id='capture')
        with self.assertRaisesRegex(ValueError, 'clock grids'):
            join.push_face(video(0, 0), track_id='f', clock_id='capture')
        for bad in (dataclasses.replace(audio(0, 0), data=np.zeros(640, np.float32)),
                    dataclasses.replace(audio(0, 0), sample_rate=16000.),
                    dataclasses.replace(audio(0, 0), subject_id='another'),
                    dataclasses.replace(audio(0, 0), timestamp_ms=10),
                    dataclasses.replace(audio(0, 0), data=np.zeros((640, 1), np.int16))):
            with self.assertRaises(ValueError):
                join.push_audio(bad, clock_id='capture')
        with self.assertRaises(ValueError):
            join.push_audio(audio(0, 0), clock_id='other-clock')
        for i in range(200):
            self.assertEqual(join.push_audio(audio(i, i*40), clock_id='capture'), [])
            self.assertLessEqual(join.status()['buffered_audio_ms'], 2000)
        self.assertGreater(join.status()['resets'], 0)
        # Huge cross-lane clock displacement must not loop once per missing frame.
        join.reset()
        join.push_audio(audio(0, 10**12), clock_id='capture')
        self.assertEqual(join.push_face(video(0, 0), track_id='f', clock_id='capture'), [])

    def test_packet_rejects_tampering_and_decoded_arrays_are_owned(self):
        join = TalkNetWindows(clock_id='capture')
        for i in range(25):
            join.push_face(video(i, i*40), track_id='f', clock_id='capture')
        join.push_audio(audio(0, 0, 1000), clock_id='capture')
        packet = join.push_audio(audio(1, 1000), clock_id='capture')[0].data
        meta, pcm, faces = decode_talknet_window(packet)
        pcm[:] = 0
        faces[:] = 0
        self.assertNotEqual(decode_talknet_window(packet)[1][0], 0)
        size = struct.unpack('!I', packet[8:12])[0]
        def replace(value):
            raw = json.dumps(value).encode()
            return packet[:8]+struct.pack('!I',len(raw))+raw+packet[12+size:]
        bads = [packet[:-1], packet+b'\0']
        for key, value in (('synchronization_verified', True), ('audio_end_ms', 1016),
                           ('frames', True), ('audio_sequence', [2, 1]), ('video_sensor', {})):
            bads.append(replace(dict(meta, **{key: value})))
        duplicate = packet[12:12+size][:-1]+b',"version":1}'
        bads.append(packet[:8]+struct.pack('!I',len(duplicate))+duplicate+packet[12+size:])
        for bad in bads:
            with self.assertRaises(ValueError):
                decode_talknet_window(bad)


if __name__ == '__main__':
    unittest.main()
