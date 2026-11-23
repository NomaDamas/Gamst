import asyncio
import math
import sys
import time
from unittest.mock import patch

import numpy as np

from gamst import CapabilitySpec, Gamst, OutputField, ProcessProvider, SensorMetadata
from gamst.providers.rotation import RotationPlugin, RotationProvider


DETECTOR = CapabilitySpec('vision.test_detector', 'vision.objects', 'video', 'test.detector', '1',
    (OutputField('objects', 'array'),), 'frame', devices=('cpu',))


def geometry(sequence, label='widget'):
    angle = min(sequence, 11) * math.pi / 8
    center = [64., 64.]
    points = [[64 + 30 * math.cos(angle + index * math.pi / 4),
               64 + 30 * math.sin(angle + index * math.pi / 4)] for index in range(8)]
    return [dict(track_id='object-1', label=label, score=.9, box=[32, 32, 64, 64],
                 center=center, observed=True, axes=[50., 50.], angle=0., points=points, points_epoch=1)]


def test_push_video_store_observations_and_rotation_watch():
    async def run():
        async with Gamst(consent=True) as human:
            human.register_capability(DETECTOR)
            base = int(time.time() * 1000) - 3000
            sensor = SensorMetadata(source_id='camera', sensor_id='detector', sequence=0,
                                    coordinate_frame='image', stream_epoch='test')
            box = dict(x=.25, y=.25, width=.5, height=.5)
            await human.push_observation(DETECTOR.id, {'objects': [dict(label='widget', score=.9, bbox=box)]},
                sensor=sensor, timestamp_ms=base)
            plugin = RotationPlugin(human, region_source='detections', targets=['widget'], expected_max_rev_s=1)
            human.register(plugin)
            with patch.object(plugin.provider.tracks, 'observe',
                              side_effect=lambda image, stamp, owner: geometry(round((stamp - base) / 50))):
                alarms = []
                human.watches.create(owner='user', description='rotation ended',
                    rule=dict(conditions=[dict(capability_id='vision.rotation', path='events', op='contains',
                                               value='rotation.ended', within_ms=1000)],
                              cooldown_ms=0, max_per_minute=60, context_window_ms=3000))
                human.watches.add_sink(alarms.append)
                for sequence in range(20):
                    await human.push_video(np.zeros((128, 128, 3), np.uint8), timestamp_ms=base + sequence * 50)
                records = await human.get_observations(capability_id='vision.rotation', window='1h',
                                                       limit=100, max_bytes=200000)
            assert records['returned_count'] == 20
            assert all(item['observation']['value']['objects'][0]['label'] == 'widget'
                       for item in records['items'])
            assert any(event['type'] == 'rotation.started' and event['axis'] == 'in_plane'
                       for item in records['items'] for event in item['observation']['value']['events'])
            assert any(event['type'] == 'rotation.ended' and event['axis'] == 'in_plane'
                       for item in records['items'] for event in item['observation']['value']['events'])
            assert alarms
            plugin.close()
    asyncio.run(run())


def test_roi_worker_profile_and_validation():
    roi = dict(x=.1, y=.1, width=.5, height=.5)
    spec = ProcessProvider.capability('vision.rotation', dict(region_source='roi', roi=roi,
                                                             expected_max_rev_s=8, targets=['wheel']))
    assert spec.id == 'vision.rotation'
    assert set(spec.devices) == {'cpu'}
    for config in ({'region_source': 'roi'}, {'region_source': 'detections'},
                   {'region_source': 'motion', 'unknown': 1}, {'roi': dict(roi, width=2)}):
        try:
            ProcessProvider.capability('vision.rotation', config)
        except ValueError:
            pass
        else:
            raise AssertionError(f'accepted invalid config: {config}')


def test_rotation_process_provider_opens_and_processes():
    async def run():
        from gamst.plugins import InputFrame
        config = dict(region_source='roi', roi=dict(x=.1, y=.1, width=.5, height=.5))
        worker = await ProcessProvider.open(sys.executable, 'vision.rotation', config)
        try:
            for sequence in range(2):
                sensor = SensorMetadata(source_id='camera', sensor_id='worker', sequence=sequence,
                                        coordinate_frame='image', stream_epoch='test')
                value = await worker.process(InputFrame(np.zeros((128, 128, 3), np.uint8), 'video',
                                                        sequence * 50, sensor=sensor))
                assert set(value) == {'objects', 'events'}
        finally:
            await worker.close()
    asyncio.run(run())


def test_roi_provider_emits_an_observation_for_every_capture():
    provider = RotationProvider(region_source='roi', roi=dict(x=.1, y=.1, width=.5, height=.5),
                                expected_max_rev_s=1)
    with patch.object(provider.tracks, 'observe', side_effect=lambda image, stamp, owner: geometry(stamp // 50, 'unknown')):
        for sequence in range(3):
            sensor = SensorMetadata(source_id='camera', sensor_id='rotation', sequence=sequence,
                                    coordinate_frame='image', stream_epoch='test')
            from gamst.plugins import InputFrame
            value = provider.process(InputFrame(np.zeros((128, 128, 3), np.uint8), 'video', sequence * 50,
                                                sensor=sensor))
            assert value['objects'][0]['label'] is None
            assert value['objects'][0]['status'] in ('idle', 'counting', 'undersampled')
    provider.close()
