import dataclasses
import json
import unittest

from humansense import HumanSense, CapabilitySpec, CapabilityRegistry, OutputField, SensorMetadata
from humansense.capabilities import bounded_json
from humansense.context import observation_packet
from humansense.plugins import InputFrame


def thermal(**changes):
    spec = CapabilitySpec('thermal.roi', 'thermal.surface', 'thermal', 'test.sensor', '1.0',
        (OutputField('mean', 'number', 'degC'), OutputField('roi', 'object'),
         OutputField('label', 'string', required=False)), 'frame',
        freshness_ms=100, interpretation='measurement', sensor_requirements=('radiometric',))
    return dataclasses.replace(spec, **changes)


class Contracts(unittest.TestCase):
    def test_registry_is_metadata_only_and_defensive(self):
        registry = CapabilityRegistry()
        spec = thermal()
        registry.register(spec)
        registry.register(spec)
        description = registry.describe()
        json.dumps(description)
        description[0]['fields'][0]['kind'] = 'anything'
        self.assertEqual(registry.get(spec.id).fields[0].kind, 'number')
        with self.assertRaises(ValueError):
            registry.register(thermal(version='2.0'))
        with self.assertRaises(ValueError):
            registry.get('missing')

    def test_field_semantics_and_limits(self):
        spec = thermal()
        original = {'mean': 32, 'roi': {'x': 1}}
        result = spec.validate(original)
        original['roi']['x'] = 100
        self.assertEqual(result['roi']['x'], 1)
        invalid = [None, {}, {'mean': True, 'roi': {}}, {'mean': '32', 'roi': {}},
                   {'mean': 32, 'roi': {}, 'extra': 1},
                   {'mean': 32, 'roi': {'temperature': float('nan')}},
                   {'mean': 32, 'roi': {'RAW': [1]}},
                   {'mean': 32, 'roi': {'values': [1]*2049}}]
        for value in invalid:
            with self.subTest(value=str(value)[:80]), self.assertRaises(ValueError):
                spec.validate(value)

    def test_json_is_bounded_before_encoding(self):
        cycle = []
        cycle.append(cycle)
        for value in (cycle, b'raw', 10**1000, {'x': float('inf')}, {'x': 'a'*8193}, {1: 'bad'}):
            with self.assertRaises(ValueError):
                bounded_json(value)
        with self.assertRaises(ValueError):
            bounded_json({'x': 'a'*10}, max_bytes=5)

    def test_spec_rejects_ambiguous_runtime_declarations(self):
        for changes in ({'streaming_mode': 'streaming'}, {'streaming_mode': 'window'},
                        {'lookahead_ms': -1}, {'freshness_ms': True},
                        {'reset_on_gap': True}, {'stateful': 1}, {'fields': []},
                        {'fields': (OutputField('x', 'number'), OutputField('x', 'number'))},
                        {'devices': ['cpu']}, {'modality': 'magic'}, {'interpretation': 'truth'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                thermal(**changes)
        with self.assertRaises(ValueError):
            OutputField('samples', 'array')
        with self.assertRaises(ValueError):
            OutputField('field', 'any')

    def test_sensor_metadata_preserves_channel_and_coordinate_identity(self):
        sensor = SensorMetadata('mic-array', 'device1', 0, 'room1', ('left', 'right'), 'cal-1')
        frame = InputFrame([[1, 2]], 'audio', 1, sensor=sensor)
        self.assertEqual(frame.sensor.channels, ('left', 'right'))
        self.assertEqual(frame.data, [[1, 2]])
        for args in (('source', 'sensor', -1), ('source', 'sensor', True),
                     ('source', 'sensor', 0, None, ('left', 'left'))):
            with self.assertRaises(ValueError):
                SensorMetadata(*args)


class RuntimeContracts(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.human = HumanSense(consent=True)
        self.human.register_capability(thermal())
        self.sensor = SensorMetadata('camera1', 'lepton1', 1, calibration_id='cal-1')

    async def asyncTearDown(self):
        await self.human.close()

    async def test_thermal_roundtrip_keeps_provenance_and_freshness(self):
        record = await self.human.push_observation('thermal.roi', {'mean': 32, 'roi': {'x': 1}},
                                                   sensor=self.sensor, timestamp_ms=1000)
        value = next(iter(observation_packet([record], 1101).values()))
        self.assertFalse(value['fresh'])
        self.assertEqual(value['value']['mean'], 32)
        self.assertEqual(value['capability']['units']['mean'], 'degC')
        self.assertEqual(value['sensor']['calibration_id'], 'cal-1')
        self.assertEqual(value['interpretation'], 'measurement')
        self.assertEqual(value['models'][0]['id'], 'test.sensor')

    async def test_different_sensors_with_same_output_are_not_overwritten(self):
        records = []
        for source in ('camera1', 'camera2'):
            records.append(await self.human.push_observation('thermal.roi', {'mean': 32, 'roi': {}},
                sensor=SensorMetadata(source, 'lepton', 1), timestamp_ms=1000))
        self.assertEqual(len(observation_packet(records, 1000)), 2)

    async def test_public_context_preserves_sources_and_providers(self):
        self.human.register_capability(thermal(id='thermal.other', provider='other.sensor'))
        for capability in ('thermal.roi', 'thermal.other'):
            for source in ('camera1', 'camera2'):
                await self.human.push_observation(capability, {'mean': 32, 'roi': {}},
                    sensor=SensorMetadata(source, 'lepton', 1))
        # Same identity is replaced, without dropping other providers or sources.
        await self.human.push_observation('thermal.roi', {'mean': 35, 'roi': {}},
            sensor=SensorMetadata('camera1', 'lepton', 2))
        observations = (await self.human.get_context())['observations']
        self.assertEqual(len(observations), 4)
        self.assertEqual(sorted(v['value']['mean'] for v in observations.values()), [32, 32, 32, 35])

    async def test_registered_classification_uses_declared_contract(self):
        self.human.register_capability(CapabilitySpec('food.label', 'classification.food',
            'video', 'food-model', '1', (OutputField('label', 'string'),),
            'frame', interpretation='classification'))
        await self.human.push_observation('food.label', {'label': 'apple'}, sensor=self.sensor)
        observations = (await self.human.get_context())['observations']
        self.assertEqual(len(observations), 1)
        self.assertEqual(next(iter(observations.values()))['value'], {'label': 'apple'})

    async def test_provenance_cannot_bypass_registry(self):
        with self.assertRaises(ValueError):
            await self.human.push_signal('thermal.surface', {}, 'thermal', capability_id='thermal.roi', sensor=self.sensor)
        with self.assertRaises(ValueError):
            await self.human.push_observation('thermal.roi', {'mean': 32, 'roi': {}}, sensor=None)
        with self.assertRaises(ValueError):
            await self.human.push_signal('unknown', 1, 'radar')
        with self.assertRaises(ValueError):
            await self.human.push_observation('unknown', {}, sensor=self.sensor)

    async def test_existing_signals_keep_existing_context_keys(self):
        record = await self.human.push_signal('audio.rms', .1, 'audio', timestamp_ms=1000)
        self.assertIn('audio.rms', observation_packet([record], 1000))

    async def test_discovery_without_consent_but_ingestion_requires_it(self):
        async with HumanSense() as human:
            human.register_capability(thermal())
            self.assertEqual(len(human.perception_capabilities.describe()), 1)
            with self.assertRaises(PermissionError):
                await human.push_observation('thermal.roi', {'mean': 32, 'roi': {}}, sensor=self.sensor)

    async def test_probability_guard_still_applies(self):
        with self.assertRaises(ValueError):
            await self.human.push_observation('thermal.roi', {'mean': 32, 'roi': {'probability': .9}}, sensor=self.sensor)
