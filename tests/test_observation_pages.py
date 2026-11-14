import json
import sqlite3
import tempfile
from pathlib import Path
import unittest

from humansense import HumanSense,SensorMetadata
from humansense.capabilities import CapabilitySpec,OutputField
from humansense.mcp import MCPServer
from humansense.observation_pages import observation_page
from humansense.storage import Store
from humansense.providers.trajectories import ImageTrajectories
from humansense.schema import timestamp
import test_image_trajectories as trajectories


def record(i,ts=100,subject='user',events=True,cap='vision.test',source='camera',text=''):
    return dict(id=str(i),subject_id=subject,timestamp_ms=ts,type='observation.signal',
        observation=dict(capability=dict(id=cap),sensor=dict(source_id=source,sensor_id='rgb'),
                         value=dict(events=[dict(kind='candidate')] if events else [],text=text)))


def page(store,**kwargs):return observation_page(store,subject_id='user',**kwargs)


class Storage(unittest.TestCase):
    def test_ingestion_order_equal_backdated_and_live_append(self):
        store=Store()
        try:
            for i,ts in enumerate((500,500,100,700)):store.add(record(i,ts))
            first=page(store,limit=2)
            self.assertEqual([r['id'] for r in first['items']],['0','1'])
            self.assertTrue(first['has_more'])
            second=page(store,cursor=first['next_cursor'],limit=2)
            self.assertEqual([r['id'] for r in second['items']],['2','3'])
            self.assertFalse(second['has_more'])
            store.add(record(4,1))
            third=page(store,cursor=second['next_cursor'])
            self.assertEqual([r['id'] for r in third['items']],['4'])
            self.assertFalse(third['retention_changed_since_cursor'])
            self.assertEqual(page(store,cursor=third['next_cursor'])['items'],[])
        finally:store.close()

    def test_filters_scope_tamper_and_byte_block_retry(self):
        store=Store()
        try:
            store.add(record(0,text='가'*700));store.add(record(1,events=False))
            store.add(record(2,cap='other'));store.add(record(3,subject='other'))
            first=page(store,capability_id='vision.test',events_only=True,max_bytes=1024)
            self.assertEqual(first['items'],[]);self.assertTrue(first['has_more'])
            self.assertGreater(first['blocked_record_bytes'],1024)
            next_=page(store,cursor=first['next_cursor'],capability_id='vision.test',events_only=True,max_bytes=10000)
            self.assertEqual([r['id'] for r in next_['items']],['0'])
            self.assertFalse(next_['has_more'])
            for args in ({'events_only':False},{'capability_id':'other'},{'source_id':'other'}):
                options=dict(capability_id='vision.test',events_only=True);options.update(args)
                with self.assertRaises(ValueError):page(store,cursor=next_['next_cursor'],**options)
            for cursor in ('!',next_['next_cursor'][:-3]+'abc','x'*2049,True):
                with self.assertRaises(ValueError):page(store,cursor=cursor)
            for args in ({'events_only':1},{'limit':True},{'max_bytes':0},{'source_id':'!'}):
                with self.assertRaises(ValueError):page(store,**args)
        finally:store.close()

    def test_retention_eviction_oversized_and_metadata_bound(self):
        store=Store(max_records=2,max_payload_bytes=3000,retention_hours=1/3600)
        try:
            store.add(record(0));first=page(store)
            for i in range(1,10):store.add(record(i))
            value=page(store,cursor=first['next_cursor'])
            self.assertTrue(value['retention_changed_since_cursor'])
            self.assertEqual([r['id'] for r in value['items']],['8','9'])
            self.assertEqual(store.connection.execute('SELECT COUNT(*) FROM record_offsets').fetchone()[0],2)
            store.prune(2000)
            empty=page(store,cursor=value['next_cursor'])
            self.assertEqual(empty['items'],[]);self.assertTrue(empty['retention_changed_since_cursor'])
            self.assertEqual(store.connection.execute('SELECT COUNT(*) FROM record_offsets').fetchone()[0],0)
            store.add(record(10,2000))
            resumed=page(store,cursor=empty['next_cursor'])
            self.assertEqual([r['id'] for r in resumed['items']],['10'])
            self.assertFalse(store.add(record(11,2000,text='x'*5000)))
            self.assertTrue(page(store,cursor=resumed['next_cursor'])['retention_changed_since_cursor'])
        finally:store.close()

    def test_legacy_database_migration_delete_and_reopen(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            path=Path(folder)/'legacy.sqlite';conn=sqlite3.connect(path)
            conn.execute('CREATE TABLE records (id TEXT PRIMARY KEY,subject TEXT NOT NULL,ts INTEGER NOT NULL,kind TEXT NOT NULL,payload TEXT NOT NULL)')
            for i,ts in ((0,700),(1,200)):
                value=record(i,ts);conn.execute('INSERT INTO records VALUES(?,?,?,?,?)',(value['id'],'user',ts,value['type'],json.dumps(value)))
            conn.commit();conn.close()
            store=Store(path)
            first=page(store);self.assertEqual([r['id'] for r in first['items']],['0','1'])
            self.assertEqual(first['items'][0],record(0,700))
            store.delete('other') # conservative cursor invalidation even no matching records
            with self.assertRaises(ValueError):page(store,cursor=first['next_cursor'])
            before=page(store)['next_cursor'];store.close();store=Store(path)
            try:
                with self.assertRaises(ValueError):page(store,cursor=before)
                self.assertEqual([r['id'] for r in page(store)['items']],['0','1'])
                store.delete('user');store.add(record(2))
                self.assertEqual(store.connection.execute('SELECT position FROM record_offsets').fetchone()[0],3)
                self.assertEqual(page(store)['items'][0]['id'],'2')
            finally:store.close()


class Runtime(unittest.IsolatedAsyncioTestCase):
    async def test_intermediate_trajectory_events_survive_latest_non_event_snapshot(self):
        p=ImageTrajectories(event_rules=[dict(id='zone',kind='roi',bounds=[.4,.3,.8,.7])])
        now=timestamp()-300
        async with HumanSense(consent=True) as human:
            human.register_capability(p.spec);server=MCPServer(human)
            cursor=(await server.call_tool('get_observation_page',dict(events_only=True)))['next_cursor']
            for i,x in enumerate((.2,.6,.2,.1)):
                f=trajectories.frame(i,now+i*50,x)
                await human.push_observation(p.spec.id,p.process(f),sensor=f.sensor,timestamp_ms=f.timestamp_ms)
            latest=await human.get_observations(capability_id=p.spec.id,limit=1)
            self.assertEqual(latest['items'][0]['observation']['value']['events'],[])
            one=await server.call_tool('get_observation_page',dict(cursor=cursor,events_only=True,limit=1))
            self.assertTrue(one['has_more'])
            two=await server.call_tool('get_observation_page',dict(cursor=one['next_cursor'],events_only=True,limit=1))
            kinds=[]
            for record_ in one['items']+two['items']:
                value=record_['observation']['value'];event=value['events'][0]
                kinds.append(event['kind'])
                samples=value['objects'][event['object_index']]['observations']
                self.assertTrue(all(samples[i]['evidence_sha256']=='a'*64 for i in event['observation_indices']))
            self.assertEqual(kinds,['roi_entry','roi_exit'])
            self.assertFalse(two['has_more'])
            await human.pause()
            self.assertEqual((await human.get_observation_page(cursor=two['next_cursor'],events_only=True))['items'],[])
            await human.delete_data()
            with self.assertRaises(ValueError):await human.get_observation_page(cursor=two['next_cursor'],events_only=True)
        with self.assertRaises(RuntimeError):await human.get_observation_page()

    async def test_prune_discloses_gap_and_subject_filter(self):
        async with HumanSense(consent=True,retention_hours=1/3600) as human:
            # The store can contain data ingested by different sources out of capture-time order.
            human.store.add(record(0,timestamp()-500,subject='local-user'))
            first=await human.get_observation_page()
            human.store.add(record(1,timestamp()-2000,subject='local-user'))
            next_=await human.get_observation_page(cursor=first['next_cursor'])
            self.assertTrue(next_['retention_changed_since_cursor'])
            self.assertEqual(next_['items'],[])
            human.store.add(record(2,timestamp(),subject='other'))
            self.assertEqual((await human.get_observation_page(cursor=next_['next_cursor']))['items'],[])
