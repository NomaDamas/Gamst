"""Actual image fitting and association with labeled fixture detections."""
import copy
from dataclasses import replace
import subprocess
import sys
import unittest
import test_camera_similarity as images
import test_detection_provider as detection
from humansense import SensorMetadata
from humansense.plugins import InputFrame
from humansense.providers.compensated_trajectories import CompensatedTrajectories,ASSUMPTION

CONFIG=dict(path='/fixture/checkpoint',sha256='a'*64,background_roi=[0,0,1,1],operating_assumption=ASSUMPTION)


@unittest.skipIf(images.cv is None,'optional OpenCV/NumPy missing')
class Paths(unittest.TestCase):
    def setUp(self):
        source=images.Similarity();source.setUp();self.image=source.texture()
        self.backend=detection.Backend()
        self.provider=CompensatedTrajectories(self.backend,version='fixture',config=CONFIG)
        self.addCleanup(self.provider.close)

    def frame(self,seq,shift=0,extra=0):
        image=images.cv.warpAffine(self.image,images.np.array([[1,0,shift],[0,1,0]],float),(480,320))
        self.backend.rows=((80+shift+extra,120,90+shift+extra,140,.9,1,'ball'),)
        return InputFrame(image,'video',1000+seq*100,'subject',sensor=SensorMetadata(
            'cam','rgb',seq,'image',stream_epoch='epoch',calibration_id='cal'))

    def test_compensated_association_preserves_raw_and_tracks_independent_motion(self):
        first=self.provider.process(self.frame(0))
        key=first['detections'][0]['track_id']
        self.assertEqual(first['objects'],[])
        second=self.provider.process(self.frame(1,8,1))
        self.assertEqual(second['geometry']['status'],'estimated',second['geometry'])
        self.assertEqual(second['detections'][0]['track_id'],key)
        path=second['objects'][0]
        self.assertEqual(len(path['observations']),2)
        self.assertGreater(path['observations'][-1]['raw_centroid'][0],path['observations'][0]['raw_centroid'][0])
        self.assertAlmostEqual(path['velocity_reference_image_s'][0],1/480/.1,delta=.003)
        frozen=copy.deepcopy(second)
        third=self.provider.process(self.frame(2,16,2))
        self.assertEqual(third['objects'][0]['segment_id'],path['segment_id'])
        self.assertEqual(second,frozen)
        self.assertFalse(third['geometry']['metric_geometry'])
        self.assertEqual(third['geometry']['composition_links'],2)

    def test_gap_cut_and_owner_changes_end_reference(self):
        self.provider.process(self.frame(0));good=self.provider.process(self.frame(1,8))
        old=good['geometry']['anchor']['id']
        gap=self.provider.process(self.frame(3,16))
        self.assertEqual(gap['objects'],[]);self.assertNotEqual(gap['geometry']['anchor']['id'],old)
        cut=self.frame(4,16);cut=replace(cut,data=images.np.zeros_like(cut.data))
        out=self.provider.process(cut)
        self.assertEqual(out['objects'],[]);self.assertFalse(out['geometry']['camera_motion_compensated'])
        changed=replace(self.frame(5),subject_id='other')
        self.assertEqual(self.provider.process(changed)['objects'],[])

    def test_horizon_missing_tracks_and_invalid_capture(self):
        self.provider.process(self.frame(0));last=self.provider.process(self.frame(1,3))
        anchor=last['geometry']['anchor']['id']
        for i in range(2,22):last=self.provider.process(self.frame(i,3))
        self.assertEqual(last['geometry']['reason'],'reference_horizon')
        self.assertNotEqual(last['geometry']['anchor']['id'],anchor)
        absent=self.frame(22,3);self.backend.rows=()
        self.assertEqual(self.provider.process(absent)['objects'],[])
        self.assertEqual(self.provider.paths,{})
        with self.assertRaises(ValueError):self.provider.process(self.frame(22,3))
        self.assertIsNone(self.provider.previous)

    def test_partial_snapshot_abstains_and_reset_closes(self):
        self.provider.process(self.frame(0))
        frame=self.frame(1,3)
        self.backend.rows=tuple((10+i*40,10,20+i*40,20,.9,i,'ball') for i in range(9))
        result=self.provider.process(frame)
        self.assertFalse(result['geometry']['complete_snapshot'])
        self.assertEqual(result['objects'],[]);self.assertIsNone(self.provider.previous)
        self.provider.close()
        with self.assertRaises(RuntimeError):self.provider.process(self.frame(2))


    def test_maximum_labels_histories_and_reference_provenance_fit_context(self):
        import json
        np,cv=images.np,images.cv
        source=images.Similarity();source.setUp();image=source.texture(1001,663)
        self.provider.detector.version='1.12.1.eager.sha256.'+'a'*64
        anchor=None
        for i in range(12):
            matrix=np.eye(3);matrix[:2]=cv.getRotationMatrix2D((500.5,331.5),i*.75,1.003**i)
            matrix[:2,2]+=np.array([i*1.23456789,i*.87654321])
            rows=[]
            for j in range(8):
                x,y=110+j*100,210
                corners=np.array([[x,y,1],[x+7,y,1],[x+7,y+7,1],[x,y+7,1]])@matrix.T
                low,high=corners[:,:2].min(axis=0),corners[:,:2].max(axis=0)
                rows.append((*low,*high,.987654321,j,chr(0x10000+j)*256))
            self.backend.rows=tuple(rows)
            seq=2**63-20+i
            frame=InputFrame(cv.warpAffine(image,matrix[:2],(1001,663)),
                'video',2**53-801+i*30,'s'*128,sensor=SensorMetadata(
                'c'*128,'i'*128,seq,'f'*128,stream_epoch='e'*128,
                calibration_id='k'*128,capture_sequence=seq))
            result=self.provider.process(frame)
            self.assertLessEqual(len(json.dumps(result,ensure_ascii=True).encode()),65536)
            if anchor is None:anchor=result['geometry']['anchor']
            self.assertEqual(result['geometry']['anchor'],anchor)
            if i:
                self.assertEqual(result['geometry']['status'],'estimated',result['geometry'])
                self.assertEqual(len(result['objects']),8)
                mappings=result['geometry']['projection_records']
                self.assertLessEqual(len(mappings),8)
                for obj in result['objects']:
                    self.assertEqual(obj['anchor_id'],anchor['id'])
                    for sample in obj['observations']:
                        records=[r for r in mappings if all(r[k]==sample[k] for k in
                            ('capture_sequence','timestamp_ms','evidence_sha256'))]
                        self.assertEqual(len(records),1)
                        self.assertNotIn('current_to_anchor_pixels',sample)
                        raw=sample['raw_centroid']
                        point=np.asarray(records[0]['current_to_anchor_pixels'])@np.array([raw[0]*1001,raw[1]*663,1])
                        np.testing.assert_allclose(point[:2]/[1001,663],sample['anchor_projection'])
        self.assertEqual([len(obj['observations']) for obj in result['objects']],[8]*8)
        self.assertEqual(len(result['detections'][0]['label']),256)
        self.assertNotIn('label',result['objects'][0])


class Discovery(unittest.TestCase):
    def test_profile_discovery_does_not_import_optional_runtime(self):
        code='from humansense import ProcessProvider;import sys;ProcessProvider.capability("rfdetr.compensated_trajectories",'+repr(CONFIG)+');assert not {"cv2","numpy","rfdetr","torch"}&sys.modules.keys()'
        result=subprocess.run([sys.executable,'-c',code],capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
