"""Association seam for camera compensation; predictor is explicitly synthetic."""
import copy
import unittest
import test_detection_provider as fixtures
from humansense.providers.detection import RFDETRObjects


@unittest.skipIf(fixtures.np is None, 'optional NumPy unavailable')
class ProjectedAssociation(unittest.TestCase):
    def test_prediction_precedes_fit_and_projected_boxes_are_not_observations(self):
        backend=fixtures.Backend();provider=RFDETRObjects(backend,version='fixture')
        first=provider.process(fixtures.frame())
        old=copy.deepcopy(provider.tracks)
        key=first['objects'][0]['track_id']
        backend.rows=((50,10,80,40,.9,1,'cup'),)
        batch=provider._predict_rows(fixtures.frame(1100))
        self.assertEqual(provider.tracks,old)
        result=provider._associate_rows(*batch,association_boxes={key:dict(x=.48,y=.1,width=.3,height=.3)},visible_only=True)
        current=result['objects'][0]
        self.assertEqual(current['track_id'],key)
        self.assertEqual(current['location']['bbox']['x'],.5)
        self.assertEqual(provider.tracks[key]['bbox']['x'],.5)
        self.assertEqual(first['objects'][0]['location']['bbox']['x'],.1)
        self.assertEqual(old[key]['bbox']['x'],.1)

    def test_missing_tracks_retire_in_compensated_mode(self):
        backend=fixtures.Backend();provider=RFDETRObjects(backend,version='fixture')
        key=provider.process(fixtures.frame())['objects'][0]['track_id']
        backend.rows=()
        result=provider._associate_rows(*provider._predict_rows(fixtures.frame(1100)),
            association_boxes={key:dict(x=.1,y=.1,width=.3,height=.3)},visible_only=True)
        self.assertEqual(result['objects'],[]);self.assertEqual(provider.tracks,{})
        backend.rows=fixtures.Backend.rows
        result=provider._associate_rows(*provider._predict_rows(fixtures.frame(1200)),association_boxes={},visible_only=True)
        self.assertNotEqual(result['objects'][0]['track_id'],key)

    def test_unknown_or_invalid_projection_does_not_commit(self):
        backend=fixtures.Backend();provider=RFDETRObjects(backend,version='fixture')
        first=provider.process(fixtures.frame());key=first['objects'][0]['track_id']
        old=copy.deepcopy(provider.tracks);batch=provider._predict_rows(fixtures.frame(1100))
        for boxes in ({'unknown':dict(x=.1,y=.1,width=.3,height=.3)},
                      {key:dict(x=-.1,y=.1,width=.3,height=.3)}):
            with self.assertRaises(ValueError):provider._associate_rows(*batch,association_boxes=boxes,visible_only=True)
            self.assertEqual(provider.tracks,old)
            self.assertEqual(provider.last_ts,1000)
