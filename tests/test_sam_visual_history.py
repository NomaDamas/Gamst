import copy
import unittest
from humansense.sam_visual_history import VisualCaptureHistory
import test_sam_visual_packet as packet_tests
from humansense.sam_visual_packet import unpack_visual_packet


@unittest.skipIf(packet_tests.np is None,'optional numpy missing')
class CaptureHistory(unittest.TestCase):
    def setUp(self):
        fixture=packet_tests.VisualPacketTests(); fixture.setUp()
        self.header=unpack_visual_packet(fixture.packet().data)[0]
        self.history=VisualCaptureHistory()

    def advance(self,start,count=2):
        h=copy.deepcopy(self.header)
        h['visual_sensor']['capture_sequence']=start
        template=h['frames'][0]
        h['frames']=[]
        for sequence in range(start,start+count):
            r=copy.deepcopy(template)
            r.update(capture_sequence=sequence,interval_us=[sequence*100000,sequence*100000])
            h['frames'].append(r)
        return h

    def test_overlapping_windows_can_reuse_original_captures(self):
        first=self.advance(10)
        pending=self.history.stage(first)
        self.assertEqual(pending.reason,'initial')
        self.history.commit(pending)
        for h in (first,self.advance(11)):
            pending=self.history.stage(h)
            self.assertIsNone(pending.reason)
            self.history.commit(pending)

    def test_same_capture_with_changed_hash_or_time_rejected(self):
        h=self.advance(10); self.history.commit(self.history.stage(h))
        for field,value in [('rgb_sha256','1'*64),('mask_sha256','2'*64),
                            ('interval_us',[999999,1000000])]:
            changed=copy.deepcopy(h); changed['frames'][0][field]=value
            with self.assertRaises(ValueError): self.history.stage(changed)
        # Failed staging did not poison previously valid evidence.
        self.history.commit(self.history.stage(h))

    def test_staged_work_cannot_survive_reset_or_cross_owner(self):
        pending=self.history.stage(self.advance(10))
        other=VisualCaptureHistory()
        with self.assertRaises(ValueError): other.commit(pending)
        self.history.reset()
        with self.assertRaises(ValueError): self.history.commit(pending)
        next_pending=self.history.stage(self.advance(10))
        self.assertEqual(next_pending.reason,'reset')
        self.history.commit(next_pending)
        with self.assertRaises(ValueError): self.history.commit(next_pending)

    def test_source_epoch_change_resets_history_not_identity_proof(self):
        h=self.advance(10); self.history.commit(self.history.stage(h))
        changed=copy.deepcopy(h)
        changed['visual_sensor']['stream_epoch']='next'
        changed['frames'][0]['rgb_sha256']='1'*64
        pending=self.history.stage(changed)
        self.assertEqual(pending.reason,'reconnect')
        self.history.commit(pending)

    def test_evicted_and_late_captures_cannot_reenter_as_unchecked_new(self):
        for first in range(0,640,128):
            self.history.commit(self.history.stage(self.advance(first,128)))
        self.assertEqual(len(self.history._records),512)
        with self.assertRaises(ValueError): self.history.stage(self.advance(0))
        fresh=VisualCaptureHistory()
        fresh.commit(fresh.stage(self.advance(10)))
        with self.assertRaises(ValueError): fresh.stage(self.advance(9))

    def test_new_ids_cannot_hide_capture_time_regression(self):
        self.history.commit(self.history.stage(self.advance(10,10)))
        for intervals in ([[1040000,1040000],[1140000,1140000]],
                          [[1900000,1900000],[2000000,2000000]],
                          [[1899999,1900001],[2000000,2000001]]):
            changed=self.advance(20)
            for frame,interval in zip(changed['frames'],intervals):
                frame['interval_us']=interval
            with self.assertRaises(ValueError):self.history.stage(changed)
        # Failed transactions preserve the frontier and valid overlapping IDs.
        self.history.commit(self.history.stage(self.advance(19)))
        changed=self.advance(1)
        changed['visual_sensor']['stream_epoch']='new-clock-epoch'
        pending=self.history.stage(changed)
        self.assertEqual(pending.reason,'reconnect')
        self.history.commit(pending)
