"""Hand joints in the movement rule language: per keypoint set, each model's own score gate, and ``hand_open``."""
import math
import unittest

from gamst.plugins import InputFrame
from gamst.private_payload import PrivatePoses
from gamst.providers.movement_rule import RuleScorer, hand_open, validate_rule
from gamst.providers.pose_exemplar import (COCO_WHOLEBODY_HANDS, GOLIATH_HANDS, HAND_JOINTS, MEDIAPIPE_BODY,
                                           ExemplarEvents, body_points, keypoint_reading)


def hand(side, opened, wrist=(.6, .5), scale=.05, visibility=.9, tips_visible=4):
    """{joint: (x, y, visibility)} of one hand, fingers up from the wrist; ``opened`` 1 stretched, 0 curled to the palm.
    Palm length (wrist to middle MCP) is ``scale``."""
    points = {f'{side}_hand_wrist': (wrist[0], wrist[1], visibility)}
    for number, finger in enumerate(('thumb', 'index', 'middle', 'ring', 'pinky')):
        x = wrist[0] + (number - 2) * scale * .3
        joints = ('cmc', 'mcp', 'ip', 'tip') if finger == 'thumb' else ('mcp', 'pip', 'dip', 'tip')
        for step, joint in enumerate(joints):
            if joint == 'mcp' and finger != 'thumb' or joint == 'cmc':
                y = wrist[1] - scale * (1. if joint == 'mcp' else .4)
            else:      # stretched: up beyond the knuckle; curled: back down toward the palm's middle
                reach = .3 * step if opened else -.25 * step
                y = wrist[1] - scale * (1. + reach)
            seen = visibility if not (joint == 'tip' and number >= 1 + tips_visible) else .05
            points[f'{side}_{finger}_{joint}'] = (x, y, seen)
    return points


def pose(points, keypoint_set='coco_wholebody133'):
    """A pose of ``keypoint_set`` with ``points`` {joint: (x, y, visibility)} placed by that set's own indices."""
    size, mapping = {'coco_wholebody133': 133, 'goliath308': 308}[keypoint_set], \
        {'coco_wholebody133': COCO_WHOLEBODY_HANDS, 'goliath308': GOLIATH_HANDS}[keypoint_set]
    landmarks = [dict(x=0., y=0., visibility=0.) for _ in range(size)]
    for name, (x, y, visibility) in points.items():
        landmarks[mapping[name]] = dict(x=x, y=y, visibility=visibility)
    return dict(keypoint_set=keypoint_set, landmarks=landmarks, image_width=100, image_height=100)


def points_of(item, hands=True):
    own, gate = keypoint_reading(item, MEDIAPIPE_BODY, .5, hands=hands)
    return body_points(item['landmarks'], own, min_visibility=gate)


class HandJoints(unittest.TestCase):
    def test_indices_follow_each_keypoint_set(self):
        self.assertEqual(len(HAND_JOINTS), 42)
        # COCO-WholeBody (rtmlib coco133.py): left_hand_root 91, left_thumb4 95, right_hand_root 112, right_pinky_finger4 132
        self.assertEqual([COCO_WHOLEBODY_HANDS[name] for name in
                          ('left_hand_wrist', 'left_thumb_tip', 'left_index_mcp', 'right_hand_wrist', 'right_pinky_tip')],
                         [91, 95, 96, 112, 132])
        self.assertEqual(sorted(COCO_WHOLEBODY_HANDS.values()), list(range(91, 133)))
        # Goliath 308 (Meta's keypoints308.py at 7e5bae8): right_thumb4 21, right_thumb_third_joint 24,
        # right_pinky_finger_third_joint 40, right_wrist 41, left_thumb4 42, left_middle_finger_third_joint 53, left_wrist 62
        self.assertEqual([GOLIATH_HANDS[name] for name in
                          ('right_thumb_tip', 'right_thumb_cmc', 'right_pinky_mcp', 'right_hand_wrist', 'left_thumb_tip',
                           'left_middle_mcp', 'left_hand_wrist')], [21, 24, 40, 41, 42, 53, 62])
        self.assertEqual(sorted(GOLIATH_HANDS.values()), list(range(21, 63)))

    def test_hands_are_read_where_the_model_has_them_with_its_own_gate(self):
        both = {**hand('left', 1), **hand('right', 0)}
        for keypoint_set in ('coco_wholebody133', 'goliath308'):
            own, gate = keypoint_reading(pose(both, keypoint_set), hands=True)
            self.assertEqual(gate, .3)
            self.assertTrue(set(HAND_JOINTS) <= set(own))
            self.assertEqual(set(HAND_JOINTS) & set(keypoint_reading(pose(both, keypoint_set))[0]), set())   # not asked
        mediapipe = dict(landmarks=[dict(x=.5, y=.5, visibility=1.)] * 33)
        own, gate = keypoint_reading(mediapipe, hands=True)
        self.assertEqual((own, gate), (MEDIAPIPE_BODY, .5))     # MediaPipe Pose has no hands: unknown there
        # a hand point at score 0.35 counts for RTMW (its gate is 0.3), not under MediaPipe's 0.5
        dim = pose(hand('right', 0, visibility=.35))
        self.assertIn('right_index_tip', points_of(dim))
        self.assertNotIn('right_index_tip', body_points(dim['landmarks'], COCO_WHOLEBODY_HANDS, min_visibility=.5))


class HandOpen(unittest.TestCase):
    def test_a_fist_is_low_an_open_hand_high_and_unknown_without_the_hand(self):
        opened, fist = points_of(pose(hand('right', 1))), points_of(pose(hand('right', 0)))
        self.assertGreater(hand_open(opened, 'right'), 1.9)       # palm + stretched fingers (1.94 here)
        self.assertLess(hand_open(fist, 'right'), .5)             # tips back in the palm (0.42 here)
        self.assertIsNone(hand_open(opened, 'left'))
        self.assertIsNotNone(hand_open(points_of(pose(hand('right', 0, tips_visible=2))), 'right'))
        self.assertIsNone(hand_open(points_of(pose(hand('right', 0, tips_visible=1))), 'right'))
        big = points_of(pose(hand('right', 1, scale=.2)))
        self.assertAlmostEqual(hand_open(big, 'right'), hand_open(opened, 'right'), places=6)   # size-free
        self.assertTrue(math.isfinite(hand_open(points_of(pose(hand('right', 1), 'goliath308')), 'right')))

    def test_rules_name_hand_joints_and_hand_open(self):
        fist = dict(measure=dict(hand_open='right'), op='lt', value=1.)
        rule = validate_rule(dict(steps=[dict(when=dict(all=[
            fist, dict(measure=dict(distance=['right_thumb_tip', 'right_index_mcp']), op='lt', value=.5),
            dict(measure=dict(rate=dict(hand_open='left')), op='gt', value=-100)]))]))
        scorer = RuleScorer(rule)
        self.assertTrue(scorer.needs_joints and scorer.needs_hands)
        self.assertFalse(RuleScorer(dict(steps=[dict(when=dict(
            measure=dict(height=['right_wrist', 'right_shoulder']), op='gt', value=0))])).needs_hands)
        for bad in (dict(hand_open='both'), dict(hand_open=['right']), dict(distance=['right_thumb', 'right_wrist']),
                    dict(angle=['right_index_tip', 'right_index_pip', 'right_index_tip'])):
            with self.assertRaises(ValueError, msg=str(bad)):
                validate_rule(dict(steps=[dict(when=dict(measure=bad, op='lt', value=1.))]))
        with self.assertRaises(ValueError):
            validate_rule(dict(channels='audio.voice_channels', steps=[dict(when=fist)]))
        one = RuleScorer(dict(steps=[dict(when=fist)]))
        channels, mask = [0.] * 13, [False] * 13
        self.assertEqual(one.violation(0, channels, mask, points_of(pose(hand('right', 0)))), 0.)
        self.assertGreater(one.violation(0, channels, mask, points_of(pose(hand('right', 1)))), .9)
        self.assertIsNone(one.violation(0, channels, mask, {}))

    def test_the_counter_counts_fists_from_the_joints_input(self):
        module = ExemplarEvents(min_hold_ms=0)
        module.set_exemplar(dict(rule=dict(steps=[
            dict(when=dict(measure=dict(hand_open='right'), op='gt', value=1.5)),
            dict(when=dict(measure=dict(hand_open='right'), op='lt', value=1.), within_ms=500)])))
        frames = ([1] * 4 + [0] * 4) * 3
        count = 0
        for index, opened in enumerate(frames):
            out = module.process(InputFrame(PrivatePoses([pose(hand('right', opened, visibility=.35))]), 'video',
                                            1000 + 33 * index))
            count += sum(event['event'] == 'exemplar.matched' for event in (out or {}).get('events', ()))
        self.assertEqual(count, 3)
        mediapipe = ExemplarEvents(min_hold_ms=0)
        mediapipe.set_exemplar(dict(rule=dict(steps=[dict(when=dict(measure=dict(hand_open='right'), op='lt', value=1.))])))
        out = mediapipe.process(InputFrame(PrivatePoses([dict(landmarks=[dict(x=.5, y=.5, visibility=1.)] * 33)]), 'video', 1000))
        self.assertNotIn('distance', out)     # MediaPipe body-only: unknown, never a match


if __name__ == '__main__':
    unittest.main()
