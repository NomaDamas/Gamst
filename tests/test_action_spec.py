"""A movement given as text (a rule), as example clips, or both (owner decision 58), and the evaluation tools that score
it against owner labels. Synthetic data and fake models only: nothing here measures accuracy on a real clip."""
import argparse
import asyncio
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from gamst.eval import multi_person as M, pose_dump as D
from gamst.plugins import InputFrame
from gamst.providers.movement_match import derive, example
from gamst.providers.movement_rule import calibrate, validate_rule
from gamst.providers.pose_exemplar import ExemplarEvents
from gamst.recipes.movement import template

STEP = 50


def level(value, other=0.):
    """Five voice channels: channel 1 carries the movement, channel 0 how it is done."""
    return [other, value, 0., 0., 0.]


def cycle(other=0., low=-.2, high=.8):
    """low -> high -> low: one repetition (450 ms)."""
    return [level(low, other)] * 2 + [level(low + (high - low) * k / 3, other) for k in (1, 2)] + \
        [level(high, other)] * 3 + [level(low + (high - low) * k / 3, other) for k in (2, 1)]


RULE = dict(channels='audio.voice_channels', steps=[
    dict(when=dict(measure=dict(channel='level_db'), op='lte', value=0.)),
    dict(when=dict(measure=dict(channel='level_db'), op='gte', value=.5), within_ms=600),
    dict(when=dict(measure=dict(channel='level_db'), op='lte', value=0.), within_ms=600)])


def samples(stream, start=1000):
    return [(start + STEP * i, channels, [True] * 5, {}) for i, channels in enumerate(stream)]


def combined_value(combine, rule=RULE, examples=None):
    examples = examples or [cycle() + [level(-.2)] * 2]
    entries = []
    for stream in examples:
        plain = [item[:3] for item in samples(stream)]
        saved = template('sequence', plain)
        entries.append(dict(saved=saved, template=example(saved['value'], span_ms=saved['span_ms']), clip=plain))
    thresholds = derive('sequence', entries, [], enter0=.12, exit0=.18)
    return dict(format=2, label='move', mode='rule+sequence', examples_mode='sequence', rule=rule, combine=combine,
                positives=[item['saved'] for item in entries], negatives=[], confirmed=True, **thresholds)


def run(module, stream, start=1000):
    found = []
    for index, channels in enumerate(stream):
        out = module.process(InputFrame(dict(channels=channels, mask=[True] * 5), 'audio', start + STEP * index))
        found += out['events'] if out else []
    out = module.finish()
    return found + (out['events'] if out else [])


def matched(found):
    return [item for item in found if item['event'] == 'exemplar.matched']


class RuleAndExamples(unittest.TestCase):
    """Count mode: the rule and an example; search mode: either (decision 58)."""

    def stream(self):
        quiet = [level(-.2)] * 6
        like = cycle()                       # the rule holds and it is a copy of the example
        unlike = cycle(other=3.)             # the rule holds, but channel 0 is far from the example
        gentle = cycle(low=.05, high=.45)    # the shape of the example, but the rule's numbers are not reached
        return quiet + like + quiet + unlike + quiet + gentle + quiet

    def module(self, combine):
        module = ExemplarEvents(modality='audio')
        module.set_exemplar(combined_value(combine))
        return module

    def test_count_mode_needs_both_and_search_mode_takes_either_once(self):
        both = matched(run(self.module('all'), self.stream()))
        either = matched(run(self.module('any'), self.stream()))
        self.assertEqual(len(both), 1)
        self.assertEqual(len(either), 3)
        self.assertTrue(both[0]['start_ms'] < 1700 < both[0]['end_ms'] + 400)
        rule_only = [item for item in either if not item['evidence']['example'] or not item['evidence']['example'].get('close', True)]
        self.assertEqual(len(rule_only), 1)         # the unlike one: by the rule alone
        # the like one is found by both (counted once, by whichever said it first), the gentle one by the example alone
        self.assertTrue(all(item['evidence']['example'] for item in either))

    def test_every_event_carries_the_rule_and_the_nearest_example(self):
        event = matched(run(self.module('all'), self.stream()))[0]
        evidence = event['evidence']
        self.assertEqual(evidence['combine'], 'all')
        self.assertTrue(evidence['rule']['satisfied'])
        self.assertEqual(len(evidence['rule']['steps_ms']), 3)
        self.assertEqual(evidence['example']['index'], 0)
        self.assertLessEqual(evidence['example']['distance'], evidence['example']['enter'])
        self.assertEqual(event['method'], 'rule+examples')

    def test_without_combine_the_runs_mode_decides(self):
        value = combined_value(None)
        for mode, expected in (('count', 1), ('search', 3)):
            module = ExemplarEvents(modality='audio')
            module._host = argparse.Namespace(mode=mode)
            module.set_exemplar(value)
            self.assertEqual(len(matched(run(module, self.stream()))), expected, mode)

    def test_repetitions_are_counted_one_by_one(self):
        module = self.module('all')
        found = matched(run(module, [level(-.2)] * 4 + cycle() * 3 + [level(-.2)] * 4))
        self.assertEqual(len(found), 3)
        self.assertEqual(len({item['start_ms'] for item in found}), 3)

    def test_plain_rules_and_examples_carry_evidence_too(self):
        module = ExemplarEvents(modality='audio')
        module.set_exemplar(dict(rule=RULE))
        event = matched(run(module, [level(-.2)] * 4 + cycle() + [level(-.2)] * 4))[0]
        self.assertTrue(event['evidence']['rule']['satisfied'])
        self.assertEqual(event['steps_ms'], event['evidence']['rule']['steps_ms'])
        value = combined_value('all')
        module = ExemplarEvents(modality='audio')
        module.set_exemplar({key: item for key, item in value.items() if key not in ('rule', 'combine')} | dict(mode='sequence'))
        event = matched(run(module, [level(-.2)] * 4 + cycle() + [level(-.2)] * 4))[0]
        self.assertEqual(event['evidence']['example']['index'], 0)

    def test_a_bad_combine_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'combine'):
            ExemplarEvents(modality='audio').set_exemplar(combined_value('both'))


class Calibration(unittest.TestCase):
    """The clips set the numbers of the text's rule: positives only widen, a negative narrows (decision 58)."""

    def test_positives_widen_what_they_do_not_pass_and_keep_the_rest(self):
        lower = cycle(high=.42)                         # reaches .42, the text asks .5
        rule, evidence = calibrate(RULE, [samples(lower), samples(cycle())])
        self.assertAlmostEqual(rule['steps'][1]['when']['value'], .42 - .05, places=4)
        self.assertEqual(rule['steps'][0]['when']['value'], 0.)       # both pass the fold as written
        self.assertEqual(evidence['leaves'][1]['text'], .5)
        self.assertGreaterEqual(rule['steps'][1]['within_ms'], 600)

    def test_a_clip_that_does_not_show_a_step_does_not_teach_it(self):
        trimmed = cycle()[3:]                           # starts at the spread: no fold before it
        rule, evidence = calibrate(dict(RULE, steps=[dict(RULE['steps'][0], when=dict(
            measure=dict(channel='level_db'), op='lte', value=-.5))] + RULE['steps'][1:]), [samples(trimmed)])
        self.assertEqual(rule['steps'][0]['when']['value'], -.5)
        self.assertFalse(evidence['shows_step'][0][0])

    def test_a_negative_the_rule_would_count_narrows_one_comparison(self):
        wide = dict(RULE, steps=[RULE['steps'][0], dict(RULE['steps'][1], when=dict(
            measure=dict(channel='level_db'), op='gte', value=.3)), RULE['steps'][2]])
        rule, evidence = calibrate(wide, [samples(cycle(high=.9))], [samples(cycle(high=.4))])
        self.assertTrue(.4 < rule['steps'][1]['when']['value'] < .9)
        self.assertEqual(evidence['negatives_moved'][0]['step'], 1)


class Enrollment(unittest.TestCase):
    """gamst recipe enroll takes a rule with clips (it refused that before decision 58)."""

    def test_a_rule_with_clips_becomes_one_combined_movement_file(self):
        from gamst.recipes import movement
        recipe = dict(name='person-action-count', modules=[dict(name='exemplar', use='exemplar')], references=[])
        with tempfile.TemporaryDirectory() as folder:
            rule_file = Path(folder, 'rule.json')
            rule_file.write_text(json.dumps(RULE))
            args = argparse.Namespace(text='up and down', name=None, rule=str(rule_file), negative=[], proposals=None,
                                      proposal=[], no_confirm=False, mode='sequence', out=str(Path(folder, 'm.json')),
                                      clips=([movement.Clip('video', 'a.mp4', 1., 2.), movement.Clip('video', 'a.mp4', 3., 4.)], []),
                                      combine=None, keep_rule_numbers=False)

            async def demonstration(recipe, args, clip, joints=False):
                stream = cycle(high=.45) if clip.start == 1. else cycle()
                return [(ts, channels, mask, {'x': (0., 0.)}) for ts, channels, mask, _ in samples(stream)], 'audio.voice_channels@1'
            with mock.patch.object(movement, 'demonstration', demonstration), \
                    mock.patch.object(movement, 'exemplar_input', lambda recipe: dict(use='camera')):
                saved = asyncio.run(movement.enroll(recipe, args))
            self.assertEqual((saved['mode'], saved['method'], saved['rule_source']), ('rule+sequence', 'rule+examples', 'text+clips'))
            self.assertEqual(len(saved['positives']), 2)
            self.assertLess(saved['rule']['steps'][1]['when']['value'], .5)
            self.assertEqual(saved['rule_text']['steps'][1]['when']['value'], .5)
            module = ExemplarEvents(modality='audio')
            module.set_exemplar(saved)
            self.assertEqual(module.mode, 'rule+sequence')

    def test_a_negative_without_clips_is_refused(self):
        from gamst.recipes import movement
        recipe = dict(name='r', modules=[dict(name='exemplar', use='exemplar')], references=[])
        with tempfile.TemporaryDirectory() as folder:
            rule_file = Path(folder, 'rule.json')
            rule_file.write_text(json.dumps(RULE))
            args = argparse.Namespace(text='t', name=None, rule=str(rule_file), negative=['b.mp4@1-2'], proposals=None,
                                      proposal=[], no_confirm=False, mode=None, out=None, clips=None)
            with mock.patch.object(movement, '_positives', lambda recipe, args: []):
                with self.assertRaisesRegex(SystemExit, 'needs example clips'):
                    asyncio.run(movement.enroll(recipe, args))


class Scoring(unittest.TestCase):
    """The owner's scorer rules (score_multi.py), ported: one to one by time, excluded ranges, halves, reasons."""

    truth = [dict(start=1., end=2., who='감스트'), dict(start=3., end=4., who='감스트'), dict(start=5., end=6., who='other'),
             dict(start=7., end=7.5, who='감스트'), dict(start=7.6, end=8., who='감스트')]

    def test_counts_precision_recall_and_times(self):
        gamst = [dict(start=1.1, end=2.1, who='감스트'), dict(start=3., end=4.2, who='unknown'),
                 dict(start=5., end=6., who='감스트'), dict(start=7., end=8., who='감스트'), dict(start=9., end=9.5, who='감스트')]
        score = M.score_action(gamst, self.truth)
        self.assertEqual((score['감스트']['truth'], score['감스트']['gamst'], score['감스트']['matched']), (4, 4, 2))
        self.assertEqual(score['모두']['matched'], 4)
        self.assertAlmostEqual(score['감스트']['start_err_s'], .05)
        errors = M.breakdown(gamst, self.truth)
        self.assertEqual(errors['wrong'], dict(identity=1, split=0, extra=1))
        self.assertEqual(errors['missed'], dict(identity=1, merged=1, missed=0))
        everyone = M.breakdown(gamst, self.truth, person=M.EVERYONE)
        self.assertEqual(everyone['missed'], dict(identity=0, merged=1, missed=0))

    def test_examples_never_score_themselves_and_halves_split_by_scene(self):
        gamst = [dict(start=1.1, end=2.1, who='감스트')]
        self.assertEqual(M.score_action(gamst, self.truth, exclude=[(0.5, 1.5)])['감스트']['truth'], 3)
        cuts = [dict(event='scene.cut', start_ms=1000 + 4500), dict(event='person.bound', start_ms=1)]
        scenes = M.scenes(cuts, 1000, 10.)
        self.assertEqual(scenes, [(0., 4.5), (4.5, 10.)])
        odd, even = M.scene_halves(scenes)
        self.assertEqual(M.score_action(gamst, self.truth, keep=even)['감스트']['truth'], 2)

    def test_occurrences_take_who_from_the_bindings_retroactively(self):
        events = [dict(event='exemplar.matched', track_id='t1', index=1, track_index=1, start_ms=1000, end_ms=1500, who='unknown'),
                  dict(event='exemplar.ended', track_id='t1', index=1, track_index=1, start_ms=1000, end_ms=1700)]
        bindings = [dict(event='person.bound', track_id='t1', who='감스트', person_id='P1', start_ms=900, end_ms=2000,
                         by='face', retroactive=True)]
        [item] = M.occurrences(events, 0, bindings)
        self.assertEqual((item['who'], item['start'], item['end']), ('감스트', 1., 1.7))


class OneOf(unittest.TestCase):
    """Owner decisions of 2026-11-28 (truth v2): two boxes labelled 감스트 at once are both truth and GAMST may name either
    ("one of"); photos and superseded segments are left out of identity scoring."""

    A, B, C = (.1, .1, .2, .6), (.6, .1, .2, .6), (.35, .5, .2, .4)

    def boxes(self, *tracks, frames=10, fps=10.):
        return [(k / fps, track, box) for k in range(frames) for track, box in tracks]

    @staticmethod
    def bound(track, who='감스트'):
        return dict(event='person.bound', track_id=track, who=who, person_id='P1', start_ms=0, end_ms=1000)

    def segment(self, label, box, frames=10, fps=10., **extra):
        return dict(label=label, boxes=[[k / fps, *box] for k in range(frames)], **extra)

    def test_two_target_boxes_at_once_count_once_and_either_is_enough(self):
        truth = dict(segments=[self.segment('G', self.A), self.segment('G', self.B)])
        boxes = self.boxes(('a', self.A), ('b', self.B))
        bindings = [self.bound('a')]                                   # GAMST names 감스트 on one of the two
        now = M.score_identity(boxes, bindings, 0, truth, fps=10.)
        self.assertEqual((now['frames']['tp'], now['frames']['fn'], now['frames']['recall']), (10, 0, 1.))
        self.assertEqual(now['several'], 10)
        before = M.score_identity(boxes, bindings, 0, truth, fps=10., one_of=False)
        self.assertEqual((before['frames']['tp'], before['frames']['fn']), (10, 10))
        both = M.score_identity(boxes, bindings + [self.bound('b')], 0, truth, fps=10.)
        self.assertEqual((both['frames']['tp'], both['frames']['fp'], both['frames']['precision']), (10, 0, 1.))

    def test_an_added_stretch_and_a_box_at_the_same_frame_are_one_unit(self):
        truth = dict(segments=[self.segment('G', self.A)], added=[dict(label='G', start=0., end=.9)])
        score = M.score_identity(self.boxes(('a', self.A)), [self.bound('a')], 0, truth, fps=10.)
        self.assertEqual((score['frames']['tp'], score['frames']['fn']), (10, 0))
        self.assertEqual(M.score_identity(self.boxes(('a', self.A)), [self.bound('a')], 0, truth, fps=10.,
                                          one_of=False)['frames']['tp'], 20)

    def test_photos_and_superseded_segments_are_neither_hits_nor_errors(self):
        truth = dict(segments=[self.segment('G', self.A), self.segment('G', self.C, photo=True),
                               self.segment('O', self.B, superseded=True)],
                     superseded=[self.segment('O', self.C, superseded=True)])
        boxes = self.boxes(('a', self.A), ('c', self.C), ('b', self.B))
        score = M.score_identity(boxes, [self.bound('a'), self.bound('b')], 0, truth, fps=10.)  # c (photo) unnamed
        self.assertEqual(score['frames'], dict(tp=10, fp=0, fn=0, tn=0, precision=1., recall=1.))
        self.assertEqual(score['left_out'], dict(photo=10, superseded=20))
        self.assertEqual([item['label'] for item in M.scored_segments(truth)], ['G'])
        self.assertEqual(M.target_moments(truth, fps=10.), [])      # the photo does not make a second 감스트

    def test_target_moments_are_the_frames_with_several_target_items(self):
        truth = dict(segments=[self.segment('G', self.A, frames=30), self.segment('O', self.B, frames=30),
                               dict(label='G', boxes=[[k / 10., *self.C] for k in range(10, 20)])])
        self.assertEqual(M.target_moments(truth, fps=10.), [(.95, 1.95)])

    def test_the_same_action_on_the_two_boxes_is_neither_a_hit_nor_an_error(self):
        truth = [dict(start=1., end=2., who='감스트'), dict(start=4., end=5., who='감스트')]
        gamst = [dict(start=1.1, end=2., who='감스트', track_id='a'), dict(start=1.2, end=2.1, who='감스트', track_id='b')]
        moments = [(.5, 2.5)]
        now = M.score_action(gamst, truth, together=moments)
        self.assertEqual((now['감스트']['gamst'], now['감스트']['matched'], now['감스트']['precision']), (1, 1, 1.))
        self.assertEqual(now['same_moment'], 1)
        self.assertEqual(M.breakdown(gamst, truth, together=moments)['wrong'], dict(identity=0, split=0, extra=0))
        # before the owner's decision (or outside such a moment) the second box's event is a split
        before = M.score_action(gamst, truth)
        self.assertEqual((before['감스트']['gamst'], before['감스트']['matched']), (2, 1))
        self.assertEqual(M.breakdown(gamst, truth)['wrong']['split'], 1)
        self.assertEqual(M.score_action(gamst, truth, together=[(5., 6.)])['감스트']['gamst'], 2)
        # one track's two events are never the same action: a repetition is not a duplicate
        same = [dict(start=1.1, end=2., who='감스트', track_id='a'), dict(start=1.5, end=2.2, who='감스트', track_id='a')]
        self.assertEqual(M.score_action(same, truth, together=moments)['감스트']['gamst'], 2)

    def test_another_box_s_event_that_matched_the_next_repetition_stays_a_match(self):
        truth = [dict(start=1., end=1.8, who='감스트'), dict(start=2., end=2.8, who='감스트')]
        gamst = [dict(start=1., end=1.9, who='감스트', track_id='a'), dict(start=1.7, end=2.8, who='감스트', track_id='b')]
        now = M.score_action(gamst, truth, together=[(0., 3.)])
        self.assertEqual((now['감스트']['gamst'], now['감스트']['matched'], now['same_moment']), (2, 2, 0))

    def test_the_v2_truth_folder_marks_photos_and_keeps_the_superseded(self):
        folder = Path(__file__).resolve().parents[1] / 'docs/ground-truth/2026-11-28/v2/man'
        if not folder.is_dir():
            self.skipTest('truth v2 not in this checkout')
        truth = M.load_truth(folder)['identity']
        self.assertEqual(truth.get('version'), 2)
        self.assertEqual(sorted(item['segment'] for item in truth['superseded']), sorted(['person-12', 'person-2~mwzau1al', 'person-27']))
        photos = sorted(item['segment'] for item in truth['segments'] if item.get('photo'))
        self.assertEqual(photos, sorted(['person-12~v2a', 'person-2~v2b', 'person-4', 'person-5', 'person-6', 'person-8']))
        late = [(item['start'], item['end'], item['label']) for item in truth['segments'] if item.get('split_of') == 'person-27']
        self.assertEqual(sorted(late), [(42.042, 44.24, 'O'), (44.24, 50.48, 'O'), (50.48, 63.53, 'G')])


class PoseDump(unittest.TestCase):
    """A recorded run replays through the same per-track counter: the same events."""

    def test_record_then_replay_gives_the_run_s_events(self):
        from gamst.private_payload import PrivatePoses
        from gamst.providers.track_exemplar import TrackExemplar
        size = 133
        rule = dict(steps=[dict(when=dict(measure=dict(height=['left_wrist', 'left_shoulder']), op='gt', value=.5),
                                hold_ms=0)])

        def pose(track, up):
            shift = 0. if track == 'a' else .5          # two people side by side
            marks = [dict(x=.5, y=.5, visibility=0.) for _ in range(size)]
            for index, (x, y) in {5: (.1, .4), 6: (.3, .4), 7: (.1, .5), 8: (.3, .5), 9: (.1, .2 if up else .6),
                                  10: (.3, .6), 11: (.15, .7), 12: (.25, .7)}.items():
                marks[index] = dict(x=x + shift, y=y, visibility=.9, raw_score=5.)
            return dict(landmarks=marks, keypoint_set='coco_wholebody133', track_id=track,
                        bbox=dict(x=.05 + shift, y=.1, width=.3, height=.8), image_width=640, image_height=640)
        frames = [PrivatePoses([pose('a', (i // 4) % 2 == 1), pose('b', (i // 6) % 2 == 1)]) for i in range(40)]
        value = dict(rule=rule)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, 'run.jsonl.gz')
            with gzip.open(path, 'wt') as out, D.Recorder(out) as recorder:
                counter = D.counter_for(value)
                live = []
                for i, data in enumerate(frames):
                    result = counter.process(InputFrame(data, 'video', 1000 + 33 * i))
                    live += result['events'] if result else []
            dump = D.load(path)
            self.assertEqual(len(dump['frames']), 40)
            replayed = D.replay(dump, value)
            key = lambda item: (item['event'], item['track_id'], item['start_ms'], item['end_ms'])
            self.assertEqual(sorted(map(key, replayed)), sorted(map(key, live + (counter.finish() or {}).get('events', []))))
            self.assertTrue(any(item['track_id'] == 'b' for item in replayed))
            self.assertIsInstance(counter, TrackExemplar)


class Combinations(unittest.TestCase):
    """compose_recipe: a person needs a text and photos; a movement takes text, clips or both (decisions 58, 59)."""

    def test_photos_and_examples_are_enrolled_beside_the_recipe(self):
        from gamst.recipes import compose as C
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / 'b.mp4').write_bytes(b'0')
            (folder / 'a.jpg').write_bytes(b'0')
            seen = {}

            def photos(who, path, profile=None):
                seen['photos'] = who
                Path(path).write_text(json.dumps(dict(kind='person', name=who['name'], consent=dict(statement='ok'),
                                                      samples=[dict(persons=1)] * 3, face=[[1., 0.], [1., 0.1], [1., .05]])))

            def examples(movement, path, profile=None, actor='user', who=None):
                seen['examples'], seen['from'] = movement, who
                Path(path).write_text(json.dumps(dict(kind='exemplar', mode='rule+sequence', id='x', rule_source='text+clips',
                                                      threshold_source='leave_one_out', enter=.2, exit=.3)))
            request = dict(name='gwan', inputs=dict(video=str(folder / 'b.mp4')),
                           who=dict(name='감스트', text='회색 맨투맨, 검은 뿔테', photos=[str(folder / 'a.jpg')], consent='owner, 2026-11-28'),
                           what=dict(movement=dict(text='관제탑', rule=RULE | dict(channels='vision.pose_channels', steps=[
                               dict(when=dict(measure=dict(distance=['left_wrist', 'right_wrist']), op='lt', value=1.2))]),
                               examples=[dict(file=str(folder / 'b.mp4'), start_s=1, end_s=2), str(folder / 'b.mp4') + '@3-4@0.5,0.4'],
                               examples_from='who')))
            with mock.patch.object(C, 'enroll_photos', photos), mock.patch.object(C, 'enroll_examples', examples):
                result = C.compose(request, out=str(folder / 'gwan.json'))
            self.assertTrue(result['ok'], result.get('missing'))
            self.assertEqual(seen['examples']['examples'], [f'{folder / "b.mp4"}@1-2', f'{folder / "b.mp4"}@3-4@0.5,0.4'])
            self.assertEqual(seen['from']['name'], '감스트')
            self.assertIn('--exemplar', result['run'])
            self.assertEqual(result['run'][result['run'].index('--person') + 1], str(folder / 'gwan.person.json'))
            self.assertEqual(json.loads((folder / 'gwan.person.json').read_text())['description'], '회색 맨투맨, 검은 뿔테')
            why = {item['capability']: item['why'] for item in result['reasons']}
            self.assertIn('decision 58', why['movement'])
            self.assertIn('photo', why['who'])

    def test_examples_need_a_file_to_be_enrolled_into(self):
        from gamst.recipes import compose as C
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / 'b.mp4').write_bytes(b'0')
            result = C.compose(dict(inputs=dict(video=str(Path(folder) / 'b.mp4')), what=dict(movement=dict(
                examples=[str(Path(folder) / 'b.mp4') + '@1-2']))))
            self.assertFalse(result['ok'])
            self.assertIn('--out', result['missing'][0])



class Instructions(unittest.TestCase):
    """set_instruction: a rule with clips, combine, and who as {name, text, photos, consent} (decisions 58, 59)."""

    def instructions(self):
        from types import SimpleNamespace
        from gamst.recipes.instructions import Instructions as Real
        found = Real.__new__(Real)
        found._busy = None
        found._name = lambda kind, reference: 'exemplar'
        found.pipeline = SimpleNamespace(references={}, _instructable=lambda: None)
        found._movement = mock.AsyncMock(return_value=dict(label='관제탑', confirmed=True))
        found._apply = mock.AsyncMock(return_value=dict(reference_revision=2))
        found._who = lambda who, kind, name=None: None
        found.enroll_person = mock.AsyncMock(return_value=dict(name='감스트'))
        return found

    def test_a_rule_with_clips_is_one_instruction(self):
        found = self.instructions()
        asyncio.run(found.set('exemplar', rule=RULE, clips=[dict(file='a.mp4')], negatives=[dict(file='b.mp4')],
                              text='관제탑', combine='any'))
        options = found._movement.call_args.kwargs
        self.assertEqual((options['rule'], options['combine'], len(options['clips'])), (RULE, 'any', 1))
        self.assertEqual(found._apply.call_args.kwargs['source'], 'rule+clips')
        with self.assertRaisesRegex(ValueError, 'combine'):
            asyncio.run(found.set('exemplar', rule=RULE, text='관제탑', combine='all'))

    def test_who_needs_words_and_photos(self):
        found = self.instructions()
        with self.assertRaisesRegex(ValueError, 'say who it is in words'):
            asyncio.run(found.set('exemplar', rule=RULE, text='t', who=dict(photos=['a.jpg'], consent='c')))
        with self.assertRaisesRegex(ValueError, 'alone never identifies a person'):
            asyncio.run(found.set('exemplar', rule=RULE, text='t', who=dict(name='감스트')))
        asyncio.run(found.set('exemplar', rule=RULE, text='t', who=dict(name='감스트', photos=['a.jpg'], consent='owner')))
        self.assertEqual(found.enroll_person.call_args.kwargs['images'], ['a.jpg'])




class ChosenPerson(unittest.TestCase):
    """Example clips learn from the clicked person or the person of who, not whoever is largest."""

    def people(self):
        box = lambda x: dict(x=x, y=.1, width=.3, height=.8)
        return {ts: [('a', box(.05), [1.] * 13, [True] * 13, {'a': (0., 0.)}),
                     ('b', box(.6), [2.] * 13, [True] * 13, {'b': (0., 0.)})] for ts in (100, 133, 166)}

    def test_a_point_picks_the_box_there_and_follows_its_track(self):
        from gamst.recipes.movement import chosen_person, parse_clip
        chosen = chosen_person(self.people(), at=(.7, .5))
        self.assertEqual([channels[0] for channels, _, _ in chosen.values()], [2.] * 3)
        clip = parse_clip('video', 'a.mp4@62.051-63.951@0.7,0.5')
        self.assertEqual((clip.start, clip.end, clip.at), (62.051, 63.951, (.7, .5)))
        self.assertEqual(clip.name(), 'a.mp4@62.051-63.951@0.7,0.5')
        with self.assertRaises(SystemExit):
            parse_clip('video', 'a.mp4@1-2@1.5,0.2')

    def test_who_picks_the_track_the_binding_names(self):
        from gamst.recipes.movement import chosen_person
        bindings = [dict(event='person.bound', track_id='a', who='감스트', start_ms=0, end_ms=500, by='face',
                         retroactive=True)]
        chosen = chosen_person(self.people(), who='감스트', bindings=bindings)
        self.assertEqual([channels[0] for channels, _, _ in chosen.values()], [1.] * 3)
        self.assertEqual(chosen_person(self.people(), who='Q', bindings=bindings), {})


HAS_CV = all(__import__('importlib.util').util.find_spec(name) for name in ('cv2', 'numpy'))


@unittest.skipUnless(HAS_CV, 'needs numpy and opencv')
class Readers(unittest.IsolatedAsyncioTestCase):
    """count_events and get_timeline give each occurrence its own span and its evidence."""

    async def test_each_occurrence_with_its_span_and_evidence(self):
        from gamst import Gamst
        from gamst.capabilities import CapabilitySpec, OutputField, SensorMetadata
        from gamst.providers.person_binding import CAPABILITY
        from tests.test_speaker_verify import now
        async with Gamst(consent=True) as human:
            start = now() - 5000
            human.register_capability(CapabilitySpec('temporal.exemplar_events', 'temporal.exemplar_events', 'video', 'test',
                                                     '1', (OutputField('count', 'integer'), OutputField('events', 'array')),
                                                     'native'))
            human.register_capability(CapabilitySpec(CAPABILITY, CAPABILITY, 'video', 'test', '1',
                                                     (OutputField('events', 'array'),), 'frame'))
            evidence = lambda ts: dict(combine='all', rule=dict(satisfied=True, steps_ms=[ts - 600, ts - 300, ts]),
                                       example=dict(index=2, distance=.2, enter=.26, close=True))
            matched = [dict(event='exemplar.matched', track_id='a', who='unknown', start_ms=ts - 600, end_ms=ts,
                            label='관제탑', method='rule+examples', evidence=evidence(ts)) for ts in (start + 700, start + 1400)]
            await human.push_observation('temporal.exemplar_events', dict(count=2, events=matched),
                                         sensor=SensorMetadata('t', 'camera', 1), timestamp_ms=start + 1400)
            await human.push_observation(CAPABILITY, dict(events=[dict(event='person.bound', track_id='a', who='감스트',
                                                                       start_ms=start, end_ms=start + 2000, by='face')]),
                                         sensor=SensorMetadata('t', 'camera', 2), timestamp_ms=start + 2000)
            found = await human.count_events('temporal.exemplar_events', event='exemplar.matched', who='감스트')
            self.assertEqual(found['count'], 2)
            spans = [(item['event']['start_ms'], item['event']['end_ms']) for item in found['evidence']]
            self.assertEqual(spans, [(start + 100, start + 700), (start + 800, start + 1400)])     # two, never merged
            self.assertEqual(found['evidence'][0]['event']['evidence']['example']['index'], 2)
            timeline = await human.get_timeline(start_ms=start, end_ms=start + 2000, step_ms=500,
                                                capabilities=['temporal.exemplar_events'], who='감스트')
            shown = [entry['value'] for entry in timeline['evidence'].values() if entry['kind'] == 'events']
            self.assertEqual([(item['start_ms'], item['end_ms']) for item in shown], spans)
            self.assertTrue(all(item['evidence']['rule']['satisfied'] for item in shown))


if __name__ == '__main__':
    unittest.main()
