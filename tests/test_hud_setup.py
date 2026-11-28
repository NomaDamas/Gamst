"""The HUD's setup step (gamst/hud_setup.py) and the live view's per-person counts, timeline and evidence card
(gamst/views.py). The setup step is validated by compose's own rules (validate_request): the same sentences, no copy.
Test files only; no model runs (photo enrollment is replaced by a stand-in that returns a checked reference)."""
import asyncio
import contextlib
import io
import json
import unittest
from types import SimpleNamespace
from unittest import mock

from gamst.hud_setup import INSTALL_CONSENT, Setup, detect_input, korean, parse_clip, run_command, setup_from
from gamst.person_reference import check
from gamst.recipes.compose import compose, validate_request
from test_compose_recipe import Files, person


class SharedValidation(Files, unittest.TestCase):
    """validate_request is the first step of compose: every refusal is the same sentence."""

    def test_compose_refuses_with_the_sentences_of_validate_request(self):
        for request in (dict(), dict(inputs=dict(camera=0), timing='post', what=dict(words=['x'])),
                        self.shotgun(who=dict(name='감스트')),
                        self.shotgun(who=dict(photos=[self.path('b.mp4')])),
                        self.shotgun(what=dict(movement=dict(text='a shotgun'))),
                        self.shotgun(what=dict(movement=dict(negatives=[self.path('b.mp4') + '@1-2']))),
                        self.shotgun(what=dict(movement=dict(examples=[self.path('b.mp4') + '@1-2'],
                                                             examples_from='who')), who=None),
                        dict(inputs=dict(audio=self.path('a.wav')))):
            checked, composed = validate_request(request), compose(request)
            self.assertFalse(checked['ok'], request)
            self.assertEqual(checked['missing'], composed['missing'], request)

    def test_a_complete_request_passes_and_a_camera_or_microphone_is_not_asked_about(self):
        """Owner, 2026-11-28: installing GAMST is the consent; only a paid hosted call waits (a cost notice)."""
        self.assertEqual(validate_request(self.shotgun()), dict(ok=True, missing=[], before_run=[]))
        camera = validate_request(self.shotgun(inputs=dict(camera=0), timing='realtime'))
        self.assertEqual(camera, dict(ok=True, missing=[], before_run=[]))
        self.assertEqual(validate_request(self.shotgun(inputs=dict(camera=0), timing='realtime',
                                                       consent=dict(capture=True)))['before_run'], [])   # still accepted
        words = validate_request(dict(inputs=dict(microphone=True), what=dict(words=['뽀또'])))
        self.assertTrue(words['ok'], words)
        self.assertEqual(len(words['before_run']), 1)
        self.assertIn('cost notice', words['before_run'][0])
        self.assertNotIn('consent.capture', ' '.join(words['before_run']))

    def test_photos_are_checked_after_they_are_enrolled_and_writing_waits_for_compose(self):
        photo = self.folder / 'p.jpg'
        photo.write_bytes(b'\xff\xd8' + bytes(16))
        request = self.shotgun(who=dict(name='감스트', photos=[str(photo)], consent='P, for this test'))
        self.assertTrue(validate_request(request)['ok'])        # enrolling and the check come with compose
        self.assertIn('--out', compose(request)['missing'][0])
        self.assertEqual(list(self.folder.glob('*.person.json')), [])


class SetupModel(Files, unittest.TestCase):
    def setup(self, **fields):
        enrolled = self.folder / 'enrolled.person.json'
        enrolled.write_text(json.dumps(person()))

        def stand_in(key):     # what photo_reference returns for photos that pass the check
            self.enrolled_keys.append(key)
            return dict(key=key, file=enrolled, check=check(person()))
        self.enrolled_keys = []
        setup = Setup(self.folder / 'hud', enroll=stand_in)
        for key, value in fields.items():
            setattr(setup, key, value)
        return setup

    def line(self, setup):
        ok, line = setup.state()
        self.assertFalse(ok, line)
        return line

    def test_nothing_chosen_names_the_input_in_korean(self):
        setup = self.setup()
        self.assertEqual(self.line(setup), korean(validate_request(setup.request())['missing'][0]))
        self.assertIn('입력을 넣으세요', self.line(setup))

    def test_a_person_needs_a_text_and_photos(self):
        video = ('video', self.folder / 'b.mp4')
        movement = dict(rule=self.path('move.json'), action='shotgun')
        self.assertIn('사람 사진을 추가하세요', self.line(self.setup(pictures=video, name='감스트', **movement)))
        photo = self.folder / 'p.jpg'
        photo.write_bytes(b'\xff\xd8' + bytes(16))
        self.assertIn('누구인지 글로도', self.line(self.setup(pictures=video, photos=[photo], **movement)))
        setup = self.setup(pictures=video, photos=[photo], name='감스트', **movement)     # no consent asked
        self.assertIn('사진 검사를 기다립니다', self.line(setup))
        self.assertEqual(setup.request()['who']['consent'], INSTALL_CONSENT)
        setup.enroll(wait=True)
        self.assertEqual(self.enrolled_keys[0][2], 'GAMST 설치·사용자 동의')     # the record the reference keeps
        self.assertEqual(setup.state(), (True, '준비됐습니다: 시작을 누르세요 (Enter).'))
        self.assertEqual(setup.request()['who'], dict(name='감스트', person=str(self.folder / 'enrolled.person.json')))
        text, _ = setup.photo_line()
        self.assertIn('사진 1장', text)
        self.assertIn('얼굴 3개 일치 1.00', text)
        setup.enroll(wait=True)
        self.assertEqual(len(self.enrolled_keys), 1)        # enrolled once for the same photos, name and consent
        setup.name = '다른 이름'
        self.assertFalse(setup.state()[0])                   # changed: enrolled again before it can start
        setup.enroll(wait=True)
        self.assertEqual(len(self.enrolled_keys), 2)

    def test_an_enrollment_that_fails_the_check_is_said_and_blocks_start(self):
        photo = self.folder / 'p.jpg'
        photo.write_bytes(b'\xff\xd8' + bytes(16))
        setup = self.setup(pictures=('video', self.folder / 'b.mp4'), photos=[photo], name='감스트',
                           action='shotgun', rule=self.path('move.json'))
        setup._enroll = lambda key: dict(key=key, error="who='감스트' refused before start: every photo must show "
                                                       'exactly one person: p.jpg: 2 people. Few-shot identity ...')
        setup.enroll(wait=True)
        self.assertIn('한 사람만', self.line(setup))
        self.assertIn('검사 실패', setup.photo_line()[0])

    def test_what_text_alone_or_negatives_alone_are_refused(self):
        video = ('video', self.folder / 'b.mp4')
        self.assertIn('글로만', self.line(self.setup(pictures=video, action='관제탑')))
        setup = self.setup(pictures=video)
        setup.set_mark('start', 1.)
        setup.set_mark('end', 1.8)
        self.assertIsNone(setup.save_clip(negative=True))
        self.assertIn('반례만', self.line(setup))

    def test_clips_are_trimmed_with_the_chosen_person_and_compose_takes_them(self):
        setup = self.setup(pictures=('video', self.folder / 'b.mp4'))
        self.assertIn('시작', setup.save_clip())
        setup.set_mark('start', 2.)
        setup.set_mark('end', 1.)            # before the start: the start is taken away
        self.assertIsNone(setup.mark['start'])
        setup.set_mark('start', .5)
        setup.mark['at'] = [.25, .6]
        self.assertIsNone(setup.save_clip())
        self.assertEqual(setup.examples, [dict(file=str(self.folder / 'b.mp4'), start_s=.5, end_s=1., at=[.25, .6])])
        self.assertEqual(setup.mark, dict(start=None, end=None, at=None))
        self.assertEqual(setup.state(), (True, '준비됐습니다: 시작을 누르세요 (Enter).'))
        self.assertEqual(setup.request()['what'], dict(movement=dict(examples=setup.examples)))

    def test_words_mode_timing_and_the_cost_notice(self):
        setup = self.setup(pictures=('camera', 0), words=['뽀또'], timing='post')
        self.assertIn('실시간', self.line(setup))
        setup.timing = 'realtime'
        self.assertIn('소리로 듣습니다', self.line(setup))
        setup.microphone = True
        setup.timing = 'realtime'
        self.assertNotIn('consent', setup.request())       # a camera and a microphone are not asked about
        line = self.line(setup)                              # the default recogniser is hosted and paid: its cost
        self.assertIn('비용', line)
        self.assertIn('--allow-remote-asr', line)
        setup.allow_remote_asr, setup.mode = True, 'search'
        self.assertEqual(setup.request()['consent'], dict(remote_asr=True))
        self.assertEqual(setup.request()['mode'], 'search')
        self.assertTrue(setup.state()[0], setup.state())

    def test_the_input_kind_is_detected_from_the_path_or_device(self):
        (self.folder / 'imu.json').write_text(json.dumps(dict(id='imu', file='imu.csv', modality='imu',
                                                              columns=[dict(name='az', unit='g')],
                                                              time=dict(column='t', unit='s'))))
        for given, kind in ((0, 'camera'), ('1', 'camera'), ('camera 2', 'camera'), ('카메라', 'camera'),
                            ('마이크', 'microphone'), ('mic', 'microphone'), (self.path('b.mp4'), 'video'),
                            (self.path('a.wav'), 'audio'), (str(self.folder / 'imu.json'), 'sensor')):
            self.assertEqual(detect_input(given)[0], kind, given)
        for given in (self.path('imu.csv'), str(self.folder / 'missing.mp4'), '', -1):
            kind, line = detect_input(given)
            self.assertIsNone(kind, given)
            self.assertIsInstance(line, str)
        setup = self.setup()
        self.assertIsNone(setup.add_input(self.path('b.mp4')))
        self.assertIsNone(setup.add_input(self.path('a.wav')))
        self.assertIsNone(setup.add_input(str(self.folder / 'imu.json')))
        self.assertEqual(setup.request()['inputs'], dict(video=self.path('b.mp4'), audio=self.path('a.wav'),
                                                         sensors=setup.request()['inputs']['sensors']))
        self.assertEqual(setup.rhythm['input'], 'imu')
        self.assertIsNone(setup.add_input('0'))              # a picture source replaces the other one
        self.assertIsNone(setup.add_input('마이크'))          # a sound source replaces the other one
        self.assertEqual((setup.pictures, setup.microphone, setup.audio), (('camera', 0), True, None))
        self.assertIn('설명 파일', setup.add_input(self.path('imu.csv')))
        setup.clear_inputs()
        self.assertEqual(setup.request()['inputs'], {})

    def test_the_command_line_has_no_consent_option_and_detects_inputs(self):
        import argparse
        from gamst.hud_setup import options
        parser = argparse.ArgumentParser()
        options(parser)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(['--consent', 'x'])
        args = parser.parse_args(['--input', self.path('b.mp4'), '--input', 'mic', '--who', '감스트',
                                  '--out-dir', str(self.folder / 'hud')])
        setup = setup_from(args)
        self.assertEqual((setup.pictures, setup.microphone, setup.consent), (('video', self.folder / 'b.mp4'), True,
                                                                             INSTALL_CONSENT))
        older = setup_from(parser.parse_args(['--camera', '0', '--out-dir', str(self.folder / 'hud')]))
        self.assertEqual(older.pictures, ('camera', 0))      # the older option is read the same way
        with self.assertRaises(SystemExit):
            setup_from(parser.parse_args(['--input', self.path('imu.csv'), '--out-dir', str(self.folder / 'hud')]))

    def test_a_sensor_is_declared_not_guessed(self):
        setup = self.setup()
        self.assertIn('설명 파일', setup.add_sensor(self.path('imu.csv')))
        spec = self.folder / 'imu.json'
        spec.write_text(json.dumps(dict(id='imu', file='imu.csv', modality='imu', columns=[dict(name='az', unit='g')],
                                        time=dict(column='t', unit='s'))))
        self.assertIsNone(setup.add_sensor(spec))
        self.assertIn('센서 리듬', self.line(setup))
        setup.cycle_rhythm('band')
        self.assertEqual(setup.rhythm, dict(input='imu', channel='az', band='breath'))
        self.assertTrue(setup.state()[0], setup.state())

    def test_the_run_command_has_a_window_the_profile_and_a_camera_stand_in(self):
        setup = self.setup(pictures=('video', self.folder / 'b.mp4'), profile='p.json')
        answer = dict(run=['gamst', 'recipe', 'run', 'r.json', '--video', 'b.mp4', '--timing', 'realtime', '--headless'])
        self.assertEqual(run_command(answer, setup, ['--report', 'x.json']),
                         ['r.json', '--video', 'b.mp4', '--timing', 'realtime', '--profile', 'p.json', '--realtime',
                          '--report', 'x.json'])
        answer['run'][7] = 'post'
        self.assertNotIn('--realtime', run_command(answer, setup))

    def test_clip_text_and_korean_lines(self):
        self.assertEqual(parse_clip('a b.mp4@62.051-63.951@0.5,0.4'),
                         dict(file='a b.mp4', start_s=62.051, end_s=63.951, at=[.5, .4]))
        self.assertIsNone(parse_clip('a.mp4'))
        self.assertTrue(korean('something new.').startswith('추가할 것: '))


class SetupWindowTest(Files, unittest.TestCase):
    """The window's buttons and keys change the model; the Start button is there only when nothing is missing."""

    def test_buttons_keys_and_a_click_on_the_person(self):
        try:
            import numpy
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest('opencv is needed')
        from gamst.hud_setup import SetupWindow
        answers = {'누구를 셀까요? 이름': '감스트'}
        setup = Setup(self.folder / 'hud')
        window = SetupWindow(setup, ask_text=lambda prompt, default='': answers.get(prompt, 'x'),
                             ask_files=lambda prompt, multiple=True: [self.path('move.json')])
        setup.pictures = ('video', self.folder / 'b.mp4')
        window.preview.frame = numpy.zeros((720, 1280, 3), numpy.uint8)
        window.preview.duration_s, window.preview.position_s = 10., 1.
        image, ok = window.draw()
        self.assertFalse(ok)
        actions = [action for _, action in window.hits]
        self.assertNotIn(('start',), actions)
        self.assertIn(('input_file',), actions)
        self.assertIn(('input_device',), actions)
        self.assertFalse([action for action in actions if action[0] in ('camera', 'video', 'microphone', 'sensor')
                          or action[:2] == ('ask', 'consent')])       # no kind to pick, no consent to give
        for action in (('ask', 'name', '누구를 셀까요? 이름'), ('rule',), ('mode', 'search'), ('timing', 'post')):
            window.act(action)
        self.assertEqual((setup.name, setup.rule, setup.mode, setup.timing),
                         ('감스트', self.folder / 'move.json', 'search', 'post'))
        window.draw()
        self.assertIn(('pick',), [action for _, action in window.hits])
        window.act(('pick',), 480, 270)         # the middle of the 960 x 540 picture
        self.assertEqual(setup.mark['at'], [.5, .5])
        window.key(ord('['))
        window.preview.position_s = 1.6
        window.key(ord(']'))
        window.key(ord('e'))
        self.assertEqual(setup.examples, [dict(file=str(self.folder / 'b.mp4'), start_s=1., end_s=1.6, at=[.5, .5])])
        self.assertFalse(window.key(ord('q')))
        setup.name = ''
        self.assertIn('글로 적어', window.setup.state()[1])     # a rule stands for a text
        setup.action = 'shotgun'
        self.assertTrue(window.draw()[1])           # nobody named: everyone is counted
        self.assertIn(('start',), [action for _, action in window.hits])


T0 = 1_000_000


def record(index, capability, value, ms, name=None):
    observation = dict(capability=dict(id=capability), value=value) if capability else dict(name=name, value=value)
    return dict(id=f'r{index}', timestamp_ms=ms, observation=observation)


def matched(index, start, end, track, **more):
    return dict(event='exemplar.matched', index=index, label='관제탑', start_ms=T0 + start, end_ms=T0 + end,
                track_id=track, who='unknown', evidence=dict(
                    combine='all', rule=dict(satisfied=True, steps_ms=[T0 + start, T0 + start + 400, T0 + end]),
                    example=dict(index=2, distance=.0123, enter=.35, negative_distance=None)), **more)


class LiveView(unittest.TestCase):
    def setUp(self):
        try:
            import cv2
            import numpy
        except ImportError:
            self.skipTest('opencv is needed')
        from gamst.views import CountView
        self.cv2, self.numpy = cv2, numpy
        pipeline = SimpleNamespace(
            name='composed-x', mode='count', inputs={'camera': dict(modality='video')},
            notes=dict(timing='realtime', pose_choice=dict(model='rtmw.l_384x288', chosen='mediapipe.full', live_fps=12.,
                                                           l2_fps=0., input_fps=30)),
            recipe_context=dict(recipe=dict(modules=[dict(name='pose', use='rtmw-pose'), dict(name='semantic', use='semantic')]),
                                notes=dict(body_identity='whole-body appearance is off: no model')),
            modules={'pose': {}}, lost=lambda capability: dict(gaps=0, frames=0),
            status=lambda: dict(modules={'pose': dict(enabled=True, last_latency_ms=30)}))
        self.view = CountView(pipeline, capabilities=('temporal.exemplar_events', 'vision.person_binding'))
        self.records = [
            record(1, 'vision.person_binding', dict(
                tracks=[dict(track_id='t1', who='감스트', person_id='P1'), dict(track_id='t2', who='unknown', person_id='P2')],
                events=[dict(event='person.bound', track_id=track, who=who, person_id=key, start_ms=T0, end_ms=T0 + 100,
                             retroactive=True) for track, who, key in (('t1', '감스트', 'P1'), ('t2', 'unknown', 'P2'))],
                bound={'감스트': 't1'}, persons={'P1': 't1', 'P2': 't2'}), T0 + 100),
            record(2, None, [dict(bbox=dict(x=.1, y=.2, width=.3, height=.7), track_id='t1'),
                             dict(bbox=dict(x=.6, y=.25, width=.25, height=.6), track_id='t2')], T0 + 29000, name='scene.poses'),
            record(3, 'temporal.exemplar_events', dict(count=3, events=[matched(1, 2000, 2900, 't1')]), T0 + 3000),
            record(4, 'temporal.exemplar_events', dict(count=3, events=[matched(2, 6000, 6800, 't2')]), T0 + 7000),
            record(5, 'temporal.exemplar_events', dict(count=3, events=[matched(3, 12000, 12900, 't1', correction=True,
                                                                                decided_by='catch_up')]), T0 + 20000)]
        self.texts, self.shown = [], []
        real = self.view._put
        self.view._put = lambda image, line, *rest: (self.texts.append(line), real(image, line, *rest))[1]
        self.frame = SimpleNamespace(modality='video', timestamp_ms=T0 + 30000,
                                     data=numpy.full((1080, 1920, 3), 90, numpy.uint8))

    def update(self, records=(), key=0):
        with mock.patch.object(self.cv2, 'imshow', lambda title, image: self.shown.append(image)), \
                mock.patch.object(self.cv2, 'waitKey', lambda ms: key), contextlib.redirect_stdout(io.StringIO()):
            self.view.first_ms = self.view.first_ms or T0
            return asyncio.run(self.view.update('camera', self.frame, list(records)))

    def test_per_person_counts_late_corrections_and_setup_notes(self):
        self.assertTrue(self.update(self.records))
        say = self.view._say
        self.assertEqual(self.view.selected, '감스트')          # the named person is selected at first
        for line in ('감스트 2', say('다른 사람 1 (P2 1)', 'others 1 (P2 1)'), 'LATE ',
                     say('자세 모델 mediapipe.full', 'pose mediapipe.full'), say('너무 느림', 'too slow'),
                     say('뺀 모듈: semantic', 'left out: semantic'), 'body_identity'):
            self.assertTrue(any(line in text for text in self.texts), line)
        self.assertTrue(any(text.startswith('  LATE ') and '감스트' in text for text in self.texts) or not self.view.korean)
        image = self.shown[-1]
        self.assertEqual(image.shape[0], 720 + self.view.strip_h)        # 1080 shown at 720, the strip below

    def test_a_click_on_the_timeline_opens_the_evidence_and_a_box_selects_a_person(self):
        self.update(self.records)
        spans = [rect for rect, action in self.view.hits if action[0] == 'span']
        self.assertEqual(len(spans), 3)
        x0, y0, x1, y1 = spans[2]
        self.view.clicks.append(((x0 + x1) // 2, (y0 + y1) // 2))
        self.texts.clear()
        self.update()
        self.update()
        say = self.view._say
        for line in (say('#3 관제탑 · 감스트 (P1)', '#3 관제탑 · 감스트 (P1)'), say('12.00초 - 12.90초', '12.00 s - 12.90 s'),
                     'LATE', say('1단계 12.00초 · 2단계 12.40초 · 3단계 12.90초', 'step 1 12.00 s · step 2 12.40 s · step 3 12.90 s'),
                     say('가장 가까운 예시: 2번, 거리 0.0123 (기준 0.350)', 'nearest example #2, distance 0.0123 (enter 0.350)')):
            self.assertTrue(any(line in text for text in self.texts), line)
        card = next(rect for rect, action in self.view.hits if action[0] == 'card')
        self.view.clicks.append((card[0] + 5, card[1] + 5))
        self.update()
        self.assertIsNone(self.view.card)
        box = next(rect for rect, action in self.view.hits if action == ('person', 'P2'))
        self.view.clicks.append((box[0] + 5, box[1] + 5))
        self.update()
        self.assertEqual(self.view.selected, 'P2')
        self.assertFalse(self.view.show_all)
        self.update(key=ord('a'))
        self.assertTrue(self.view.show_all)
        self.assertFalse(self.update(key=ord('q')))            # the existing keys still work

    def test_who_is_resolved_retroactively_and_one_action_on_two_tracks_counts_once(self):
        from gamst.views import occurrences
        rows = {'temporal.exemplar_events': [self.records[2], record(6, 'temporal.exemplar_events', dict(events=[
            matched(4, 2000, 2950, 't3')]), T0 + 3100)]}
        binding = [self.records[0], record(7, 'vision.person_binding', dict(events=[dict(
            event='person.bound', track_id='t3', who='감스트', person_id='P1', start_ms=T0, end_ms=T0 + 3050,
            retroactive=True)]), T0 + 3050)]
        found = occurrences(rows, binding)
        self.assertEqual([(item['index'], item['who'], item['person_id']) for item in found], [(1, '감스트', 'P1')])
        self.assertEqual([item['who'] for item in occurrences(rows)], ['unknown', 'unknown'])   # no binding: as committed


if __name__ == '__main__':
    unittest.main()
