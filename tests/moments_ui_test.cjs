const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');

(async () => {
  const {
    HandRaiseDetector, PenDropDetector, SwingPeakDetector, oneFpsBadge, measureMotion,
  } = await import(pathToFileURL(path.resolve('site/moments.js')).href);

  const geometry = (stamp, leftRaised = false, rightRaised = false) => ({
    timestamp_ms: stamp, source_epoch: 'pose-fixture',
    left: { valid: true, visibility: .99, wrist_y: leftRaised ? .2 : .8, shoulder_y: .5 },
    right: { valid: true, visibility: .99, wrist_y: rightRaised ? .2 : .8, shoulder_y: .5 },
  });
  const hand = new HandRaiseDetector();
  [0, 40, 80].forEach((stamp) => assert.deepEqual(hand.update(geometry(stamp)), []));
  assert.deepEqual(hand.update(geometry(120, true)), [], 'first crossing is retained before confirmation');
  const onset = hand.update(geometry(160, true));
  assert.equal(onset.length, 1);
  assert.equal(onset[0].type, 'hand_raise');
  assert.equal(onset[0].capture_ms, 120, 'confirmation never retimestamps onset');
  assert.equal(onset[0].confirmation_ms, 160);
  assert.equal(onset[0].side, 'left');
  [200, 240, 280, 320].forEach((stamp) => assert.deepEqual(hand.update(geometry(stamp, true)), [], 'holding does not repeatedly escalate'));
  for (let stamp = 360; stamp <= 1280; stamp += 40) hand.update(geometry(stamp));
  assert.deepEqual(hand.update(geometry(1320, true)), []);
  assert.equal(hand.update(geometry(1360, true))[0].capture_ms, 1320, 'lower/re-arm permits a new onset after cooldown');
  console.log('Hand raise: first crossing, hysteresis, re-arm and cooldown passed');

  const occluded = new HandRaiseDetector();
  [0, 40, 80].forEach((stamp) => occluded.update(geometry(stamp)));
  occluded.update(geometry(120, false, true));
  const hidden = geometry(160, false, true);
  hidden.right.valid = false;
  hidden.right.visibility = .1;
  assert.deepEqual(occluded.update(hidden), []);
  assert.deepEqual(occluded.update(geometry(200, false, true)), [], 'visibility loss discards an unconfirmed candidate');
  const rightHand = new HandRaiseDetector();
  [0, 40, 80].forEach((stamp) => rightHand.update(geometry(stamp)));
  rightHand.update(geometry(120, false, true));
  assert.equal(rightHand.update(geometry(160, false, true))[0].side, 'right');
  console.log('Hand raise: left/right visibility gates and occlusion abstention passed');

  const quiet = (stamp) => ({ timestamp_ms: stamp, source_epoch: 'drop-fixture', valid: true,
    dy_pixels: 0, moving_fraction: 0, sample_count: 14400, interval_ms: 50,
    dbfs: -80, spectral_flux: 0, camera_motion: false });
  const drop = new PenDropDetector();
  for (let stamp = 0; stamp <= 5000; stamp += 50) assert.equal(drop.update(quiet(stamp)), null);
  assert(drop.baseline, 'five seconds of quiet arms robust baselines');
  const downward = (stamp) => ({ ...quiet(stamp), dy_pixels: 2, moving_fraction: .01 });
  assert.equal(drop.update(downward(5100)), null);
  assert.equal(drop.update(downward(5150)), null, 'motion alone is not impact');
  const impact = drop.update({ ...quiet(5180), dbfs: -20, spectral_flux: .5, audio_start_ms: 5140, audio_end_ms: 5180 });
  assert.equal(impact.type, 'pen_drop');
  assert.equal(impact.capture_ms, 5180);
  assert.deepEqual(impact.audio_interval, [5140, 5180]);
  assert.equal(impact.downward_started_ms, 5100);
  assert.equal(drop.update({ ...quiet(5210), dbfs: -20, spectral_flux: .5 }), null, 'audio alone does not retrigger during cooldown');
  console.log('Pen drop: downward run plus bounded audio transient emits one impact candidate');

  const silent = new PenDropDetector();
  for (let stamp = 0; stamp <= 35000; stamp += 50) assert.equal(silent.update(quiet(stamp)), null, '30-second quiet control never fires');
  const missingAudio = new PenDropDetector();
  for (let stamp = 0; stamp <= 5000; stamp += 50) missingAudio.update(quiet(stamp));
  missingAudio.update(downward(5100));
  assert.equal(missingAudio.update({ ...downward(5150), dbfs: NaN, spectral_flux: NaN, valid: false }), null);
  assert.equal(missingAudio.lastDegraded, 'missing audio/video observation');
  const shaken = new PenDropDetector();
  for (let stamp = 0; stamp <= 5000; stamp += 50) shaken.update(quiet(stamp));
  shaken.update(downward(5100));
  assert.equal(shaken.update({ ...downward(5150), camera_motion: true }), null);
  assert.equal(shaken.lastDegraded, 'camera motion');
  console.log('Pen drop: quiet, missing-audio and camera-shake controls abstain');

  const swing = new SwingPeakDetector();
  const handPoint = (stamp, x, moving = 0) => ({ timestamp_ms: stamp, source_epoch: 'swing-fixture', valid: true, point: { x, y: 100 }, moving_fraction: moving });
  [0, 33, 66, 99, 132, 165].forEach((stamp) => assert.equal(swing.update(handPoint(stamp, 0)), null));
  assert(swing.baseline, 'stationary baseline is armed');
  [[198, 0, 0], [231, 50, .2], [264, 100, .2], [297, 140, .2], [330, 200, .2], [363, 230, .2], [396, 230.2, 0], [429, 230.4, 0]].forEach(([stamp, x, moving]) => swing.update(handPoint(stamp, x, moving)));
  const peak = swing.update(handPoint(462, 230.6, 0));
  assert.equal(peak.type, 'swing_peak');
  assert.equal(peak.capture_ms, 330, 'original endpoint of maximum observed interval is retained');
  assert.equal(peak.interval_start_ms, 297);
  assert.equal(peak.interval_end_ms, 330);
  assert(Math.abs(peak.peak_speed_px_s - 60 * 1000 / 33) < 1e-9);
  assert(!('peak_speed_m_s' in peak), 'image velocity is never relabeled as metric velocity');
  const lost = new SwingPeakDetector();
  [0, 33, 66, 99, 132, 165].forEach((stamp) => lost.update(handPoint(stamp, 0)));
  lost.update(handPoint(198, 50, .2));
  lost.update(handPoint(231, 100, .2));
  assert.equal(lost.update({ timestamp_ms: 264, source_epoch: 'swing-fixture', valid: false, point: null }), null);
  assert.equal(lost.lastDegraded, 'hand loss or tracking reset during segment');
  console.log('Swing peak: maximum px/s interval, clean close and tracking-loss abstention passed');

  assert.deepEqual(oneFpsBadge(1500, [0, 1000]), { status: 'seen', nearest_sample_ms: 1000, distance_ms: 500 });
  assert.equal(oneFpsBadge(1501, [0, 1000]).status, 'missed');
  assert.equal(oneFpsBadge(400, []).status, 'missed');
  const width = 8, height = 8;
  const before = new Uint8ClampedArray(width * height * 4);
  const after = new Uint8ClampedArray(before);
  for (let y = 2; y < 5; y += 1) for (let x = 2; x < 5; x += 1) after[(y * width + x) * 4] = 255;
  const motion = measureMotion(after, before, width, height, 33);
  assert(motion.valid && motion.changed_centroid.x === 3 && motion.changed_centroid.y === 3);
  assert.equal(motion.sample_count, 64);
  assert.equal(motion.moving_fraction, 9 / 64);
  console.log('Sampling badge and deterministic changed-pixel ROI measurement passed');

  const html = fs.readFileSync('site/moments.html', 'utf8');
  for (const id of ['video', 'overlay', 'momentsList', 'framesSeen', 'modelCalls', 'estimatedCost', 'sampledFrame', 'answerText']) assert(html.includes(`id="${id}"`));
  assert(html.includes('@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs'));
  assert(!html.includes('<script src="http'), 'remote scripts are not permitted');
  console.log('Moment Catcher shell: native video, sampling lane, evidence and counters present');
  console.log('All Moment Catcher detector tests passed');
})().catch((error) => { console.error(error); process.exitCode = 1; });
