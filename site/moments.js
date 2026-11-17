import { chatEndpoint, normalizeLocalModelConfig } from "./playground.js";

export const MOMENT_LABELS = Object.freeze({
  hand_raise: "hand-raise geometry onset",
  pen_drop: "impact candidate",
  swing_peak: "maximum observed image velocity for this segment",
});

const median = (values) => {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
};

export function robustBaseline(values, floor = 0) {
  const center = median(values.filter(Number.isFinite));
  return { median: center, mad: Math.max(floor, median(values.filter(Number.isFinite).map((value) => Math.abs(value - center)))) };
}

export function oneFpsBadge(captureMs, sampleTimes, toleranceMs = 500) {
  const valid = (sampleTimes || []).filter(Number.isFinite);
  if (!valid.length) return { status: "missed", nearest_sample_ms: null, distance_ms: null };
  const nearest = valid.reduce((best, stamp) => Math.abs(stamp - captureMs) < Math.abs(best - captureMs) ? stamp : best, valid[0]);
  const distance = Math.abs(nearest - captureMs);
  return { status: distance <= toleranceMs ? "seen" : "missed", nearest_sample_ms: nearest, distance_ms: distance };
}

function finitePoint(point) {
  return point && Number.isFinite(point.x) && Number.isFinite(point.y);
}

export class HandRaiseDetector {
  constructor({ visibilityThreshold = .6, maxGapMs = 100, cooldownMs = 1000 } = {}) {
    this.visibilityThreshold = visibilityThreshold;
    this.maxGapMs = maxGapMs;
    this.cooldownMs = cooldownMs;
    this.reset();
  }

  reset() {
    this.lastTimestamp = null;
    this.sourceEpoch = null;
    this.armed = false;
    this.baselineRun = 0;
    this.sides = Object.fromEntries(["left", "right"].map((side) => [side, this.freshSide()]));
  }

  freshSide() {
    return { previousRaised: false, candidate: null, history: [], latched: false, belowRun: 0, cooldownUntil: 0 };
  }

  resetStream() {
    const epoch = this.sourceEpoch;
    this.reset();
    this.sourceEpoch = epoch;
  }

  update(observation) {
    const stamp = observation?.timestamp_ms;
    if (!Number.isFinite(stamp)) return [];
    if (this.sourceEpoch !== null && observation.source_epoch !== this.sourceEpoch) this.reset();
    this.sourceEpoch = observation.source_epoch ?? "default";
    const sides = ["left", "right"].map((name) => ({ name, ...observation[name] }));
    const valid = sides.every((side) => side.valid !== false && Number.isFinite(side.wrist_y) && Number.isFinite(side.shoulder_y)
      && (side.visibility ?? 1) >= this.visibilityThreshold);
    const discontinuous = this.lastTimestamp !== null && (stamp <= this.lastTimestamp || stamp - this.lastTimestamp > this.maxGapMs);
    if (!valid || discontinuous) {
      this.lastTimestamp = discontinuous ? null : stamp;
      this.resetStream();
      return [];
    }
    this.lastTimestamp = stamp;
    const raised = Object.fromEntries(sides.map((side) => [side.name, side.wrist_y < side.shoulder_y]));
    this.baselineRun = !raised.left && !raised.right ? this.baselineRun + 1 : 0;
    if (this.baselineRun >= 3) this.armed = true;
    const events = [];
    for (const side of sides) {
      const state = this.sides[side.name];
      const isRaised = raised[side.name];
      if (state.latched) {
        state.belowRun = isRaised ? 0 : state.belowRun + 1;
        if (state.belowRun >= 3) {
          Object.assign(state, this.freshSide(), { previousRaised: false, cooldownUntil: state.cooldownUntil });
        }
        state.previousRaised = isRaised;
        continue;
      }
      if (this.armed && !state.candidate && isRaised && !state.previousRaised && stamp >= state.cooldownUntil) {
        state.candidate = { capture_ms: stamp, side: side.name };
        state.history = [];
      }
      if (state.candidate) {
        state.history.push(isRaised);
        if (state.history.length > 3) state.history.shift();
        const confirmed = state.history.filter(Boolean).length >= 2;
        if (confirmed) {
          events.push({ type: "hand_raise", capture_ms: state.candidate.capture_ms, side: side.name,
            confirmation_ms: stamp, confirmation_latency_ms: stamp - state.candidate.capture_ms,
            evidence_observations: [state.candidate.capture_ms, stamp] });
          state.latched = true;
          state.candidate = null;
          state.cooldownUntil = stamp + this.cooldownMs;
          state.belowRun = 0;
        } else if (state.history.length === 3) {
          state.candidate = null;
          state.history = [];
        }
      }
      state.previousRaised = isRaised;
    }
    return events;
  }
}

export class PenDropDetector {
  constructor({ quietMs = 5000, maxGapMs = 100, impactWindowMs = 120, cooldownMs = 1000,
    minDownwardSpeed = 20, minMovingFraction = .0008, absoluteDbfsFloor = -45 } = {}) {
    this.options = { quietMs, maxGapMs, impactWindowMs, cooldownMs, minDownwardSpeed, minMovingFraction, absoluteDbfsFloor };
    this.reset();
  }

  reset() {
    this.lastTimestamp = null;
    this.sourceEpoch = null;
    this.samples = [];
    this.baseline = null;
    this.armedAt = null;
    this.downRun = 0;
    this.pending = null;
    this.cooldownUntil = 0;
    this.lastDegraded = null;
  }

  update(observation) {
    const stamp = observation?.timestamp_ms;
    if (!Number.isFinite(stamp)) return null;
    if (this.sourceEpoch !== null && observation.source_epoch !== this.sourceEpoch) this.reset();
    this.sourceEpoch = observation.source_epoch ?? "default";
    const gap = this.lastTimestamp !== null && (stamp <= this.lastTimestamp || stamp - this.lastTimestamp > this.options.maxGapMs);
    const valid = observation.valid !== false && Number.isFinite(observation.dy_pixels) && Number.isFinite(observation.moving_fraction)
      && Number.isFinite(observation.interval_ms) && observation.interval_ms > 0 && (observation.sample_count ?? 1) > 0
      && Number.isFinite(observation.dbfs) && Number.isFinite(observation.spectral_flux) && !observation.camera_motion;
    if (gap || !valid) {
      const degraded = gap ? "video gap or non-monotonic capture time" : observation.camera_motion ? "camera motion" : "missing audio/video observation";
      this.reset();
      this.lastDegraded = degraded;
      return null;
    }
    this.lastTimestamp = stamp;
    if (stamp < this.cooldownUntil) return null;
    const speed = observation.dy_pixels * 1000 / observation.interval_ms;
    if (!this.baseline) {
      this.samples.push({ stamp, speed, moving: observation.moving_fraction, dbfs: observation.dbfs, flux: observation.spectral_flux });
      if (this.samples.length > 900) this.samples.shift();
      const first = this.samples[0]?.stamp ?? stamp;
      const quiet = speed <= this.options.minDownwardSpeed && observation.moving_fraction <= this.options.minMovingFraction * 2;
      if (!quiet) {
        this.samples = [];
        this.armedAt = null;
        return null;
      }
      if (stamp - first >= this.options.quietMs) {
        this.baseline = {
          speed: robustBaseline(this.samples.map((sample) => sample.speed), .5),
          moving: robustBaseline(this.samples.map((sample) => sample.moving), .0001),
          dbfs: robustBaseline(this.samples.map((sample) => sample.dbfs), .5),
          flux: robustBaseline(this.samples.map((sample) => sample.flux), .005),
        };
        this.armedAt = stamp;
      }
      return null;
    }
    const downward = speed > this.baseline.speed.median + 6 * this.baseline.speed.mad
      && speed >= this.options.minDownwardSpeed
      && observation.moving_fraction > Math.max(this.options.minMovingFraction, this.baseline.moving.median + 6 * this.baseline.moving.mad);
    this.downRun = downward ? this.downRun + 1 : 0;
    if (this.downRun >= 2) {
      this.pending ||= { downward_started_ms: this.pending?.downward_started_ms ?? stamp - observation.interval_ms };
      this.pending.last_motion_ms = stamp;
      this.pending.downward_speed_px_s = speed;
      this.pending.moving_fraction = observation.moving_fraction;
    }
    if (!this.pending || stamp - this.pending.last_motion_ms > this.options.impactWindowMs) {
      this.pending = null;
      return null;
    }
    const transient = observation.dbfs >= this.options.absoluteDbfsFloor
      && observation.dbfs >= this.baseline.dbfs.median + 12
      && observation.spectral_flux > this.baseline.flux.median + 6 * this.baseline.flux.mad;
    if (!transient) return null;
    const event = { type: "pen_drop", capture_ms: stamp, impact_boundary_ms: stamp,
      downward_started_ms: this.pending.downward_started_ms,
      audio_interval: [observation.audio_start_ms ?? stamp - 40, observation.audio_end_ms ?? stamp],
      downward_speed_px_s: this.pending.downward_speed_px_s, moving_fraction: this.pending.moving_fraction, dbfs: observation.dbfs,
      spectral_flux: observation.spectral_flux };
    this.pending = null;
    this.downRun = 0;
    this.cooldownUntil = stamp + this.options.cooldownMs;
    return event;
  }
}

export class SwingPeakDetector {
  constructor({ maxGapMs = 100, maxSegmentMs = 1500, cooldownMs = 1000, minSpeedPxS = 80, minMovingFraction = .03 } = {}) {
    this.options = { maxGapMs, maxSegmentMs, cooldownMs, minSpeedPxS, minMovingFraction };
    this.reset();
  }

  reset() {
    this.lastTimestamp = null;
    this.lastPoint = null;
    this.sourceEpoch = null;
    this.baselineSamples = [];
    this.baseline = null;
    this.recent = [];
    this.segment = null;
    this.cooldownUntil = 0;
    this.lastDegraded = null;
  }

  update(observation) {
    const stamp = observation?.timestamp_ms;
    if (!Number.isFinite(stamp)) return null;
    if (this.sourceEpoch !== null && observation.source_epoch !== this.sourceEpoch) this.reset();
    this.sourceEpoch = observation.source_epoch ?? "default";
    const gap = this.lastTimestamp !== null && (stamp <= this.lastTimestamp || stamp - this.lastTimestamp > this.options.maxGapMs);
    if (gap || observation.valid === false || !finitePoint(observation.point)) {
      const degraded = this.segment ? "hand loss or tracking reset during segment" : gap ? "capture gap" : "invalid hand observation";
      this.reset();
      this.lastDegraded = degraded;
      return null;
    }
    const intervalMs = this.lastTimestamp === null ? null : stamp - this.lastTimestamp;
    const speed = intervalMs && this.lastPoint ? Math.hypot(observation.point.x - this.lastPoint.x, observation.point.y - this.lastPoint.y) * 1000 / intervalMs : 0;
    const moving = Number.isFinite(observation.moving_fraction) ? observation.moving_fraction : speed > 0 ? 1 : 0;
    const interval = intervalMs ? { t0: this.lastTimestamp, t1: stamp, speed_px_s: speed, point: { ...observation.point }, moving_fraction: moving } : null;
    this.lastTimestamp = stamp;
    this.lastPoint = { ...observation.point };
    if (stamp < this.cooldownUntil) return null;
    if (!this.baseline) {
      if (interval && speed > this.options.minSpeedPxS) {
        this.baselineSamples = [];
        return null;
      }
      if (interval) this.baselineSamples.push({ speed, moving });
      if (this.baselineSamples.length >= 5) {
        this.baseline = { speed: robustBaseline(this.baselineSamples.map((sample) => sample.speed), 2),
          moving: robustBaseline(this.baselineSamples.map((sample) => sample.moving), .002) };
      }
      return null;
    }
    const entry = Math.max(this.options.minSpeedPxS, this.baseline.speed.median + 6 * this.baseline.speed.mad);
    const exit = entry * .45;
    const flowGate = Math.max(this.options.minMovingFraction, this.baseline.moving.median + 6 * this.baseline.moving.mad);
    const qualifies = interval && speed > entry && moving > flowGate;
    this.recent.push({ qualifies, interval });
    if (this.recent.length > 3) this.recent.shift();
    if (!this.segment) {
      if (this.recent.filter((item) => item.qualifies).length >= 2) {
        this.segment = { started_ms: this.recent[0].interval.t0, intervals: this.recent.filter((item) => item.interval).map((item) => item.interval), belowRun: 0 };
      }
      return null;
    }
    if (interval) this.segment.intervals.push(interval);
    this.segment.belowRun = speed < exit ? this.segment.belowRun + 1 : 0;
    const capped = stamp - this.segment.started_ms >= this.options.maxSegmentMs;
    if (this.segment.belowRun < 3 && !capped) return null;
    const peak = this.segment.intervals.reduce((best, item) => !best || item.speed_px_s > best.speed_px_s ? item : best, null);
    this.segment = null;
    this.recent = [];
    this.cooldownUntil = stamp + this.options.cooldownMs;
    this.baseline = null;
    this.baselineSamples = [];
    this.lastPoint = { ...observation.point };
    return peak ? { type: "swing_peak", capture_ms: peak.t1, interval_start_ms: peak.t0, interval_end_ms: peak.t1,
      peak_speed_px_s: peak.speed_px_s, closed_ms: stamp, close_reason: capped ? "segment cap" : "below exit threshold" } : null;
  }
}

export function measureMotion(current, previous, width, height, intervalMs) {
  if (!current || !previous || current.length !== previous.length || !(intervalMs > 0)) {
    return { valid: false, dx_pixels: 0, dy_pixels: 0, moving_fraction: 0, sample_count: 0, median_speed_pixels_per_second: 0 };
  }
  let count = 0, sx = 0, sy = 0;
  const total = width * height;
  for (let pixel = 0; pixel < total; pixel += 1) {
    const index = pixel * 4;
    const now = current[index] * .299 + current[index + 1] * .587 + current[index + 2] * .114;
    const before = previous[index] * .299 + previous[index + 1] * .587 + previous[index + 2] * .114;
    if (Math.abs(now - before) < 35) continue;
    const x = pixel % width, y = Math.floor(pixel / width);
    count += 1; sx += x; sy += y;
  }
  if (count < 3) return { valid: true, dx_pixels: 0, dy_pixels: 0, moving_fraction: 0, sample_count: total, median_speed_pixels_per_second: 0 };
  return { valid: true, changed_centroid: { x: sx / count, y: sy / count }, dx_pixels: 0, dy_pixels: 0,
    moving_fraction: count / total, sample_count: total, median_speed_pixels_per_second: 0 };
}

const $ = (id) => document.getElementById(id);
const MEDIAPIPE_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";
const MEDIAPIPE_WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const POSE_MODEL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";
const HAND_MODEL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const FRAME_BUFFER_SIZE = 48;

const runtime = {
  active: false, generation: 0, sourceKind: null, sourceName: null, sourceEpoch: null, stream: null, objectUrl: null, replayFile: null,
  videoCallback: null, fallbackCallback: null, framesSeen: 0, modelCalls: 0, moments: [], selectedMoment: null, sequence: 0,
  samples: [], nextSampleMs: 0, lastFrameMs: null, frameBuffer: [], frameCursor: 0,
  processCanvas: null, previousPixels: null, previousCentroid: null,
  audioContext: null, audioSource: null, elementAudioSource: null, analyser: null, muteGain: null, audioSamples: null, audioSpectrum: null, previousSpectrum: null,
  decodedAudio: null, previousDecodedSpectrum: null, frameStep: false, frameIndex: 0,
  poseModel: null, handModel: null, poseSegment: 0, lastPoseMs: null, handPath: [],
  handRaise: new HandRaiseDetector(), penDrop: new PenDropDetector(), swingPeak: new SwingPeakDetector(),
  quietControl: null, modelConfigLoaded: false,
};

function setStatus(message, degraded = false) {
  $("runStatus").textContent = message;
  $("runStatus").classList.toggle("degraded", degraded);
}

function updateCounters() {
  $("framesSeen").textContent = runtime.framesSeen.toLocaleString();
  $("modelCalls").textContent = runtime.modelCalls.toLocaleString();
  const raw = $("pricePerCall").value.trim();
  const price = raw === "" ? null : Number(raw);
  $("estimatedCost").textContent = price === null || !Number.isFinite(price) || price < 0 ? "unknown" : `$${(runtime.modelCalls * price).toFixed(6)}`;
  $("momentCount").textContent = `${runtime.moments.length} caught`;
}

function logRun(message, kind = "observation") {
  const item = document.createElement("li");
  item.className = kind;
  item.textContent = message;
  $("runLog").prepend(item);
  while ($("runLog").children.length > 30) $("runLog").lastElementChild.remove();
}

function providerConfig() {
  return { baseUrl: $("providerUrl").value.trim(), model: $("providerModel").value.trim(), key: $("providerKey").value };
}

async function loadOptionalProvider() {
  if (runtime.modelConfigLoaded) return;
  runtime.modelConfigLoaded = true;
  try {
    const module = await import("./local-config.js");
    const config = normalizeLocalModelConfig(module);
    if (config && /^https?:\/\/(127\.0\.0\.1|localhost)(:|\/|$)/.test(config.baseUrl)) {
      $("providerUrl").value = config.baseUrl;
      $("providerModel").value = config.model;
      $("providerKey").value = config.apiKey;
      setStatus("Loopback provider configuration loaded; replay remains sensor-only until a moment fires.");
    }
  } catch { /* Optional ignored local configuration. */ }
}

async function ensureAudio() {
  if (runtime.audioContext) {
    runtime.audioSource?.disconnect();
    const source = runtime.sourceKind === "camera" ? runtime.audioContext.createMediaStreamSource(runtime.stream) : (runtime.elementAudioSource ||= runtime.audioContext.createMediaElementSource($("video")));
    runtime.audioSource = source;
    source.connect(runtime.analyser);
    runtime.previousSpectrum.fill(0);
    await runtime.audioContext.resume();
    return;
  }
  const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextCtor) return;
  const context = new AudioContextCtor();
  const analyser = context.createAnalyser();
  analyser.fftSize = 2048;
  analyser.smoothingTimeConstant = 0;
  const gain = context.createGain();
  gain.gain.value = 0;
  const source = runtime.sourceKind === "camera" ? context.createMediaStreamSource(runtime.stream) : context.createMediaElementSource($("video"));
  source.connect(analyser);
  analyser.connect(gain);
  gain.connect(context.destination);
  runtime.audioContext = context;
  runtime.audioSource = source;
  runtime.elementAudioSource = runtime.sourceKind === "replay" ? source : null;
  runtime.analyser = analyser;
  runtime.muteGain = gain;
  runtime.audioSamples = new Float32Array(analyser.fftSize);
  runtime.audioSpectrum = new Uint8Array(analyser.frequencyBinCount);
  runtime.previousSpectrum = new Uint8Array(analyser.frequencyBinCount);
  await context.resume();
}

const DFT_SIZE = 512, DFT_BINS = 32;
const DFT_COS = Array.from({ length: DFT_BINS }, (_, frequency) => Float64Array.from({ length: DFT_SIZE }, (_, index) => Math.cos(2 * Math.PI * frequency * index / DFT_SIZE)));
const DFT_SIN = Array.from({ length: DFT_BINS }, (_, frequency) => Float64Array.from({ length: DFT_SIZE }, (_, index) => Math.sin(2 * Math.PI * frequency * index / DFT_SIZE)));

async function prepareReplayAudio() {
  runtime.decodedAudio = null;
  runtime.previousDecodedSpectrum = new Float64Array(DFT_BINS);
  if (!runtime.replayFile || !runtime.audioContext) return;
  const bytes = await runtime.replayFile.arrayBuffer();
  try { runtime.decodedAudio = await runtime.audioContext.decodeAudioData(bytes); }
  catch { runtime.decodedAudio = null; }
}

function decodedAudioMeasurement(stamp) {
  const buffer = runtime.decodedAudio;
  if (!buffer) return null;
  const data = buffer.getChannelData(0), rate = buffer.sampleRate;
  const end = Math.max(0, Math.min(data.length, Math.round(stamp * rate / 1000)));
  const rmsStart = Math.max(0, end - Math.round(.04 * rate));
  let energy = 0;
  for (let index = rmsStart; index < end; index += 1) energy += data[index] * data[index];
  const rms = Math.sqrt(energy / Math.max(1, end - rmsStart));
  const start = Math.max(0, end - DFT_SIZE);
  const spectrum = new Float64Array(DFT_BINS);
  for (let frequency = 0; frequency < DFT_BINS; frequency += 1) {
    let real = 0, imaginary = 0;
    for (let index = 0; index < DFT_SIZE; index += 1) {
      const sample = data[start + index] || 0;
      real += sample * DFT_COS[frequency][index];
      imaginary -= sample * DFT_SIN[frequency][index];
    }
    spectrum[frequency] = Math.hypot(real, imaginary) / DFT_SIZE;
  }
  let flux = 0;
  for (let index = 0; index < DFT_BINS; index += 1) flux += Math.max(0, spectrum[index] - runtime.previousDecodedSpectrum[index]);
  runtime.previousDecodedSpectrum.set(spectrum);
  return { valid: true, rms, dbfs: rms > 0 ? 20 * Math.log10(rms) : -100,
    spectral_flux: flux, audio_start_ms: stamp - 40, audio_end_ms: stamp };
}

function audioMeasurement(stamp) {
  if (runtime.sourceKind === "replay") {
    const decoded = decodedAudioMeasurement(stamp);
    if (decoded) return decoded;
  }
  if (!runtime.analyser) return { valid: false, dbfs: NaN, rms: NaN, spectral_flux: NaN };
  runtime.analyser.getFloatTimeDomainData(runtime.audioSamples);
  runtime.analyser.getByteFrequencyData(runtime.audioSpectrum);
  let energy = 0, flux = 0;
  for (const sample of runtime.audioSamples) energy += sample * sample;
  for (let index = 0; index < runtime.audioSpectrum.length; index += 1) {
    flux += Math.max(0, runtime.audioSpectrum[index] - runtime.previousSpectrum[index]) / 255;
  }
  const rms = Math.sqrt(energy / runtime.audioSamples.length);
  runtime.previousSpectrum.set(runtime.audioSpectrum);
  return { valid: true, rms, dbfs: rms > 0 ? 20 * Math.log10(rms) : -100,
    spectral_flux: flux / runtime.audioSpectrum.length, audio_start_ms: stamp - 40, audio_end_ms: stamp };
}

async function closeModels() {
  runtime.poseModel?.close();
  runtime.handModel?.close();
  runtime.poseModel = runtime.handModel = null;
}

async function loadGeometryModels(generation) {
  $("geometryStatus").textContent = "geometry providers loading";
  try {
    const { PoseLandmarker, HandLandmarker, FilesetResolver } = await import(MEDIAPIPE_URL);
    const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM);
    const create = async (Factory, modelAssetPath, extra) => {
      for (const delegate of ["GPU", "CPU"]) {
        try { return await Factory.createFromOptions(vision, { baseOptions: { modelAssetPath, delegate }, ...extra }); }
        catch { /* Try portable CPU. */ }
      }
      return null;
    };
    const [pose, hand] = await Promise.all([
      create(PoseLandmarker, POSE_MODEL, { runningMode: "VIDEO", numPoses: 1 }),
      create(HandLandmarker, HAND_MODEL, { runningMode: "VIDEO", numHands: 1 }),
    ]);
    if (generation !== runtime.generation || !runtime.active) { pose?.close(); hand?.close(); return; }
    runtime.poseModel = pose;
    runtime.handModel = hand;
    $("geometryStatus").textContent = pose && hand ? "MediaPipe pose + hand · VIDEO · 1 entity" : "geometry unavailable · motion/audio remain active";
  } catch {
    if (generation === runtime.generation) $("geometryStatus").textContent = "geometry unavailable · motion/audio remain active";
  }
}

function clearRuntimeMedia() {
  if (runtime.videoCallback && $("video").cancelVideoFrameCallback) $("video").cancelVideoFrameCallback(runtime.videoCallback);
  clearInterval(runtime.fallbackCallback);
  runtime.videoCallback = runtime.fallbackCallback = null;
  runtime.stream?.getTracks().forEach((track) => track.stop());
  runtime.stream = null;
  if (runtime.objectUrl) URL.revokeObjectURL(runtime.objectUrl);
  runtime.objectUrl = null;
  $("video").pause();
  $("video").removeAttribute("src");
  $("video").srcObject = null;
  $("video").load();
  runtime.audioSource?.disconnect();
  runtime.audioSource = null;
  runtime.previousSpectrum?.fill(0);
  closeModels();
}

function resetRun() {
  runtime.framesSeen = 0;
  runtime.modelCalls = 0;
  runtime.moments = [];
  runtime.selectedMoment = null;
  runtime.sequence = 0;
  runtime.samples = [];
  runtime.nextSampleMs = 0;
  runtime.lastFrameMs = null;
  runtime.lastPresentedFrames = null;
  runtime.frameBuffer = [];
  runtime.frameCursor = 0;
  runtime.previousPixels = null;
  runtime.previousCentroid = null;
  runtime.previousDecodedSpectrum?.fill(0);
  runtime.frameStep = false;
  runtime.frameIndex = 0;
  runtime.handPath = [];
  runtime.lastPoseMs = null;
  runtime.poseSegment += 1;
  runtime.handRaise.reset();
  runtime.penDrop.reset();
  runtime.swingPeak.reset();
  $("momentsList").replaceChildren();
  $("emptyMoments").hidden = false;
  $("answerText").textContent = "Select a caught moment to inspect its evidence-pinned answer.";
  $("answerMeta").textContent = "No moment selected";
  $("sampledFrame").hidden = true;
  $("samplePlaceholder").hidden = false;
  $("runLog").replaceChildren();
  clearTimeout(runtime.quietControl?.timer);
  runtime.quietControl = null;
  $("quietResult").textContent = "No control run";
  updateCounters();
}

async function startCamera() {
  stopCapture();
  resetRun();
  const generation = runtime.generation;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: true });
    if (generation !== runtime.generation) { stream.getTracks().forEach((track) => track.stop()); return; }
    runtime.stream = stream;
    runtime.sourceKind = "camera";
    runtime.sourceName = "live camera";
    $("video").srcObject = stream;
    $("video").muted = true;
    await $("video").play();
    await beginProcessing("live smoke");
    loadGeometryModels(generation);
  } catch (error) {
    setStatus(`Camera/audio unavailable: ${error.message}. Choose a replay file instead.`, true);
  }
}

async function loadReplay(file) {
  if (!file) return;
  stopCapture();
  resetRun();
  runtime.sourceKind = "replay";
  runtime.sourceName = file.name;
  runtime.replayFile = file;
  runtime.decodedAudio = null;
  runtime.objectUrl = URL.createObjectURL(file);
  $("video").src = runtime.objectUrl;
  $("video").muted = false;
  $("replayButton").disabled = false;
  $("frameReplayButton").disabled = false;
  $("sourceLabel").textContent = `${file.name} · synthetic/file replay`;
  setStatus("Replay loaded. Press Replay; detection uses video motion and its audio lane.");
}

async function startReplay() {
  if (!runtime.replayFile) return;
  try {
    await loadReplay(runtime.replayFile);
    await ensureAudio();
    await prepareReplayAudio();
    await $("video").play();
    await beginProcessing("synthetic/file replay");
  } catch (error) { setStatus(`Replay failed: ${error.message}`, true); }
}

async function waitForMetadata() {
  if ($("video").readyState >= 1) return;
  await new Promise((resolve, reject) => {
    $("video").addEventListener("loadedmetadata", resolve, { once: true });
    $("video").addEventListener("error", () => reject(new Error("video metadata could not be decoded")), { once: true });
  });
}

async function startFrameReplay() {
  if (!runtime.replayFile) return;
  try {
    await loadReplay(runtime.replayFile);
    await waitForMetadata();
    await ensureAudio();
    await prepareReplayAudio();
    runtime.frameStep = true;
    await beginProcessing("synthetic · 30 FPS frame-step replay");
  } catch (error) { setStatus(`Frame-step replay failed: ${error.message}`, true); }
}

function processNextSeekedFrame() {
  if (!runtime.active || !runtime.frameStep) return;
  const video = $("video"), generation = runtime.generation;
  const frameCount = Math.ceil(video.duration * 30);
  if (runtime.frameIndex >= frameCount) {
    runtime.active = false;
    runtime.frameStep = false;
    $("stopButton").disabled = true;
    $("quietButton").disabled = true;
    setStatus(`Replay complete · ${runtime.moments.length} moments · ${runtime.modelCalls} model calls`);
    return;
  }
  const target = Math.min((runtime.frameIndex + .5) / 30, Math.max(0, video.duration - .001));
  const advance = () => {
    if (generation !== runtime.generation || !runtime.active) return;
    processVideoFrame({ mediaTime: runtime.frameIndex / 30 });
    runtime.frameIndex += 1;
    setTimeout(processNextSeekedFrame, 1000 / 30);
  };
  if (Math.abs(video.currentTime - target) < .0001) advance();
  else {
    video.addEventListener("seeked", advance, { once: true });
    video.currentTime = target;
  }
}

async function beginProcessing(label) {
  runtime.active = true;
  runtime.sourceEpoch = `${runtime.sourceKind}:${runtime.generation}`;
  $("artifactLabel").textContent = label;
  $("sourceLabel").textContent = runtime.sourceKind === "camera" ? "Live camera · smoke until separately measured" : `${runtime.sourceName} · replay timeline, not wall-clock capture`;
  $("stopButton").disabled = false;
  $("quietButton").disabled = false;
  $("stageEmpty").hidden = true;
  if (runtime.sourceKind === "replay") await ensureAudio();
  else await ensureAudio().catch(() => {});
  setStatus(runtime.sourceKind === "replay" ? "Catching at native replay fps · model fires only on a moment" : "Live sensors active · no event-triggered call while quiet");
  if (runtime.frameStep) processNextSeekedFrame();
  else scheduleVideoFrame();
}

function stopCapture() {
  runtime.generation += 1;
  runtime.active = false;
  clearTimeout(runtime.quietControl?.timer);
  runtime.quietControl = null;
  clearRuntimeMedia();
  $("stopButton").disabled = true;
  $("quietButton").disabled = true;
  $("replayButton").disabled = !runtime.replayFile;
  $("frameReplayButton").disabled = !runtime.replayFile;
  if ($("runStatus")) setStatus("Stopped. Choose a camera or replay input.");
}

function replayTimestamp(metadata) {
  return runtime.sourceKind === "replay" ? Math.round((metadata?.mediaTime ?? $("video").currentTime) * 1000) : Date.now();
}

function scheduleVideoFrame() {
  if (!runtime.active) return;
  const video = $("video");
  if (video.requestVideoFrameCallback && runtime.videoCallback === null) {
    runtime.videoCallback = video.requestVideoFrameCallback((now, metadata) => {
      runtime.videoCallback = null;
      runtime.lastPresentedFrames = metadata.presentedFrames ?? video.getVideoPlaybackQuality?.().totalVideoFrames ?? null;
      processVideoFrame(metadata);
      scheduleVideoFrame();
    });
  }
  if (runtime.fallbackCallback === null) {
    runtime.fallbackCallback = setInterval(() => {
      const presented = video.getVideoPlaybackQuality?.().totalVideoFrames ?? null;
      if (video.readyState >= 2 && !video.paused && (presented === null || presented !== runtime.lastPresentedFrames)) {
        runtime.lastPresentedFrames = presented;
        processVideoFrame({ mediaTime: video.currentTime });
      }
    }, 16);
  }
}

function retainFrame(stamp) {
  const video = $("video");
  const sourceWidth = video.videoWidth || 640, sourceHeight = video.videoHeight || 360;
  const scale = Math.min(1, 640 / sourceWidth);
  const width = Math.max(1, Math.round(sourceWidth * scale)), height = Math.max(1, Math.round(sourceHeight * scale));
  let frame = runtime.frameBuffer[runtime.frameCursor % FRAME_BUFFER_SIZE];
  if (!frame) {
    const canvas = document.createElement("canvas");
    frame = { canvas, context: canvas.getContext("2d", { alpha: false }) };
    runtime.frameBuffer[runtime.frameCursor % FRAME_BUFFER_SIZE] = frame;
  }
  if (frame.canvas.width !== width || frame.canvas.height !== height) {
    frame.canvas.width = width; frame.canvas.height = height;
  }
  frame.context.drawImage(video, 0, 0, width, height);
  frame.capture_ms = stamp;
  frame.geometry = null;
  runtime.frameCursor += 1;
  return frame;
}

function currentMotion(stamp) {
  const canvas = runtime.processCanvas;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.drawImage($("video"), 0, 0, canvas.width, canvas.height);
  const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
  const interval = runtime.lastFrameMs === null ? 0 : stamp - runtime.lastFrameMs;
  const motion = measureMotion(pixels, runtime.previousPixels, canvas.width, canvas.height, interval);
  if (motion.changed_centroid && runtime.previousCentroid) {
    motion.dx_pixels = motion.changed_centroid.x - runtime.previousCentroid.x;
    motion.dy_pixels = motion.changed_centroid.y - runtime.previousCentroid.y;
    motion.median_speed_pixels_per_second = Math.hypot(motion.dx_pixels, motion.dy_pixels) * 1000 / interval;
  }
  runtime.previousPixels = new Uint8ClampedArray(pixels);
  runtime.previousCentroid = motion.changed_centroid || null;
  motion.interval_ms = interval;
  motion.camera_motion = motion.moving_fraction > .35;
  return motion;
}

function poseObservation(stamp) {
  if (!runtime.poseModel || runtime.sourceKind !== "camera") return null;
  if (runtime.lastPoseMs !== null && stamp - runtime.lastPoseMs > 100) runtime.poseSegment += 1;
  runtime.lastPoseMs = stamp;
  const result = runtime.poseModel.detectForVideo($("video"), performance.now());
  const landmarks = result.landmarks?.[0];
  const makeSide = (wristIndex, shoulderIndex) => {
    const wrist = landmarks?.[wristIndex], shoulder = landmarks?.[shoulderIndex];
    if (!wrist || !shoulder) return { valid: false };
    const visibility = Math.min(wrist.visibility ?? 1, shoulder.visibility ?? 1);
    return { valid: Number.isFinite(wrist.x) && Number.isFinite(wrist.y) && Number.isFinite(shoulder.x) && Number.isFinite(shoulder.y),
      wrist_y: wrist.y, shoulder_y: shoulder.y, visibility };
  };
  return { timestamp_ms: stamp, source_epoch: `${runtime.sourceEpoch}:pose:${runtime.poseSegment}`,
    left: makeSide(15, 11), right: makeSide(16, 12), landmarks };
}

function handObservation(stamp, motion) {
  if (!runtime.handModel || runtime.sourceKind !== "camera") return null;
  const result = runtime.handModel.detectForVideo($("video"), performance.now());
  const landmarks = result.landmarks?.length === 1 ? result.landmarks[0] : null;
  const point = landmarks?.[8];
  return { timestamp_ms: stamp, source_epoch: runtime.sourceEpoch, valid: Boolean(point),
    point: point ? { x: point.x * $("video").videoWidth, y: point.y * $("video").videoHeight } : null,
    moving_fraction: motion.moving_fraction, sample_count: motion.sample_count };
}

function processVideoFrame(metadata) {
  if (!runtime.active || $("video").readyState < 2) return;
  const stamp = replayTimestamp(metadata);
  if (runtime.lastFrameMs === stamp) return;
  const frame = retainFrame(stamp);
  const motion = currentMotion(stamp);
  const audio = audioMeasurement(stamp);
  runtime.framesSeen += 1;
  if (stamp >= runtime.nextSampleMs) {
    captureSample(stamp, frame);
    runtime.nextSampleMs = (Math.floor(stamp / 1000) + 1) * 1000;
  }
  runtime.lastFrameMs = stamp;
  let pose = null, hand = null;
  try { pose = poseObservation(stamp); hand = handObservation(stamp, motion); }
  catch { $("geometryStatus").textContent = "geometry frame rejected · motion/audio remain active"; }
  frame.geometry = { pose, hand, motion, audio };
  const events = [];
  if (pose) events.push(...runtime.handRaise.update(pose));
  if (hand) {
    const peak = runtime.swingPeak.update(hand);
    if (peak) events.push(peak);
    if (hand.valid) runtime.handPath.push({ stamp, ...hand.point });
  }
  if (motion.valid && motion.interval_ms > 0) {
    const impact = runtime.penDrop.update({ timestamp_ms: stamp, source_epoch: runtime.sourceEpoch,
      ...motion, ...audio, valid: motion.valid && audio.valid });
    if (impact) events.push(impact);
  }
  if (runtime.sourceKind === "replay") $("geometryStatus").textContent = runtime.penDrop.baseline ? `impact armed · ${runtime.penDrop.downRun} downward` : "impact quiet baseline";
  $("motionReadout").textContent = `dy ${motion.dy_pixels.toFixed(1)} px · moving ${(motion.moving_fraction * 100).toFixed(2)}% · flux ${Number.isFinite(audio.spectral_flux) ? audio.spectral_flux.toFixed(3) : "—"}`;
  for (const event of events) addCaughtMoment(event);
  if (runtime.handPath.length > 120) runtime.handPath.splice(0, runtime.handPath.length - 120);
  drawOverlay(stamp, frame);
  $("nativeFps").textContent = motion.interval_ms > 0 ? `${(1000 / motion.interval_ms).toFixed(1)} fps` : "native fps";
  $("audioReadout").textContent = audio.valid ? `audio ${audio.dbfs.toFixed(1)} dBFS` : "audio unavailable · impact abstains";
  updateCounters();
}

function captureSample(sampleMs, frame) {
  runtime.samples.push(sampleMs);
  if (runtime.samples.length > 120) runtime.samples.shift();
  const image = frame.canvas.toDataURL("image/jpeg", .72);
  $("sampledFrame").src = image;
  $("sampledFrame").hidden = false;
  $("samplePlaceholder").hidden = true;
  $("sampleTime").textContent = `sample ${sampleMs} ms`;
  $("sampleGrid").classList.remove("tick");
  void $("sampleGrid").offsetWidth;
  $("sampleGrid").classList.add("tick");
}

function drawOverlay(stamp, frame) {
  const canvas = $("overlay");
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(rect.width * ratio)), height = Math.max(1, Math.round(rect.height * ratio));
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, width, height);
  const lineY = height - 24 * ratio;
  context.strokeStyle = "rgba(255,255,255,.55)";
  context.fillStyle = "rgba(255,255,255,.85)";
  context.lineWidth = ratio;
  context.font = `${9 * ratio}px ui-monospace, monospace`;
  context.beginPath(); context.moveTo(12 * ratio, lineY); context.lineTo(width - 12 * ratio, lineY); context.stroke();
  for (let index = 0; index <= 5; index += 1) {
    const x = (12 + (width / ratio - 24) * index / 5) * ratio;
    context.beginPath(); context.moveTo(x, lineY - 5 * ratio); context.lineTo(x, lineY + 5 * ratio); context.stroke();
    context.fillText(`${Math.max(0, Math.floor(stamp / 1000) - 5 + index)}s`, x - 7 * ratio, lineY + 16 * ratio);
  }
  const phase = (stamp % 1000) / 1000;
  const playX = (12 + (width / ratio - 24) * phase) * ratio;
  context.strokeStyle = "#b9b2ff";
  context.beginPath(); context.moveTo(playX, 0); context.lineTo(playX, height); context.stroke();
  const pose = frame.geometry?.pose;
  if (pose?.landmarks) {
    for (const index of [11, 12, 15, 16]) {
      const point = pose.landmarks[index];
      context.fillStyle = index >= 15 ? "#ffb55f" : "#a9a0ff";
      context.beginPath(); context.arc(point.x * width, point.y * height, 4 * ratio, 0, Math.PI * 2); context.fill();
    }
  }
}

function frameAt(captureMs) {
  return runtime.frameBuffer.filter(Boolean).reduce((best, frame) => {
    if (!best) return frame;
    return Math.abs(frame.capture_ms - captureMs) < Math.abs(best.capture_ms - captureMs) ? frame : best;
  }, null);
}

function evidenceImage(event, frame) {
  const canvas = document.createElement("canvas");
  canvas.width = frame.canvas.width; canvas.height = frame.canvas.height;
  const context = canvas.getContext("2d", { alpha: false });
  context.drawImage(frame.canvas, 0, 0);
  context.strokeStyle = "#b9b2ff"; context.lineWidth = 2;
  context.strokeRect(canvas.width * .1, canvas.height * .1, canvas.width * .8, canvas.height * .8);
  const geometry = frame.geometry;
  if (event.type === "hand_raise" && geometry?.pose?.landmarks) {
    const wrist = geometry.pose.landmarks[event.side === "left" ? 15 : 16];
    const shoulder = geometry.pose.landmarks[event.side === "left" ? 11 : 12];
    if (wrist && shoulder) {
      context.strokeStyle = "#ffb55f"; context.beginPath(); context.moveTo(shoulder.x * canvas.width, shoulder.y * canvas.height);
      context.lineTo(wrist.x * canvas.width, wrist.y * canvas.height); context.stroke();
    }
  }
  if (event.type === "swing_peak" && geometry?.hand?.point) {
    const point = geometry.hand.point;
    const x = point.x / $("video").videoWidth * canvas.width, y = point.y / $("video").videoHeight * canvas.height;
    context.fillStyle = "#ffb55f"; context.beginPath(); context.arc(x, y, 6, 0, Math.PI * 2); context.fill();
  }
  return canvas.toDataURL("image/jpeg", .86);
}

function metricText(event) {
  if (event.type === "hand_raise") return `${event.side} wrist/shoulder · confirmed +${event.confirmation_latency_ms} ms`;
  if (event.type === "pen_drop") return `${event.downward_speed_px_s.toFixed(1)} px/s downward · ${event.dbfs.toFixed(1)} dBFS · audio [${event.audio_interval.join(", ")}] ms`;
  return `${event.peak_speed_px_s.toFixed(1)} px/s · interval [${event.interval_start_ms}, ${event.interval_end_ms}] ms`;
}

function addCaughtMoment(event) {
  const frame = frameAt(event.capture_ms);
  if (!frame || Math.abs(frame.capture_ms - event.capture_ms) > 50) {
    logRun(`Missed evidence binding at ${event.capture_ms} ms; no thumbnail substituted.`, "degraded");
    return;
  }
  event.capture_ms = frame.capture_ms;
  event.evidence_id = `mc:${runtime.generation}:${String(++runtime.sequence).padStart(3, "0")}:${event.capture_ms}`;
  event.thumbnail = evidenceImage(event, frame);
  event.source = runtime.sourceKind;
  event.artifact = runtime.sourceKind === "replay" ? "synthetic/file replay" : "live smoke";
  event.badge = oneFpsBadge(event.capture_ms, runtime.samples);
  event.modelState = "pending";
  runtime.moments.unshift(event);
  $("emptyMoments").hidden = true;
  renderMoment(event);
  selectMoment(event);
  logRun(`${MOMENT_LABELS[event.type]} caught at ${event.capture_ms} ms · ${event.evidence_id}`, "catch");
  if ($("describeMoments").checked) describeMoment(event);
  else { event.modelState = "not requested"; updateMomentAnswer(event); }
}

function renderMoment(event) {
  const card = document.createElement("article");
  card.className = "moment-card";
  card.id = `moment-${event.evidence_id}`;
  card.dataset.evidence = event.evidence_id;
  const image = document.createElement("img");
  image.src = event.thumbnail; image.alt = `${MOMENT_LABELS[event.type]} at ${event.capture_ms} ms`;
  const body = document.createElement("div");
  body.className = "moment-body";
  const title = document.createElement("div"); title.className = "moment-title";
  const heading = document.createElement("h3"); heading.textContent = MOMENT_LABELS[event.type];
  const badge = document.createElement("span"); badge.className = `fps-badge ${event.badge.status}`;
  badge.textContent = `1 FPS: ${event.badge.status}`;
  title.append(heading, badge);
  const time = document.createElement("code"); time.textContent = `capture_ms=${event.capture_ms}`;
  const evidence = document.createElement("a"); evidence.href = `#${card.id}`; evidence.textContent = event.evidence_id;
  evidence.addEventListener("click", () => selectMoment(event));
  const metrics = document.createElement("p"); metrics.textContent = metricText(event);
  const answer = document.createElement("p"); answer.className = "moment-answer"; answer.dataset.answer = event.evidence_id; answer.textContent = "VLM call pending…";
  body.append(title, time, evidence, metrics, answer);
  card.append(image, body);
  card.addEventListener("click", () => selectMoment(event));
  $("momentsList").prepend(card);
}

function selectMoment(event) {
  runtime.selectedMoment = event;
  document.querySelectorAll(".moment-card").forEach((card) => card.classList.toggle("selected", card.dataset.evidence === event.evidence_id));
  $("answerMeta").textContent = `capture_ms=${event.capture_ms} · ${event.evidence_id}`;
  updateMomentAnswer(event);
}

function updateMomentAnswer(event) {
  const cardAnswer = document.querySelector(`[data-answer="${CSS.escape(event.evidence_id)}"]`);
  if (cardAnswer) cardAnswer.textContent = event.answer || (event.modelState === "pending" ? "VLM call pending…" : event.modelState);
  if (runtime.selectedMoment === event) {
    $("answerText").textContent = event.answer || (event.modelState === "pending" ? "Describing the frozen evidence once…" : event.modelState);
  }
}

async function describeMoment(event) {
  if (event.modelRequested) return;
  event.modelRequested = true;
  const config = providerConfig();
  if (!config.baseUrl || !config.model) {
    event.modelState = "provider not configured · detection retained";
    updateMomentAnswer(event);
    return;
  }
  runtime.modelCalls += 1;
  updateCounters();
  try {
    const evidence = { ...event };
    delete evidence.thumbnail; delete evidence.answer; delete evidence.modelRequested;
    const response = await fetch(chatEndpoint(config.baseUrl), {
      method: "POST",
      headers: { "Content-Type": "application/json", ...(config.key ? { Authorization: `Bearer ${config.key}` } : {}) },
      body: JSON.stringify({ model: config.model, temperature: .1, max_tokens: 140, messages: [
        { role: "system", content: "Describe only observable image and sensor measurements. Do not infer emotion, intent, identity, material, force, or calibrated physical speed. Cite the supplied evidence_id and capture_ms. Keep the answer under 60 words." },
        { role: "user", content: [{ type: "text", text: `One caught moment. Evidence: ${JSON.stringify(evidence)}. Describe this frozen frame and its measurements.` },
          { type: "image_url", image_url: { url: event.thumbnail } }] },
      ] }),
    });
    if (!response.ok) throw new Error(`${response.status} ${(await response.text()).slice(0, 180)}`);
    const payload = await response.json();
    event.answer = payload.choices?.[0]?.message?.content;
    if (typeof event.answer !== "string" || !event.answer.trim()) throw new Error("provider returned no description");
    event.modelState = "answered";
  } catch (error) {
    event.modelState = "model call failed";
    event.answer = `Model call failed: ${error.message}. Sensor evidence is unchanged.`;
  }
  updateMomentAnswer(event);
}

function startQuietControl() {
  if (!runtime.active) return;
  clearTimeout(runtime.quietControl?.timer);
  const startingCalls = runtime.modelCalls;
  const startingMoments = runtime.moments.length;
  runtime.quietControl = { startingCalls, startingMoments, timer: null };
  $("quietButton").disabled = true;
  $("quietResult").textContent = "30 s control running · heartbeat traffic: none";
  logRun("30-second no-event control started; no heartbeat/baseline model traffic configured.");
  runtime.quietControl.timer = setTimeout(() => {
    const newCalls = runtime.modelCalls - startingCalls;
    const newMoments = runtime.moments.length - startingMoments;
    $("quietResult").textContent = newCalls === 0 ? "PASS · 0 event-triggered model calls in 30 s" : `CHECK · ${newCalls} model calls / ${newMoments} moments`;
    logRun(newCalls === 0 ? "30-second quiet control: no event-triggered model calls." : `30-second control: ${newCalls} calls and ${newMoments} detected moments; inspect as potential false triggers.`, newCalls === 0 ? "observation" : "degraded");
    $("quietButton").disabled = !runtime.active;
    runtime.quietControl = null;
  }, 30000);
}

function bindControls() {
  $("cameraButton").addEventListener("click", startCamera);
  $("replayInput").addEventListener("change", (event) => loadReplay(event.target.files?.[0]));
  $("replayButton").addEventListener("click", startReplay);
  $("frameReplayButton").addEventListener("click", startFrameReplay);
  $("stopButton").addEventListener("click", stopCapture);
  $("quietButton").addEventListener("click", startQuietControl);
  $("pricePerCall").addEventListener("input", updateCounters);
  $("describeMoments").addEventListener("change", () => {
    if ($("describeMoments").checked) for (const event of runtime.moments) if (!event.modelRequested) describeMoment(event);
  });
  $("video").addEventListener("ended", () => {
    if (!runtime.active) return;
    runtime.active = false;
    if (runtime.videoCallback && $("video").cancelVideoFrameCallback) $("video").cancelVideoFrameCallback(runtime.videoCallback);
    clearInterval(runtime.fallbackCallback);
    runtime.videoCallback = runtime.fallbackCallback = null;
    $("stopButton").disabled = true;
    $("quietButton").disabled = true;
    setStatus(`Replay complete · ${runtime.moments.length} moments · ${runtime.modelCalls} model calls`);
  });
  window.addEventListener("beforeunload", clearRuntimeMedia);
}

function boot() {
  runtime.processCanvas = document.createElement("canvas");
  runtime.processCanvas.width = 160;
  runtime.processCanvas.height = 90;
  bindControls();
  updateCounters();
  loadOptionalProvider();
}

if (typeof window !== "undefined") window.addEventListener("DOMContentLoaded", boot);
