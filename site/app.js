import {
  LOCALES, t, utf8Bytes, gaugeValues, packetMetrics, packetRate, evidenceIds,
  compactPacket, buildChatMessages, chatEndpoint, createSSEParser, extractDelta, captureSnapshotDecision, captureTransitionCanClear, captureGenerationOwnsMedia, normalizeLocalModelConfig,
  sparklinePoints,
} from "./playground.js";

const $ = (id) => document.getElementById(id);
// Controls are intentionally looked up at the point of use because the same
// client is served by the static browser demo and the authenticated local
// dashboard. A partially loaded/older page must not turn a missing control
// into a capture failure (for example, `undefined.value`).
function controlValue(id, fallback = "") {
  const node = $(id);
  return node && "value" in node ? node.value : fallback;
}
function setControlValue(id, value) {
  const node = $(id);
  if (node && "value" in node) node.value = value;
}
const sessionToken = document.querySelector('meta[name="gamst-session-token"]')?.content || "";
const isLocal = Boolean(sessionToken && sessionToken !== "__SESSION_TOKEN__");
const MEDIAPIPE_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";
const MEDIAPIPE_WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const FACE_MODEL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const POSE_MODEL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";
const STORAGE = "gamst.playground.";
const TAB_NAMES = ["response", "context", "json", "metrics"];
const GAUGE_NAMES = ["smile", "eyeClosure", "jawOpen", "browRaise", "audioRms", "speechActivity"];
const emptyPacket = () => ({ source: isLocal ? "local_dashboard" : "browser_preview", timestamp_ms: Date.now(), observations: {}, raw_media_included: false, live: false });

const state = {
  locale: "en", active: false, consent: false, generation: 0, packet: emptyPacket(), sentPacket: null,
  stream: null, audioContext: null, analyser: null, processor: null, source: null, mute: null,
  faceModel: null, poseModel: null, captureTimer: null, audioBuffer: [], audioSending: false,
  videoBusy: false, localController: null, localData: null, localReconnect: null,
  localCaptureRevision: 0, localCaptureTransition: null, localCaptureTransitionId: 0,
  arrivals: [], droppedFrames: 0, histories: Object.fromEntries(GAUGE_NAMES.map((name) => [name, []])),
  metricHistory: [], audioTimeline: [], sceneTimeline: [], lastTimelineStamp: 0,
  sending: false, chatController: null, featureSignature: "", guidePreview: null, localMyProxyConfig: null, localMyProxyActive: false,
};
let settingsTimer = null;

function readStorage(kind, key) {
  try { return (kind === "session" ? sessionStorage : localStorage).getItem(STORAGE + key) || ""; } catch { return ""; }
}
function writeStorage(kind, key, value) {
  try { (kind === "session" ? sessionStorage : localStorage).setItem(STORAGE + key, value); } catch { /* Storage can be disabled without blocking the tool. */ }
}
function removeStorage(kind, key) {
  try { (kind === "session" ? sessionStorage : localStorage).removeItem(STORAGE + key); } catch { /* no-op */ }
}

async function loadLocalMyProxyConfig() {
  try {
    const module = await import("./local-config.js");
    return normalizeLocalModelConfig(module);
  } catch {
    // The file is intentionally optional and ignored by git. A missing local
    // config is a normal state for public/demo deployments.
    return null;
  }
}

function updateLocalMyProxyButton() {
  const button = $("useLocalMyProxy");
  if (!button) return;
  const available = Boolean(state.localMyProxyConfig);
  button.disabled = !available;
  button.title = available ? "" : t(state.locale, "localMyProxyUnavailable");
}

function applyLocalMyProxyConfig() {
  const config = state.localMyProxyConfig;
  if (!config) return showError("errorLocalMyProxy");
  setControlValue("provider", "byok");
  setControlValue("baseUrl", config.baseUrl);
  setControlValue("model", config.model);
  setControlValue("apiKey", config.apiKey);
  state.localMyProxyActive = true;
  if ($("rememberKey")) $("rememberKey").checked = false;
  // Endpoint/model are harmless UI preferences; keep the credential in the
  // session only so the local config never causes a persistent key write.
  writeStorage("local", "baseUrl", config.baseUrl);
  writeStorage("local", "model", config.model);
  writeStorage("local", "provider", "byok");
  writeStorage("local", "rememberKey", "false");
  writeStorage("session", "apiKey", config.apiKey);
  removeStorage("local", "apiKey");
  updateProviderNotice();
  renderResponseSetup(false);
  $("settingsStatus").textContent = t(state.locale, "localMyProxyApplied");
  clearTimeout(settingsTimer);
  settingsTimer = setTimeout(() => { $("settingsStatus").textContent = ""; }, 2000);
}

try {
  const saved = localStorage.getItem("gamst.uiLanguage");
  if (LOCALES.includes(saved)) state.locale = saved;
} catch { /* default en */ }

function applyI18n() {
  document.documentElement.lang = state.locale;
  document.title = t(state.locale, "appTitle");
  setControlValue("language", state.locale);
  document.querySelectorAll("[data-i18n]").forEach((node) => { node.textContent = t(state.locale, node.dataset.i18n); });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((node) => { node.placeholder = t(state.locale, node.dataset.i18nPlaceholder); });
  $("send").setAttribute("aria-label", t(state.locale, "send"));
  $("connectionLabel").textContent = isLocal ? t(state.locale, "connectionLocal", { port: location.port || "8766" }) : t(state.locale, "connectionBrowser");
  updateProviderNotice();
  updateLocalMyProxyButton();
  renderPacket();
  updateCaptureUI();
  renderResponseSetup(false);
}

function setStatus(key, variables) {
  $("status").textContent = key ? t(state.locale, key, variables) : "";
}
function showError(key, error) {
  const message = error?.message || (typeof error === "string" ? error : "");
  $("error").textContent = t(state.locale, key, { message });
}
function clearError() { $("error").textContent = ""; }
function pressed(id) { return $(id).getAttribute("aria-pressed") === "true"; }
function setPressed(id, value) { $(id).setAttribute("aria-pressed", String(Boolean(value))); }
function bindToggle(id, onChange) {
  $(id).addEventListener("click", () => {
    if ($(id).disabled) return;
    setPressed(id, !pressed(id));
    onChange?.(pressed(id));
  });
}
function selectTab(name, focus = false) {
  if (!TAB_NAMES.includes(name)) return;
  for (const key of TAB_NAMES) {
    const selected = key === name;
    $("tab-" + key).setAttribute("aria-selected", String(selected));
    $("tab-" + key).tabIndex = selected ? 0 : -1;
    $("panel-" + key).hidden = !selected;
  }
  if (focus) $("tab-" + name).focus();
}
for (const [index, name] of TAB_NAMES.entries()) {
  const tab = $("tab-" + name);
  tab.addEventListener("click", () => selectTab(name));
  tab.addEventListener("keydown", (event) => {
    let next;
    if (event.key === "ArrowRight") next = (index + 1) % TAB_NAMES.length;
    else if (event.key === "ArrowLeft") next = (index + TAB_NAMES.length - 1) % TAB_NAMES.length;
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = TAB_NAMES.length - 1;
    else return;
    event.preventDefault();
    selectTab(TAB_NAMES[next], true);
  });
}

for (const id of ["inputCamera", "inputMicrophone", "inputText", "overlayFace", "overlayBody", "overlayPerson"]) {
  bindToggle(id, () => {
    if (id.startsWith("overlay")) drawGuides(state.guidePreview || state.localData || { guides: state.packet.observations, state: { paused: !state.active } });
  });
}

function updateCaptureUI() {
  $("start").disabled = state.active;
  $("stop").disabled = !state.active;
  $("inputCamera").disabled = state.active;
  $("inputMicrophone").disabled = state.active;
  document.querySelectorAll("[data-feature]").forEach((button) => { button.disabled = state.active; });
  $("captureState").textContent = t(state.locale, state.active ? "observing" : "stopped");
  $("stageEmpty").hidden = state.active || Boolean(state.stream) || Boolean(state.guidePreview);
  if (state.active) $("stageEmpty").hidden = true;
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
}
function currentDrops() {
  const workerDrops = (state.localData?.workers || []).reduce((sum, worker) => sum + (Number(worker.dropped) || 0), 0);
  return workerDrops + state.droppedFrames;
}

function renderGauges(values) {
  for (const name of GAUGE_NAMES) {
    const value = values[name];
    const valid = Number.isFinite(value);
    const history = state.histories[name];
    if (valid && state.active) {
      const stamp = state.packet.timestamp_ms;
      if (history.at(-1)?.stamp !== stamp) {
        history.push({ stamp, value });
        if (history.length > 60) history.shift();
      }
    }
    $("value-" + name).textContent = valid ? name === "audioRms" ? value.toFixed(3) : name === "speechActivity" ? t(state.locale, value >= .5 ? "on" : "off") : `${Math.round(value * 100)}` : "—";
    const normalized = valid ? name === "audioRms" ? Math.min(1, value * 4) : Math.max(0, Math.min(1, value)) : 0;
    $("bar-" + name).style.width = `${normalized * 100}%`;
    $("spark-" + name).setAttribute("points", sparklinePoints(history.map((point) => point.value), 120, 28, name === "audioRms" ? null : 1));
    const gauge = document.querySelector(`[data-gauge="${name}"]`);
    gauge.setAttribute("role", "meter");
    gauge.setAttribute("aria-label", t(state.locale, name));
    gauge.setAttribute("aria-valuemin", "0");
    gauge.setAttribute("aria-valuemax", name === "audioRms" ? "1" : "100");
    if (valid) gauge.setAttribute("aria-valuenow", String(name === "audioRms" ? value : value * 100));
    else gauge.removeAttribute("aria-valuenow");
  }
}

function renderMetrics(now = Date.now(), record = false) {
  const metrics = packetMetrics(state.packet, now, currentDrops());
  const rate = packetRate(state.arrivals, now);
  $("metricRate").textContent = `${rate.toFixed(1)} Hz`;
  $("metricBytes").textContent = formatBytes(metrics.bytes);
  $("metricAge").textContent = metrics.ageMs === null ? "—" : `${Math.round(metrics.ageMs)} ms`;
  $("metricDrops").textContent = String(metrics.dropped);
  $("jsonSize").textContent = formatBytes(metrics.bytes);
  $("detailBytes").textContent = formatBytes(metrics.bytes);
  $("detailAge").textContent = metrics.ageMs === null ? "—" : `${Math.round(metrics.ageMs)} ms`;
  $("detailDrops").textContent = String(metrics.dropped);
  $("detailObservations").textContent = String(metrics.observations);
  $("observationCount").textContent = `${metrics.observations} ${t(state.locale, "observations")}`;
  const ids = evidenceIds(state.packet);
  $("detailEvidence").textContent = ids.at(-1) || "—";
  if (record && (state.active || metrics.observations)) {
    state.metricHistory.push({ bytes: metrics.bytes, age: metrics.ageMs });
    if (state.metricHistory.length > 60) state.metricHistory.shift();
  }
  $("bytesSpark").setAttribute("points", sparklinePoints(state.metricHistory.map((point) => point.bytes), 240, 42));
  $("ageSpark").setAttribute("points", sparklinePoints(state.metricHistory.map((point) => point.age), 240, 42));
}

function observationText(name) {
  const observation = state.packet.observations?.[name];
  return observation?.fresh === false ? "" : observation?.value;
}
function renderTimeline() {
  const stamp = state.packet.timestamp_ms;
  if (state.active && stamp !== state.lastTimelineStamp) {
    state.lastTimelineStamp = stamp;
    const rms = state.packet.observations?.["audio.rms"];
    const speech = state.packet.observations?.["speech.activity"];
    const audioValue = Number.isFinite(rms?.value) ? rms.value : Number.isFinite(speech?.value) ? speech.value : typeof speech?.value === "boolean" ? Number(speech.value) : null;
    const faces = state.packet.observations?.["scene.faces"]?.value;
    const poses = state.packet.observations?.["scene.poses"]?.value;
    const sceneValue = Array.isArray(faces) || Array.isArray(poses) ? Math.min(1, ((faces?.length || 0) + (poses?.length || 0)) / 4) : observationText("scene.description") ? 1 : null;
    if (audioValue !== null) state.audioTimeline.push(audioValue);
    if (sceneValue !== null) state.sceneTimeline.push(sceneValue);
    state.audioTimeline = state.audioTimeline.slice(-60);
    state.sceneTimeline = state.sceneTimeline.slice(-60);
  }
  $("audioTimeline").querySelector("polyline").setAttribute("points", sparklinePoints(state.audioTimeline, 600, 30));
  $("sceneTimeline").querySelector("polyline").setAttribute("points", sparklinePoints(state.sceneTimeline, 600, 30));
  const transcript = observationText("text.transcript") || observationText("speech.transcript") || observationText("text.utterance");
  const rms = state.packet.observations?.["audio.rms"]?.value;
  $("audioTimelineLabel").textContent = transcript || (Number.isFinite(rms) && state.active ? `RMS ${rms.toFixed(3)}` : "—");
  const scene = observationText("scene.description");
  const faces = state.active ? state.packet.observations?.["scene.faces"]?.value : null;
  $("sceneTimelineLabel").textContent = scene || (Array.isArray(faces) ? `${faces.length} · ${t(state.locale, "faceLandmarks")}` : "—");
  $("timelineStatus").textContent = state.lastTimelineStamp ? new Date(state.lastTimelineStamp).toLocaleTimeString(state.locale) : t(state.locale, "noTimeline");
}

function renderVoiceAndWorkers() {
  const emotion = state.packet.observations?.["classification.voice_emotion"];
  const labels = emotion?.value?.labels;
  $("voiceEmotion").textContent = state.active && emotion?.fresh !== false && labels ? Object.entries(labels).sort((a, b) => b[1] - a[1]).map(([name, value]) => `${name} ${Number(value).toFixed(3)}`).join(" · ") : t(state.locale, "noEmotion");
  const workers = state.localData?.workers || [];
  $("workers").textContent = workers.length ? workers.map((worker) => `${worker.name.replace("local_vlm_scene", "VLM")} · ${worker.last_latency_ms === null ? "—" : Math.round(worker.last_latency_ms) + "ms"} · Q ${worker.queued} · ${t(state.locale, "dropped")} ${worker.dropped}`).join(" / ") : t(state.locale, "noWorkers");
}

function renderPacket(record = false) {
  const values = gaugeValues(state.packet, state.active);
  renderGauges(values);
  renderMetrics(Date.now(), record);
  renderTimeline();
  renderVoiceAndWorkers();
  $("rawJson").textContent = Object.keys(state.packet.observations || {}).length ? JSON.stringify(state.packet, null, 2) : t(state.locale, "noPacket");
  const faces = state.active ? state.packet.observations?.["scene.faces"]?.value : null;
  $("faceCount").textContent = Array.isArray(faces) ? `${faces.length} · ${t(state.locale, "faceLandmarks")}` : `— · ${t(state.locale, "faceLandmarks")}`;
  const width = state.guidePreview?.width || $("video").videoWidth || 640;
  const height = state.guidePreview?.height || $("video").videoHeight || 360;
  $("stageDimensions").textContent = `${width} × ${height}`;
  if (state.sentPacket) {
    $("contextJson").textContent = JSON.stringify(state.sentPacket, null, 2);
    $("contextSize").textContent = formatBytes(utf8Bytes(JSON.stringify(state.sentPacket)));
  } else {
    $("contextJson").textContent = t(state.locale, "noPacket");
    $("contextSize").textContent = "0 B";
  }
}
function publishPacket(packet, data = null) {
  state.packet = packet || emptyPacket();
  if (data) state.localData = data;
  const now = Date.now();
  state.arrivals.push(now);
  state.arrivals = state.arrivals.filter((time) => now - time <= 5000);
  renderPacket(true);
  if (!state.guidePreview) drawGuides(data || { guides: state.packet.observations, state: { paused: !state.active } });
}

function makeSvg(tag, attributes = {}) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
  return node;
}
function featureEnabled(name) {
  return state.localData?.features?.find((feature) => feature.name === name)?.enabled !== false;
}
function drawGuides(data) {
  const overlay = $("cameraOverlay");
  overlay.replaceChildren();
  const observations = data?.guides || {};
  const paused = data?.state?.paused ?? !state.active;
  const preview = Boolean(data?.preview);
  if ((paused && !preview) || (!preview && !$("video").videoWidth)) return;
  const width = preview ? data.width : $("video").videoWidth;
  const height = preview ? data.height : $("video").videoHeight;
  if (!width || !height) return;
  overlay.setAttribute("viewBox", `0 0 ${width} ${height}`);
  const faces = observations["scene.faces"];
  if (pressed("overlayFace") && faces && faces.fresh !== false && featureEnabled("mediapipe_face_landmarker")) {
    for (const face of Array.isArray(faces.value) ? faces.value : []) {
      const path = (face.landmarks || []).filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y)).map((point) => `M${(point.x * width).toFixed(1)} ${(point.y * height).toFixed(1)}h.01`).join(" ");
      if (path) overlay.append(makeSvg("path", { d: path, fill: "none", stroke: "#68ede4", "stroke-width": 1.6, "stroke-linecap": "round", opacity: .85 }));
    }
  }
  const poses = observations["scene.poses"];
  if (poses && poses.fresh !== false && featureEnabled("mediapipe_pose_landmarker")) {
    const links = [[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],[23,25],[25,27],[24,26],[26,28],[27,29],[29,31],[28,30],[30,32]];
    for (const [poseIndex, pose] of (Array.isArray(poses.value) ? poses.value : []).entries()) {
      const points = pose.landmarks || [];
      const visible = (point) => point && Number.isFinite(point.x) && Number.isFinite(point.y) && (point.visibility ?? 1) >= .5 && (point.presence ?? 1) >= .5;
      if (pressed("overlayBody")) {
        for (const [a, b] of links) if (visible(points[a]) && visible(points[b])) overlay.append(makeSvg("line", { x1: points[a].x * width, y1: points[a].y * height, x2: points[b].x * width, y2: points[b].y * height, stroke: "#bca8ff", "stroke-width": 2 }));
        const path = points.filter(visible).map((point) => `M${(point.x * width).toFixed(1)} ${(point.y * height).toFixed(1)}h.01`).join(" ");
        if (path) overlay.append(makeSvg("path", { d: path, stroke: "#d8ceff", "stroke-width": 5, "stroke-linecap": "round" }));
      }
      if (pressed("overlayPerson")) {
        const visiblePoints = points.filter(visible);
        if (visiblePoints.length) {
          const xs = visiblePoints.map((point) => point.x * width), ys = visiblePoints.map((point) => point.y * height);
          const x = Math.max(0, Math.min(...xs) - 12), y = Math.max(0, Math.min(...ys) - 12);
          const right = Math.min(width, Math.max(...xs) + 12), bottom = Math.min(height, Math.max(...ys) + 12);
          overlay.append(makeSvg("rect", { x, y, width: right - x, height: bottom - y, rx: 5, fill: "none", stroke: "#66e5df", "stroke-width": 1.5 }));
        }
      }
      if (pressed("overlayBody")) {
        const raised = Object.entries(pose.geometry || {}).filter(([, value]) => value === true).map(([key]) => key.startsWith("left") ? "Left hand raised" : "Right hand raised");
        const anchor = points[11];
        if (raised.length && anchor) {
          const label = makeSvg("text", { x: Math.max(5, anchor.x * width), y: Math.max(18, anchor.y * height - 16), fill: "#fff", "font-size": 13, "paint-order": "stroke", stroke: "#353042", "stroke-width": 3 });
          label.textContent = `${pose.track_id || `Body ${poseIndex + 1}`} · ${raised.join(" · ")}`;
          overlay.append(label);
        }
      }
    }
  }
  const tracks = observations["scene.person_tracks"];
  if (pressed("overlayPerson") && tracks && tracks.fresh !== false && featureEnabled("opencv_hog_person")) {
    for (const track of Array.isArray(tracks.value) ? tracks.value : []) {
      if (!track.visible || !track.bbox) continue;
      const box = track.bbox;
      overlay.append(makeSvg("rect", { x: box.x * width, y: box.y * height, width: box.width * width, height: box.height * height, rx: 4, fill: "none", stroke: track.association_quality === "uncertain" ? "#ffd078" : "#66e5df", "stroke-width": 2 }));
      const label = makeSvg("text", { x: box.x * width + 4, y: Math.max(14, box.y * height + 14), fill: "#66e5df", "font-size": 11 });
      label.textContent = String(track.track_id || "person").split(":").at(-1) + (track.association_quality === "uncertain" ? " ?" : "");
      overlay.append(label);
    }
  }
}

async function api(body, options = {}) {
  const response = await fetch("/api", { method: "POST", headers: { "Content-Type": "application/json", "X-Gamst-Token": sessionToken }, body: JSON.stringify(body), ...options });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}

async function loadVisionModels(generation) {
  // Camera ownership must not depend on optional CDN/model availability. A
  // failed GPU delegate or offline model fetch used to bubble into
  // beginCapture(), which stopped an otherwise valid camera immediately.
  try {
    const { FaceLandmarker, PoseLandmarker, FilesetResolver } = await import(MEDIAPIPE_URL);
    const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM);
    const create = async (Factory, modelAssetPath, extra) => {
      for (const delegate of ["GPU", "CPU"]) {
        try {
          return await Factory.createFromOptions(vision, { baseOptions: { modelAssetPath, delegate }, ...extra });
        } catch { /* try the portable CPU delegate before giving up */ }
      }
      return null;
    };
    const [face, pose] = await Promise.all([
      create(FaceLandmarker, FACE_MODEL, { runningMode: "VIDEO", numFaces: 4, outputFaceBlendshapes: true }),
      create(PoseLandmarker, POSE_MODEL, { runningMode: "VIDEO", numPoses: 4 }),
    ]);
    if (generation !== state.generation) { face?.close(); pose?.close(); return; }
    state.faceModel = face;
    state.poseModel = pose;
  } catch {
    // Perception is best effort in browser mode; retain the live media stream
    // and continue publishing audio/text observations when models are absent.
    state.faceModel = null;
    state.poseModel = null;
  }
}

function signal(name, value, stamp) {
  state.packet.observations[name] = { value, timestamp_ms: stamp, fresh: true, age_ms: 0, evidence_id: `browser:${name}:${stamp}`, interpretation: "observation_not_inner_state" };
}
function browserTick() {
  if (!state.active || isLocal) return;
  const stamp = Date.now();
  state.packet.timestamp_ms = stamp;
  state.packet.live = true;
  if (state.analyser) {
    const samples = new Float32Array(state.analyser.fftSize);
    state.analyser.getFloatTimeDomainData(samples);
    let sum = 0;
    for (const sample of samples) sum += sample * sample;
    const rms = Math.sqrt(sum / samples.length);
    signal("audio.rms", Number(rms.toFixed(5)), stamp);
    // Transparent energy gate, not a claim about words or speaker state.
    signal("speech.activity", rms >= .015, stamp);
  }
  if ($("video").readyState >= 2 && (state.faceModel || state.poseModel)) {
    const time = performance.now();
    if (state.faceModel) {
      const result = state.faceModel.detectForVideo($("video"), time);
      signal("scene.faces", result.faceLandmarks.map((landmarks, index) => ({
        frame_face_index: index,
        landmarks: landmarks.map((point) => ({ x: point.x, y: point.y, z: point.z })),
        blendshapes: Object.fromEntries((result.faceBlendshapes[index]?.categories || []).map((category) => [category.categoryName, Number(category.score.toFixed(4))])),
      })), stamp);
    }
    if (state.poseModel) {
      const result = state.poseModel.detectForVideo($("video"), time);
      signal("scene.poses", result.landmarks.map((landmarks, index) => ({
        frame_pose_index: index,
        landmark_count: landmarks.length,
        landmarks: landmarks.map((point) => ({ x: point.x, y: point.y, z: point.z, visibility: point.visibility, presence: point.presence })),
        geometry: {
          left_wrist_above_shoulder: Boolean(landmarks[15] && landmarks[11] && landmarks[15].y < landmarks[11].y),
          right_wrist_above_shoulder: Boolean(landmarks[16] && landmarks[12] && landmarks[16].y < landmarks[12].y),
        },
      })), stamp);
    }
  }
  publishPacket(state.packet);
}
function localVideoTick() {
  if (!state.active || !isLocal || state.videoBusy || !$("video").videoWidth) {
    if (state.active && isLocal && state.videoBusy) state.droppedFrames++;
    return;
  }
  state.videoBusy = true;
  const canvas = document.createElement("canvas");
  canvas.width = 640;
  canvas.height = 360;
  const context = canvas.getContext("2d");
  context.drawImage($("video"), 0, 0, 640, 360);
  const jpeg = canvas.toDataURL("image/jpeg", .65).split(",")[1];
  api({ op: "video", jpeg }).catch((error) => { state.droppedFrames++; showError("errorRequest", error); }).finally(() => { state.videoBusy = false; });
}
function restartCaptureTimer() {
  clearInterval(state.captureTimer);
  state.captureTimer = null;
  if (!state.active) return;
  const rate = Number(controlValue("packetRate", "4"));
  const interval = 1000 / (Number.isFinite(rate) && rate > 0 ? rate : 4);
  state.captureTimer = setInterval(isLocal ? localVideoTick : browserTick, interval);
  if (!isLocal) browserTick();
}

async function setupAudio(generation) {
  const AudioContextCtor = globalThis.AudioContext || globalThis.webkitAudioContext;
  if (!AudioContextCtor) throw new Error("Audio input is unavailable in this browser");
  state.audioContext = new AudioContextCtor();
  await state.audioContext.resume();
  if (generation !== state.generation) return;
  state.source = state.audioContext.createMediaStreamSource(state.stream);
  if (!isLocal) {
    state.analyser = state.audioContext.createAnalyser();
    state.analyser.fftSize = 2048;
    state.source.connect(state.analyser);
    return;
  }
  state.processor = state.audioContext.createScriptProcessor(4096, 1, 1);
  state.mute = state.audioContext.createGain();
  const gain = state.mute?.gain;
  if (!gain) throw new Error("Audio output control is unavailable");
  gain.value = 0;
  state.source.connect(state.processor);
  state.processor.connect(state.mute);
  state.mute.connect(state.audioContext.destination);
  const rate = state.audioContext.sampleRate;
  state.processor.onaudioprocess = (event) => {
    if (!state.active || generation !== state.generation) return;
    state.audioBuffer.push(...event.inputBuffer.getChannelData(0));
    if (state.audioBuffer.length > rate * 2) {
      state.audioBuffer = state.audioBuffer.slice(-rate);
      state.droppedFrames++;
    }
    if (state.audioBuffer.length >= rate / 2 && !state.audioSending) {
      const block = state.audioBuffer.splice(0, Math.floor(rate / 2));
      const count = Math.floor(block.length * 16000 / rate);
      const samples = new Array(count);
      for (let index = 0; index < count; index++) {
        const start = Math.floor(index * rate / 16000);
        const end = Math.min(block.length, Math.max(start + 1, Math.floor((index + 1) * rate / 16000)));
        let sum = 0;
        for (let i = start; i < end; i++) sum += block[i];
        samples[index] = sum / (end - start);
      }
      state.audioSending = true;
      api({ op: "audio", samples, sample_rate: 16000 }).catch((error) => { state.droppedFrames++; showError("errorRequest", error); }).finally(() => { state.audioSending = false; });
    }
  };
}

function releaseMedia() {
  state.generation++;
  clearInterval(state.captureTimer);
  state.captureTimer = null;
  state.faceModel?.close();
  state.poseModel?.close();
  state.faceModel = null;
  state.poseModel = null;
  if (state.processor) { state.processor.onaudioprocess = null; state.processor.disconnect(); state.processor = null; }
  state.source?.disconnect();
  state.source = null;
  state.mute?.disconnect();
  state.mute = null;
  state.audioContext?.close().catch(() => {});
  state.audioContext = null;
  state.analyser = null;
  state.stream?.getTracks().forEach((track) => track.stop());
  state.stream = null;
  $("video").srcObject = null;
  state.audioBuffer = [];
  state.audioSending = false;
  state.videoBusy = false;
}
async function stopCapture(notify = true) {
  const wasActive = state.active;
  const transitionId = isLocal && wasActive && notify ? ++state.localCaptureTransitionId : null;
  if (transitionId !== null) state.localCaptureTransition = transitionId;
  state.active = false;
  releaseMedia();
  clearGuidePreview();
  state.packet.live = false;
  for (const observation of Object.values(state.packet.observations || {})) observation.fresh = false;
  updateCaptureUI();
  renderPacket();
  $("cameraOverlay").replaceChildren();
  if (isLocal && notify && wasActive) {
    try {
      const result = await api({ op: "stop" });
      if (Number.isFinite(result.capture_revision)) state.localCaptureRevision = Math.max(state.localCaptureRevision, result.capture_revision);
    } catch (error) { showError("errorRequest", error); }
  }
  if (transitionId !== null && captureTransitionCanClear(state.localCaptureTransition, transitionId)) state.localCaptureTransition = null;
}

async function beginCapture() {
  if (state.active) return;
  clearError();
  if (!pressed("inputCamera") && !pressed("inputMicrophone") && !pressed("inputText")) return showError("errorInput");
  const generation = ++state.generation;
  state.active = true;
  const transitionId = isLocal ? ++state.localCaptureTransitionId : null;
  if (transitionId !== null) state.localCaptureTransition = transitionId;
  clearGuidePreview();
  updateCaptureUI();
  setStatus("waitingModel");
  try {
    if (isLocal) {
      const result = await api({ op: "start", consent: true });
      if (Number.isFinite(result.capture_revision)) state.localCaptureRevision = Math.max(state.localCaptureRevision, result.capture_revision);
    }
    if (generation !== state.generation || !state.active || (transitionId !== null && state.localCaptureTransition !== transitionId)) return;
    // Keep the selected inputs in the beginCapture scope.  Previously these
    // declarations lived inside the device block, while the status update
    // below read them after the block; that ReferenceError entered the catch
    // path and released a freshly granted camera immediately.
    const camera = pressed("inputCamera"), microphone = pressed("inputMicrophone");
    if (camera || microphone) {
      // Do not publish the result into shared state until this request still
      // owns the active capture. A late old request must never overwrite a
      // newer stream before its generation guard runs.
      const stream = await navigator.mediaDevices.getUserMedia({ video: camera ? { width: { ideal: 640 }, height: { ideal: 360 } } : false, audio: microphone });
      if (!captureGenerationOwnsMedia(state.generation, generation, state.active)) { stream.getTracks().forEach((track) => track.stop()); return; }
      state.stream = stream;
      // A browser or device can end a track independently of our start/stop
      // controls. Treat that as a real capture stop, while the generation
      // guard prevents cleanup from a superseded transition from racing the
      // current stream.
      for (const track of stream.getTracks()) {
        track.addEventListener("ended", () => {
          if (generation !== state.generation || !state.active) return;
          stopCapture().catch((error) => showError("errorRequest", error));
        }, { once: true });
      }
      if (camera) {
        $("video").srcObject = state.stream;
        // A browser autoplay policy must not revoke an already granted camera.
        // The stream remains owned even when the preview element cannot play.
        await $("video").play().catch(() => {});
      }
      if (microphone) {
        // AudioContext startup is an optional input path. Browsers can reject
        // it independently (autoplay policy, device contention, or an
        // unsupported backend) after granting the camera. Keep the camera
        // stream alive and report the audio-specific failure instead of
        // tearing down the whole capture session.
        try {
          await setupAudio(generation);
        } catch (error) {
          state.processor?.disconnect();
          state.processor = null;
          state.source?.disconnect();
          state.source = null;
          state.analyser = null;
          state.mute?.disconnect();
          state.mute = null;
          state.audioContext?.close().catch(() => {});
          state.audioContext = null;
          if (generation === state.generation && state.active) showError("errorRequest", error);
        }
      }
      if (camera && !isLocal) await loadVisionModels(generation);
    }
    if (generation !== state.generation || !state.active) return;
    state.packet = { ...emptyPacket(), observations: state.packet.observations || {}, live: true, timestamp_ms: Date.now() };
    restartCaptureTimer();
    updateCaptureUI();
    setStatus(camera || microphone ? "observing" : "noDevices");
    if (isLocal) {
      const data = await api({ op: "state" });
      if (transitionId !== null && captureTransitionCanClear(state.localCaptureTransition, transitionId)) state.localCaptureTransition = null;
      handleLocalState(data);
    }
  } catch (error) {
    if (generation === state.generation) {
      await stopCapture(false);
      if (isLocal) {
        try {
          const result = await api({ op: "stop" });
          if (Number.isFinite(result.capture_revision)) state.localCaptureRevision = Math.max(state.localCaptureRevision, result.capture_revision);
        } catch { /* preserve the original capture error */ }
      }
      if (transitionId !== null && captureTransitionCanClear(state.localCaptureTransition, transitionId)) state.localCaptureTransition = null;
      showError("errorCapture", error);
      setStatus("stopped");
    }
  }
}
function requestStart() {
  clearError();
  if (!state.consent) {
    $("consentDialog").showModal();
    return;
  }
  beginCapture();
}

function clearGuidePreview() {
  state.guidePreview = null;
  $("guidePreview").hidden = true;
  $("guidePreview").removeAttribute("src");
  $("previewNote").hidden = true;
  $("video").hidden = false;
}

function addTextObservation(text) {
  const stamp = Date.now();
  if (!isLocal) {
    signal("text.utterance", text.slice(0, 4000), stamp);
    state.packet.timestamp_ms = stamp;
    publishPacket(state.packet);
  }
}
function packetWithQuestion(question) {
  const packet = typeof structuredClone === "function" ? structuredClone(state.packet) : JSON.parse(JSON.stringify(state.packet));
  if (pressed("inputText")) {
    packet.observations ||= {};
    const stamp = Date.now();
    packet.observations["text.utterance"] = { value: question.slice(0, 4000), timestamp_ms: stamp, fresh: true, age_ms: 0, evidence_id: `browser:text.utterance:${stamp}`, interpretation: "observation_not_inner_state" };
    packet.timestamp_ms = stamp;
  }
  return packet;
}

function providerConfig() {
  const baseUrl = controlValue("baseUrl").trim();
  const model = controlValue("model").trim();
  return { baseUrl, model, key: controlValue("apiKey"), provider: controlValue("provider", "byok"), reasoning_effort: state.localMyProxyActive && state.localMyProxyConfig?.baseUrl === baseUrl && state.localMyProxyConfig?.model === model ? state.localMyProxyConfig.reasoning_effort : undefined };
}
function updateProviderNotice() {
  const config = providerConfig();
  const local = isLocal && config.provider === "local";
  $("byokFields").hidden = local;
  $("providerNotice").textContent = local ? t(state.locale, "providerNoticeLocal") : t(state.locale, "providerNotice", { endpoint: config.baseUrl || "OpenAI-compatible endpoint" });
  $("responseProvider").textContent = local ? t(state.locale, "responseLocal") : t(state.locale, "responseByok");
}
function renderResponseSetup(show = true) {
  const config = providerConfig();
  const localReady = isLocal && config.provider === "local" && state.localData?.companion_available;
  const byokReady = config.provider === "byok" && config.baseUrl && config.model && config.key;
  const needsSetup = !localReady && !byokReady;
  if (show && needsSetup) {
    selectTab("response");
    $("setupCard").hidden = false;
    $("responseIdle").hidden = true;
    $("responseText").textContent = "";
    $("responseState").textContent = "";
  } else if (!state.sending && !$("responseText").textContent) {
    $("setupCard").hidden = !needsSetup;
    $("responseIdle").hidden = needsSetup;
  } else if (!needsSetup) $("setupCard").hidden = true;
  return !needsSetup;
}
function renderEvidence(packet, localIds = null) {
  const ids = localIds?.length ? localIds : evidenceIds(packet);
  $("evidenceBlock").hidden = false;
  $("evidenceIds").replaceChildren();
  if (!ids.length) {
    const span = document.createElement("span");
    span.className = "muted";
    span.textContent = t(state.locale, "noEvidence");
    $("evidenceIds").append(span);
    return;
  }
  for (const id of ids) {
    const code = document.createElement("code");
    code.textContent = id;
    code.title = id;
    $("evidenceIds").append(code);
  }
}

async function streamBYOK(question, packet) {
  const config = providerConfig();
  const endpoint = chatEndpoint(config.baseUrl);
  state.chatController = new AbortController();
  const requestBody = { model: config.model, messages: buildChatMessages(question, packet), stream: true, temperature: .3 };
  if (config.reasoning_effort) requestBody.reasoning_effort = config.reasoning_effort;
  const response = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${config.key}` },
    body: JSON.stringify(requestBody),
    signal: state.chatController.signal,
  });
  if (!response.ok) {
    const detail = (await response.text()).slice(0, 2000);
    throw new Error(`${response.status} ${response.statusText}${detail ? ` · ${detail}` : ""}`);
  }
  if (!response.body) throw new Error(t(state.locale, "errorStream"));
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let answer = "", done = false;
  const parser = createSSEParser((payload) => {
    const delta = extractDelta(payload);
    if (delta.done) { done = true; return; }
    if (delta.text) {
      answer += delta.text;
      $("responseText").textContent = answer;
    }
  });
  while (!done) {
    const { value, done: finished } = await reader.read();
    if (finished) break;
    parser.push(decoder.decode(value, { stream: true }));
  }
  parser.push(decoder.decode());
  parser.finish();
  if (!answer && !done) throw new Error(t(state.locale, "errorStream"));
  return answer;
}

async function sendQuestion() {
  const question = controlValue("question").trim();
  clearError();
  if (!question) return showError("emptyQuestion");
  selectTab("response");
  if (!renderResponseSetup(true)) return;
  const config = providerConfig();
  if (config.provider === "local") {
    if (!state.localData?.companion_available) { showError("errorNoCompanion"); return; }
    if (state.localData?.text_companion_enabled === false) { showError("companionDisabled"); return; }
    if (!state.active) { showError("errorConsent"); return; }
  }
  if (state.sending) state.chatController?.abort();
  state.sending = true;
  state.chatController = null;
  $("send").disabled = true;
  $("send").querySelector("[data-i18n]").textContent = t(state.locale, "sending");
  $("setupCard").hidden = true;
  $("responseIdle").hidden = true;
  $("responseText").textContent = "";
  $("responseText").classList.add("streaming");
  $("responseState").textContent = t(state.locale, "sending");
  state.sentPacket = compactPacket(packetWithQuestion(question), Number(controlValue("byteBudget", "16384")));
  renderPacket();
  renderEvidence(state.sentPacket);
  try {
    let answer;
    if (config.provider === "local") {
      const result = await api({ op: "chat", text: question });
      answer = result.assistant_text || "";
      renderEvidence(state.sentPacket, result.evidence_ids);
    } else {
      addTextObservation(question);
      answer = await streamBYOK(question, state.sentPacket);
    }
    $("responseText").textContent = answer;
    setControlValue("question", "");
    setStatus("response");
  } catch (error) {
    if (error.name !== "AbortError") {
      $("responseText").textContent = "";
      showError("errorProvider", error);
      $("responseState").textContent = t(state.locale, "errorProvider", { message: error.message });
    }
  } finally {
    state.sending = false;
    state.chatController = null;
    $("send").disabled = false;
    $("send").querySelector("[data-i18n]").textContent = t(state.locale, "send");
    $("responseText").classList.remove("streaming");
    $("responseState").textContent = "";
  }
}

function renderLocalSettings(data) {
  if (!isLocal || !data) return;
  const features = (data.features || []).filter((feature) => feature.name !== "text_companion");
  features.push({ name: "text_companion", enabled: data.text_companion_enabled !== false });
  const signature = JSON.stringify(features.map((feature) => [feature.name, feature.enabled]));
  if (signature !== state.featureSignature) {
    state.featureSignature = signature;
    $("featureSwitches").replaceChildren();
    for (const feature of features) {
      const row = document.createElement("div");
      row.className = "feature-row";
      const name = document.createElement("span");
      name.textContent = feature.name.replaceAll("_", " ");
      name.title = feature.name;
      const button = document.createElement("button");
      button.type = "button";
      button.className = "toggle-mini";
      button.dataset.feature = feature.name;
      button.setAttribute("aria-pressed", String(Boolean(feature.enabled)));
      button.textContent = feature.enabled ? t(state.locale, "on") : t(state.locale, "off");
      button.disabled = state.active;
      button.addEventListener("click", async () => {
        clearError();
        try {
          const enabled = button.getAttribute("aria-pressed") !== "true";
          await api({ op: "configure_feature", name: feature.name, enabled });
          button.setAttribute("aria-pressed", String(enabled));
          button.textContent = t(state.locale, enabled ? "on" : "off");
          state.featureSignature = "";
          handleLocalState(await api({ op: "state" }));
        } catch (error) { showError("errorRequest", error); }
      });
      row.append(name, button);
      $("featureSwitches").append(row);
    }
  }
  $("asrLanguage").disabled = !data.asr?.available || state.active;
  if (document.activeElement !== $("asrLanguage")) setControlValue("asrLanguage", data.asr?.language || "auto");
  $("sam3Status").textContent = data.sam3?.message || t(state.locale, "unavailable");
  $("testGuides").disabled = state.active || !data.guide_test_available;
  const vlm = data.vlm || {};
  setPressed("vlmEnabled", Boolean(vlm.enabled));
  $("vlmEnabled").disabled = !vlm.available;
  $("applyVlm").disabled = !vlm.available;
  $("testVlm").disabled = !vlm.available;
  for (const [id, key] of [["vlmInterval", "interval"], ["vlmWindow", "window_seconds"], ["vlmFps", "sample_fps"]]) if (document.activeElement !== $(id) && vlm[key] !== undefined) setControlValue(id, vlm[key]);
  const option = $("localCompanionOption");
  option.disabled = !data.companion_available;
  option.textContent = `${t(state.locale, "localCompanion")}${data.companion_available ? "" : ` · ${t(state.locale, "unavailable")}`}`;
}

function handleLocalState(data) {
  // A state poll can complete after a start/stop request that superseded it.
  // Ignore all snapshots during the short transition and older revisions
  // thereafter, otherwise a queued paused snapshot stops fresh camera tracks.
  const decision = captureSnapshotDecision(state.localCaptureRevision, Boolean(state.localCaptureTransition), data);
  if (!decision.accept) return;
  state.localCaptureRevision = decision.revision;
  state.localData = data;
  const running = !data.state?.paused;
  if (state.active && !running) {
    state.active = false;
    releaseMedia();
    updateCaptureUI();
  } else if (!state.active && running) {
    state.active = true;
    updateCaptureUI();
  }
  state.packet = data.context || emptyPacket();
  state.packet.live = running;
  publishPacket(state.packet, data);
  renderLocalSettings(data);
  renderResponseSetup(false);
}
async function connectLocal() {
  clearTimeout(state.localReconnect);
  state.localController = new AbortController();
  $("connectionDot").classList.add("connecting");
  try {
    const response = await fetch("/api", { method: "POST", headers: { "Content-Type": "application/json", "X-Gamst-Token": sessionToken }, body: JSON.stringify({ op: "stream" }), signal: state.localController.signal });
    if (!response.ok || !response.body) throw new Error(`Local stream ${response.status}`);
    $("connectionDot").classList.remove("connecting");
    const reader = response.body.getReader(), decoder = new TextDecoder();
    let rest = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      rest += decoder.decode(value, { stream: true });
      let index;
      while ((index = rest.indexOf("\n")) >= 0) {
        const line = rest.slice(0, index).trim();
        rest = rest.slice(index + 1);
        if (line) handleLocalState(JSON.parse(line));
      }
    }
  } catch (error) {
    if (error.name === "AbortError") return;
    $("connectionDot").classList.add("connecting");
  }
  state.localReconnect = setTimeout(connectLocal, 1500);
}

async function copyText(text) {
  await navigator.clipboard.writeText(text);
  setStatus("copied");
  clearTimeout(settingsTimer);
  settingsTimer = setTimeout(() => setStatus(""), 1400);
}
async function clearData() {
  clearError();
  if (isLocal && !window.confirm(t(state.locale, "deleteConfirm"))) return;
  await stopCapture();
  state.chatController?.abort();
  if (isLocal) {
    try { await api({ op: "delete" }); } catch (error) { showError("errorRequest", error); return; }
  }
  state.packet = emptyPacket();
  state.sentPacket = null;
  state.arrivals = [];
  state.metricHistory = [];
  state.audioTimeline = [];
  state.sceneTimeline = [];
  state.lastTimelineStamp = 0;
  state.droppedFrames = 0;
  for (const history of Object.values(state.histories)) history.length = 0;
  $("responseText").textContent = "";
  $("evidenceBlock").hidden = true;
  renderPacket();
  renderResponseSetup(false);
  setStatus(isLocal ? "contextDeleted" : "browserDeleted");
}

const presets = {
  clarify: { camera: false, microphone: true, text: true, face: false, body: false, person: false },
  hand: { camera: true, microphone: false, text: true, face: false, body: true, person: true },
  that: { camera: true, microphone: true, text: true, face: true, body: true, person: true },
  stale: { camera: false, microphone: false, text: true, face: false, body: false, person: false },
};
for (const button of document.querySelectorAll("[data-preset]")) {
  button.addEventListener("click", () => {
    if (state.active) stopCapture();
    const name = button.dataset.preset, preset = presets[name];
    document.querySelectorAll("[data-preset]").forEach((node) => node.classList.toggle("active", node === button));
    setPressed("inputCamera", preset.camera);
    setPressed("inputMicrophone", preset.microphone);
    setPressed("inputText", preset.text);
    setPressed("overlayFace", preset.face);
    setPressed("overlayBody", preset.body);
    setPressed("overlayPerson", preset.person);
    const key = { clarify: "presetClarifyQuestion", hand: "presetHandQuestion", that: "presetThatQuestion", stale: "presetStaleQuestion" }[name];
    setControlValue("question", t(state.locale, key));
    $("question")?.focus();
    clearError();
  });
}

$("start").addEventListener("click", requestStart);
$("stop").addEventListener("click", () => stopCapture());
$("clear").addEventListener("click", clearData);
$("consentDialog").addEventListener("close", () => {
  if ($("consentDialog").returnValue === "allow") {
    state.consent = true;
    beginCapture();
  }
});
$("send").addEventListener("click", sendQuestion);
$("question").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing && event.keyCode !== 229) {
    event.preventDefault();
    sendQuestion();
  }
});
$("copyJson").addEventListener("click", () => copyText(JSON.stringify(state.packet, null, 2)).catch(() => showError("errorClipboard")));
$("share").addEventListener("click", async () => {
  const text = JSON.stringify(state.packet, null, 2);
  try {
    if (!navigator.share) throw new Error("share unavailable");
    await navigator.share({ title: t(state.locale, "appTitle"), text });
  } catch (error) {
    if (error.name === "AbortError") return;
    try { await copyText(text); setStatus("errorShare"); } catch { showError("errorClipboard"); }
  }
});
$("language").addEventListener("change", () => {
  const language = controlValue("language");
  state.locale = LOCALES.includes(language) ? language : "en";
  try { localStorage.setItem("gamst.uiLanguage", state.locale); } catch { /* no-op */ }
  applyI18n();
});
$("configureModel").addEventListener("click", () => { $("modelSettings").open = true; $("baseUrl").focus(); });
$("provider").addEventListener("change", () => { updateProviderNotice(); renderResponseSetup(false); });
for (const id of ["baseUrl", "model", "apiKey"]) $(id).addEventListener("input", () => { state.localMyProxyActive = false; updateProviderNotice(); renderResponseSetup(false); });
$("useLocalMyProxy")?.addEventListener("click", applyLocalMyProxyConfig);
$("saveSettings").addEventListener("click", () => {
  const config = providerConfig();
  writeStorage("local", "baseUrl", config.baseUrl);
  writeStorage("local", "model", config.model);
  writeStorage("local", "provider", config.provider);
  writeStorage("local", "rate", controlValue("packetRate", "4"));
  writeStorage("local", "budget", controlValue("byteBudget", "16384"));
  if ($("rememberKey").checked) {
    writeStorage("local", "apiKey", config.key);
    removeStorage("session", "apiKey");
  } else {
    writeStorage("session", "apiKey", config.key);
    removeStorage("local", "apiKey");
  }
  writeStorage("local", "rememberKey", String($("rememberKey").checked));
  $("settingsStatus").textContent = t(state.locale, "settingsSaved");
  clearTimeout(settingsTimer);
  settingsTimer = setTimeout(() => { $("settingsStatus").textContent = ""; }, 1500);
  restartCaptureTimer();
  renderResponseSetup(false);
});
$("packetRate").addEventListener("change", restartCaptureTimer);

$("openLocalSettings").addEventListener("click", () => { $("localSettingsDialog").showModal(); });
$("closeLocalSettings").addEventListener("click", () => $("localSettingsDialog").close());
$("localSettingsDialog").addEventListener("close", () => $("openLocalSettings").focus());
setPressed("vlmEnabled", false);
bindToggle("vlmEnabled");
$("asrLanguage").addEventListener("change", async () => {
  try { await api({ op: "configure_asr", language: controlValue("asrLanguage", "auto") }); clearError(); } catch (error) { showError("errorRequest", error); }
});
$("applyVlm").addEventListener("click", async () => {
  try {
    await api({ op: "configure_vlm", enabled: pressed("vlmEnabled"), interval: Number(controlValue("vlmInterval", "3")), window_seconds: Number(controlValue("vlmWindow", "4")), sample_fps: Number(controlValue("vlmFps", "2")) });
    clearError();
    handleLocalState(await api({ op: "state" }));
  } catch (error) { showError("errorRequest", error); }
});
$("testVlm").addEventListener("click", async () => {
  $("testVlm").disabled = true;
  $("vlmTestResult").textContent = t(state.locale, "testing");
  try {
    const result = await api({ op: "test_vlm" });
    $("vlmTestResult").textContent = `${result.latency_ms}ms\n${result.description}`;
  } catch (error) { $("vlmTestResult").textContent = error.message; }
  finally { $("testVlm").disabled = !state.localData?.vlm?.available; }
});
$("testGuides").addEventListener("click", async () => {
  $("testGuides").disabled = true;
  $("localSettingsDialog").close();
  try {
    const data = await api({ op: "test_guides" });
    state.guidePreview = data;
    $("guidePreview").onload = () => drawGuides(data);
    $("guidePreview").src = data.image;
    $("guidePreview").hidden = false;
    $("previewNote").hidden = false;
    $("previewNote").textContent = `${t(state.locale, "guidePreview")} · ${data.latency_ms}ms`;
    $("video").hidden = true;
    $("stageEmpty").hidden = true;
    drawGuides(data);
  } catch (error) { showError("errorRequest", error); }
  finally { $("testGuides").disabled = !state.localData?.guide_test_available; }
});
$("loadCapabilities").addEventListener("click", async () => {
  $("capabilitiesOutput").textContent = t(state.locale, "testing");
  try { $("capabilitiesOutput").textContent = JSON.stringify(await api({ op: "capabilities" }), null, 2); }
  catch (error) { $("capabilitiesOutput").textContent = error.message; }
});

function restoreSettings() {
  setControlValue("baseUrl", readStorage("local", "baseUrl"));
  setControlValue("model", readStorage("local", "model"));
  const remembered = readStorage("local", "rememberKey") === "true";
  if ($("rememberKey")) $("rememberKey").checked = remembered;
  setControlValue("apiKey", remembered ? readStorage("local", "apiKey") : readStorage("session", "apiKey"));
  setControlValue("packetRate", readStorage("local", "rate") || "4");
  setControlValue("byteBudget", readStorage("local", "budget") || "16384");
  const savedProvider = readStorage("local", "provider");
  setControlValue("provider", isLocal && savedProvider !== "byok" ? "local" : "byok");
}

window.addEventListener("pagehide", () => {
  releaseMedia();
  state.localController?.abort();
  state.chatController?.abort();
  if (isLocal && state.active) fetch("/api", { method: "POST", headers: { "Content-Type": "application/json", "X-Gamst-Token": sessionToken }, body: JSON.stringify({ op: "stop" }), keepalive: true }).catch(() => {});
});
setInterval(() => renderMetrics(), 500);

restoreSettings();
loadLocalMyProxyConfig().then((config) => {
  state.localMyProxyConfig = config;
  updateLocalMyProxyButton();
}).catch(() => updateLocalMyProxyButton());
for (const id of ["openLocalSettings"]) $(id).hidden = !isLocal;
$("localCompanionOption").hidden = !isLocal;
applyI18n();
updateCaptureUI();
renderResponseSetup(false);
if (isLocal) connectLocal();
