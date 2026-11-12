const $ = (id) => document.getElementById(id);
let stream = null,
  audio = null,
  analyser = null,
  faceModel = null,
  timer = null,
  active = false,
  generation = 0,
  count = 0;
let packet = {
  source: "browser_preview",
  observations: {},
  raw_media_included: false,
};
function error(e) {
  $("error").textContent = e.message || String(e);
}
const charts = new ContextCharts($("categoryCharts"), "en");
function paint() {
  charts.render(packet.observations, active);
  $("context").textContent = JSON.stringify(packet, null, 2);
  $("updates").textContent = String(count);
}
function signal(name, value) {
  packet.observations[name] = {
    value,
    timestamp_ms: Date.now(),
    interpretation: "observation_not_inner_state",
  };
  packet.timestamp_ms = Date.now();
}
function stop() {
  active = false;
  generation++;
  clearInterval(timer);
  timer = null;
  stream?.getTracks().forEach((t) => t.stop());
  stream = null;
  audio?.close();
  audio = null;
  analyser = null;
  faceModel?.close();
  faceModel = null;
  $("camera").srcObject = null;
  $("cameraHint").hidden = false;
  $("start").disabled = false;
  $("stop").disabled = true;
  $("status").textContent = "Stopped · last observations retained";
  $("liveDot").classList.remove("live");
  $("volume").style.width = "0%";
  packet.live = false;
  paint();
}
$("start").onclick = async () => {
  if (!$("consent").checked)
    return error("Please agree to local processing first.");
  if (!$("mic").checked && !$("face").checked)
    return error("Select a microphone or camera, or add text below.");
  $("error").textContent = "";
  $("start").disabled = true;
  $("stop").disabled = false;
  const epoch = ++generation;
  active = true;
  try {
    const media = await navigator.mediaDevices.getUserMedia({
      audio: $("mic").checked,
      video: $("face").checked ? { width: 640, height: 360 } : false,
    });
    if (epoch !== generation) {
      media.getTracks().forEach((t) => t.stop());
      return;
    }
    stream = media;
    if ($("mic").checked) {
      audio = new AudioContext();
      await audio.resume();
      if (epoch !== generation) return;
      analyser = audio.createAnalyser();
      analyser.fftSize = 2048;
      audio.createMediaStreamSource(stream).connect(analyser);
    }
    if ($("face").checked) {
      $("status").textContent = "Loading local face model…";
      $("camera").srcObject = stream;
      await $("camera").play();
      $("cameraHint").hidden = true;
      const { FaceLandmarker, FilesetResolver } = await import(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.3/vision_bundle.mjs"
      );
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.3/wasm",
      );
      const model = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
          delegate: "GPU",
        },
        runningMode: "VIDEO",
        numFaces: 1,
        outputFaceBlendshapes: true,
      });
      if (epoch !== generation) {
        model.close();
        return;
      }
      faceModel = model;
    }
    if (epoch !== generation) return;
    $("status").textContent = "Live · processed in this browser";
    $("liveDot").classList.add("live");
    packet.live = true;
    timer = setInterval(() => {
      if (!active) return;
      try {
        if (analyser) {
          const data = new Float32Array(analyser.fftSize);
          analyser.getFloatTimeDomainData(data);
          const rms = Math.sqrt(
            data.reduce((s, x) => s + x * x, 0) / data.length,
          );
          signal("audio.rms", Number(rms.toFixed(5)));
          $("rms").textContent = rms.toFixed(3);
          $("volume").style.width = Math.min(100, rms * 500) + "%";
        }
        if (faceModel && $("camera").readyState >= 2) {
          const r = faceModel.detectForVideo($("camera"), performance.now());
          const faces = r.faceBlendshapes.map((f, i) => ({
            frame_face_index: i,
            blendshapes: Object.fromEntries(
              [...f.categories]
                .sort((a, b) => b.score - a.score)
                .map((x) => [x.categoryName, Number(x.score.toFixed(3))]),
            ),
          }));
          signal("scene.faces", faces);
          $("faces").textContent = String(faces.length);
        }
        count++;
        paint();
      } catch (e) {
        stop();
        error(e);
      }
    }, 250);
  } catch (e) {
    if (epoch === generation) {
      stop();
      error(e);
    }
  }
};
$("stop").onclick = stop;
$("add").onclick = () => {
  if (!$("consent").checked)
    return error("Please agree to local processing first.");
  const text = $("utterance").value.trim();
  if (!text) return error("Type a message first.");
  $("error").textContent = "";
  signal("text.utterance", text.slice(0, 2000));
  count++;
  paint();
  $("status").textContent = active ? "Live · text added" : "Text context ready";
};
$("clear").onclick = () => {
  stop();
  packet = {
    source: "browser_preview",
    observations: {},
    raw_media_included: false,
  };
  count = 0;
  $("utterance").value = "";
  $("rms").textContent = "—";
  $("faces").textContent = "—";
  $("error").textContent = "";
  $("status").textContent = "Session cleared";
  paint();
};
async function copy(id, text) {
  try {
    await navigator.clipboard.writeText(text);
    const old = $(id).textContent;
    $(id).textContent = "Copied";
    setTimeout(() => ($(id).textContent = old), 1400);
  } catch (e) {
    error("Clipboard unavailable. Select and copy the displayed text.");
  }
}
$("copy").onclick = () => copy("copy", JSON.stringify(packet, null, 2));
$("copyInstall").onclick = () =>
  copy("copyInstall", $("installCode").textContent);
window.addEventListener("pagehide", stop);
document.addEventListener("visibilitychange", () => {
  if (document.hidden && active) stop();
});
