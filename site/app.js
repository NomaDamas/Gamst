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
const demoMessages = {
  preview: ["BROWSER PREVIEW", "브라우저 미리보기", "浏览器预览"],
  heading: ["Your next input. A live context packet.", "지금의 입력을 실시간 문맥으로.", "把当前输入转换为实时上下文。"],
  scope: ["Browser preview: microphone levels and up to four faces. Speech transcription, voice emotion, VLM and MCP run in the full local app.", "브라우저에서는 마이크 음량과 최대 4명의 얼굴을 관찰합니다. 음성 인식·목소리 감정·VLM·MCP는 전체 로컬 앱에서 실행됩니다.", "浏览器预览支持麦克风音量和最多四张面部。语音转写、语音情绪、VLM和MCP在完整本地应用中运行。"],
  language: ["Demo language", "데모 화면 언어", "演示界面语言"],
  local: ["Open full local demo ↗", "전체 로컬 데모 열기 ↗", "打开完整本地演示 ↗"],
  localHint: ["Local installation must be running.", "로컬 설치 후 서버를 실행해야 합니다.", "需先安装并启动本地服务器。"],
  devicesOff: ["Your devices are off.", "장치가 꺼져 있습니다.", "设备已关闭。"],
  chooseStart: ["You choose when to start.", "원할 때 시작하세요.", "由您选择何时开始。"],
  microphone: ["Microphone", "마이크", "麦克风"],
  camera: ["Camera + facial movement", "카메라 · 얼굴 움직임", "摄像头 · 面部动作"],
  assets: ["Camera mode downloads MediaPipe assets from jsDelivr and Google. Frames and audio are processed here; they are not uploaded.", "카메라 모드는 jsDelivr와 Google에서 MediaPipe 파일을 받습니다. 영상과 음성은 이 브라우저에서 처리하며 업로드하지 않습니다.", "摄像头模式从jsDelivr和Google下载MediaPipe资源。视频与音频在浏览器中处理，不会上传。"],
  consent: ["I agree to process my selected inputs in this browser.", "선택한 입력을 이 브라우저에서 처리하는 데 동의합니다.", "我同意在此浏览器中处理所选输入。"],
  start: ["Start live preview", "실시간 미리보기 시작", "开始实时预览"],
  stop: ["Stop", "정지", "停止"],
  textLabel: ["Text to add to context", "문맥에 추가할 텍스트", "添加到上下文的文本"],
  placeholder: ["Or type: Could you explain that again?", "또는 입력: 다시 설명해 주실래요?", "或输入：可以再解释一次吗？"],
  add: ["Add text", "텍스트 추가", "添加文本"],
  clear: ["Clear session", "기록 지우기", "清除记录"],
  copy: ["Copy JSON", "JSON 복사", "复制JSON"],
  micLevel: ["MIC LEVEL", "마이크 음량", "麦克风音量"],
  faces: ["FACES", "얼굴 수", "面部数量"],
  updates: ["UPDATES", "업데이트", "更新次数"],
  contextJson: ["Context JSON", "문맥 JSON", "上下文JSON"],
  copyHint: ["Copy this context into your agent.", "이 문맥을 에이전트에게 전달하세요.", "将此上下文交给您的智能体。"],
  sdkHint: ["The local SDK adds transcription, scene descriptions, evidence IDs and streaming.", "로컬 SDK는 음성 인식·장면 설명·근거 ID·스트리밍을 제공합니다.", "本地SDK提供语音转写、场景描述、证据ID与流式传输。"],
  measurementHint: ["Live browser measurements. Facial movements are measured separately from voice emotion in the local app.", "실시간 브라우저 관측입니다. 얼굴 움직임과 목소리 감정은 로컬 앱에서도 별도로 측정합니다.", "实时浏览器观测。面部动作与本地应用中的语音情绪分别测量。"],
  ready: ["Ready when you are", "시작 대기", "等待开始"],
  stopped: ["Stopped · last observations retained", "중지됨 · 마지막 관측 유지", "已停止 · 保留最近观测"],
  loading: ["Loading local face model…", "얼굴 모델 불러오는 중…", "正在加载面部模型…"],
  live: ["Live · processed in this browser", "관찰 중 · 브라우저에서 처리", "实时观测 · 浏览器内处理"],
  textAdded: ["Live · text added", "관찰 중 · 텍스트 추가됨", "实时观测 · 已添加文本"],
  textReady: ["Text context ready", "텍스트 문맥 준비됨", "文本上下文已就绪"],
  cleared: ["Session cleared", "기록 삭제됨", "记录已清除"],
  copied: ["Copied", "복사됨", "已复制"],
  agree: ["Please agree to local processing first.", "먼저 로컬 처리에 동의해 주세요.", "请先同意本地处理。"],
  selectInput: ["Select a microphone or camera, or add text below.", "마이크나 카메라를 선택하거나 아래에 텍스트를 입력하세요.", "请选择麦克风或摄像头，或在下方添加文本。"],
  typeFirst: ["Type a message first.", "먼저 메시지를 입력하세요.", "请先输入消息。"],
  clipboard: ["Clipboard unavailable. Select and copy the displayed text.", "클립보드를 사용할 수 없습니다. 표시된 텍스트를 선택해 복사하세요.", "剪贴板不可用。请选择并复制显示的文本。"]
};
let demoLanguage = "en", statusKey = "ready", errorText = "";
try { const saved = localStorage.getItem("cym.publicDemoLanguage"); if (["en", "ko", "zh"].includes(saved)) demoLanguage = saved; } catch {}
const demoText = (key) => demoMessages[key][["en", "ko", "zh"].indexOf(demoLanguage)];
function translateError(text) {
  const key = Object.keys(demoMessages).find((k) => demoMessages[k][0] === text);
  return key ? demoText(key) : text;
}
function status(key) { statusKey = key; $("status").textContent = demoText(key); }
function error(e) { errorText = e.message || String(e); $("error").textContent = translateError(errorText); }
const charts = new ContextCharts($("categoryCharts"), demoLanguage);
function applyDemoLanguage() {
  $("demoLanguage").value = demoLanguage;
  $("try").lang = demoLanguage;
  document.querySelectorAll("#try [data-demo-i18n]").forEach((node) => { node.textContent = demoText(node.dataset.demoI18n); });
  document.querySelectorAll("#try [data-demo-placeholder]").forEach((node) => { node.placeholder = demoText(node.dataset.demoPlaceholder); });
  status(statusKey);
  if ($("error").textContent) $("error").textContent = translateError(errorText);
  charts.locale = demoLanguage;
  charts.render(packet.observations, active);
}
$("demoLanguage").onchange = () => {
  demoLanguage = $("demoLanguage").value;
  if (!["en", "ko", "zh"].includes(demoLanguage)) demoLanguage = "en";
  try { localStorage.setItem("cym.publicDemoLanguage", demoLanguage); } catch {}
  applyDemoLanguage();
};
applyDemoLanguage();
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
  status("stopped");
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
      status("loading");
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
        numFaces: 4,
        outputFaceBlendshapes: true,
      });
      if (epoch !== generation) {
        model.close();
        return;
      }
      faceModel = model;
    }
    if (epoch !== generation) return;
    status("live");
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
  status(active ? "textAdded" : "textReady");
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
  status("cleared");
  paint();
};
async function copy(id, text) {
  try {
    await navigator.clipboard.writeText(text);
    const old = $(id).textContent;
    $(id).textContent = demoText("copied");
    setTimeout(() => { $(id).textContent = id === "copy" ? demoText("copy") : old; }, 1400);
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
