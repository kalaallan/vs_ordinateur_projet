const state = {
  models: [],
  selectedModel: null,
  uploadId: null,
  currentJobId: null,
  pollTimer: null,
  commentaryLog: [],
  lastLiveComment: "",
  ttsBusy: false,
  ttsObjectUrl: null,
};

const modelListEl = document.getElementById("model-list");
const videoInputEl = document.getElementById("video-input");
const uploadZoneEl = document.getElementById("upload-zone");
const analyzeButtonEl = document.getElementById("analyze-button");
const statusValueEl = document.getElementById("status-value");
const summaryValueEl = document.getElementById("summary-value");
const previewVideoEl = document.getElementById("preview-video");
const liveStreamEl = document.getElementById("live-stream");
const placeholderEl = document.getElementById("placeholder");
const speakButtonEl = document.getElementById("speak-button");
const ttsAudioEl = document.getElementById("tts-audio");

function currentCommentaryText() {
  if (state.commentaryLog.length > 0) {
    return state.commentaryLog[state.commentaryLog.length - 1];
  }
  return (summaryValueEl.textContent || "").trim();
}

function updateSpeakButtonState() {
  const hasText = currentCommentaryText().length >= 3;
  speakButtonEl.disabled = !hasText || state.ttsBusy;
  speakButtonEl.classList.toggle("playing", state.ttsBusy);
}

function toHypeSpeechText(text) {
  const clean = (text || "").trim();
  if (!clean) {
    return "";
  }
  if (/[.!?]$/.test(clean)) {
    return `Oh la la, ${clean}`;
  }
  return `Oh la la, ${clean}.`;
}

function speakWithBrowserTTS(text) {
  if (!("speechSynthesis" in window) || typeof SpeechSynthesisUtterance === "undefined") {
    return false;
  }
  const phrase = toHypeSpeechText(text);
  if (!phrase) {
    return false;
  }
  const utter = new SpeechSynthesisUtterance(phrase);
  utter.lang = "fr-FR";
  utter.rate = 1.08;
  utter.pitch = 1.12;
  utter.volume = 1.0;

  const voices = window.speechSynthesis.getVoices() || [];
  const french = voices.find((v) => (v.lang || "").toLowerCase().startsWith("fr"));
  if (french) {
    utter.voice = french;
  }
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utter);
  return true;
}

function setStatus(text, level = "default") {
  statusValueEl.textContent = text;
  statusValueEl.className = "value";
  if (level === "running") {
    statusValueEl.classList.add("status-running");
  }
  if (level === "ok") {
    statusValueEl.classList.add("status-ok");
  }
  if (level === "error") {
    statusValueEl.classList.add("status-error");
  }
}

function setVideoSource(src) {
  liveStreamEl.src = "";
  liveStreamEl.style.display = "none";
  previewVideoEl.src = `${src}${src.includes("?") ? "&" : "?"}t=${Date.now()}`;
  previewVideoEl.style.display = "block";
  placeholderEl.style.display = "none";
}

function setLiveStream(jobId) {
  previewVideoEl.pause();
  previewVideoEl.removeAttribute("src");
  previewVideoEl.load();
  previewVideoEl.style.display = "none";
  liveStreamEl.src = `/api/stream/${jobId}?t=${Date.now()}`;
  liveStreamEl.style.display = "block";
  placeholderEl.style.display = "none";
}

function clearStats() {
  return;
}

function resetCommentary(text = "") {
  state.commentaryLog = [];
  state.lastLiveComment = "";
  summaryValueEl.textContent = text;
  updateSpeakButtonState();
}

function appendCommentary(line) {
  const clean = (line || "").trim();
  if (!clean || clean === state.lastLiveComment) {
    return;
  }
  state.lastLiveComment = clean;
  state.commentaryLog.push(clean);
  summaryValueEl.textContent = state.commentaryLog.join("\n");
  summaryValueEl.scrollTop = summaryValueEl.scrollHeight;
  updateSpeakButtonState();
}

async function playCommentary() {
  const text = currentCommentaryText();
  if (!text || state.ttsBusy) {
    return;
  }

  state.ttsBusy = true;
  updateSpeakButtonState();

  try {
    const response = await fetch("/api/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        hype: true,
        response_format: "mp3",
        speed: 1.04,
      }),
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.detail || "Impossible de générer l'audio.");
    }

    const blob = await response.blob();
    if (!blob || blob.size < 256) {
      throw new Error("Audio TTS invalide (source vide).");
    }
    if (state.ttsObjectUrl) {
      URL.revokeObjectURL(state.ttsObjectUrl);
      state.ttsObjectUrl = null;
    }
    state.ttsObjectUrl = URL.createObjectURL(blob);
    ttsAudioEl.src = state.ttsObjectUrl;
    await ttsAudioEl.play();
  } catch (error) {
    if (speakWithBrowserTTS(text)) {
      setStatus("Lecture vocale via navigateur", "ok");
    } else {
      setStatus(error.message || "Erreur TTS", "error");
    }
  } finally {
    state.ttsBusy = false;
    updateSpeakButtonState();
  }
}

function updateAnalyzeState() {
  analyzeButtonEl.disabled = !(state.uploadId && state.selectedModel);
}

function renderModels() {
  modelListEl.innerHTML = "";
  state.models.forEach((model) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `model-option ${state.selectedModel === model.id ? "selected" : ""}`;
    button.disabled = !model.available;
    button.innerHTML = `
      <div class="name-row">
        <span>${model.name}</span>
        <span class="check-dot"></span>
      </div>
    `;
    button.title = model.available ? model.description : "Poids non disponibles";
    button.addEventListener("click", () => {
      state.selectedModel = model.id;
      renderModels();
      updateAnalyzeState();
    });
    modelListEl.appendChild(button);
  });
}

async function fetchModels() {
  const response = await fetch("/api/models");
  if (!response.ok) {
    throw new Error("Impossible de charger les modèles.");
  }
  const payload = await response.json();
  state.models = payload.items || [];
  const preferred = state.models.find((item) => item.id === payload.default && item.available);
  const fallback = state.models.find((item) => item.available);
  state.selectedModel = preferred ? preferred.id : fallback ? fallback.id : null;
  renderModels();
  updateAnalyzeState();
}

async function uploadVideo(file) {
  const form = new FormData();
  form.append("file", file);
  setStatus("Upload en cours...", "running");

  const response = await fetch("/api/upload", {
    method: "POST",
    body: form,
  });

  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || "Upload impossible.");
  }

  state.uploadId = payload.upload_id;
  setVideoSource(payload.video_url);
  setStatus("Vidéo chargée", "ok");
  resetCommentary("Vidéo chargée. Sélectionnez un modèle puis lancez l'analyse.");
  clearStats();
  updateAnalyzeState();
  updateSpeakButtonState();
}

async function startAnalysis() {
  if (!state.uploadId || !state.selectedModel) {
    return;
  }

  setStatus("Initialisation de l'analyse...", "running");
  clearStats();

  const response = await fetch("/api/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      upload_id: state.uploadId,
      model_id: state.selectedModel,
    }),
  });

  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || "L'analyse ne peut pas démarrer.");
  }

  state.currentJobId = payload.id;
  setLiveStream(state.currentJobId);
  resetCommentary("");
  pollJob();
}

async function pollJob() {
  if (!state.currentJobId) {
    return;
  }

  if (state.pollTimer) {
    clearInterval(state.pollTimer);
  }

  const tick = async () => {
    const response = await fetch(`/api/jobs/${state.currentJobId}`);
    const payload = await response.json();
    if (!response.ok) {
      setStatus(payload.detail || "Erreur de suivi du job.", "error");
      clearInterval(state.pollTimer);
      return;
    }

    if (payload.status === "queued" || payload.status === "running") {
      const pct = Math.round((payload.progress || 0) * 100);
      setStatus(`${payload.detail} (${pct}%)`, "running");
      if (payload.live_comment) {
        appendCommentary(payload.live_comment);
      }
      return;
    }

    if (payload.status === "failed") {
      setStatus(payload.detail || "Analyse échouée.", "error");
      if (!state.commentaryLog.length) {
        summaryValueEl.textContent = "Le backend a retourné une erreur durant l'analyse.";
      }
      liveStreamEl.src = "";
      liveStreamEl.style.display = "none";
      clearInterval(state.pollTimer);
      return;
    }

    if (payload.status === "completed") {
      const result = payload.result || {};
      appendCommentary(result.commentary);
      if (result.video_url) {
        setVideoSource(result.video_url);
      }
      setStatus("Analyse terminée", "ok");
      clearInterval(state.pollTimer);
    }
  };

  await tick();
  state.pollTimer = setInterval(tick, 1500);
}

uploadZoneEl.addEventListener("click", () => videoInputEl.click());
videoInputEl.addEventListener("change", async () => {
  const file = videoInputEl.files?.[0];
  if (!file) {
    return;
  }
  try {
    await uploadVideo(file);
  } catch (error) {
    setStatus(error.message || "Erreur upload", "error");
  }
});

analyzeButtonEl.addEventListener("click", async () => {
  try {
    await startAnalysis();
  } catch (error) {
    setStatus(error.message || "Erreur analyse", "error");
  }
});

(async function init() {
  try {
    await fetchModels();
    setStatus("En attente", "default");
    updateSpeakButtonState();
  } catch (error) {
    setStatus(error.message || "Erreur d'initialisation", "error");
    resetCommentary("API indisponible. Vérifiez que le backend FastAPI est lancé.");
  }
})();

speakButtonEl.addEventListener("click", async () => {
  await playCommentary();
});
