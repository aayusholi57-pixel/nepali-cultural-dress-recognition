const $ = (id) => document.getElementById(id);

const imageInput = $("imageInput");
const dropzone = $("dropzone");
const preview = $("preview");
const analyzeButton = $("analyzeButton");
const clearButton = $("clearButton");
const statusText = $("status");

const ALLOWED_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);
const MAX_FILE_SIZE = 5 * 1024 * 1024;
let selectedFile = null;
let previewUrl = null;
let isAnalyzing = false;

$("year").textContent = new Date().getFullYear();

function setStatus(message, isError = false) {
  statusText.textContent = message;
  statusText.classList.toggle("error", isError);
}

function setApiStatus(message, state = "idle") {
  $("apiStatus").textContent = message;
  $("apiDot").className = "status-dot" + (state === "ok" ? " is-online" : state === "error" ? " is-error" : "");
}

function setHealth(healthy, message) {
  $("healthText").textContent = message;
  $("healthDot").className = "status-dot" + (healthy ? " is-online" : " is-error");
}

function setBusy(busy) {
  isAnalyzing = busy;
  analyzeButton.disabled = busy || !selectedFile;
  clearButton.disabled = busy || !selectedFile;
  analyzeButton.innerHTML = busy
    ? '<span class="spinner" aria-hidden="true"></span> Analyzing image…'
    : 'Recognize attire <span aria-hidden="true">→</span>';
}

function chooseFile(file) {
  if (!file) return;

  if (!ALLOWED_TYPES.has(file.type)) {
    setStatus("Please choose a JPG, PNG, or WEBP image.", true);
    return;
  }
  if (file.size > MAX_FILE_SIZE) {
    setStatus("This image is larger than 5 MB. Please choose a smaller file.", true);
    return;
  }
  if (file.size === 0) {
    setStatus("This file is empty. Please choose another image.", true);
    return;
  }

  selectedFile = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  preview.src = previewUrl;
  preview.hidden = false;
  $("uploadTitle").textContent = file.name;
  $("uploadHint").textContent = "Image ready to analyze";
  $("fileMeta").textContent = (file.size / 1024 / 1024).toFixed(2) + " MB · " + file.type.replace("image/", "").toUpperCase();
  analyzeButton.disabled = false;
  clearButton.disabled = false;
  setStatus("");
  setApiStatus("Image ready — waiting to analyze");
  $("emptyState").hidden = false;
  $("resultContent").hidden = true;
  $("emptyState").querySelector("strong").textContent = "Ready when you are";
  $("emptyState").querySelector("p").textContent = "Select Recognize attire to run the model.";
}

function clearSelection() {
  if (isAnalyzing) return;
  selectedFile = null;
  imageInput.value = "";
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  preview.removeAttribute("src");
  preview.hidden = true;
  $("uploadTitle").textContent = "Drop your image here";
  $("uploadHint").textContent = "or browse files from your device";
  $("fileMeta").textContent = "No image selected";
  analyzeButton.disabled = true;
  clearButton.disabled = true;
  setStatus("");
  setApiStatus("Waiting for an image");
  $("emptyState").hidden = false;
  $("resultContent").hidden = true;
  $("emptyState").querySelector("strong").textContent = "Your result will appear here";
  $("emptyState").querySelector("p").innerHTML = 'Choose an image, then select <b>Recognize attire</b>.';
  $("rawResponse").textContent = "No prediction has been requested yet.";
}

imageInput.addEventListener("change", (event) => chooseFile(event.target.files[0]));

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragover");
  });
});
["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragover");
  });
});
dropzone.addEventListener("drop", (event) => {
  const file = event.dataTransfer?.files?.[0];
  chooseFile(file);
});
clearButton.addEventListener("click", clearSelection);

function showResults(data) {
  $("emptyState").hidden = true;
  $("resultContent").hidden = false;

  const confidence = Number(data.confidence);
  const confidenceIsValid = Number.isFinite(confidence);
  const confidenceLabel = confidenceIsValid ? confidence.toFixed(2) + "%" : "Not available";
  const confidenceWidth = confidenceIsValid ? Math.max(0, Math.min(100, confidence)) : 0;

  $("resultTitle").textContent = data.top_prediction || "No class returned";
  $("resultDescription").textContent = data.is_out_of_distribution
    ? "The model is uncertain about this image. Treat this prediction with caution."
    : "Best matching class returned by the model.";
  $("resultKicker").textContent = data.is_out_of_distribution ? "UNCERTAIN MATCH" : "TOP MATCH";
  $("confidence").textContent = confidenceLabel;
  $("confidenceBar").style.width = confidenceWidth + "%";

  const matches = Array.isArray(data.top_k) ? data.top_k : [];
  const otherMatches = $("otherMatches");
  otherMatches.replaceChildren();
  if (matches.length > 1) {
    const heading = document.createElement("p");
    heading.className = "matches-heading";
    heading.textContent = "Other likely matches";
    otherMatches.appendChild(heading);

    matches.slice(1, 3).forEach((item, index) => {
      const row = document.createElement("div");
      row.className = "match-row";
      const label = document.createElement("span");
      label.textContent = item.class_name || "Unknown class";
      const score = document.createElement("strong");
      const value = Number(item.confidence);
      score.textContent = Number.isFinite(value) ? value.toFixed(2) + "%" : "—";
      row.append(label, score);
      otherMatches.appendChild(row);
    });
  }
  $("rawResponse").textContent = JSON.stringify(data, null, 2);
}

async function checkBackend() {
  try {
    const response = await fetch("/health", { headers: { Accept: "application/json" } });
    if (!response.ok) throw new Error("HTTP " + response.status);
    const data = await response.json();
    if (data.model_loaded) {
      setHealth(true, "Model ready");
      setApiStatus("Connected to model", "ok");
    } else {
      setHealth(false, "Model not loaded");
      setApiStatus("API online, model not loaded", "error");
    }
  } catch {
    setHealth(false, "API unavailable");
    setApiStatus("Could not reach the backend", "error");
  }
}

analyzeButton.addEventListener("click", async () => {
  if (!selectedFile || isAnalyzing) return;

  setBusy(true);
  setStatus("Sending image to the model…");
  setApiStatus("Predicting…");
  $("rawResponse").textContent = "Waiting for the model response…";

  try {
    const body = new FormData();
    body.append("file", selectedFile, selectedFile.name);

    // Relative URL means the browser uses this same FastAPI server—no manual endpoint or CORS setup.
    const response = await fetch("/predict", { method: "POST", body });
    const raw = await response.text();
    let data;
    try {
      data = JSON.parse(raw);
    } catch {
      data = { detail: raw || "The server returned an empty response." };
    }

    if (!response.ok) {
      const detail = typeof data.detail === "string" ? data.detail : "Request failed (HTTP " + response.status + ").";
      throw new Error(detail);
    }

    showResults(data);
    setApiStatus(data.is_out_of_distribution ? "Prediction returned — uncertain" : "Prediction complete", "ok");
    setStatus(data.is_out_of_distribution
      ? "Prediction complete, but the model is not confident enough for a reliable match."
      : "Done — your prediction is ready.");
  } catch (error) {
    $("emptyState").hidden = false;
    $("resultContent").hidden = true;
    $("emptyState").querySelector("strong").textContent = "Prediction could not be completed";
    $("emptyState").querySelector("p").textContent = error.message || "Check that the backend and model are running.";
    $("rawResponse").textContent = error.message || "Unknown error";
    setApiStatus("Prediction failed", "error");
    setStatus(error.message || "Something went wrong. Please try again.", true);
    if (error instanceof TypeError) {
      setStatus("Could not reach the API. Check the Docker container and try again.", true);
    }
  } finally {
    setBusy(false);
  }
});

checkBackend();
