const urlInput = document.getElementById("repo-url");
const generateBtn = document.getElementById("generate-btn");
const generatorPanel = document.getElementById("generator-panel");
const progressPanel = document.getElementById("progress-panel");
const resultPanel = document.getElementById("result-panel");
const errorBox = document.getElementById("error-box");
const progressMessage = document.getElementById("progress-message");
const providerSelect = document.getElementById("provider");
const modelSelect = document.getElementById("model");
const apiKeyInput = document.getElementById("api-key");
const testProviderBtn = document.getElementById("test-provider-btn");
const providerStatus = document.getElementById("provider-status");
const modeCards = [...document.querySelectorAll(".mode-card")];

let selectedMode = "readme";
let lastResult = null;

const modelsByProvider = {
  huggingface: [
    {value: "Qwen/Qwen2.5-Coder-32B-Instruct", label: "Qwen2.5 Coder 32B Instruct"},
    {value: "Qwen/Qwen2.5-Coder-7B-Instruct", label: "Qwen2.5 Coder 7B Instruct"},
    {value: "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", label: "DeepSeek Coder V2 Lite"}
  ],
  mistral: [
    {value: "mistral-small-latest", label: "Mistral Small"},
    {value: "mistral-large-latest", label: "Mistral Large"},
    {value: "codestral-latest", label: "Codestral (coding)"}
  ]
};

function populateModels() {
  const provider = providerSelect.value;
  modelSelect.replaceChildren();
  modelsByProvider[provider].forEach(model => {
    const option = document.createElement("option");
    option.value = model.value;
    option.textContent = model.label;
    modelSelect.appendChild(option);
  });
  apiKeyInput.value = "";
  apiKeyInput.placeholder = provider === "mistral"
    ? "Paste your Mistral API key"
    : "Paste your Hugging Face access token";
}
providerSelect.addEventListener("change", populateModels);
populateModels();

testProviderBtn.addEventListener("click", async () => {
  const provider = providerSelect.value;
  const model = modelSelect.value;
  const apiKey = apiKeyInput.value.trim();
  providerStatus.textContent = "";
  providerStatus.className = "provider-status";

  if (!apiKey) {
    providerStatus.textContent = "Enter your API key first.";
    providerStatus.classList.add("status-error");
    apiKeyInput.focus();
    return;
  }

  testProviderBtn.disabled = true;
  testProviderBtn.textContent = "Testing connection…";
  try {
    const response = await fetch("/api/test-connection", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({provider, model, api_key: apiKey})
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Provider connection failed.");
    providerStatus.textContent = data.message;
    providerStatus.classList.add("status-success");
  } catch (error) {
    providerStatus.textContent = error.message || "Could not test the provider connection.";
    providerStatus.classList.add("status-error");
  } finally {
    testProviderBtn.disabled = false;
    testProviderBtn.textContent = "Test provider connection";
  }
});

document.getElementById("toggle-key").addEventListener("click", () => {
  const showing = apiKeyInput.type === "text";
  apiKeyInput.type = showing ? "password" : "text";
  document.getElementById("toggle-key").textContent = showing ? "Show" : "Hide";
  document.getElementById("toggle-key").setAttribute("aria-label", showing ? "Show API key" : "Hide API key");
});

modeCards.forEach(card => {
  card.addEventListener("click", () => {
    selectedMode = card.dataset.mode;
    modeCards.forEach(item => {
      const selected = item === card;
      item.classList.toggle("selected", selected);
      item.setAttribute("aria-pressed", String(selected));
    });
  });
});

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function setBusy(busy) {
  generateBtn.disabled = busy;
  generateBtn.innerHTML = busy
    ? '<span class="spinner-small"></span> Analyzing repository…'
    : '<span>✦</span> Generate with AI <span class="btn-arrow">→</span>';
}

generateBtn.addEventListener("click", async () => {
  const repoUrl = urlInput.value.trim();
  const provider = providerSelect.value;
  const model = modelSelect.value;
  const apiKey = apiKeyInput.value.trim();
  errorBox.hidden = true;

  if (!repoUrl) {
    showError("Please paste a GitHub repository URL first.");
    urlInput.focus();
    return;
  }
  if (!apiKey) {
    showError("Enter your API key for the selected provider.");
    apiKeyInput.focus();
    return;
  }

  generatorPanel.hidden = true;
  resultPanel.hidden = true;
  progressPanel.hidden = false;
  setBusy(true);
  progressMessage.textContent = "Cloning repository and reading project files…";

  const messages = [
    "Cloning repository and reading project files…",
    `Analyzing the architecture with ${provider === "mistral" ? "Mistral AI" : "Hugging Face"}…`, 
    "Preparing your selected output…"
  ];
  let messageIndex = 0;
  const messageTimer = setInterval(() => {
    messageIndex = Math.min(messageIndex + 1, messages.length - 1);
    progressMessage.textContent = messages[messageIndex];
  }, 7000);

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        repo_url: repoUrl,
        mode: selectedMode,
        provider,
        model,
        api_key: apiKey
      })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Generation failed. Please try again.");

    lastResult = data;
    apiKeyInput.value = "";
    document.getElementById("result-title").textContent =
      data.mode === "linkedin" ? "Generated LinkedIn post" : "Generated README";
    document.getElementById("result-kind").textContent =
      data.mode === "linkedin" ? "PLAIN TEXT" : "MARKDOWN";
    document.getElementById("result-repo").textContent = `${data.provider || provider} · ${data.model || model}`;
    document.getElementById("result-output").textContent = data.output;
    document.getElementById("analysis-output").textContent = data.analysis || "No analysis returned.";
    progressPanel.hidden = true;
    resultPanel.hidden = false;
    resultPanel.scrollIntoView({behavior: "smooth", block: "start"});
  } catch (error) {
    progressPanel.hidden = true;
    generatorPanel.hidden = false;
    showError(error.message || "Unable to connect to the server.");
  } finally {
    apiKeyInput.value = "";
    clearInterval(messageTimer);
    setBusy(false);
  }
});

document.getElementById("copy-btn").addEventListener("click", async () => {
  if (!lastResult) return;
  const button = document.getElementById("copy-btn");
  try {
    await navigator.clipboard.writeText(lastResult.output);
    button.textContent = "✓ Copied!";
    setTimeout(() => button.textContent = "▢ Copy output", 1800);
  } catch {
    showError("Clipboard access was blocked. Select the output text and copy it manually.");
  }
});

document.getElementById("download-btn").addEventListener("click", async () => {
  if (!lastResult) return;
  const response = await fetch("/api/download", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({content: lastResult.output, filename: lastResult.filename})
  });
  if (!response.ok) {
    showError("Could not download the file. Please try again.");
    return;
  }
  const blob = await response.blob();
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = lastResult.filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(link.href);
});

document.getElementById("new-btn").addEventListener("click", () => {
  resultPanel.hidden = true;
  generatorPanel.hidden = false;
  urlInput.focus();
  window.scrollTo({top: 0, behavior: "smooth"});
});
urlInput.addEventListener("keydown", event => {
  if (event.key === "Enter") generateBtn.click();
});
