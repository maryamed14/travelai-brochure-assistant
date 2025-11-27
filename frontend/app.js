const questionInput = document.getElementById("question");
const askBtn = document.getElementById("ask-btn");
const answerEl = document.getElementById("answer");
const contextEl = document.getElementById("context");
const contextCard = document.getElementById("context-card");
const statusEl = document.getElementById("status");

function getMode() {
  const checked = document.querySelector('input[name="mode"]:checked');
  return checked ? checked.value : "qa";
}

function setLoading(isLoading, mode) {
  askBtn.disabled = isLoading;
  statusEl.classList.toggle("hidden", !isLoading);
  statusEl.classList.remove("error");

  if (isLoading) {
    statusEl.textContent = mode === "qa" ? "Running RAG QA..." : "Running agent with tools...";
  } else {
    statusEl.textContent = "";
  }
}

function showError(message) {
  statusEl.classList.remove("hidden");
  statusEl.classList.add("error");
  statusEl.textContent = message;
}

async function callBackend(question, mode) {
  const endpoint = mode === "qa" ? "/qa" : "/agent";

  const res = await fetch(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ question }),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Request failed: ${res.status} ${text}`);
  }

  return res.json();
}

function renderQAResponse(data) {
  answerEl.textContent = data.answer ?? "";

  const ctx = data.context ?? [];
  contextEl.innerHTML = "";

  if (!ctx.length) {
    contextEl.textContent = "No context chunks returned.";
    return;
  }

  ctx.forEach((chunk, idx) => {
    const div = document.createElement("div");
    div.className = "context-item";

    const meta = document.createElement("div");
    meta.className = "context-meta";
    meta.textContent = `#${idx + 1} | ${chunk.city} | ${chunk.source_file} | score=${chunk.score.toFixed?.(3) ?? chunk.score}`;

    const text = document.createElement("div");
    text.className = "context-text";
    text.textContent = chunk.text;

    div.appendChild(meta);
    div.appendChild(text);
    contextEl.appendChild(div);
  });
}

function renderAgentResponse(data) {
  answerEl.textContent = data.answer ?? JSON.stringify(data, null, 2);
  contextEl.innerHTML = "";
  contextEl.textContent = "Agent mode does not expose retrieval chunks directly.";
}

askBtn.addEventListener("click", async () => {
  const question = questionInput.value.trim();
  const mode = getMode();

  if (!question) {
    showError("Please enter a question.");
    return;
  }

  setLoading(true, mode);
  answerEl.textContent = "";
  contextEl.textContent = "";
  contextEl.innerHTML = "";
  contextCard.style.display = mode === "qa" ? "block" : "block"; // keep card but change content

  try {
    const data = await callBackend(question, mode);
    if (mode === "qa") {
      renderQAResponse(data);
    } else {
      renderAgentResponse(data);
    }
    setLoading(false, mode);
  } catch (err) {
    console.error(err);
    setLoading(false, mode);
    showError(err.message || "Something went wrong.");
  }
});
