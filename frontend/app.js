const questionInput = document.getElementById("question");
const askBtn = document.getElementById("ask-btn");
const answerEl = document.getElementById("answer");
const statusEl = document.getElementById("status");

// Basic sanity check
console.log("questionInput:", questionInput);
console.log("askBtn:", askBtn);
console.log("answerEl:", answerEl);
console.log("statusEl:", statusEl);

function getMode() {
  const checked = document.querySelector('input[name="mode"]:checked');
  return checked ? checked.value : "qa";
}

function setLoading(isLoading, mode) {
  askBtn.disabled = isLoading;
  statusEl.classList.toggle("hidden", !isLoading);
  statusEl.classList.remove("error");

  if (isLoading) {
    statusEl.textContent =
      mode === "qa" ? "Running RAG QA..." : "Running agent with tools...";
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
  console.log("Calling", endpoint, "with question:", question);

  const res = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question: question }),
  });

  console.log("Response status:", res.status);

  if (!res.ok) {
    const text = await res.text();
    throw new Error("Request failed: " + res.status + " " + text);
  }

  const data = await res.json();
  console.log("Response JSON:", data);
  return data;
}

function renderQAResponse(data) {
  if (data && typeof data.answer === "string") {
    answerEl.textContent = data.answer;
  } else {
    answerEl.textContent = JSON.stringify(data, null, 2);
  }
}

function renderAgentResponse(data) {
  if (data && typeof data.answer === "string") {
    answerEl.textContent = data.answer;
  } else {
    answerEl.textContent = JSON.stringify(data, null, 2);
  }
}

askBtn.addEventListener("click", async function () {
  const question = questionInput.value.trim();
  const mode = getMode();

  if (!question) {
    showError("Please enter a question.");
    return;
  }

  setLoading(true, mode);
  answerEl.textContent = "";
  statusEl.classList.remove("error");

  try {
    const data = await callBackend(question, mode);

    if (mode === "qa") {
      renderQAResponse(data);
    } else {
      renderAgentResponse(data);
    }

    setLoading(false, mode);
  } catch (err) {
    console.error("Frontend error:", err);
    setLoading(false, mode);
    showError(err.message || "Something went wrong.");
  }
});
