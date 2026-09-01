const chatWindow = document.getElementById("chat-window");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const splash = document.getElementById("splash");
const chatScreen = document.getElementById("chat-screen");
const startChatBtn = document.getElementById("start-chat");

startChatBtn.addEventListener("click", () => {
  splash.classList.add("hidden");
  chatScreen.classList.remove("hidden");
  chatInput.focus();
});

function addMessage(text, sender, meta, forms) {
  const turn = document.createElement("div");
  turn.className = "turn";

  const wrapper = document.createElement("div");
  wrapper.className = `message ${sender}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  wrapper.appendChild(bubble);

  turn.appendChild(wrapper);

  if (forms && forms.length > 0) {
    const formsEl = document.createElement("div");
    formsEl.className = "forms";
    forms.forEach((form) => {
      const link = document.createElement("a");
      link.href = form.download_url;
      link.className = "form-link";
      link.setAttribute("download", "");
      link.innerHTML = `<svg class="file-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M6 2h9l5 5v13a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2z" stroke="white" stroke-width="1.6" stroke-linejoin="round"/>
          <path d="M14 2v5h5" stroke="white" stroke-width="1.6" stroke-linejoin="round"/>
        </svg><span>${form.name}</span>`;
      formsEl.appendChild(link);
    });
    turn.appendChild(formsEl);
  }

  if (meta) {
    const metaEl = document.createElement("div");
    metaEl.className = "meta";
    metaEl.textContent = meta;
    turn.appendChild(metaEl);
  }

  chatWindow.appendChild(turn);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return turn;
}

function addThinkingBubble() {
  const turn = document.createElement("div");
  turn.className = "turn";

  const wrapper = document.createElement("div");
  wrapper.className = "message bot";

  const bubble = document.createElement("div");
  bubble.className = "bubble thinking";
  bubble.innerHTML = `<span class="dot"></span><span class="dot"></span><span class="dot"></span>`;
  wrapper.appendChild(bubble);

  turn.appendChild(wrapper);
  chatWindow.appendChild(turn);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return turn;
}

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const message = chatInput.value.trim();
  if (!message) return;

  addMessage(message, "user");
  chatInput.value = "";
  chatInput.disabled = true;

  const thinkingTurn = addThinkingBubble();

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    const data = await res.json();

    chatWindow.removeChild(thinkingTurn);

    if (data.error) {
      addMessage(`⚠️ ${data.error}`, "bot");
    } else {
      const meta = `Detected intent: ${data.intent_label} (${(data.intent_confidence * 100).toFixed(0)}% confidence)`;
      addMessage(data.answer, "bot", meta, data.forms);
    }
  } catch (err) {
    chatWindow.removeChild(thinkingTurn);
    addMessage("⚠️ Could not reach the server. Is the Flask app running?", "bot");
  } finally {
    chatInput.disabled = false;
    chatInput.focus();
  }
});
