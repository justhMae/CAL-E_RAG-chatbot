const chatWindow = document.getElementById("chat-window");
const hero = document.getElementById("hero");
const messagesEl = document.getElementById("messages");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");

const splash = document.getElementById("splash");
const chatScreen = document.getElementById("chat-screen");
const startChatBtn = document.getElementById("start-chat");

const menuBtn = document.getElementById("menu-btn");
const drawer = document.getElementById("drawer");
const scrim = document.getElementById("scrim");
const newChatBtn = document.getElementById("new-chat-btn");
const historyList = document.getElementById("history-list");
const chipsRow = document.getElementById("chips-row");

let categories = [];        // [{ key, label, services: [{id, title, office}] }]
let historyEntries = [];    // [{ id, text, turnEl }], most recent first
let currentServicesTurn = null; // DOM node of the currently-visible services card

/* ---------- Splash / welcome screen ---------- */
startChatBtn.addEventListener("click", () => {
  splash.classList.add("hidden");
  chatScreen.classList.remove("hidden");
  chatInput.focus();
});

/* ---------- Chips: let mouse-wheel scroll horizontally too ---------- */
chipsRow.addEventListener("wheel", (e) => {
  if (e.deltaY === 0) return;
  e.preventDefault();
  chipsRow.scrollLeft += e.deltaY;
}, { passive: false });

/* ---------- Drawer open/close ---------- */
function openDrawer() {
  drawer.classList.add("open");
  scrim.classList.add("visible");
}

function closeDrawer() {
  drawer.classList.remove("open");
  scrim.classList.remove("visible");
}

menuBtn.addEventListener("click", openDrawer);
scrim.addEventListener("click", closeDrawer);

/* ---------- Chat history (drawer) ---------- */
// The chat is one continuous transcript. Every question the user sends
// gets logged here, and clicking a logged entry jumps back up to that
// exact message in the transcript above (scroll + brief highlight) —
// it does not swap out the conversation.
function renderHistory() {
  historyList.innerHTML = "";
  if (historyEntries.length === 0) {
    const empty = document.createElement("li");
    empty.className = "history-empty";
    empty.textContent = "No chats yet";
    historyList.appendChild(empty);
    return;
  }
  historyEntries.forEach((entry) => {
    const li = document.createElement("li");
    li.className = "history-item";
    li.textContent = entry.text;
    li.addEventListener("click", () => jumpToMessage(entry));
    historyList.appendChild(li);
  });
}

function addToHistory(text, turnEl) {
  historyEntries.unshift({ id: `h${Date.now()}${Math.random().toString(16).slice(2)}`, text, turnEl });
  renderHistory();
}

function jumpToMessage(entry) {
  closeDrawer();
  if (!entry.turnEl || !entry.turnEl.isConnected) return;

  // Wait for the drawer's close transition so scrollIntoView measures
  // the chat window at its final (drawer-closed) size.
  setTimeout(() => {
    entry.turnEl.scrollIntoView({ behavior: "smooth", block: "center" });
    entry.turnEl.classList.add("jump-highlight");
    setTimeout(() => entry.turnEl.classList.remove("jump-highlight"), 1400);
  }, 260);
}

renderHistory();

/* ---------- New chat ---------- */
newChatBtn.addEventListener("click", () => {
  messagesEl.innerHTML = "";
  hero.classList.remove("hidden");
  historyEntries = [];
  currentServicesTurn = null;
  chipsRow.querySelectorAll(".chip").forEach((c) => c.classList.remove("active"));
  renderHistory();
  closeDrawer();
  chatInput.value = "";
  chatInput.focus();
});

/* ---------- Category chips ---------- */
async function loadCategories() {
  try {
    const res = await fetch("/api/categories");
    if (!res.ok) throw new Error("no backend");
    categories = await res.json();
  } catch (err) {
    try {
      const res = await fetch("static/categories.sample.json");
      categories = await res.json();
    } catch (fallbackErr) {
      console.error("Could not load categories (no backend and no sample file):", fallbackErr);
    }
  }
  renderChips();
}

function renderChips() {
  chipsRow.innerHTML = "";
  categories.forEach((cat) => {
    const chip = document.createElement("button");
    chip.className = "chip";
    chip.type = "button";
    chip.dataset.key = cat.key;
    chip.textContent = cat.label;
    chipsRow.appendChild(chip);
  });
}

chipsRow.addEventListener("click", (e) => {
  const chip = e.target.closest(".chip");
  if (!chip) return;
  const category = categories.find((c) => c.key === chip.dataset.key);
  if (!category) return;

  chipsRow.querySelectorAll(".chip").forEach((c) => c.classList.remove("active"));
  chip.classList.add("active");

  showServicesForCategory(category);
});

const slider = document.getElementById("chips-row");

let isDown = false;
let startX;
let scrollLeft;

slider.addEventListener("mousedown", (e) => {
  isDown = true;
  slider.style.cursor = "grabbing";
  startX = e.pageX - slider.offsetLeft;
  scrollLeft = slider.scrollLeft;
});

slider.addEventListener("mouseleave", () => {
  isDown = false;
  slider.style.cursor = "grab";
});

slider.addEventListener("mouseup", () => {
  isDown = false;
  slider.style.cursor = "grab";
});

slider.addEventListener("mousemove", (e) => {
  if (!isDown) return;

  e.preventDefault();

  const x = e.pageX - slider.offsetLeft;
  const walk = (x - startX) * 2;
  slider.scrollLeft = scrollLeft - walk;
});

function showServicesForCategory(category) {
  hero.classList.add("hidden");

  if (currentServicesTurn && currentServicesTurn.parentNode) {
    currentServicesTurn.remove();
  }

  const turn = document.createElement("div");
  turn.className = "turn";

  const wrapper = document.createElement("div");
  wrapper.className = "message bot";

  const card = document.createElement("div");
  card.className = "services-card";

  const title = document.createElement("p");
  title.className = "services-title";
  title.textContent = `Services under "${category.label}"`;
  card.appendChild(title);

  if (!category.services || category.services.length === 0) {
    const empty = document.createElement("p");
    empty.className = "services-empty";
    empty.textContent = "No services listed under this category yet.";
    card.appendChild(empty);
  } else {
    const list = document.createElement("ul");
    category.services.forEach((service) => {
      const li = document.createElement("li");
      li.textContent = service.title;
      if (service.office) {
        const office = document.createElement("span");
        office.className = "service-office";
        office.textContent = service.office;
        li.appendChild(office);
      }
      li.addEventListener("click", () => {
        chatInput.value = service.title;
        chatForm.requestSubmit();
      });
      list.appendChild(li);
    });
    card.appendChild(list);
  }

  wrapper.appendChild(card);
  turn.appendChild(wrapper);
  messagesEl.appendChild(turn);
  chatWindow.scrollTop = chatWindow.scrollHeight;

  currentServicesTurn = turn;
}

loadCategories();

/* ---------- Messages ---------- */
function addMessage(text, sender, meta, forms) {
  hero.classList.add("hidden");

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

  messagesEl.appendChild(turn);
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
  messagesEl.appendChild(turn);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return turn;
}

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const message = chatInput.value.trim();
  if (!message) return;

  const userTurn = addMessage(message, "user");
  addToHistory(message, userTurn);
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

    messagesEl.removeChild(thinkingTurn);

    if (data.error) {
      addMessage(`⚠️ ${data.error}`, "bot");
    } else {
      const meta = `Detected intent: ${data.intent_label} (${(data.intent_confidence * 100).toFixed(0)}% confidence)`;
      addMessage(data.answer, "bot", meta, data.forms);
    }
  } catch (err) {
    messagesEl.removeChild(thinkingTurn);
    addMessage("⚠️ Could not reach the server. Is the Flask app running?", "bot");
  } finally {
    chatInput.disabled = false;
    chatInput.focus();
  }
});
