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
const drawerSearchBtn = document.getElementById("drawer-search-btn");
const historySearchWrap = document.getElementById("history-search-wrap");
const historySearchInput = document.getElementById("history-search-input");
const appEl = document.querySelector(".app");
const viewToggleBtn = document.getElementById("view-toggle-btn");

const VIEW_STORAGE_KEY = "cal-e-view-mode";

function isDesktopView() {
  return appEl.classList.contains("desktop-view");
}

function applyViewMode(mode) {
  const desktop = mode === "desktop";
  appEl.classList.toggle("desktop-view", desktop);
  if (desktop) {
    drawer.classList.remove("open");
    scrim.classList.remove("visible");
  }
  if (viewToggleBtn) {
    viewToggleBtn.setAttribute("aria-label", desktop ? "Switch to mobile view" : "Switch to desktop view");
    viewToggleBtn.setAttribute("title", desktop ? "Switch to mobile view" : "Switch to desktop view");
    viewToggleBtn.innerHTML = desktop
      ? `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><rect x="7" y="2" width="10" height="20" rx="2" stroke="white" stroke-width="2"/><path d="M11 18h2" stroke="white" stroke-width="2" stroke-linecap="round"/></svg><span>Mobile</span>`
      : `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><rect x="2" y="3" width="20" height="14" rx="2" stroke="white" stroke-width="2"/><path d="M8 21h8M12 17v4" stroke="white" stroke-width="2" stroke-linecap="round"/></svg><span>Desktop</span>`;
  }
}

const savedView = localStorage.getItem(VIEW_STORAGE_KEY);
applyViewMode(savedView || (window.innerWidth >= 860 ? "desktop" : "mobile"));

if (viewToggleBtn) {
  viewToggleBtn.addEventListener("click", () => {
    const next = isDesktopView() ? "mobile" : "desktop";
    localStorage.setItem(VIEW_STORAGE_KEY, next);
    applyViewMode(next);
  });
}

let categories = [];        // [{ key, label, services: [{id, title, office}] }]
let historyEntries = [];    // [{ id, text, turnEl }], most recent first
let currentServicesTurn = null;
let historyQuery = "";

/* splash / welcome screen */
startChatBtn.addEventListener("click", () => {
  splash.classList.add("hidden");
  chatScreen.classList.remove("hidden");
  chatInput.focus();
});

/* chips; let mouse-wheel scroll horizontally too */
chipsRow.addEventListener("wheel", (e) => {
  if (e.deltaY === 0) return;
  e.preventDefault();
  chipsRow.scrollLeft += e.deltaY;
}, { passive: false });

/* drawer open/close */
function openDrawer() {
  drawer.classList.add("open");
  scrim.classList.add("visible");
}

function closeDrawer() {
  drawer.classList.remove("open");
  scrim.classList.remove("visible");
  closeHistorySearch();
}

menuBtn.addEventListener("click", openDrawer);
scrim.addEventListener("click", closeDrawer);

/* chat history */
function renderHistory() {
  historyList.innerHTML = "";

  const query = historyQuery.trim().toLowerCase();
  const visibleEntries = query
    ? historyEntries.filter((entry) => entry.text.toLowerCase().includes(query))
    : historyEntries;

  if (visibleEntries.length === 0) {
    const empty = document.createElement("li");
    empty.className = "history-empty";
    empty.textContent = historyEntries.length === 0
      ? "No chats yet"
      : "No matching chats";
    historyList.appendChild(empty);
    return;
  }
  visibleEntries.forEach((entry) => {
    const li = document.createElement("li");
    li.className = "history-item";
    li.textContent = entry.text;
    li.addEventListener("click", () => jumpToMessage(entry));
    historyList.appendChild(li);
  });
}

/* chat history search */
function closeHistorySearch() {
  historySearchWrap.classList.add("hidden");
  historySearchInput.value = "";
  historyQuery = "";
  renderHistory();
}

drawerSearchBtn.addEventListener("click", () => {
  const isHidden = historySearchWrap.classList.contains("hidden");
  if (isHidden) {
    historySearchWrap.classList.remove("hidden");
    historySearchInput.focus();
  } else {
    closeHistorySearch();
  }
});

historySearchInput.addEventListener("input", () => {
  historyQuery = historySearchInput.value;
  renderHistory();
});

function addToHistory(text, turnEl) {
  historyEntries.unshift({ id: `h${Date.now()}${Math.random().toString(16).slice(2)}`, text, turnEl });
  renderHistory();
}

function jumpToMessage(entry) {
  if (!isDesktopView()) {
    closeDrawer();
  }
  if (!entry.turnEl || !entry.turnEl.isConnected) return;

  const delay = isDesktopView() ? 40 : 260;
  setTimeout(() => {
    entry.turnEl.scrollIntoView({ behavior: "smooth", block: "center" });
    entry.turnEl.classList.add("jump-highlight");
    setTimeout(() => entry.turnEl.classList.remove("jump-highlight"), 1400);
  }, delay);
}

renderHistory();

/* new chat */
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

/* category chips */
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
        chatInput.dataset.categoryHint = category.key;
        chatInput.dataset.docIdHint = service.id;
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

/* lightweight markdown-to-HTML for bot replies */
function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function renderMarkdownLite(text) {
  const escaped = escapeHtml(text);
  const lines = escaped.split("\n");
  const htmlParts = [];
  let listBuffer = [];
  let listType = null; // "ul" | "ol"

  function flushList() {
    if (listBuffer.length > 0) {
      htmlParts.push(`<${listType}>` + listBuffer.join("") + `</${listType}>`);
      listBuffer = [];
      listType = null;
    }
  }

  function inline(str) {
    let out = str.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    out = out.replace(/\*(.+?)\*/g, "<em>$1</em>");
    return out;
  }
  function isTableRow(line) {
    return line.startsWith("|") && line.endsWith("|") && line.length > 1;
  }
  function isTableSeparatorRow(line) {
    return isTableRow(line) && /^\|[\s:|-]+\|$/.test(line);
  }
  function splitTableRow(line) {
    return line.slice(1, -1).split("|").map((cell) => cell.trim());
  }

  let i = 0;
  while (i < lines.length) {
    const line = lines[i].trim();

    if (line === "") {
      flushList();
      i++;
      continue;
    }

    if (isTableRow(line) && i + 1 < lines.length && isTableSeparatorRow(lines[i + 1].trim())) {
      flushList();
      const headerCells = splitTableRow(line);
      i += 2;
      const bodyRows = [];
      while (i < lines.length && isTableRow(lines[i].trim())) {
        bodyRows.push(splitTableRow(lines[i].trim()));
        i++;
      }
      const thead = "<thead><tr>" +
        headerCells.map((c) => `<th>${inline(c)}</th>`).join("") +
        "</tr></thead>";
      const tbody = "<tbody>" +
        bodyRows.map((row) =>
          "<tr>" + row.map((c) => `<td>${inline(c)}</td>`).join("") + "</tr>"
        ).join("") +
        "</tbody>";
      htmlParts.push(`<div class="table-wrap"><table>${thead}${tbody}</table></div>`);
      continue;
    }

    const headingMatch = line.match(/^(#{1,6})\s+(.*)$/);
    if (headingMatch) {
      flushList();
      const level = Math.min(headingMatch[1].length + 2, 4);
      htmlParts.push(`<h${level}>${inline(headingMatch[2])}</h${level}>`);
      i++;
      continue;
    }

    const bulletMatch = line.match(/^[-*]\s+(.*)$/);
    const numberedMatch = line.match(/^\d+\.\s+(.*)$/);

    if (bulletMatch) {
      if (listType !== "ul") flushList();
      listType = "ul";
      listBuffer.push(`<li>${inline(bulletMatch[1])}</li>`);
      i++;
      continue;
    }
    if (numberedMatch) {
      if (listType !== "ol") flushList();
      listType = "ol";
      listBuffer.push(`<li>${inline(numberedMatch[1])}</li>`);
      i++;
      continue;
    }

    flushList();
    htmlParts.push(`<p>${inline(line)}</p>`);
    i++;
  }
  flushList();

  return htmlParts.join("");
}

/* messages */
function addMessage(text, sender, meta, forms) {
  hero.classList.add("hidden");

  const turn = document.createElement("div");
  turn.className = "turn";

  const wrapper = document.createElement("div");
  wrapper.className = `message ${sender}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  if (sender === "bot") {
    bubble.innerHTML = renderMarkdownLite(text);
  } else {
    bubble.textContent = text;
  }
  wrapper.appendChild(bubble);

  turn.appendChild(wrapper);

  if (forms && forms.length > 0) {
    const formsEl = document.createElement("div");
    formsEl.className = "forms";

    const label = document.createElement("div");
    label.className = "forms-label";
    label.textContent = "You may find these forms from the relevant office useful:";
    formsEl.appendChild(label);
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

  const categoryHint = chatInput.dataset.categoryHint || null;
  const docIdHint = chatInput.dataset.docIdHint || null;
  delete chatInput.dataset.categoryHint;
  delete chatInput.dataset.docIdHint;

  const userTurn = addMessage(message, "user");
  addToHistory(message, userTurn);
  chatInput.value = "";
  chatInput.disabled = true;

  const thinkingTurn = addThinkingBubble();

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, category_hint: categoryHint, doc_id_hint: docIdHint }),
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