const chat = document.getElementById("chat");
const input = document.getElementById("msg");
const sendBtn = document.getElementById("send");

// Settings elements
const settingsBtn = document.getElementById("settingsBtn");
const settingsModal = document.getElementById("settingsModal");
const saveTokenBtn = document.getElementById("saveToken");
const cancelSettingsBtn = document.getElementById("cancelSettings");
const tokenInput = document.getElementById("oauthToken");
const tokenStatus = document.getElementById("tokenStatus");

const TOKEN_KEY = "oauthToken";

// Busy/typing state
let busy = false;
let typingRow = null;

// Track current phase (used for typing bubble prediction)
let currentPhase = "eligibility";
let ceLoaded = false;

function lockComposer(reason) {
  input.setAttribute("readonly", "readonly");
  input.placeholder =
    reason || "Dynamics CE not loaded – open Settings and paste an OAuth token";
  sendBtn.disabled = true;
}

function unlockComposer() {
  input.removeAttribute("readonly");
  input.placeholder =
    "Type a message… e.g. I earn 110k, debts 450/mo, savings 40k; budget 800k in Brunswick";
  sendBtn.disabled = false;
}

function getToken() {
  return window.localStorage.getItem(TOKEN_KEY) || "";
}
function setToken(val) {
  if (!val) {
    window.localStorage.removeItem(TOKEN_KEY);
  } else {
    window.localStorage.setItem(TOKEN_KEY, val);
  }
  updateTokenStatus();
}
function updateTokenStatus() {
  const t = getToken();
  if (t) {
    tokenStatus.classList.add("token-dot--ok");
    tokenStatus.title = "OAuth token set";
  } else {
    tokenStatus.classList.remove("token-dot--ok");
    tokenStatus.title = "OAuth token not set";
  }
}
function setBusy(on) {
  busy = on;
  input.disabled = on;
  sendBtn.disabled = on;
  chat.setAttribute("aria-busy", on ? "true" : "false");
  document.querySelector(".app")?.classList.toggle("is-busy", on);
  document.querySelectorAll(".chip").forEach((btn) => {
    btn.disabled = on;
  });
}

// ---------- Chat rendering ----------
function addUserMessage(text) {
  const row = document.createElement("div");
  row.className = "row user";
  row.innerHTML = `
    <div class="avatar user">🧑</div>
    <div class="bubble user"><div class="content"></div></div>
  `;
  row.querySelector(".content").textContent = text;
  chat.appendChild(row);
  chat.scrollTop = chat.scrollHeight;
}
function addBotMessage(text, phaseClass) {
  const row = document.createElement("div");
  row.className = "row";
  row.innerHTML = `
    <div class="avatar">🤖</div>
    <div class="bubble ${phaseClass || "eligibility"}">
      <div class="meta"><span class="badge ${
        phaseClass || "eligibility"
      }">${labelFor(phaseClass)}</span></div>
      <div class="content"></div>
    </div>
  `;
  row.querySelector(".content").textContent = text;
  chat.appendChild(row);
  chat.scrollTop = chat.scrollHeight;
}
function showTyping(phaseClass) {
  removeTyping();
  typingRow = document.createElement("div");
  typingRow.className = "row";
  typingRow.innerHTML = `
    <div class="avatar">🤖</div>
    <div class="bubble ${phaseClass || "eligibility"}">
      <div class="meta"><span class="badge ${
        phaseClass || "eligibility"
      }">${labelFor(phaseClass)}</span></div>
      <div class="content">
        <span class="typing" aria-label="Assistant is typing">
          <span class="dot"></span><span class="dot"></span><span class="dot"></span>
        </span>
      </div>
    </div>
  `;
  chat.appendChild(typingRow);
  chat.scrollTop = chat.scrollHeight;
}
function removeTyping() {
  if (typingRow) {
    typingRow.remove();
    typingRow = null;
  }
}
function labelFor(phase) {
  switch (phase) {
    case "eligibility":
      return "Eligibility";
    case "listings":
      return "Listings";
    case "area":
      return "Area";
    case "done":
      return "Done";
    default:
      return "Assistant";
  }
}

// Predict the phase to display while waiting for the server
function predictPhase(text) {
  const t = text.trim().toLowerCase();
  if (t.startsWith("/reset")) return "eligibility";
  if (t.startsWith("/list")) return "listings";
  if (t.startsWith("/pick")) return "area";
  if (t.startsWith("/next")) {
    // Only confidently predict eligibility -> listings (your reported issue)
    if (currentPhase === "eligibility") return "listings";
    // For other transitions, stay on current to avoid wrong colour when no /pick yet
    return currentPhase;
  }
  return currentPhase;
}

// ---------- API send ----------
async function send() {
  if (busy) return; // prevent double-send

  if (!ceLoaded || input.hasAttribute("readonly")) {
    // Quietly refuse to send while CE is unavailable
    return;
  }

  const text = input.value.trim();
  if (!text) return;

  setBusy(true);
  addUserMessage(text);
  input.value = "";

  // Show typing using predicted phase
  const phaseForTyping = predictPhase(text);
  showTyping(phaseForTyping);

  const headers = { "Content-Type": "application/json" };
  const token = getToken();
  if (token) headers["Authorization"] = `Bearer ${token}`;

  try {
    const res = await fetch("/api/message", {
      method: "POST",
      headers,
      body: JSON.stringify({ message: text }),
    });
    const data = await res.json();
    removeTyping();
    if (!res.ok) {
      addBotMessage(data.error || "Error", "eligibility");
    } else {
      // Update currentPhase from server
      if (data.phaseClass) currentPhase = data.phaseClass;
      addBotMessage(data.reply, data.phaseClass);
    }
  } catch (err) {
    removeTyping();
    addBotMessage("Network error", "eligibility");
  } finally {
    setBusy(false);
    input.focus();
  }
}

sendBtn.addEventListener("click", send);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter") send();
});
document.querySelectorAll(".chip").forEach((btn) => {
  btn.addEventListener("click", () => {
    if (busy) return;
    input.value = btn.dataset.text;
    input.focus();
  });
});

// ---------- Settings modal ----------
function openSettings() {
  if (busy) return;
  tokenInput.value = getToken();
  settingsModal.classList.remove("hidden");
  settingsModal.setAttribute("aria-hidden", "false");
  tokenInput.focus();
}
function closeSettings() {
  settingsModal.classList.add("hidden");
  settingsModal.setAttribute("aria-hidden", "true");
}
settingsBtn.addEventListener("click", openSettings);
saveTokenBtn.addEventListener("click", () => {
  setToken(tokenInput.value.trim());
  closeSettings();
  addBotMessage(
    "Token saved locally. Future API calls will include Authorization: Bearer <token>.",
    "eligibility"
  );
});
cancelSettingsBtn.addEventListener("click", closeSettings);
settingsModal.addEventListener("click", (e) => {
  if (e.target.classList.contains("modal__backdrop")) closeSettings();
});
window.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !settingsModal.classList.contains("hidden"))
    closeSettings();
});

const dvStatus = document.getElementById("dvStatus");

function getToken() {
  return window.localStorage.getItem(TOKEN_KEY) || "";
}
function setToken(val) {
  if (!val) window.localStorage.removeItem(TOKEN_KEY);
  else window.localStorage.setItem(TOKEN_KEY, val);
  updateTokenStatus();
}

function updateTokenStatus() {
  const t = getToken();
  const dot = document.getElementById("tokenStatus");
  if (t) {
    dot.classList.add("token-dot--ok");
    dot.title = "OAuth token set";
  } else {
    dot.classList.remove("token-dot--ok");
    dot.title = "OAuth token not set";
  }
}

// ---- NEW: fetch contacts snapshot from backend on page load
async function fetchDynamicsHousingInfo() {
  const headers = { Accept: "application/json" };
  const t = typeof getToken === "function" ? getToken() : "";
  if (t) headers["Authorization"] = `Bearer ${t}`;

  try {
    // 1) Probe PROPERTIES first — if it fails, bail immediately
    const propsRes = await fetch("/api/dynamics/properties", { headers });

    if (!propsRes.ok) {
      markCeFailed(`HTTP ${propsRes.status}`);
      return; // ⛔️ do NOT continue to other endpoints
    }

    const propsData = await safeJson(propsRes);
    if (!propsData || propsData.error) {
      markCeFailed(propsData?.error || "Unexpected response");
      return; // ⛔️
    }

    // 2) Only now fetch the other three, sequentially
    const all = {
      properties: propsData,
      ptv: null,
      restaurants: null,
      schools: null,
    };
    const labelMap = {
      "/api/dynamics/ptv": "ptv",
      "/api/dynamics/restaurants": "restaurants",
      "/api/dynamics/schools": "schools",
    };

    for (const path of [
      "/api/dynamics/ptv",
      "/api/dynamics/restaurants",
      "/api/dynamics/schools",
    ]) {
      const res = await fetch(path, { headers });
      const data = await safeJson(res);
      if (!res.ok) {
        all[labelMap[path]] = { error: data?.error || `HTTP ${res.status}` };
        console.warn(
          `Dynamics CE partial failure for ${path}:`,
          all[labelMap[path]]
        );
        continue;
      }
      all[labelMap[path]] = data;
    }

    // 3) Success UI state
    window.__dvData = all;
    console.log("Dynamics data loaded (CE):", all);
    ceLoaded = true;

    if (dvStatus) dvStatus.textContent = "Data loaded from Dynamics CE";
    const dot = document.getElementById("tokenStatus");
    if (dot) {
      dot.classList.add("token-dot--ok");
      dot.title = "OAuth token present and CE responded";
    }
    unlockComposer();
  } catch (err) {
    markCeFailed("Network error");
  }
}

// NEW: Reset button
const resetBtn = document.getElementById("resetBtn");

async function resetApp() {
  if (busy) return;
  setBusy(true);

  if (!ceLoaded) {
    // Keep the composer locked and show a nudge message
    addBotMessage(
      "Dynamics CE is not available. Open the Settings (gear) and paste a valid OAuth token, then hit the refresh button.",
      "eligibility"
    );
  } else {
    unlockComposer();
  }

  // Clear chat UI and typing bubble
  removeTyping();
  chat.innerHTML = "";

  // Reset state
  currentPhase = "eligibility";
  if (dvStatus) dvStatus.textContent = "Reloading Dynamics from CE…";
  // clear any cached CE data
  try {
    delete window.__dvData;
  } catch {}

  // Tell backend to reset its session (silently)
  try {
    const headers = { "Content-Type": "application/json" };
    const t = getToken && typeof getToken === "function" ? getToken() : "";
    if (t) headers["Authorization"] = `Bearer ${t}`;
    await fetch("/api/message", {
      method: "POST",
      headers,
      body: JSON.stringify({ message: "/reset" }),
    });
  } catch (e) {
    console.warn("Backend /reset failed (continuing):", e);
  }

  // Re-run the initializers you wanted
  try {
    updateTokenStatus();
    await fetchDynamicsHousingInfo();
  } catch (e) {
    console.warn("Reload sequence issue:", e);
  }

  // Re-add greeting
  addBotMessage(
    'Hi! Tell me about your income, debts, savings and budget (e.g., "I earn 110k, debts 450/mo, savings 40k; budget 800k in Brunswick"). Then use /next to see listings.',
    "eligibility"
  );

  setBusy(false);
  input.focus();
}

if (resetBtn) {
  resetBtn.addEventListener("click", resetApp);
}

function markCeFailed(reason) {
  ceLoaded = false;
  // CE status text
  if (dvStatus) dvStatus.textContent = "Not loaded from CE";
  // Force the dot red even if a token exists
  const dot = document.getElementById("tokenStatus");
  if (dot) {
    dot.classList.remove("token-dot--ok");
    dot.title = reason
      ? `Dynamics CE error: ${reason}`
      : "Dynamics CE not loaded";
  }
  // Lock the composer
  lockComposer(
    "Dynamics CE not loaded – open Settings and paste an OAuth token"
  );
  // Clear stale cache
  window.__dvData = null;
  console.warn("Dynamics CE failed:", reason);
}

async function safeJson(res) {
  try {
    return await res.json();
  } catch {
    return null;
  }
}

// ---------- Init ----------
updateTokenStatus();
fetchDynamicsHousingInfo(); // <-- NEW: do the call on page load
lockComposer("Loading Dynamics CE…");

// Greeting
addBotMessage(
  'Hi! Tell me about your income, debts, savings and budget (e.g., "I earn 110k, debts 450/mo, savings 40k; budget 800k in Brunswick"). Then use /next to see listings.',
  "eligibility"
);
