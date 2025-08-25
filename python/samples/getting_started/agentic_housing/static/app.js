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
let contactCreated = false; // NEW: to avoid duplicate creates
let profileComplete = false; // NEW: gate /next
let eligFormRow = null;
let eligFormRendered = false;

function lockEligibilityForm() {
  const form = document.getElementById("eligForm");
  if (!form) return;

  // Visually indicate locked state
  form.classList.add("is-locked");

  // Disable all fields inside the form (but not the post-form CTA button)
  Array.from(form.elements).forEach((el) => (el.disabled = true));

  // Tweak the Save button label
  const saveBtn = form.querySelector('button[type="submit"]');
  if (saveBtn) {
    saveBtn.disabled = true;
    saveBtn.textContent = "Saved";
  }
}

function getDvSuburbs() {
  const arr = window.__dvData?.properties?.value || [];
  const set = new Set(
    arr.map((r) => (r.cr54b_suburb || "").trim()).filter(Boolean)
  );
  return Array.from(set).sort((a, b) => a.localeCompare(b));
}

function enforceComposerLockByPhase() {
  // Lock only while we're in eligibility AND the form hasn't been completed yet.
  if (currentPhase === "eligibility" && !profileComplete) {
    lockComposer("Complete the eligibility form first");
  } else {
    unlockComposer();
  }
}
// programmatic sender that can bypass the composer lock
async function sendText(text, { overrideLock = false } = {}) {
  if (busy) return;
  if ((!ceLoaded || input.hasAttribute("readonly")) && !overrideLock) return;

  setBusy(true);
  addUserMessage(text);
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
      if (data.phaseClass) currentPhase = data.phaseClass;
      addBotMessage(data.reply, data.phaseClass);
      // unlock only once we reach Listings
      enforceComposerLockByPhase();
    }
    return data;
  } catch (err) {
    removeTyping();
    addBotMessage("Network error", "eligibility");
  } finally {
    setBusy(false);
  }
}

function parseMoney(s) {
  if (s == null) return NaN;
  return Number(
    String(s)
      .replace(/,/g, "")
      .replace(/[^\d.]/g, "")
  );
}

function uniqueSuburbsFromCE() {
  const vals = (window.__dvData?.properties?.value || []).map((v) =>
    (v.cr54b_suburb || "").trim()
  );
  return [...new Set(vals.filter(Boolean))].sort((a, b) => a.localeCompare(b));
}

function getCE() {
  return window.__dvData || {};
}

function ceSuburbOptions() {
  const dv = getCE();
  const seen = new Map(); // lower -> Canonical
  const add = (s) => {
    if (!s) return;
    let v = String(s).trim();
    if (!v) return;
    // normalize one known misspelling
    if (v.toLowerCase() === "glen waverley") v = "Glen Waverley";
    const key = v.toLowerCase();
    if (!seen.has(key)) seen.set(key, v);
  };

  dv.properties?.value?.forEach((r) => add(r.cr54b_suburb));
  dv.ptv?.value?.forEach((r) => add(r.cr54b_suburb));
  dv.restaurants?.value?.forEach((r) => add(r.cr54b_suburb));
  dv.schools?.value?.forEach((r) => add(r.cr54b_suburb));

  return Array.from(seen.values()).sort((a, b) => a.localeCompare(b));
}

async function submitEligibilityForm(e) {
  e.preventDefault();
  if (busy) return;

  const form = e.currentTarget;
  const hint = form.querySelector("#eligFormHint");

  // read + normalize
  const fd = new FormData(form);
  const payload = {
    firstname: (fd.get("firstname") || "").trim(),
    lastname: (fd.get("lastname") || "").trim(),
    cr54b_suburb: (fd.get("cr54b_suburb") || "").trim(),
    cr54b_target_price: parseMoney(fd.get("cr54b_target_price")),
    cr54b_income: parseMoney(fd.get("cr54b_income")),
    cr54b_debts: parseMoney(fd.get("cr54b_debts")),
    cr54b_expenses: parseMoney(fd.get("cr54b_expenses")),
    cr54b_savings: parseMoney(fd.get("cr54b_savings")),
    dependents: Number(fd.get("dependents") || 0),
  };

  // validate
  const missing = Object.entries(payload).filter(
    ([k, v]) => v === "" || v == null || (typeof v === "number" && !isFinite(v))
  );
  if (missing.length) {
    hint.textContent = "Please complete all fields with valid numbers.";
    return;
  }

  setBusy(true);
  hint.textContent = "Saving to Dynamics CE…";

  // auth header (use same token as other calls)
  const headers = {
    "Content-Type": "application/json",
    Accept: "application/json",
  };
  const t = getToken();
  if (t) headers["Authorization"] = `Bearer ${t}`;

  try {
    // 1) Create contact in Dataverse
    if (!contactCreated) {
      const resp = await fetch("/api/dynamics/contacts", {
        method: "POST",
        headers,
        body: JSON.stringify(payload),
      });
      const j = await resp.json().catch(() => ({}));
      if (!resp.ok || !j.ok) {
        hint.textContent = `Contact create failed: ${j.error || resp.status}`;
        setBusy(false);
        return;
      }
      contactCreated = true;
      if (dvStatus) dvStatus.textContent = "Contact created in Dynamics CE";
    }

    // 2) Seed the agent’s facts by sending a structured message
    const seedMsg = `My name is ${payload.firstname} ${payload.lastname}. I earn ${payload.cr54b_income} income, have ${payload.cr54b_debts}/mo debts and ${payload.cr54b_expenses}/mo expenses, savings ${payload.cr54b_savings}. Budget ${payload.cr54b_target_price} in ${payload.cr54b_suburb}. Dependents ${payload.dependents}.`;
    const res1 = await fetch("/api/message", {
      method: "POST",
      headers,
      body: JSON.stringify({ message: seedMsg }),
    });
    const d1 = await res1.json();
    if (res1.ok) addBotMessage(d1.reply, d1.phaseClass || "eligibility");

    // 3) Move to Listings
    const res2 = await fetch("/api/message", {
      method: "POST",
      headers,
      body: JSON.stringify({ message: "/next" }),
    });
    const d2 = await res2.json();
    if (res2.ok) {
      currentPhase = d2.phaseClass || "listings";
      addBotMessage(d2.reply, d2.phaseClass);
    }

    // 4) Unlock chat + remove form
    if (currentPhase !== "eligibility") {
      unlockComposer();
      if (eligFormRow) eligFormRow.remove();
    }
  } catch (err) {
    console.warn("Eligibility submit error:", err);
    hint.textContent = "Network error while saving.";
  } finally {
    setBusy(false);
  }
}

function renderEligibilityForm() {
  if (eligFormRendered) return;
  eligFormRendered = true;

  const suburbs = getDvSuburbs();
  const row = document.createElement("div");
  row.className = "row";
  row.innerHTML = `
    <div class="avatar">🤖</div>
    <div class="bubble eligibility elig-bubble">
      <div class="meta"><span class="badge eligibility">Eligibility</span></div>
      <div class="content">
        <form id="eligForm" class="elig-form" autocomplete="off" novalidate>
          <div class="row-2">
            <div class="field">
              <label>First name</label>
              <input id="ef_first" required placeholder="e.g., Alex" />
            </div>
            <div class="field">
              <label>Last name</label>
              <input id="ef_last" required placeholder="e.g., Nguyen" />
            </div>
          </div>

          <div class="field">
            <label>Suburb or city</label>
            <select id="ef_suburb" required>
              <option value="">Select a suburb</option>
              ${suburbs
                .map((s) => `<option value="${s}">${s}</option>`)
                .join("")}
            </select>
          </div>

          <div class="row-2">
            <div class="field">
              <label>Target purchase price (A$)</label>
              <input id="ef_target" inputmode="numeric" required placeholder="e.g., 750,000" required />
            </div>
            <div class="field">
              <label>Gross annual income (A$)</label>
              <input id="ef_income" inputmode="numeric" placeholder="e.g., 120,000" required />
            </div>
          </div>

          <div class="row-2">
            <div class="field">
              <label>Monthly debts (A$/mo)</label>
              <input id="ef_debts" inputmode="numeric" placeholder="e.g., 450" required />
            </div>
            <div class="field">
              <label>Monthly living expenses (A$/mo)</label>
              <input id="ef_expenses" inputmode="numeric" placeholder="e.g., 2,000" required />
            </div>
          </div>

          <div class="row-2">
            <div class="field">
              <label>Savings for deposit (A$)</label>
              <input id="ef_savings" inputmode="numeric" placeholder="e.g., 80,000" required />
            </div>
            <div class="field">
              <label>Dependents</label>
              <input id="ef_deps" inputmode="numeric" value="0" required placeholder="e.g., 0" required />
            </div>
          </div>

          <div class="actions">
            <button type="submit" class="btn btn--primary">Save details</button>
          </div>
          <div class="hint" id="eligStatus" aria-live="polite"></div>
        </form>
      </div>
    </div>`;
  chat.appendChild(row);
  chat.scrollTop = chat.scrollHeight;

  const $ = (id) => row.querySelector(id);
  const toNum = (v) => Number(String(v).replace(/[^\d.]/g, "")) || 0;

  const eligFormRow = row; // <-- add this

  $("form#eligForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const firstname = $("#ef_first").value.trim();
    const lastname = $("#ef_last").value.trim();
    const suburb = $("#ef_suburb").value.trim();

    const target = toNum($("#ef_target").value);
    const income = toNum($("#ef_income").value);
    const debts = toNum($("#ef_debts").value);
    const expenses = toNum($("#ef_expenses").value);
    const savings = toNum($("#ef_savings").value);
    const deps = toNum($("#ef_deps").value);

    const eligStatus = $("#eligStatus");

    if (
      !firstname ||
      !lastname ||
      !suburb ||
      !target ||
      !income ||
      !debts ||
      !expenses ||
      !savings
    ) {
      eligStatus.textContent = "Please complete all fields.";
      return;
    }

    // 1) Save to Dataverse
    try {
      const headers = {
        "Content-Type": "application/json",
        Accept: "application/json",
      };
      const t = getToken();
      if (t) headers["Authorization"] = `Bearer ${t}`;

      const payload = {
        firstname,
        lastname,
        cr54b_income: income,
        cr54b_debts: debts,
        cr54b_expense: expenses,
        cr54b_savings: savings,
        cr54b_target_price: target,
        cr54b_suburb: suburb,
      };

      const resp = await fetch("/api/dynamics/contacts", {
        method: "POST",
        headers,
        body: JSON.stringify(payload),
      });
      const j = await resp.json().catch(() => ({}));
      if (resp.ok && j.ok) {
        contactCreated = true;
        eligStatus.textContent = "Saved to Dynamics CE.";
        if (dvStatus) dvStatus.textContent = "Contact saved to Dynamics CE";
      } else {
        eligStatus.textContent = "Could not save to CE (continuing anyway).";
        console.warn("CE create failed", j);
      }
    } catch (err) {
      eligStatus.textContent =
        "Network error saving to CE (continuing anyway).";
      console.warn(err);
    }

    // 2) Auto-send a canonical message so the bot computes eligibility
    // Mark profile complete but stay in Eligibility
    profileComplete = true;
    currentPhase = "eligibility";
    enforceComposerLockByPhase(); // this will now UNLOCK the chat box

    lockEligibilityForm();

    // 3) Auto-send a single, canonical "My details are:" message so the Eligibility agent
    //    parses everything deterministically (includes "in <suburb>" to hit the suburb regex)
    const pretty = (n) => new Intl.NumberFormat("en-AU").format(Number(n));
    const detailsMsg =
      `My details are: My name is ${firstname} ${lastname}. ` +
      `I earn $${pretty(income)} income, have $${pretty(
        debts
      )}/mo debts and $${pretty(expenses)}/mo expenses, ` +
      `savings $${pretty(savings)}. Budget $${pretty(
        target
      )} in ${suburb}. Dependents ${deps}.`;

    await sendText(detailsMsg, { overrideLock: true });

    // Offer both options: user can type /next OR click a button
    //     const actionBar = document.createElement("div");
    //     actionBar.className = "elig-nextbar";
    //     actionBar.innerHTML = `
    //   <div class="hint" style="margin-right:auto">Details saved. Type <b>/next</b> to see listings, or use the button.</div>
    //   <button type="button" class="btn btn--primary" id="goListings">Continue to listings</button>
    // `;
    //     row.querySelector(".content").appendChild(actionBar);
    //     row.querySelector("#goListings").addEventListener("click", async () => {
    //       const data = await sendText("/next", { overrideLock: true });
    //       if (data?.phaseClass === "listings") {
    //         currentPhase = "listings";
    //         unlockComposer();
    //       }
    //     });
  });
}

function showFormError(msg) {
  const el = document.getElementById("eligFormError");
  if (el) el.textContent = msg || "";
}

function disableEligibilityForm() {
  const form = document.getElementById("eligForm");
  if (!form) return;
  Array.from(form.elements).forEach((el) => (el.disabled = true));
}

function toNumber(v) {
  const s = String(v || "").replace(/[,\s$]/g, "");
  const n = Number(s);
  return Number.isFinite(n) ? n : NaN;
}

async function handleEligibilitySubmit(e) {
  e.preventDefault();
  const form = e.currentTarget;
  showFormError("");

  // Gather + validate
  const payload = Object.fromEntries(new FormData(form).entries());
  const numericKeys = [
    "target_price",
    "gross_annual_income",
    "monthly_debts",
    "monthly_expenses",
    "savings",
    "dependents",
  ];
  for (const k of numericKeys) {
    const n = toNumber(payload[k]);
    if (!Number.isFinite(n)) {
      showFormError(`"${k.replaceAll("_", " ")}" must be a number`);
      return;
    }
    payload[k] = n;
  }
  if (!payload.first_name || !payload.last_name || !payload.location_hint) {
    showFormError("Please fill all fields");
    return;
  }

  // POST to the new seed endpoint
  try {
    const headers = {
      "Content-Type": "application/json",
      Accept: "application/json",
    };
    const t = getToken();
    if (t) headers["Authorization"] = `Bearer ${t}`;

    const res = await fetch("/api/session/seed", {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
    });
    const j = await res.json().catch(() => ({}));

    if (!res.ok || !j.ok) {
      showFormError(j.error || `Save failed (HTTP ${res.status})`);
      return;
    }

    // Lock the form, unlock composer, gate passes
    disableEligibilityForm();
    profileComplete = true;

    addBotMessage(
      `Thanks, ${payload.first_name}. I’ve saved your details. When you’re ready, type /next to see listings for ${payload.location_hint}.`,
      "eligibility"
    );
  } catch (err) {
    showFormError("Network error while saving");
  }
}

function hasMinProfile(facts) {
  const req = [
    "first_name",
    "last_name",
    "location_hint",
    "target_price",
    "gross_annual_income",
    "monthly_debts",
    "monthly_expenses",
    "savings",
    "dependents",
  ];
  return req.every(
    (k) =>
      facts && facts[k] !== null && facts[k] !== undefined && facts[k] !== ""
  );
}

function mapFactsToContactPayload(f) {
  return {
    firstname: f.first_name,
    lastname: f.last_name,
    cr54b_suburb: f.location_hint,
    cr54b_target_price: Number(f.target_price),
    cr54b_income: Number(f.gross_annual_income),
    cr54b_debts: Number(f.monthly_debts),
    cr54b_expense: Number(f.monthly_expenses),
    cr54b_savings: Number(f.savings),
  };
}
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
  if (!ceLoaded) return; // no CE → no send

  const text = input.value.trim();
  if (!text) return;

  // In Eligibility, block free chat until the form is completed
  if (currentPhase === "eligibility" && !profileComplete) {
    lockComposer("Complete the eligibility form first");
    return;
  }

  // clear the box and delegate to the canonical sender
  input.value = "";
  await sendText(text); // <-- this already handles UI + POST
  input.focus();
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

    // After CE snapshots load, push them into the agent module
    try {
      const syncHeaders = { Accept: "application/json" };
      const t = getToken?.() || "";
      if (t) syncHeaders["Authorization"] = `Bearer ${t}`;

      const syncRes = await fetch("/api/dynamics/sync?props=50&support=100", {
        method: "POST",
        headers: syncHeaders,
      });
      const syncJson = await syncRes.json().catch(() => ({}));
      console.log("Agent sync:", syncJson);
      if (!syncRes.ok || !syncJson.ok) {
        console.warn(
          "⚠️ Agent sync failed; listings will be empty until fixed."
        );
      }
    } catch (e) {
      console.warn("⚠️ Agent sync call errored", e);
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

    // NEW: show the eligibility form immediately and keep chat disabled
    renderEligibilityForm();
    currentPhase = "eligibility";
    enforceComposerLockByPhase();
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

  eligFormRendered = false;
  renderEligibilityForm();
  currentPhase = "eligibility";
  enforceComposerLockByPhase();

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
    if (currentPhase === "eligibility" || !contactCreated) {
      renderEligibilityForm();
    }
    profileComplete = false;
    contactCreated = false;
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
