const state = {
  sessionId: null,
  requestId: null,
  lastSequence: -1,
  source: null,
  busy: false,
  authConfig: null,
  authClient: null,
  user: null,
  conversations: JSON.parse(localStorage.getItem("easyteaching-conversations") || "[]"),
};

const LOCAL_DEMO_SCOPE = {
  teacher_id: "teacher-001",
  class_id: "kangaroo-room",
};

const ui = {
  composer: document.querySelector("#composer"),
  input: document.querySelector("#message-input"),
  send: document.querySelector("#send-button"),
  attach: document.querySelector("#attach-button"),
  fileInput: document.querySelector("#file-input"),
  messages: document.querySelector("#messages"),
  welcome: document.querySelector("#welcome"),
  chatScroll: document.querySelector("#chat-scroll"),
  connection: document.querySelector("#connection-state"),
  conversations: document.querySelector("#conversation-list"),
  sidebar: document.querySelector("#sidebar"),
  toast: document.querySelector("#toast"),
  loginScreen: document.querySelector("#login-screen"),
  loginForm: document.querySelector("#login-form"),
  loginEmail: document.querySelector("#login-email"),
  loginPassword: document.querySelector("#login-password"),
  loginButton: document.querySelector("#login-button"),
  loginError: document.querySelector("#login-error"),
  userMenu: document.querySelector("#user-menu"),
  userEmail: document.querySelector("#user-email"),
  logout: document.querySelector("#logout-button"),
};

function escapeHtml(value = "") {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function paragraphs(value = "") {
  return value
    .split(/\n{2,}/)
    .filter(Boolean)
    .map((part) => `<p>${escapeHtml(part).replaceAll("\n", "<br>")}</p>`)
    .join("");
}

function setStatus(label, kind = "ready") {
  ui.connection.className = `connection-state ${kind}`;
  ui.connection.innerHTML = `<span class="status-dot"></span>${escapeHtml(label)}`;
}

function setBusy(busy) {
  state.busy = busy;
  ui.input.disabled = busy;
  ui.attach.disabled = busy;
  ui.send.disabled = busy || !ui.input.value.trim();
  setStatus(busy ? "Working" : "Ready", busy ? "busy" : "ready");
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    ui.chatScroll.scrollTop = ui.chatScroll.scrollHeight;
  });
}

function showToast(message) {
  ui.toast.textContent = message;
  ui.toast.classList.add("show");
  window.setTimeout(() => ui.toast.classList.remove("show"), 2600);
}

function saveConversations() {
  localStorage.setItem("easyteaching-conversations", JSON.stringify(state.conversations.slice(0, 8)));
  renderConversationList();
}

function renderConversationList() {
  ui.conversations.innerHTML = state.conversations.length
    ? state.conversations.map((item) => `
      <button class="conversation-item ${item.sessionId === state.sessionId ? "active" : ""}"
        data-session-id="${escapeHtml(item.sessionId)}"
        data-request-id="${escapeHtml(item.requestId || "")}">${escapeHtml(item.title)}</button>`).join("")
    : '<div class="conversation-item">No conversations yet</div>';
}

function rememberConversation(prompt) {
  const title = prompt.length > 42 ? `${prompt.slice(0, 42)}…` : prompt;
  state.conversations = state.conversations.filter((item) => item.sessionId !== state.sessionId);
  state.conversations.unshift({
    sessionId: state.sessionId,
    requestId: state.requestId,
    prompt,
    title,
  });
  saveConversations();
}

function addUserMessage(message) {
  const article = document.createElement("article");
  article.className = "message user";
  article.innerHTML = `<div class="bubble">${escapeHtml(message)}</div>`;
  ui.messages.appendChild(article);
  scrollToBottom();
}

function addAssistantShell() {
  const article = document.createElement("article");
  article.className = "message assistant";
  article.innerHTML = `
    <div class="avatar">E</div>
    <div class="bubble">
      <div class="thinking"><span class="thinking-dots"><i></i><i></i><i></i></span><span>Understanding your request…</span></div>
      <details class="trace-panel">
        <summary>View agent activity</summary>
        <ol class="trace-list"></ol>
      </details>
      <div class="assistant-result"></div>
    </div>`;
  ui.messages.appendChild(article);
  scrollToBottom();
  return article;
}

function addTrace(shell, event) {
  const list = shell.querySelector(".trace-list");
  const item = document.createElement("li");
  const data = event.data || {};
  item.textContent = data.message || data.step || data.phase || event.event.replaceAll("_", " ");
  list.appendChild(item);
  const thinking = shell.querySelector(".thinking span:last-child");
  if (thinking) {
    thinking.textContent = data.message || "EasyTeaching is working…";
  }
}

function citationsHtml(citations = []) {
  if (!citations.length) return "";
  return `<div class="citation-list">${citations.map((citation, index) => {
    const label = citation.title || citation.source || `Source ${index + 1}`;
    const page = citation.page ? ` · page ${citation.page}` : "";
    const content = `<strong>[${index + 1}] ${escapeHtml(label)}</strong>${escapeHtml(page)}`;
    return citation.url
      ? `<a class="citation" href="${escapeHtml(citation.url)}" target="_blank" rel="noreferrer">${content}</a>`
      : `<div class="citation">${content}</div>`;
  }).join("")}</div>`;
}

function fieldLabel(value = "") {
  return String(value)
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function previewValue(value) {
  if (value === null || value === undefined || value === "") return "Not provided";
  if (Array.isArray(value)) return value.map((item) => escapeHtml(item)).join(", ");
  if (typeof value === "object") {
    return `<pre>${escapeHtml(JSON.stringify(value, null, 2))}</pre>`;
  }
  return escapeHtml(value);
}

function approvalHtml(approval = {}) {
  if (!approval || approval.status === "not_required") return "";
  const status = approval.status || "required";
  const rows = Object.entries(approval.preview || {}).map(([key, value]) => `
    <div class="approval-field">
      <dt>${escapeHtml(fieldLabel(key))}</dt>
      <dd>${previewValue(value)}</dd>
    </div>`).join("");

  if (status === "required") {
    return `
      <section class="approval-card" aria-live="polite">
        <div class="approval-heading">
          <span class="approval-icon" aria-hidden="true">✓</span>
          <div>
            <p class="approval-eyebrow">YOUR APPROVAL IS REQUIRED</p>
            <h3>Review before saving</h3>
          </div>
        </div>
        <p class="approval-copy">Nothing has been written yet. Check every field, then approve or reject this action.</p>
        <dl class="approval-preview">${rows || '<div class="approval-empty">No preview fields were supplied.</div>'}</dl>
        <div class="approval-actions">
          <button class="approval-button reject" type="button" data-approval-decision="reject">Reject</button>
          <button class="approval-button approve" type="button" data-approval-decision="approve">Approve and save</button>
        </div>
        <p class="approval-progress" role="status"></p>
      </section>`;
  }

  const labels = {
    approved: "Approved and completed",
    rejected: "Rejected — nothing was saved",
    failed: "The approved action could not be completed",
  };
  return `<div class="approval-outcome ${escapeHtml(status)}">${escapeHtml(labels[status] || fieldLabel(status))}</div>`;
}

function renderDraft(shell, payload) {
  shell.querySelector(".thinking")?.remove();
  shell.dataset.sessionId = payload.session_id;
  shell.dataset.requestId = payload.request_id;
  const result = shell.querySelector(".assistant-result");
  result.innerHTML = `
    ${payload.draft.is_draft ? '<span class="draft-badge">DRAFT · REVIEW BEFORE USE</span>' : ""}
    <h2 class="message-title">${escapeHtml(payload.draft.title)}</h2>
    <div class="assistant-copy">${paragraphs(payload.draft.content)}</div>
    ${citationsHtml(payload.citations)}
    ${approvalHtml(payload.approval)}`;

  scrollToBottom();
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    if (response.status === 401 && state.authConfig?.enabled) showLogin();
    throw new Error(payload.error?.message || payload.detail || `Request failed (${response.status})`);
  }
  return payload;
}

function showLogin(message = "") {
  state.source?.close();
  state.user = null;
  ui.userMenu.hidden = true;
  ui.loginError.textContent = message;
  ui.loginScreen.hidden = false;
  ui.loginEmail.focus();
}

function showAuthenticatedApp(user) {
  state.user = user;
  ui.loginScreen.hidden = true;
  ui.userEmail.textContent = user.email || "Teacher";
  ui.userMenu.hidden = false;
}

async function loadSupabaseSdk() {
  if (window.supabase) return;
  await new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = "https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2";
    script.onload = resolve;
    script.onerror = () => reject(new Error("Supabase login library could not be loaded."));
    document.head.appendChild(script);
  });
}

async function initializeAuth() {
  try {
    const config = await api("/auth/config");
    state.authConfig = config;
    if (!config.enabled) return;
    await loadSupabaseSdk();
    state.authClient = window.supabase.createClient(
      config.supabase_url,
      config.supabase_publishable_key,
      { auth: { persistSession: false, autoRefreshToken: false, detectSessionInUrl: false } },
    );
    try {
      const user = await api("/auth/me");
      showAuthenticatedApp(user);
    } catch (_error) {
      showLogin();
    }
  } catch (error) {
    showToast(error.message);
  }
}

async function submitLogin(event) {
  event.preventDefault();
  if (!state.authClient) return;
  ui.loginButton.disabled = true;
  ui.loginError.textContent = "";
  try {
    const { data, error } = await state.authClient.auth.signInWithPassword({
      email: ui.loginEmail.value.trim(),
      password: ui.loginPassword.value,
    });
    if (error) throw error;
    const user = await api("/auth/session", {
      method: "POST",
      body: JSON.stringify({ access_token: data.session.access_token }),
    });
    ui.loginPassword.value = "";
    resetConversation();
    showAuthenticatedApp(user);
  } catch (error) {
    showLogin(error.message || "Sign in failed.");
  } finally {
    ui.loginButton.disabled = false;
  }
}

async function logout() {
  state.source?.close();
  try {
    await api("/auth/session", { method: "DELETE" });
  } catch (_error) {
    // The local login screen still closes access if the remote session is unavailable.
  }
  state.conversations = [];
  localStorage.removeItem("easyteaching-conversations");
  resetConversation();
  showLogin();
}

async function ensureSession() {
  if (state.sessionId) return state.sessionId;
  const sessionScope = state.authConfig?.enabled ? {} : LOCAL_DEMO_SCOPE;
  const session = await api("/sessions", {
    method: "POST",
    body: JSON.stringify(sessionScope),
  });
  state.sessionId = session.session_id;
  return state.sessionId;
}

async function uploadFile(file) {
  const sessionId = await ensureSession();
  const body = new FormData();
  body.append("file", file);
  const response = await fetch(`/sessions/${sessionId}/uploads`, { method: "POST", body });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error?.message || `Upload failed (${response.status})`);
  return payload;
}

async function getDraft(shell, sessionId = state.sessionId, requestId = state.requestId) {
  try {
    const payload = await api(`/sessions/${sessionId}/drafts/${requestId}`);
    renderDraft(shell, payload);
    return payload;
  } catch (error) {
    const thinking = shell.querySelector(".thinking span:last-child");
    if (thinking) thinking.textContent = "The draft is still being prepared…";
  }
}

function connectEvents(shell, afterSequence = state.lastSequence) {
  state.source?.close();
  const url = `/sessions/${state.sessionId}/events?request_id=${encodeURIComponent(state.requestId)}&after_sequence=${afterSequence}`;
  const source = new EventSource(url);
  shell.dataset.sessionId = state.sessionId;
  shell.dataset.requestId = state.requestId;
  state.source = source;
  let lastSequence = afterSequence;

  const handle = (event) => {
    const payload = JSON.parse(event.data);
    lastSequence = payload.sequence;
    state.lastSequence = payload.sequence;
    addTrace(shell, payload);
    if (payload.event === "draft_ready") getDraft(shell, shell.dataset.sessionId, shell.dataset.requestId);
    if (payload.event === "approval_required") {
      source.close();
      getDraft(shell, shell.dataset.sessionId, shell.dataset.requestId);
      setBusy(false);
      setStatus("Review required", "review");
    }
    if (["completed", "failed", "cancelled"].includes(payload.event)) {
      source.close();
      if (payload.event === "completed") getDraft(shell, shell.dataset.sessionId, shell.dataset.requestId);
      if (payload.event === "failed") showRunError(shell, "EasyTeaching could not complete this draft. Please try again.");
      setBusy(false);
    }
  };

  ["run_started", "route_selected", "trace", "draft_ready", "approval_required", "completed", "failed", "cancelled"]
    .forEach((name) => source.addEventListener(name, handle));

  source.onerror = () => {
    source.close();
    if (state.busy) {
      window.setTimeout(() => connectEvents(shell, lastSequence), 700);
    }
  };
}

async function submitApproval(shell, decision) {
  const sessionId = shell.dataset.sessionId;
  const requestId = shell.dataset.requestId;
  const card = shell.querySelector(".approval-card");
  const progress = card?.querySelector(".approval-progress");
  const buttons = card?.querySelectorAll("[data-approval-decision]") || [];
  if (!sessionId || !requestId || !card) return;

  buttons.forEach((button) => { button.disabled = true; });
  if (progress) progress.textContent = decision === "approve" ? "Saving approved fields…" : "Rejecting this action…";
  setStatus(decision === "approve" ? "Saving" : "Rejecting", "busy");
  try {
    await api(`/sessions/${sessionId}/approvals`, {
      method: "POST",
      body: JSON.stringify({ request_id: requestId, decision }),
    });
    await getDraft(shell, sessionId, requestId);
    showToast(decision === "approve" ? "Approved action completed." : "Action rejected. Nothing was saved.");
    setStatus("Ready", "ready");
  } catch (error) {
    buttons.forEach((button) => { button.disabled = false; });
    if (progress) progress.textContent = error.message;
    setStatus("Needs attention", "error");
  }
}

function showRunError(shell, message) {
  shell.querySelector(".thinking")?.remove();
  shell.querySelector(".assistant-result").innerHTML = `<div class="assistant-copy"><p>${escapeHtml(message)}</p></div>`;
  setBusy(false);
  setStatus("Needs attention", "error");
}

async function submitMessage(message) {
  ui.welcome.hidden = true;
  addUserMessage(message);
  const shell = addAssistantShell();
  setBusy(true);
  try {
    await ensureSession();
    state.requestId = crypto.randomUUID();
    state.lastSequence = -1;
    rememberConversation(message);
    await api(`/sessions/${state.sessionId}/messages`, {
      method: "POST",
      body: JSON.stringify({ message, request_id: state.requestId }),
    });
    connectEvents(shell, state.lastSequence);
  } catch (error) {
    showRunError(shell, error.message);
  }
}

function resetConversation() {
  state.source?.close();
  state.sessionId = null;
  state.requestId = null;
  state.lastSequence = -1;
  ui.messages.innerHTML = "";
  ui.welcome.hidden = false;
  ui.input.value = "";
  setBusy(false);
  renderConversationList();
  ui.sidebar.classList.remove("open");
  ui.input.focus();
}

ui.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  const message = ui.input.value.trim();
  if (!message || state.busy) return;
  ui.input.value = "";
  ui.input.style.height = "auto";
  submitMessage(message);
});

ui.input.addEventListener("input", () => {
  ui.input.style.height = "auto";
  ui.input.style.height = `${Math.min(ui.input.scrollHeight, 180)}px`;
  ui.send.disabled = state.busy || !ui.input.value.trim();
});

ui.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    ui.composer.requestSubmit();
  }
});

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    ui.input.value = button.dataset.prompt;
    ui.input.dispatchEvent(new Event("input"));
    ui.input.focus();
  });
});

document.querySelector("#new-chat").addEventListener("click", resetConversation);
document.querySelector("#open-sidebar").addEventListener("click", () => ui.sidebar.classList.add("open"));
document.querySelector("#close-sidebar").addEventListener("click", () => ui.sidebar.classList.remove("open"));
ui.loginForm.addEventListener("submit", submitLogin);
ui.logout.addEventListener("click", logout);

ui.conversations.addEventListener("click", (event) => {
  const sessionId = event.target.dataset.sessionId;
  if (!sessionId || sessionId === state.sessionId) return;
  const remembered = state.conversations.find((item) => item.sessionId === sessionId);
  state.sessionId = sessionId;
  state.requestId = event.target.dataset.requestId || remembered?.requestId || null;
  state.lastSequence = -1;
  ui.messages.innerHTML = "";
  ui.welcome.hidden = Boolean(state.requestId);
  renderConversationList();
  if (remembered?.prompt) addUserMessage(remembered.prompt);
  if (state.requestId) {
    const shell = addAssistantShell();
    getDraft(shell, state.sessionId, state.requestId).then((payload) => {
      if (payload?.approval?.status === "required") {
        setStatus("Review required", "review");
      }
    });
  } else {
    showToast("Session restored. Send a message to continue its LangGraph thread.");
  }
  ui.sidebar.classList.remove("open");
});

ui.attach.addEventListener("click", () => {
  if (!state.busy) ui.fileInput.click();
});

ui.fileInput.addEventListener("change", async () => {
  const file = ui.fileInput.files?.[0];
  if (!file) return;
  ui.attach.disabled = true;
  setStatus("Uploading", "busy");
  try {
    const uploaded = await uploadFile(file);
    const reference = `[Uploaded ${uploaded.category}: ${uploaded.filename}; file_id: ${uploaded.file_id}]`;
    ui.input.value = `${ui.input.value.trim()}${ui.input.value.trim() ? "\n" : ""}${reference}`;
    ui.input.dispatchEvent(new Event("input"));
    showToast(`${uploaded.filename} is ready for this conversation.`);
  } catch (error) {
    showToast(error.message);
  } finally {
    ui.fileInput.value = "";
    ui.attach.disabled = state.busy;
    setStatus(state.busy ? "Working" : "Ready", state.busy ? "busy" : "ready");
  }
});

ui.messages.addEventListener("click", (event) => {
  const button = event.target.closest("[data-approval-decision]");
  if (!button || button.disabled) return;
  const shell = button.closest(".message.assistant");
  if (!shell) return;
  submitApproval(shell, button.dataset.approvalDecision);
});

renderConversationList();
setBusy(false);
initializeAuth();
