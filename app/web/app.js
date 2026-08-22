const state = {
  sessionId: null,
  requestId: null,
  lastSequence: -1,
  source: null,
  busy: false,
  authConfig: null,
  authClient: null,
  user: null,
  conversations: JSON.parse(localStorage.getItem("eduflow-conversations") || "[]"),
};

const ui = {
  composer: document.querySelector("#composer"),
  input: document.querySelector("#message-input"),
  send: document.querySelector("#send-button"),
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
  localStorage.setItem("eduflow-conversations", JSON.stringify(state.conversations.slice(0, 8)));
  renderConversationList();
}

function renderConversationList() {
  ui.conversations.innerHTML = state.conversations.length
    ? state.conversations.map((item) => `
      <button class="conversation-item ${item.sessionId === state.sessionId ? "active" : ""}"
        data-session-id="${escapeHtml(item.sessionId)}">${escapeHtml(item.title)}</button>`).join("")
    : '<div class="conversation-item">No conversations yet</div>';
}

function rememberConversation(prompt) {
  const title = prompt.length > 42 ? `${prompt.slice(0, 42)}…` : prompt;
  state.conversations = state.conversations.filter((item) => item.sessionId !== state.sessionId);
  state.conversations.unshift({ sessionId: state.sessionId, title });
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
        <summary>View workflow activity</summary>
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
    thinking.textContent = data.message || "Working through the educator workflow…";
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

function renderDraft(shell, payload) {
  shell.querySelector(".thinking")?.remove();
  const result = shell.querySelector(".assistant-result");
  result.innerHTML = `
    ${payload.draft.is_draft ? '<span class="draft-badge">DRAFT · REVIEW BEFORE USE</span>' : ""}
    <h2 class="message-title">${escapeHtml(payload.draft.title)}</h2>
    <div class="assistant-copy">${paragraphs(payload.draft.content)}</div>
    ${citationsHtml(payload.citations)}`;

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
  localStorage.removeItem("eduflow-conversations");
  resetConversation();
  showLogin();
}

async function ensureSession() {
  if (state.sessionId) return state.sessionId;
  const session = await api("/sessions", { method: "POST", body: "{}" });
  state.sessionId = session.session_id;
  return state.sessionId;
}

async function getDraft(shell) {
  try {
    const payload = await api(`/sessions/${state.sessionId}/drafts/${state.requestId}`);
    renderDraft(shell, payload);
  } catch (error) {
    const thinking = shell.querySelector(".thinking span:last-child");
    if (thinking) thinking.textContent = "The draft is still being prepared…";
  }
}

function connectEvents(shell, afterSequence = state.lastSequence) {
  state.source?.close();
  const url = `/sessions/${state.sessionId}/events?request_id=${encodeURIComponent(state.requestId)}&after_sequence=${afterSequence}`;
  const source = new EventSource(url);
  state.source = source;
  let lastSequence = afterSequence;

  const handle = (event) => {
    const payload = JSON.parse(event.data);
    lastSequence = payload.sequence;
    state.lastSequence = payload.sequence;
    addTrace(shell, payload);
    if (payload.event === "draft_ready") getDraft(shell);
    if (["completed", "failed", "cancelled"].includes(payload.event)) {
      source.close();
      if (payload.event === "completed") getDraft(shell);
      if (payload.event === "failed") showRunError(shell, "EasyTeaching could not complete this draft. Please try again.");
      setBusy(false);
    }
  };

  ["run_started", "route_selected", "trace", "draft_ready", "completed", "failed", "cancelled"]
    .forEach((name) => source.addEventListener(name, handle));

  source.onerror = () => {
    source.close();
    if (state.busy) {
      window.setTimeout(() => connectEvents(shell, lastSequence), 700);
    }
  };
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
  state.sessionId = sessionId;
  state.requestId = null;
  state.lastSequence = -1;
  ui.messages.innerHTML = "";
  ui.welcome.hidden = false;
  renderConversationList();
  showToast("Session restored. Send a message to continue its LangGraph thread.");
  ui.sidebar.classList.remove("open");
});

renderConversationList();
setBusy(false);
initializeAuth();
