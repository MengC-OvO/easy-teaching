const state = {
  sessionId: null,
  requestId: null,
  lastSequence: -1,
  lastEventId: null,
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
  // Escape first: model text cannot inject HTML. Render a small Markdown subset.
  const inline = (text) => escapeHtml(text).replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>").replace(/`([^`]+)`/g, "<code>$1</code>");
  return value.split(/\n{2,}/).filter(Boolean).map((block) => {
    const lines = block.split("\n");
    if (lines.every((line) => /^\s*[-*]\s+/.test(line))) return `<ul>${lines.map((line) => `<li>${inline(line.replace(/^\s*[-*]\s+/, ""))}</li>`).join("")}</ul>`;
    if (lines.every((line) => /^\s*\d+[.)]\s+/.test(line))) return `<ol>${lines.map((line) => `<li>${inline(line.replace(/^\s*\d+[.)]\s+/, ""))}</li>`).join("")}</ol>`;
    if (lines.length === 1 && /^#{1,6}\s+/.test(block)) return `<h3>${inline(block.replace(/^#{1,6}\s+/, ""))}</h3>`;
    return `<p>${lines.map(inline).join("<br>")}</p>`;
  }).join("");
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
      <div class="thinking"><span class="thinking-dots"><i></i><i></i><i></i></span><span>正在理解你的请求…</span></div>
      <details class="trace-panel">
        <summary>查看执行过程</summary>
        <ol class="trace-list"></ol>
      </details>
      <div class="assistant-result"></div>
    </div>`;
  ui.messages.appendChild(article);
  scrollToBottom();
  return article;
}

function addTrace(shell, event) {
  const messages = {
    initialize: "正在准备上下文…", main_react: "正在分析下一步…",
    merge_observations: "正在整理查询结果…", context_update: "正在更新对话上下文…",
    long_memory_update: "正在完成收尾…", finalize_draft: "正在准备回答…",
    run_started: "请求已接收…", completed: "处理完成", draft_ready: "正在输出回答…",
    prepare_approval: "正在准备审批预览…", decision_feedback: "正在调整执行方式…",
  };
  const list = shell.querySelector(".trace-list");
  const item = document.createElement("li");
  const data = event.data || {};
  const message = messages[data.step] || messages[event.event] || data.message || data.step || "正在处理…";
  item.textContent = message;
  list.appendChild(item);
  const thinking = shell.querySelector(".thinking span:last-child");
  if (thinking) {
    thinking.textContent = message;
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
  if (approval.result?.execution_status === "queued") {
    return '<div class="approval-outcome">Approved — waiting for background execution.</div>';
  }
  if (approval.result?.error?.code === "action_outcome_unknown") {
    return '<div class="approval-outcome failed">Execution result needs verification. Do not repeat the action until its previous result has been checked.</div>';
  }
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

function streamAnswer(shell, sessionId, requestId) {
  if (shell._answerPromise) return shell._answerPromise;
  shell._answerPromise = new Promise((resolve) => {
    const source = new EventSource(`/sessions/${sessionId}/drafts/${requestId}/stream`);
    shell._answerSource = source;
    let text = "", offset = 0, metadata = null, failures = 0;
    let displayed = 0, finished = false, frame = null, previousTime = null, credit = 0;
    const characters = [];
    // Network chunks may arrive together. Keep a visible render queue and never
    // replace it with the full answer merely because answer_done arrived.
    const paint = (now) => {
      frame = null;
      if (!shell.isConnected) { close(); resolve(); return; }
      const elapsed = previousTime === null ? 16 : Math.min(now - previousTime, 80);
      previousTime = now;
      credit += elapsed * 0.05; // 50 Unicode code points per second.
      const count = Math.floor(credit);
      const end = Math.min(displayed + count, characters.length);
      credit = end === characters.length ? 0 : credit - count;
      if (end > displayed) {
        displayed = end;
        const copy = shell.querySelector(".assistant-copy");
        if (copy) copy.textContent = characters.slice(0, displayed).join("");
        scrollToBottom();
      }
      if (finished && displayed === characters.length) {
        if (metadata) renderDraft(shell, {...metadata, draft: {...metadata.draft, content: text}});
        resolve(metadata);
        return;
      }
      if (displayed < characters.length) frame = requestAnimationFrame(paint);
      else previousTime = null;
    };
    const schedulePaint = () => { if (frame === null) frame = requestAnimationFrame(paint); };
    const close = () => { source.close(); shell._answerSource = null; };
    source.addEventListener("answer_start", (event) => {
      metadata = JSON.parse(event.data);
      if (!shell.isConnected) { close(); resolve(); return; }
      if (!offset) renderDraft(shell, metadata);
      shell.querySelector(".assistant-copy")?.classList.add("streaming-answer");
      setStatus("正在输出", "busy");
    });
    source.addEventListener("answer_delta", (event) => {
      if (!shell.isConnected) { close(); resolve(); return; }
      const data = JSON.parse(event.data);
      if (data.offset <= offset) return;
      text += data.text;
      characters.push(...Array.from(data.text));
      offset = data.offset;
      failures = 0;
      schedulePaint();
    });
    source.addEventListener("answer_done", () => {
      close();
      finished = true;
      schedulePaint();
    });
    source.onerror = async () => {
      if (!shell.isConnected) { close(); resolve(); return; }
      if (++failures < 3) return;
      close();
      try {
        const payload = await api(`/sessions/${sessionId}/drafts/${requestId}`);
        if (!shell.isConnected) { resolve(); return; }
        const fullText = payload.draft.content;
        const visibleText = characters.slice(0, displayed).join("");
        if (!metadata || !fullText.startsWith(visibleText)) {
          displayed = 0;
          renderDraft(shell, {...payload, draft: {...payload.draft, content: ""}});
        }
        metadata = payload;
        text = fullText;
        characters.length = 0;
        for (const character of fullText) characters.push(character);
        shell.querySelector(".assistant-copy")?.classList.add("streaming-answer");
        finished = true;
        schedulePaint();
      } catch (error) {
        if (frame !== null) cancelAnimationFrame(frame);
        if (shell.isConnected) showRunError(shell, error.message);
        resolve();
      }
    };
  });
  return shell._answerPromise;
}

function connectEvents(shell, afterEventId = state.lastEventId) {
  state.source?.close();
  if (!shell.isConnected) return;
  const cursor = afterEventId ? `&after_event_id=${encodeURIComponent(afterEventId)}` : "";
  const url = `/sessions/${shell.dataset.sessionId || state.sessionId}/events?request_id=${encodeURIComponent(shell.dataset.requestId || state.requestId)}${cursor}&after_sequence=${state.lastSequence ?? -1}`;
  const source = new EventSource(url);
  shell.dataset.sessionId = shell.dataset.sessionId || state.sessionId;
  shell.dataset.requestId = shell.dataset.requestId || state.requestId;
  state.source = source;
  let lastEventId = afterEventId, ended = false;

  const handle = async (event) => {
    if (!shell.isConnected) { source.close(); return; }
    const payload = JSON.parse(event.data);
    lastEventId = event.lastEventId || payload.event_id || lastEventId;
    state.lastEventId = lastEventId;
    state.lastSequence = payload.sequence;
    addTrace(shell, payload);
    if (payload.event === "draft_ready") streamAnswer(shell, shell.dataset.sessionId, shell.dataset.requestId);
    if (payload.event === "approval_required") {
      ended = true;
      source.close();
      await shell._answerPromise;
      if (!shell.isConnected) return;
      await getDraft(shell, shell.dataset.sessionId, shell.dataset.requestId);
      setBusy(false);
      setStatus("Review required", "review");
    }
    if (["completed", "failed", "cancelled"].includes(payload.event)) {
      ended = true;
      source.close();
      if (payload.event === "completed") {
        await streamAnswer(shell, shell.dataset.sessionId, shell.dataset.requestId);
        if (shell.isConnected) setStatus("Ready", "ready");
      }
      if (payload.event === "failed") showRunError(shell, "EasyTeaching could not complete this draft. Please try again.");
      if (shell.isConnected) setBusy(false);
    }
  };

  ["run_started", "route_selected", "trace", "draft_ready", "approval_required", "completed", "failed", "cancelled"]
    .forEach((name) => source.addEventListener(name, handle));

  source.onerror = () => {
    source.close();
    if (!ended && state.busy && shell.isConnected) {
      window.setTimeout(() => connectEvents(shell, lastEventId), 700);
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
    const admission = await api(`/sessions/${sessionId}/approvals`, {
      method: "POST",
      body: JSON.stringify({ request_id: requestId, decision }),
    });
    if (admission.status === "running") {
      setBusy(true);
      if (progress) progress.textContent = "Approved. Waiting for background execution…";
      // Read durable state: the preceding approval_required SSE event belongs
      // to the graph phase and must not terminate this action-phase wait.
      while (shell.isConnected) {
        const outcome = await api(`/sessions/${sessionId}/drafts/${requestId}`);
        if (!["accepted", "running"].includes(outcome.status)) {
          renderDraft(shell, outcome);
          setBusy(false);
          setStatus(outcome.status === "completed" ? "Ready" : "Needs attention",
            outcome.status === "completed" ? "ready" : "error");
          if (outcome.status === "completed") showToast("Approved action completed.");
          return;
        }
        await new Promise((resolve) => setTimeout(resolve, 1500));
      }
      return;
    }
    await getDraft(shell, sessionId, requestId);
    if (admission.status === "completed") {
      showToast(decision === "approve" ? "Approved action completed." : "Action rejected. Nothing was saved.");
      setStatus("Ready", "ready");
    } else {
      setStatus("Needs attention", "error");
    }
  } catch (error) {
    setBusy(false);
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
    state.lastEventId = null;
    rememberConversation(message);
    await api(`/sessions/${state.sessionId}/messages`, {
      method: "POST",
      body: JSON.stringify({ message, request_id: state.requestId }),
    });
    connectEvents(shell, state.lastEventId);
  } catch (error) {
    showRunError(shell, error.message);
  }
}

function resetConversation() {
  state.source?.close();
  document.querySelectorAll(".message.assistant").forEach((shell) => shell._answerSource?.close());
  state.sessionId = null;
  state.requestId = null;
  state.lastSequence = -1;
  state.lastEventId = null;
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
  state.lastEventId = null;
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
