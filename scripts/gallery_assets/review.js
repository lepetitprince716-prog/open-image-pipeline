"use strict";

const params = new URLSearchParams(window.location.search);
const batch = params.get("batch") || "";
const state = {
  items: [],
  result: null,
  index: 0,
  saving: false,
  imageKey: "",
  imageReady: false,
  imageError: false,
};

const $ = (selector) => document.querySelector(selector);
let image = $("#image");
const empty = $("#empty");
const nameEl = $("#name");
const captionEl = $("#caption");
const progress = $("#progress");
const position = $("#position");
const status = $("#status");
const editor = $("#commentEditor");
const commentInput = $("#commentInput");
const loveBtn = $("#love");
const yesBtn = $("#yes");
const noBtn = $("#no");
const commentBtn = $("#comment");
const prevBtn = $("#prev");
const nextBtn = $("#next");

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok || body.ok === false) {
    throw new Error(body.error || `HTTP ${response.status}`);
  }
  return body;
}

function decisionFor(index) {
  if (!state.result) return null;
  if ((state.result.l || []).includes(index)) return "love";
  if ((state.result.y || []).includes(index)) return "yes";
  if ((state.result.n || []).includes(index)) return "no";
  if (Object.prototype.hasOwnProperty.call(state.result.c || {}, String(index))) {
    return "comment";
  }
  return null;
}

function reviewedCount() {
  if (!state.result) return 0;
  return (state.result.l || []).length
    + (state.result.y || []).length
    + (state.result.n || []).length
    + Object.keys(state.result.c || {}).length;
}

function firstPendingIndex(start = 0) {
  const total = state.items.length;
  for (let offset = 0; offset < total; offset += 1) {
    const candidate = (start + offset) % total;
    if (!decisionFor(state.items[candidate].i)) return candidate;
  }
  return -1;
}

function closeEditor() {
  editor.hidden = true;
  commentInput.value = "";
}

function setStatus(message, done = false) {
  status.textContent = message;
  status.classList.toggle("done", done);
}

/* Build the element for one item and wire its ready/error callbacks. Images are ready on
   load (plus decode when supported); videos are ready on loadedmetadata — waiting for a
   full load would stall the buttons until the whole clip downloaded. */
function buildMedia(item, key) {
  const label = item.name || `媒体 ${item.i + 1}`;
  const src = `/api/review/media?batch=${encodeURIComponent(batch)}&i=${item.i}`;
  const isVideo = item.media === "video";
  const node = isVideo ? document.createElement("video") : new Image();
  node.id = "image";
  node.className = "loading";
  if (isVideo) {
    node.controls = true;
    node.preload = "metadata";
    node.playsInline = true;
    node.setAttribute("aria-label", label);
  } else {
    node.alt = label;
  }

  const ready = async () => {
    if (!isVideo) {
      try {
        await node.decode();
      } catch {
        // A completed load is still usable when decode() is unsupported.
      }
    }
    if (state.imageKey !== key || image !== node) return;
    state.imageReady = true;
    state.imageError = false;
    render();
  };
  const failed = () => {
    if (state.imageKey !== key || image !== node) return;
    state.imageReady = false;
    state.imageError = true;
    render();
  };

  if (isVideo) {
    node.addEventListener("loadedmetadata", () => void ready());
  } else {
    node.onload = ready;
  }
  node.onerror = failed;
  node.src = src;
  return node;
}

function showFatal(message) {
  image.hidden = true;
  nameEl.hidden = true;
  if (captionEl) captionEl.hidden = true;
  empty.hidden = false;
  empty.textContent = message;
  loveBtn.disabled = true;
  yesBtn.disabled = true;
  noBtn.disabled = true;
  commentBtn.disabled = true;
  prevBtn.disabled = true;
  nextBtn.disabled = true;
  setStatus("");
}

function render() {
  const total = state.items.length;
  const reviewed = reviewedCount();
  progress.textContent = `${reviewed} / ${total}`;
  position.textContent = total ? `${state.index + 1} / ${total}` : "0 / 0";

  if (!total) {
    image.hidden = true;
    nameEl.hidden = true;
    if (captionEl) captionEl.hidden = true;
    empty.hidden = false;
    empty.textContent = "这个批次没有内容";
    loveBtn.disabled = true;
    yesBtn.disabled = true;
    noBtn.disabled = true;
    commentBtn.disabled = true;
    prevBtn.disabled = true;
    nextBtn.disabled = true;
    setStatus("");
    return;
  }

  const item = state.items[state.index];
  const decision = decisionFor(item.i);
  const imageKey = String(item.i);
  image.hidden = false;
  nameEl.hidden = false;
  empty.hidden = true;
  if (state.imageKey !== imageKey) {
    state.imageKey = imageKey;
    state.imageReady = false;
    state.imageError = false;
    const next = buildMedia(item, imageKey);
    image.replaceWith(next);
    image = next;
  }
  image.classList.toggle("loading", !state.imageReady);
  nameEl.textContent = item.name || `媒体 ${item.i + 1}`;
  if (captionEl) {
    captionEl.textContent = item.caption || "";
    captionEl.hidden = !item.caption;
  }
  loveBtn.disabled = state.saving || !state.imageReady;
  yesBtn.disabled = state.saving || !state.imageReady;
  noBtn.disabled = state.saving || !state.imageReady;
  commentBtn.disabled = state.saving || !state.imageReady;
  prevBtn.disabled = state.saving || total < 2;
  nextBtn.disabled = state.saving || total < 2;
  loveBtn.classList.toggle("active", decision === "love");
  yesBtn.classList.toggle("active", decision === "yes");
  noBtn.classList.toggle("active", decision === "no");
  commentBtn.classList.toggle("active", decision === "comment");

  if (state.imageError) {
    setStatus("加载失败");
  } else if (!state.imageReady) {
    setStatus("加载中…");
  } else if (reviewed === total) {
    setStatus("完成，回到会话说一声即可读取结果", true);
  } else if (decision === "love") {
    setStatus("已标记非常喜欢");
  } else if (decision === "yes") {
    setStatus("已认可");
  } else if (decision === "no") {
    setStatus("已拒绝，不采用");
  } else if (decision === "comment") {
    setStatus("已评论：部分认可，需修改");
  } else {
    setStatus("");
  }
}

function move(delta) {
  if (!state.items.length || state.saving) return;
  state.index = (state.index + delta + state.items.length) % state.items.length;
  closeEditor();
  render();
}

async function saveDecision(decision, comment = "") {
  if (!state.items.length || state.saving || !state.imageReady) return;
  state.saving = true;
  render();
  const item = state.items[state.index];
  let errorMessage = "";
  try {
    const body = await api("/api/review/decision", {
      method: "POST",
      body: JSON.stringify({ batch, i: item.i, decision, comment }),
    });
    state.result = body.result;
    closeEditor();
    const next = firstPendingIndex(state.index + 1);
    if (next >= 0) state.index = next;
  } catch (error) {
    errorMessage = `保存失败：${error.message}`;
  } finally {
    state.saving = false;
    render();
    if (errorMessage) setStatus(errorMessage);
  }
}

function openEditor() {
  if (!state.items.length || state.saving || !state.imageReady) return;
  const item = state.items[state.index];
  commentInput.value = state.result?.c?.[String(item.i)] || "";
  editor.hidden = false;
  commentInput.focus();
  commentInput.select();
}

async function load() {
  if (!batch) {
    showFatal("缺少批次编号");
    return;
  }
  try {
    const body = await api(`/api/review/batch?batch=${encodeURIComponent(batch)}`);
    state.items = body.items || [];
    state.result = body.result;
    const pending = firstPendingIndex();
    state.index = pending >= 0 ? pending : 0;
    render();
  } catch (error) {
    showFatal(`加载失败：${error.message}`);
  }
}

loveBtn.addEventListener("click", () => void saveDecision("love"));
yesBtn.addEventListener("click", () => void saveDecision("yes"));
noBtn.addEventListener("click", () => void saveDecision("no"));
commentBtn.addEventListener("click", openEditor);
prevBtn.addEventListener("click", () => move(-1));
nextBtn.addEventListener("click", () => move(1));
$("#commentCancel").addEventListener("click", closeEditor);
$("#commentSave").addEventListener("click", () => {
  const text = commentInput.value.trim();
  if (!text) {
    commentInput.focus();
    return;
  }
  void saveDecision("comment", text);
});

document.addEventListener("keydown", (event) => {
  if (!editor.hidden) {
    if (event.key === "Escape") {
      event.preventDefault();
      closeEditor();
    } else if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      $("#commentSave").click();
    }
    return;
  }
  if (event.repeat || state.saving) return;
  const key = event.key.toLowerCase();
  if (key === "l") {
    event.preventDefault();
    void saveDecision("love");
  } else if (key === "y") {
    event.preventDefault();
    void saveDecision("yes");
  } else if (key === "n") {
    event.preventDefault();
    void saveDecision("no");
  } else if (key === "c") {
    event.preventDefault();
    openEditor();
  } else if (event.key === "ArrowLeft") {
    event.preventDefault();
    move(-1);
  } else if (event.key === "ArrowRight") {
    event.preventDefault();
    move(1);
  }
});

void load();
