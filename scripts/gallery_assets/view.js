"use strict";

/* Look-only page for one batch: Atelier-themed proof sheet. A natural-aspect grid, a
   zoomable lightbox (wheel / double-tap to 1:1, drag pan, swipe nav) and an A/B compare
   overlay with synced zoom. It never POSTs, so a batch registered in view mode is never
   asked for a decision the server would reject. */

const params = new URLSearchParams(window.location.search);
const batch = params.get("batch") || "";

const $ = (selector) => document.querySelector(selector);
const batchIdEl = $("#batchId");
const countEl = $("#count");
const reviewLink = $("#reviewLink");
const statePanel = $("#statePanel");
const grid = $("#grid");
const compareBar = $("#compareBar");
const cbLabel = $("#cbLabel");
const cbCancel = $("#cbCancel");
const lightbox = $("#lightbox");
const lbViewport = $("#stageViewport");
const lbClose = $("#lbClose");
const lbPrev = $("#lbPrev");
const lbNext = $("#lbNext");
const lbName = $("#lbName");
const lbZoom = $("#lbZoom");
const lbPosText = $("#lbPosText");
const lbCaption = $("#lbCaption");
const lbPos = $("#lbPos");
const compareDlg = $("#compare");
const cmpSync = $("#cmpSync");
const cmpSwap = $("#cmpSwap");
const cmpClose = $("#cmpClose");

const state = {
  items: [],          // payload order; pos = array index, item.i = server media index
  mode: "grid",       // grid | lightbox | compare — the keydown dispatcher switches on this
  lbPos: 0,
  compare: { a: null, b: null, synced: true },
};

const HOVER_FINE = window.matchMedia("(hover: hover) and (pointer: fine)").matches;
const DOT_LIMIT = 12;

const mediaUrl = (item) =>
  `/api/review/media?batch=${encodeURIComponent(batch)}&i=${item.i}`;
const labelFor = (item) => item.name || `媒体 ${item.i + 1}`;
const pad2 = (pos) => String(pos + 1).padStart(2, "0");

const BROKEN_SVG =
  '<svg viewBox="0 0 24 24" fill="none" stroke-width="1.4" stroke-linecap="round">' +
  '<path d="M3 7v10a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2Z"/>' +
  '<path d="m3 15 4-4 3 3 5-5 6 6"/><path d="m9 3 6 18" stroke-dasharray="2 3"/></svg>';
const EMPTY_SVG =
  '<svg viewBox="0 0 24 24" fill="none" stroke-width="1.4" stroke-linecap="round">' +
  '<rect x="3.5" y="3.5" width="17" height="17" rx="2.5"/>' +
  '<path d="M8 12h8" stroke-dasharray="2 3"/></svg>';

/* Broken-media placeholder. The label carries the item's filename, which is untrusted, so only
   the static SVG goes through innerHTML and the text is set with textContent. */
function brokenStage(item) {
  const broken = document.createElement("div");
  broken.className = "stage-broken";
  const inner = document.createElement("div");
  inner.className = "inner";
  inner.insertAdjacentHTML("afterbegin", BROKEN_SVG);
  const label = document.createElement("div");
  label.textContent = `无法加载 · ${labelFor(item)}`;
  inner.appendChild(label);
  broken.appendChild(inner);
  return broken;
}

/* ---------- zoom viewport (shared by lightbox and compare panes) ----------
   The viewport clips; .stage-media carries translate(tx,ty) scale(s) with origin 0 0.
   s = 1 means "fit"; sOne is the scale at which one media pixel equals one CSS pixel. */
function createZoomView(viewport, opts = {}) {
  const media = viewport.querySelector(".stage-media");
  // kind gates the gesture layer (swipe works while an image is still loading);
  // zoom additionally needs a computed fit (z.fitW > 0).
  let kind = "none"; // none | image | video
  const z = { s: 1, tx: 0, ty: 0, fitW: 0, fitH: 0, sOne: 1 };
  const zoomReady = () => kind === "image" && z.fitW > 0;
  const pointers = new Map();
  let panStart = null;      // {x, y, tx, ty, moved}
  let pinchDist = 0;
  let lastTap = { t: 0, x: 0, y: 0 };
  let suppressTap = false;

  const rect = () => viewport.getBoundingClientRect();

  function applyTransform(animated = false, notify = true) {
    media.classList.toggle("is-animated", animated);
    media.style.transform = `translate(${z.tx}px, ${z.ty}px) scale(${z.s})`;
    // notify=false for load-time fitting: a pane finishing its own load must not be
    // mirrored onto the other compare pane as if the user had moved it.
    if (notify && opts.onChange) opts.onChange();
  }

  function clampPan() {
    const r = rect();
    const w = z.fitW * z.s;
    const h = z.fitH * z.s;
    z.tx = w <= r.width ? (r.width - w) / 2 : Math.min(0, Math.max(r.width - w, z.tx));
    z.ty = h <= r.height ? (r.height - h) / 2 : Math.min(0, Math.max(r.height - h, z.ty));
  }

  function computeFit(natW, natH) {
    const r = rect();
    if (!r.width || !r.height || !natW || !natH) return;
    const fit = Math.min(r.width / natW, r.height / natH);
    z.fitW = natW * fit;
    z.fitH = natH * fit;
    z.sOne = natW / z.fitW;
    const img = media.firstElementChild;
    if (img) {
      img.style.width = `${z.fitW}px`;
      img.style.height = `${z.fitH}px`;
    }
    reset(false, false);
  }

  function reset(animated = false, notify = true) {
    const r = rect();
    z.s = 1;
    z.tx = (r.width - z.fitW) / 2;
    z.ty = (r.height - z.fitH) / 2;
    applyTransform(animated, notify);
  }

  function zoomAt(cx, cy, factor, animated = false) {
    if (!zoomReady()) return;
    const maxS = Math.max(z.sOne, 4);
    const next = Math.min(maxS, Math.max(1, z.s * factor));
    const real = next / z.s;
    z.tx = cx - (cx - z.tx) * real;
    z.ty = cy - (cy - z.ty) * real;
    z.s = next;
    clampPan();
    applyTransform(animated);
  }

  function toggleOne(cx, cy) {
    if (!zoomReady()) return;
    const r = rect();
    const px = cx ?? r.width / 2;
    const py = cy ?? r.height / 2;
    if (z.s > 1.01) reset(true);
    else zoomAt(px, py, z.sOne / z.s, true);
  }

  function localPoint(event) {
    const r = rect();
    return { x: event.clientX - r.left, y: event.clientY - r.top };
  }

  viewport.addEventListener("wheel", (event) => {
    if (!zoomReady()) return;
    event.preventDefault();
    const p = localPoint(event);
    zoomAt(p.x, p.y, Math.exp(-event.deltaY * 0.0015));
  }, { passive: false });

  viewport.addEventListener("pointerdown", (event) => {
    if (kind !== "image") return;
    try {
      viewport.setPointerCapture(event.pointerId);
    } catch {
      // Synthetic events carry no active pointer; capture is an optimization only.
    }
    pointers.set(event.pointerId, localPoint(event));
    if (pointers.size === 1) {
      const p = localPoint(event);
      panStart = { x: p.x, y: p.y, tx: z.tx, ty: z.ty, moved: false };
      suppressTap = false;
    } else if (pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      pinchDist = Math.hypot(a.x - b.x, a.y - b.y);
      suppressTap = true;
    }
  });

  viewport.addEventListener("pointermove", (event) => {
    if (kind !== "image" || !pointers.has(event.pointerId)) return;
    pointers.set(event.pointerId, localPoint(event));
    if (pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      const dist = Math.hypot(a.x - b.x, a.y - b.y);
      if (pinchDist > 0) {
        zoomAt((a.x + b.x) / 2, (a.y + b.y) / 2, dist / pinchDist);
      }
      pinchDist = dist;
      return;
    }
    if (!panStart) return;
    const p = localPoint(event);
    const dx = p.x - panStart.x;
    const dy = p.y - panStart.y;
    if (Math.hypot(dx, dy) > 8) panStart.moved = true;
    if (z.s > 1) {
      viewport.classList.add("is-panning");
      z.tx = panStart.tx + dx;
      z.ty = panStart.ty + dy;
      clampPan();
      applyTransform();
    }
  });

  function endPointer(event) {
    if (!pointers.has(event.pointerId)) return;
    const p = pointers.get(event.pointerId);
    pointers.delete(event.pointerId);
    viewport.classList.remove("is-panning");
    if (pointers.size > 0 || !panStart) { pinchDist = 0; return; }
    const start = panStart;
    panStart = null;
    if (event.type === "pointercancel") return;
    const dx = p.x - start.x;
    const dy = p.y - start.y;
    if (!start.moved && !suppressTap) {
      const now = performance.now();
      if (now - lastTap.t < 300 && Math.hypot(p.x - lastTap.x, p.y - lastTap.y) < 24) {
        lastTap.t = 0;
        toggleOne(p.x, p.y);
      } else {
        lastTap = { t: now, x: p.x, y: p.y };
      }
      return;
    }
    if (z.s === 1 && opts.onSwipe && !suppressTap) {
      if (Math.abs(dx) > 48 && Math.abs(dx) > 1.5 * Math.abs(dy)) {
        opts.onSwipe(dx < 0 ? "left" : "right");
      } else if (dy > 80) {
        opts.onSwipe("down");
      }
    }
  }
  viewport.addEventListener("pointerup", endPointer);
  viewport.addEventListener("pointercancel", endPointer);
  media.addEventListener("transitionend", () => media.classList.remove("is-animated"));

  return {
    media,
    get isImage() { return zoomReady(); },
    get zoomPercent() { return Math.round((z.s / z.sOne) * 100); },
    get isZoomed() { return z.s > 1.001; },
    setImage(item, { onLoad, onBroken } = {}) {
      this.teardown();
      viewport.classList.remove("is-video", "is-plain");
      kind = "image";
      const img = document.createElement("img");
      img.alt = labelFor(item);
      img.decoding = "async";
      img.addEventListener("load", () => {
        computeFit(img.naturalWidth, img.naturalHeight);
        if (onLoad) onLoad();
      });
      img.addEventListener("error", () => {
        this.teardown();
        if (onBroken) onBroken();
      });
      img.src = mediaUrl(item);
      media.replaceChildren(img);
    },
    setVideo(item) {
      this.teardown();
      kind = "video";
      viewport.classList.remove("is-plain");
      viewport.classList.add("is-video");
      const video = document.createElement("video");
      video.controls = true;
      video.playsInline = true;
      video.setAttribute("aria-label", labelFor(item));
      video.src = mediaUrl(item);
      media.replaceChildren(video);
      video.play().catch(() => {
        video.muted = true;
        video.play().catch(() => {});
      });
    },
    /* Dropping the source stops a closed video from streaming on in the background —
       always tear down before hiding the dialog, and again from its "close" event. */
    teardown() {
      kind = "none";
      z.fitW = 0;
      z.fitH = 0;
      z.s = 1;
      pointers.clear();
      panStart = null;
      media.style.transform = "";
      media.replaceChildren();
    },
    reset: () => reset(false),
    refit(natW, natH) { computeFit(natW, natH); },
    zoomStep(factor) {
      const r = rect();
      zoomAt(r.width / 2, r.height / 2, factor, true);
    },
    toggleOne: () => toggleOne(),
    getNormalized() {
      const r = rect();
      return {
        k: z.s / z.sOne,
        nx: (r.width / 2 - z.tx) / (z.fitW * z.s),
        ny: (r.height / 2 - z.ty) / (z.fitH * z.s),
      };
    },
    setNormalized(shared) {
      if (!zoomReady() || !shared || !(shared.k > 0)) return;
      const r = rect();
      z.s = Math.min(Math.max(z.sOne, 4), Math.max(1, shared.k * z.sOne));
      z.tx = r.width / 2 - shared.nx * z.fitW * z.s;
      z.ty = r.height / 2 - shared.ny * z.fitH * z.s;
      clampPan();
      media.classList.remove("is-animated");
      media.style.transform = `translate(${z.tx}px, ${z.ty}px) scale(${z.s})`;
    },
  };
}

/* ---------- states ---------- */
function renderState(kind, detail) {
  grid.hidden = true;
  statePanel.hidden = false;
  statePanel.replaceChildren();
  const kicker = document.createElement("div");
  kicker.className = "kicker";
  const title = document.createElement("h2");
  title.className = "state-title";
  const sub = document.createElement("p");
  sub.className = "state-detail";
  sub.textContent = batch || "—";
  if (kind === "loading") {
    kicker.textContent = "载入中 · LOADING";
    const track = document.createElement("div");
    track.className = "loading-track";
    statePanel.append(kicker, sub, track);
    return;
  }
  const glyph = document.createElement("div");
  glyph.className = "state-glyph";
  glyph.innerHTML = kind === "empty" ? EMPTY_SVG : BROKEN_SVG;
  if (kind === "empty") {
    kicker.textContent = "空批次 · EMPTY";
    title.textContent = "这个批次没有内容";
    statePanel.append(glyph, kicker, title, sub);
    return;
  }
  kicker.textContent = "载入失败 · ERROR";
  title.textContent = "无法载入批次";
  const panel = document.createElement("div");
  panel.className = "state-error-panel";
  panel.textContent = detail.message;
  statePanel.append(glyph, kicker, title, sub, panel);
  if (detail.retry) {
    const retry = document.createElement("button");
    retry.type = "button";
    retry.className = "ctl";
    retry.textContent = "重试";
    retry.addEventListener("click", () => {
      renderState("loading");
      void load();
    });
    statePanel.append(retry);
  }
}

/* ---------- grid ---------- */
let previewingVideo = null;

function buildCard(item, pos) {
  const li = document.createElement("li");
  li.className = "card";
  li.dataset.pos = String(pos);

  const open = document.createElement("button");
  open.type = "button";
  open.className = "card-open";
  open.setAttribute("aria-label", `媒体 ${pos + 1}：${labelFor(item)}`);

  const name = document.createElement("span");
  name.className = "name";
  name.textContent = labelFor(item);

  const box = document.createElement("span");
  box.className = "media-box";
  let preview;
  if (item.media === "video") {
    preview = document.createElement("video");
    preview.preload = "metadata";
    preview.muted = true;
    preview.playsInline = true;
    preview.addEventListener("loadedmetadata", () => preview.classList.add("is-loaded"));
    if (HOVER_FINE) {
      open.addEventListener("mouseenter", () => {
        if (previewingVideo && previewingVideo !== preview) {
          previewingVideo.pause();
          previewingVideo.currentTime = 0.03;
        }
        previewingVideo = preview;
        preview.play().catch(() => {});
      });
      open.addEventListener("mouseleave", () => {
        preview.pause();
        preview.currentTime = 0.03;
      });
    }
  } else {
    preview = document.createElement("img");
    preview.alt = "";
    preview.loading = "lazy";
    preview.decoding = "async";
    preview.addEventListener("load", () => preview.classList.add("is-loaded"));
  }
  preview.addEventListener("error", () => {
    li.classList.add("is-broken");
    const glyph = document.createElement("span");
    glyph.className = "broken-glyph";
    glyph.innerHTML = BROKEN_SVG;
    box.appendChild(glyph);
    name.textContent = `无法加载 · ${labelFor(item)}`;
  });
  preview.src = mediaUrl(item);
  box.appendChild(preview);

  if (item.media === "video") {
    const badge = document.createElement("span");
    badge.className = "play-badge";
    badge.textContent = "▶ 视频";
    box.appendChild(badge);
  }
  open.appendChild(box);

  const foot = document.createElement("span");
  foot.className = "card-foot";
  const idx = document.createElement("span");
  idx.className = "idx";
  idx.textContent = pad2(pos);
  foot.append(idx, name);
  open.appendChild(foot);

  if (item.caption) {
    const caption = document.createElement("span");
    caption.className = "caption";
    caption.textContent = item.caption;
    open.appendChild(caption);
  }

  open.addEventListener("click", () => openLightbox(pos));
  li.appendChild(open);

  const chip = document.createElement("button");
  chip.type = "button";
  chip.className = "cmp-chip";
  chip.textContent = "⇄";
  chip.setAttribute("aria-pressed", "false");
  chip.setAttribute("aria-label", `加入对比：${labelFor(item)}`);
  chip.addEventListener("click", () => armCompare(pos));
  li.appendChild(chip);
  return li;
}

function renderGrid() {
  statePanel.hidden = true;
  grid.hidden = false;
  grid.dataset.count = String(state.items.length);
  grid.replaceChildren(...state.items.map(buildCard));
}

/* ---------- lightbox ---------- */
const lbView = createZoomView(lbViewport, {
  onChange: () => {
    lbZoom.hidden = !lbView.isZoomed;
    lbZoom.textContent = `${lbView.zoomPercent}%`;
  },
  onSwipe: (dir) => {
    if (dir === "left") move(1);
    else if (dir === "right") move(-1);
    else closeLightbox();
  },
});

function renderPosIndicator() {
  const total = state.items.length;
  lbPos.replaceChildren();
  if (total <= 1) return;
  if (total <= DOT_LIMIT) {
    state.items.forEach((_, pos) => {
      const dot = document.createElement("button");
      dot.type = "button";
      dot.className = "dot" + (pos === state.lbPos ? " on" : "");
      dot.setAttribute("aria-label", `第 ${pos + 1} 项`);
      dot.addEventListener("click", () => {
        state.lbPos = pos;
        renderStage();
      });
      lbPos.appendChild(dot);
    });
    return;
  }
  const track = document.createElement("div");
  track.className = "track";
  const fill = document.createElement("div");
  fill.className = "fill";
  fill.style.width = `${((state.lbPos + 1) / total) * 100}%`;
  track.appendChild(fill);
  lbPos.appendChild(track);
}

function renderStage() {
  const item = state.items[state.lbPos];
  if (!item) return;
  lbZoom.hidden = true;
  if (item.media === "video") {
    lbView.setVideo(item);
  } else {
    lbView.setImage(item, {
      onBroken: () => {
        lbViewport.classList.add("is-plain");
        lbView.media.replaceChildren(brokenStage(item));
      },
    });
  }
  lbName.textContent = `${pad2(state.lbPos)} · ${labelFor(item)}`;
  lbPosText.textContent = `${state.lbPos + 1} / ${state.items.length}`;
  lbCaption.textContent = item.caption || "";
  lbCaption.hidden = !item.caption;
  const single = state.items.length < 2;
  lbPrev.disabled = single;
  lbNext.disabled = single;
  renderPosIndicator();
}

function openLightbox(pos) {
  state.lbPos = pos;
  state.mode = "lightbox";
  renderStage();
  if (!lightbox.open) lightbox.showModal();
}

function closeLightbox() {
  lbView.teardown();
  if (lightbox.open) lightbox.close();
  state.mode = "grid";
}

function move(delta) {
  const total = state.items.length;
  if (!total) return;
  state.lbPos = (state.lbPos + delta + total) % total;
  renderStage();
}

lbPrev.addEventListener("click", () => move(-1));
lbNext.addEventListener("click", () => move(1));
lbClose.addEventListener("click", closeLightbox);
lightbox.addEventListener("close", () => {
  // Safety net for close paths that skip closeLightbox (form method dialogs, UA quirks).
  lbView.teardown();
  state.mode = "grid";
});
lightbox.addEventListener("click", (event) => {
  if (event.target === lightbox) closeLightbox();
});
window.addEventListener("resize", () => {
  if (!lightbox.open) return;
  const img = lbView.media.querySelector("img");
  if (img && img.naturalWidth) lbView.refit(img.naturalWidth, img.naturalHeight);
});

/* ---------- compare ---------- */
const cmpPanes = {};
for (const pane of compareDlg.querySelectorAll(".cmp-pane")) {
  const slot = pane.dataset.slot;
  const viewport = pane.querySelector(".stage-viewport");
  cmpPanes[slot] = {
    pane,
    label: pane.querySelector(".cmp-label"),
    view: null,
    viewport,
  };
}
let syncGuard = false;
for (const slot of ["a", "b"]) {
  const other = slot === "a" ? "b" : "a";
  cmpPanes[slot].view = createZoomView(cmpPanes[slot].viewport, {
    onChange: () => {
      if (!state.compare.synced || syncGuard) return;
      const source = cmpPanes[slot].view;
      const target = cmpPanes[other].view;
      if (!source.isImage || !target.isImage) return;
      syncGuard = true;
      target.setNormalized(source.getNormalized());
      syncGuard = false;
    },
  });
}

function armCompare(pos) {
  const { compare } = state;
  if (compare.a === pos) {
    disarmCompare();
    return;
  }
  if (compare.a === null) {
    compare.a = pos;
    const card = grid.querySelector(`.card[data-pos="${pos}"]`);
    if (card) {
      card.classList.add("is-armed");
      const chip = card.querySelector(".cmp-chip");
      chip.classList.add("on");
      chip.setAttribute("aria-pressed", "true");
    }
    cbLabel.textContent = `${pad2(pos)} · ${labelFor(state.items[pos])}`;
    compareBar.hidden = false;
    return;
  }
  openCompare(compare.a, pos);
}

function disarmCompare() {
  const pos = state.compare.a;
  state.compare.a = null;
  state.compare.b = null;
  compareBar.hidden = true;
  if (pos === null) return;
  const card = grid.querySelector(`.card[data-pos="${pos}"]`);
  if (card) {
    card.classList.remove("is-armed");
    const chip = card.querySelector(".cmp-chip");
    chip.classList.remove("on");
    chip.setAttribute("aria-pressed", "false");
  }
}

function renderComparePane(slot, pos) {
  const { view, label, viewport } = cmpPanes[slot];
  const item = state.items[pos];
  label.textContent = `${pad2(pos)} · ${labelFor(item)}`;
  if (item.media === "video") {
    view.setVideo(item);
    return;
  }
  view.setImage(item, {
    onBroken: () => {
      viewport.classList.add("is-plain");
      view.media.replaceChildren(brokenStage(item));
    },
  });
}

function updateSyncAvailability() {
  const bothImages =
    state.items[state.compare.a]?.media === "image" &&
    state.items[state.compare.b]?.media === "image";
  cmpSync.disabled = !bothImages;
  setSync(bothImages);
}

function openCompare(a, b) {
  state.compare.b = b;
  state.mode = "compare";
  renderComparePane("a", a);
  renderComparePane("b", b);
  updateSyncAvailability();
  compareBar.hidden = true;
  if (!compareDlg.open) compareDlg.showModal();
}

function closeCompare() {
  // Capture before close(): the dialog's "close" event runs disarmCompare synchronously.
  const armedPos = state.compare.a;
  cmpPanes.a.view.teardown();
  cmpPanes.b.view.teardown();
  if (compareDlg.open) compareDlg.close();
  state.mode = "grid";
  disarmCompare();
  const chip = armedPos === null
    ? null
    : grid.querySelector(`.card[data-pos="${armedPos}"] .cmp-chip`);
  if (chip) chip.focus();
}

function setSync(on) {
  state.compare.synced = on;
  cmpSync.classList.toggle("on", on);
  cmpSync.setAttribute("aria-pressed", String(on));
}

function swapPanes() {
  const { compare } = state;
  if (compare.a === null || compare.b === null) return;
  [compare.a, compare.b] = [compare.b, compare.a];
  renderComparePane("a", compare.a);
  renderComparePane("b", compare.b);
}

cmpSync.addEventListener("click", () => setSync(!state.compare.synced));
cmpSwap.addEventListener("click", swapPanes);
cmpClose.addEventListener("click", closeCompare);
compareDlg.addEventListener("close", () => {
  cmpPanes.a.view.teardown();
  cmpPanes.b.view.teardown();
  state.mode = "grid";
  disarmCompare();
});
cbCancel.addEventListener("click", disarmCompare);

/* ---------- keyboard ---------- */
document.addEventListener("keydown", (event) => {
  const target = event.target;
  if (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement) return;
  const onVideo = target instanceof HTMLVideoElement;
  if (state.mode === "grid") {
    if (event.key === "Escape" && state.compare.a !== null) {
      event.preventDefault();
      disarmCompare();
    } else if (event.key === "c" || event.key === "C") {
      const card = target instanceof Element ? target.closest(".card") : null;
      if (card) {
        event.preventDefault();
        armCompare(Number(card.dataset.pos));
      }
    }
    return;
  }
  if (state.mode === "lightbox") {
    if (event.key === "Escape") {
      // Handle Escape rather than letting the dialog close itself, so the media
      // source is always released on the way out.
      event.preventDefault();
      closeLightbox();
      return;
    }
    if (onVideo) return;
    if (event.key === "ArrowLeft") { event.preventDefault(); move(-1); }
    else if (event.key === "ArrowRight") { event.preventDefault(); move(1); }
    else if (event.key === "Home") { event.preventDefault(); state.lbPos = 0; renderStage(); }
    else if (event.key === "End") { event.preventDefault(); state.lbPos = state.items.length - 1; renderStage(); }
    else if (event.key === "z" || event.key === "Z") { event.preventDefault(); lbView.toggleOne(); }
    else if (event.key === "+" || event.key === "=") { event.preventDefault(); lbView.zoomStep(1.4); }
    else if (event.key === "-") { event.preventDefault(); lbView.zoomStep(1 / 1.4); }
    else if (event.key === "0") { event.preventDefault(); lbView.reset(); }
    return;
  }
  if (state.mode === "compare") {
    if (event.key === "Escape") {
      event.preventDefault();
      closeCompare();
      return;
    }
    if (onVideo) return;
    const both = [cmpPanes.a.view, cmpPanes.b.view];
    if (event.key === "s" || event.key === "S") {
      if (!cmpSync.disabled) { event.preventDefault(); setSync(!state.compare.synced); }
    } else if (event.key === "x" || event.key === "X") { event.preventDefault(); swapPanes(); }
    else if (event.key === "z" || event.key === "Z") {
      event.preventDefault();
      both.forEach((view) => view.isImage && view.toggleOne());
    } else if (event.key === "+" || event.key === "=") {
      event.preventDefault();
      both.forEach((view) => view.isImage && view.zoomStep(1.4));
    } else if (event.key === "-") {
      event.preventDefault();
      both.forEach((view) => view.isImage && view.zoomStep(1 / 1.4));
    } else if (event.key === "0") {
      event.preventDefault();
      both.forEach((view) => view.isImage && view.reset());
    }
  }
});

/* ---------- load ---------- */
async function load() {
  if (!batch) {
    renderState("error", { message: "缺少批次编号（?batch=）", retry: false });
    return;
  }
  batchIdEl.textContent = batch;
  try {
    const response = await fetch(`/api/review/batch?batch=${encodeURIComponent(batch)}`);
    const body = await response.json().catch(() => ({}));
    if (!response.ok || body.ok === false) {
      throw new Error(body.error || `HTTP ${response.status}`);
    }
    state.items = body.items || [];
    countEl.textContent = `${state.items.length} 项`;
    if (body.mode === "review") {
      reviewLink.href = `/review?batch=${encodeURIComponent(batch)}`;
      reviewLink.hidden = false;
    }
    if (!state.items.length) {
      renderState("empty");
      return;
    }
    renderGrid();
  } catch (error) {
    renderState("error", { message: error.message, retry: true });
  }
}

renderState("loading");
void load();
