(() => {
  const MODEL_RE = /\/models\/(\d+)/;
  const IMAGE_RE = /\/images\/(\d+)/;
  const BADGE_ATTR = "data-scb";
  let scanTimer = 0;
  let tickBusy = false;
  let tickAgain = false;
  // dockHref is the href whose dock actually rendered. A failed render leaves it
  // unset so a later tick retries instead of the page staying stuck on "未连接".
  let dockHref = "";
  let dockPendingHref = "";
  let dockRetryAt = 0;
  const DOCK_RETRY_MS = 5000;

  function siteFromHost(host) {
    host = (host || location.hostname).toLowerCase();
    if (host.includes("civitai.red") || host.includes("civitai.green")) return "red";
    if (host.includes("civarchive") || host.includes("civitaiarchive")) return "archive";
    return "com";
  }

  function parseHref(href) {
    if (!href) return null;
    let url;
    try {
      url = new URL(href, location.origin);
    } catch {
      return null;
    }
    const image = url.pathname.match(IMAGE_RE);
    const model = url.pathname.match(MODEL_RE);
    const versionId = Number(url.searchParams.get("modelVersionId") || "") || null;
    if (image) {
      return { kind: "image", imageId: Number(image[1]), modelId: null, versionId, site: siteFromHost(url.hostname) };
    }
    if (model) {
      return { kind: "model", imageId: null, modelId: Number(model[1]), versionId, site: siteFromHost(url.hostname) };
    }
    return null;
  }

  function sendOnce(type, payload) {
    return new Promise((resolve) => {
      chrome.runtime.sendMessage({ type, ...payload }, (resp) => {
        if (chrome.runtime.lastError) {
          resolve({ ok: false, error: chrome.runtime.lastError.message });
          return;
        }
        resolve(resp || { ok: false, error: "no response" });
      });
    });
  }

  // MV3 shuts the service worker down when idle, so the first message after that can
  // come back before it has booted. Retry that class once. Not "message port closed":
  // there the worker may have received the message and only the reply was lost, and a
  // retried download or recipe save would queue it twice.
  const COLD_START_RE = /receiving end does not exist|could not establish connection/i;

  async function send(type, payload) {
    const first = await sendOnce(type, payload);
    if (first.ok || !COLD_START_RE.test(String(first.error || ""))) return first;
    await new Promise((done) => setTimeout(done, 150));
    return sendOnce(type, payload);
  }

  // Civitai lays a position:absolute inset-0 link over each card. Overwriting that with
  // relative collapses the card's layout, so only promote a genuinely static box.
  function ensurePositioned(node) {
    if (getComputedStyle(node).position === "static") node.style.position = "relative";
  }

  function badgeEl(status, title) {
    const el = document.createElement("span");
    el.className = `scb-badge scb-${status}`;
    el.title = title || status;
    if (status === "exact") el.textContent = "已有";
    else if (status === "other_version") el.textContent = "其它版本";
    else el.textContent = "未下";
    return el;
  }

  function attachBadge(node, status, title) {
    const old = node.querySelector(":scope > .scb-badge");
    if (old) old.remove();
    const badge = badgeEl(status, title);
    ensurePositioned(node);
    node.appendChild(badge);
    node.setAttribute(BADGE_ATTR, status);
  }

  function cardLinks() {
    return [...document.querySelectorAll('a[href*="/models/"]')].filter((a) => {
      const href = a.getAttribute("href") || "";
      return MODEL_RE.test(href) && a.querySelector("img");
    });
  }

  // Per card, not per page: an infinite-scroll feed appends one card at a time, and a
  // whole-page key re-queried every card on the page on every append.
  const ownedCache = new Map(); // "modelId:versionId" -> { row, at }
  const OWNED_CACHE_MAX = 3000;
  const OWNED_BATCH = 60;
  const OWNED_RETRY_MS = 5000;
  // What the library owns changes underneath us — this tab's own downloads, another
  // tab's, the Studio UI. Entries expire so a card cannot sit on a stale 未下, and a
  // download finishing anywhere drops the cache outright (see pollProgress).
  const OWNED_TTL_MS = 300000;
  let ownedRetryAt = 0;

  function ownedTitle(row) {
    return row.file ? `${row.status}: ${row.name || row.file}` : row.status;
  }

  async function markCards() {
    const pending = [];
    const nodesByKey = new Map();
    for (const a of cardLinks()) {
      const parsed = parseHref(a.href);
      if (!parsed || parsed.kind !== "model") continue;
      const key = `${parsed.modelId}:${parsed.versionId || ""}`;
      const cached = ownedCache.get(key);
      if (cached && Date.now() - cached.at < OWNED_TTL_MS) {
        if (a.getAttribute(BADGE_ATTR) !== cached.row.status) attachBadge(a, cached.row.status, ownedTitle(cached.row));
        continue;
      }
      if (!nodesByKey.has(key)) {
        nodesByKey.set(key, []);
        pending.push({ key, modelId: parsed.modelId, versionId: parsed.versionId });
      }
      nodesByKey.get(key).push(a);
    }
    if (!pending.length) return;
    if (Date.now() < ownedRetryAt) return;
    const batch = pending.slice(0, OWNED_BATCH);
    // A page can hold more cards than one request should carry; the rest come in on
    // the next pass, which this asks for since our own badges no longer wake the
    // observer.
    if (pending.length > batch.length) tickAgain = true;
    const resp = await send("owned", {
      ids: batch.map((x) => ({ modelId: x.modelId, versionId: x.versionId })),
      enrich: true,
    });
    if (!resp.ok || !resp.data || !Array.isArray(resp.data.results)) {
      ownedRetryAt = Date.now() + OWNED_RETRY_MS;
      // A settled page produces no mutations, so nothing else would come back for the
      // retry and the cards would stay unbadged.
      setTimeout(schedule, OWNED_RETRY_MS);
      return;
    }
    resp.data.results.forEach((row, i) => {
      const item = batch[i];
      if (!item || !row) return;
      ownedCache.set(item.key, { row, at: Date.now() });
      for (const a of nodesByKey.get(item.key) || []) attachBadge(a, row.status, ownedTitle(row));
    });
    if (ownedCache.size > OWNED_CACHE_MAX) {
      for (const k of [...ownedCache.keys()].slice(0, ownedCache.size - OWNED_CACHE_MAX)) ownedCache.delete(k);
    }
  }

  let progressTimer = 0;

  function ensureDock() {
    let dock = document.getElementById("scb-dock");
    if (dock) return dock;
    dock = document.createElement("div");
    dock.id = "scb-dock";
    const progress = document.createElement("div");
    progress.id = "scb-progress";
    progress.className = "scb-dock-inner scb-progress-box";
    progress.hidden = true;
    const page = document.createElement("div");
    page.id = "scb-page";
    page.className = "scb-dock-inner";
    dock.appendChild(progress);
    dock.appendChild(page);
    document.documentElement.appendChild(dock);
    return dock;
  }

  function renderDock(html) {
    ensureDock();
    const page = document.getElementById("scb-page");
    page.innerHTML = html;
    page.hidden = !html;
    return page;
  }

  function formatBytes(n) {
    if (n == null || !Number.isFinite(Number(n))) return "";
    const v = Number(n);
    if (v < 1024) return `${v.toFixed(0)} B`;
    if (v < 1024 * 1024) return `${(v / 1024).toFixed(0)} KB`;
    if (v < 1024 * 1024 * 1024) return `${(v / 1024 / 1024).toFixed(1)} MB`;
    return `${(v / 1024 / 1024 / 1024).toFixed(2)} GB`;
  }

  function formatSpeed(n) {
    if (n == null || !Number.isFinite(Number(n)) || Number(n) <= 0) return "";
    return `${formatBytes(n)}/s`;
  }

  function statusLabel(st) {
    return ({
      downloading: "下载中",
      queued: "排队",
      paused: "暂停",
      completed: "完成",
      failed: "失败",
      canceled: "已取消",
    })[st] || st;
  }

  function renderProgress(data) {
    ensureDock();
    const box = document.getElementById("scb-progress");
    const jobs = ((data && data.jobs) || []).filter((j) => j && j.downloadId);
    const active = jobs.filter((j) => j.active);
    const recent = jobs.filter((j) => !j.active).slice(0, 3);
    const show = active.concat(recent);
    if (!show.length) {
      box.hidden = true;
      box.innerHTML = "";
      return;
    }
    box.hidden = false;
    box.innerHTML = show.map((j) => {
      const pct = Math.round(Number(j.progress) || 0);
      const size = [formatBytes(j.bytesDownloaded), formatBytes(j.totalBytes)].filter(Boolean).join(" / ");
      const speed = formatSpeed(j.bytesPerSecond);
      return `<div class="scb-job">
        <div class="scb-row">
          <span class="scb-badge scb-${j.active ? "missing" : (j.status === "completed" ? "exact" : "other_version")}">${statusLabel(j.status)}</span>
          <span class="scb-label">${escapeHtml(j.name || j.downloadId)}</span>
          <span class="scb-pct">${j.active ? pct + "%" : ""}</span>
        </div>
        <div class="scb-bar"><i style="width:${j.active ? pct : 100}%"></i></div>
        <div class="scb-meta">${escapeHtml([size, speed].filter(Boolean).join(" · "))}</div>
      </div>`;
    }).join("");
  }

  let hadActiveJobs = false;
  let pollFailures = 0;
  // Long enough to ride out a backend restart, short enough to stop eventually. The
  // owned cache expires on its own, so giving up here is not a stuck badge.
  const POLL_MAX_FAILURES = 12;
  const POLL_RETRY_MS = 5000;

  async function pollProgress() {
    const resp = await send("progress", {});
    if (!resp.ok) {
      // One failed poll must not end the chain. The re-arm used to come from every
      // page scan, so a transient error healed itself; without this the progress box
      // freezes and the owned-cache clear a finishing download depends on never runs.
      pollFailures += 1;
      if (hadActiveJobs && pollFailures < POLL_MAX_FAILURES) {
        clearTimeout(progressTimer);
        progressTimer = setTimeout(() => { pollProgress().catch(() => {}); }, POLL_RETRY_MS);
      }
      return;
    }
    pollFailures = 0;
    renderProgress(resp.data);
    const active = (resp.data && resp.data.active) || [];
    if (hadActiveJobs && !active.length) {
      ownedCache.clear();
      ownedRetryAt = 0;
      schedule();
    }
    hadActiveJobs = active.length > 0;
    if (active.length) {
      clearTimeout(progressTimer);
      progressTimer = setTimeout(() => { pollProgress().catch(() => {}); }, 1500);
    }
  }

  function watchProgress() {
    clearTimeout(progressTimer);
    progressTimer = setTimeout(() => { pollProgress().catch(() => {}); }, 300);
  }

  async function onModelPage(parsed) {
    const resp = await send("inspect", {
      url: location.href,
      modelId: parsed.modelId,
      versionId: parsed.versionId,
      site: parsed.site,
    });
    const owned = resp.ok && resp.data && resp.data.owned ? resp.data.owned : null;
    const status = owned ? owned.status : "missing";
    const label = owned && owned.file
      ? `${owned.name || owned.file}`
      : (resp.ok ? "不在 Z3 库里" : (resp.error || "Studio 未连接"));
    const box = renderDock(`
      <div class="scb-row">
        <span class="scb-badge scb-${status}">${status === "exact" ? "已有" : status === "other_version" ? "其它版本" : "未下"}</span>
        <span class="scb-label">${escapeHtml(label)}</span>
        <button type="button" class="scb-btn" data-scb-dl>下到 Z3</button>
      </div>
    `);
    box.querySelector("[data-scb-dl]").addEventListener("click", async (ev) => {
      const btn = ev.currentTarget;
      btn.disabled = true;
      btn.textContent = "排队中…";
      const dl = await send("download", {
        modelId: parsed.modelId,
        versionId: parsed.versionId,
        site: parsed.site,
        allowNsfw: parsed.site === "red",
      });
      if (!dl.ok) {
        btn.disabled = false;
        btn.textContent = "失败";
        btn.title = dl.error || "";
        return;
      }
      if (dl.data && dl.data.already) {
        btn.disabled = false;
        btn.textContent = "库里已有";
        return;
      }
      btn.textContent = "下载中";
      watchProgress();
    });
    return resp.ok;
  }

  async function onImagePage(parsed) {
    const resp = await send("inspect", { url: location.href, imageId: parsed.imageId, site: parsed.site });
    if (!resp.ok) {
      renderDock(`<div class="scb-row"><span class="scb-label">${escapeHtml(resp.error || "Studio 未连接")}</span></div>`);
      return false;
    }
    const resources = (resp.data && resp.data.resources) || [];
    const genParams = resp.data && resp.data.genParams;
    const hasParams = Boolean(genParams && Object.keys(genParams).length);
    const missing = resources.filter((r) => {
      const st = (r.owned && r.owned.status) || "missing";
      return st === "missing" && (r.modelVersionId || r.modelId);
    });
    // recipeReady === null means the backend could not tell (LM-direct); keep the
    // button enabled and let the save itself report.
    const ready = resp.data && resp.data.recipeReady;
    const saveLabel = resources.length || ready === null ? "存配方" : hasParams ? "存参数" : "无可存内容";
    const saveDisabled = !resources.length && !hasParams && ready !== null;
    const rows = resources.map((r, i) => {
      const st = (r.owned && r.owned.status) || "missing";
      const name = r.modelName || `resource ${i + 1}`;
      const canDl = st === "missing" && (r.modelVersionId || r.modelId);
      const dlBtn = canDl
        ? `<button type="button" class="scb-btn scb-btn-mini" data-scb-res-dl="${i}">下载</button>`
        : "";
      return `<div class="scb-res"><span class="scb-badge scb-${st}">${st === "exact" ? "已有" : st === "other_version" ? "其它版本" : "未下"}</span><span>${escapeHtml(name)}</span>${dlBtn}</div>`;
    }).join("");
    const fillRow = missing.length
      ? `<div class="scb-row"><button type="button" class="scb-btn" data-scb-fill>补齐缺件 (${missing.length})</button></div>`
      : "";
    const box = renderDock(`
      <div class="scb-row">
        <span class="scb-label">${escapeHtml(resp.data.title || "配方")}</span>
        <button type="button" class="scb-btn" data-scb-recipe ${saveDisabled ? "disabled" : ""}>${saveLabel}</button>
      </div>
      <div class="scb-res-list">${rows || "<div class='scb-res'>没有识别到 LoRA / 底模</div>"}</div>
      ${fillRow}
    `);
    const saveBtn = box.querySelector("[data-scb-recipe]");
    if (saveBtn && !saveDisabled) {
      saveBtn.addEventListener("click", async (ev) => {
        const btn = ev.currentTarget;
        btn.disabled = true;
        btn.textContent = "保存中…";
        const out = await send("recipe", {
          imageId: parsed.imageId,
          url: location.href,
          site: parsed.site,
          name: resp.data.title,
        });
        btn.disabled = false;
        btn.textContent = out.ok ? "已存" : "失败";
        if (!out.ok) btn.title = out.error || "";
      });
    }
    box.querySelectorAll("[data-scb-res-dl]").forEach((btn) => {
      btn.addEventListener("click", async (ev) => {
        const el = ev.currentTarget;
        const res = resources[Number(el.getAttribute("data-scb-res-dl"))];
        if (!res) return;
        el.disabled = true;
        el.textContent = "排队…";
        const dl = await send("download", {
          modelId: res.modelId,
          versionId: res.modelVersionId,
          site: parsed.site,
          allowNsfw: parsed.site === "red",
        });
        if (!dl.ok) {
          el.disabled = false;
          el.textContent = "失败";
          el.title = dl.error || "";
          return;
        }
        el.textContent = dl.data && dl.data.already ? "已有" : "下载中";
        watchProgress();
      });
    });
    const fillBtn = box.querySelector("[data-scb-fill]");
    if (fillBtn) {
      fillBtn.addEventListener("click", async (ev) => {
        const btn = ev.currentTarget;
        btn.disabled = true;
        btn.textContent = "排队中…";
        const out = await send("downloadMissing", {
          resources: missing.map((r) => ({
            modelId: r.modelId,
            modelVersionId: r.modelVersionId,
            modelName: r.modelName,
          })),
          allowNsfw: parsed.site === "red",
        });
        if (!out.ok) {
          btn.disabled = false;
          btn.textContent = "失败";
          btn.title = out.error || "";
          return;
        }
        const queued = (out.data && out.data.queued || []).length;
        const skipped = (out.data && out.data.skipped || []).length;
        btn.textContent = `已排 ${queued}${skipped ? ` · 跳过 ${skipped}` : ""}`;
        if (skipped) {
          btn.title = (out.data.skipped || []).map((s) => `${s.name}: ${s.reason}`).join("\n");
        }
        watchProgress();
      });
    }
    return true;
  }

  const SAVED_REFRESH_MS = 60000;
  const SAVED_BACKOFF_MS = 300000;
  const savedImageIds = new Set();
  // Ids saved from this page, kept across refreshes: the store may not list a
  // just-saved recipe yet, and a card must not flip back to 存 in the meantime.
  const locallySaved = new Set();
  let savedFetchedAt = 0;
  let savedRefreshMs = SAVED_REFRESH_MS;
  let savedInFlight = false;

  // Which images already have a recipe in the store. Refreshed on a timer and never
  // awaited by the render path: a slow or unreachable backend must not stop the
  // buttons from appearing.
  async function refreshSavedImages() {
    if (savedInFlight || Date.now() - savedFetchedAt < savedRefreshMs) return false;
    savedInFlight = true;
    try {
      const resp = await send("savedImages", {});
      if (!resp.ok || !resp.data || !Array.isArray(resp.data.imageIds)) {
        // Back off rather than re-probing on every debounced tick while a backend is down.
        savedFetchedAt = Date.now();
        savedRefreshMs = SAVED_BACKOFF_MS;
        return false;
      }
      savedFetchedAt = Date.now();
      savedRefreshMs = SAVED_REFRESH_MS;
      // Replace rather than accumulate, so a recipe deleted from the store loses its mark.
      const next = new Set(locallySaved);
      for (const id of resp.data.imageIds) {
        const n = Number(id);
        if (n) next.add(n);
      }
      let changed = next.size !== savedImageIds.size;
      if (!changed) {
        for (const n of next) {
          if (!savedImageIds.has(n)) { changed = true; break; }
        }
      }
      savedImageIds.clear();
      for (const n of next) savedImageIds.add(n);
      return changed;
    } catch {
      savedFetchedAt = Date.now();
      savedRefreshMs = SAVED_BACKOFF_MS;
      return false;
    } finally {
      savedInFlight = false;
    }
  }

  function imageCardLinks() {
    return [...document.querySelectorAll('a[href*="/images/"]')].filter((a) => {
      const href = a.getAttribute("href") || "";
      return IMAGE_RE.test(href) && a.querySelector("img");
    });
  }

  function savedMarkEl() {
    const el = document.createElement("span");
    el.className = "scb-card-saved";
    el.textContent = "已存";
    el.title = "这张图的配方已经在 LoRA Manager 里";
    return el;
  }

  function saveButtonEl(a, parsed) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "scb-card-save";
    btn.textContent = "存";
    btn.title = "存配方到 Z3 LoRA Manager";
    btn.addEventListener("click", async (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      btn.disabled = true;
      btn.textContent = "…";
      const out = await send("recipe", {
        imageId: parsed.imageId,
        url: new URL(a.href, location.origin).toString(),
        site: parsed.site,
      });
      if (out.ok) {
        // Mark it locally so the card flips to 已存 on the next pass without waiting
        // for the 60s store refresh.
        savedImageIds.add(parsed.imageId);
        locallySaved.add(parsed.imageId);
        btn.textContent = "已存";
        btn.title = "已存";
      } else {
        btn.textContent = "×";
        btn.title = out.error || "保存失败";
        btn.disabled = false;
      }
    });
    return btn;
  }

  // One mechanism covers the feed, model-page galleries, and lightbox thumbnails:
  // a 存 button on every image card, saving without navigating, and a persistent
  // 已存 mark on the ones already in the store.
  function markImageCards() {
    for (const a of imageCardLinks()) {
      const parsed = parseHref(a.href);
      if (!parsed || parsed.kind !== "image") continue;
      const state = savedImageIds.has(parsed.imageId) ? "saved" : "new";
      if (a.getAttribute("data-scb-recipe-card") === state) continue;
      a.setAttribute("data-scb-recipe-card", state);
      a.querySelector(":scope > .scb-card-save")?.remove();
      a.querySelector(":scope > .scb-card-saved")?.remove();
      ensurePositioned(a);
      a.appendChild(state === "saved" ? savedMarkEl() : saveButtonEl(a, parsed));
    }
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (ch) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[ch]));
  }

  async function runTick() {
    const href = location.href;
    const parsed = parseHref(href);
    if (href !== dockHref) {
      if (href !== dockPendingHref) {
        dockPendingHref = href;
        dockRetryAt = 0;
      }
      if (Date.now() >= dockRetryAt) {
        let rendered = true;
        if (parsed && parsed.kind === "model") rendered = await onModelPage(parsed);
        else if (parsed && parsed.kind === "image") rendered = await onImagePage(parsed);
        else {
          const page = document.getElementById("scb-page");
          if (page) {
            page.hidden = true;
            page.innerHTML = "";
          }
        }
        if (rendered) dockHref = href;
        else dockRetryAt = Date.now() + DOCK_RETRY_MS;
      }
    }
    // Image cards first, and without awaiting the store lookup: markCards() blocks on
    // the backend, and a stalled call used to mean no buttons rendered at all.
    markImageCards();
    refreshSavedImages().then((changed) => { if (changed) markImageCards(); }).catch(() => {});
    await markCards();
  }

  // Two overlapping ticks would both renderDock, and the first one's querySelector
  // would then bind a second listener to the replaced node: one click, two POSTs.
  async function tick() {
    if (tickBusy) {
      tickAgain = true;
      return;
    }
    tickBusy = true;
    try {
      await runTick();
    } finally {
      tickBusy = false;
      if (tickAgain) {
        tickAgain = false;
        schedule();
      }
    }
  }

  function schedule() {
    if (tickBusy) {
      tickAgain = true;
      return;
    }
    clearTimeout(scanTimer);
    scanTimer = setTimeout(() => {
      tick().catch(() => {});
    }, 400);
  }

  function isOurNode(node) {
    if (node.nodeType !== 1) return false;
    if (node.id === "scb-dock") return true;
    return [...node.classList].some((c) => c.startsWith("scb-"));
  }

  // Our own badges and buttons are DOM writes too. Without this filter the observer
  // retriggers itself and the scan loop never settles, even on an idle page.
  const mo = new MutationObserver((records) => {
    for (const r of records) {
      // Anything inside our own elements is our own write: the class-less <i> in a
      // progress bar, and the text nodes an innerHTML template leaves behind. Testing
      // the parent also covers removals, whose parentNode is already null.
      if (isOurNode(r.target)) continue;
      for (const n of r.addedNodes) if (!isOurNode(n)) return schedule();
      for (const n of r.removedNodes) if (!isOurNode(n)) return schedule();
    }
  });
  mo.observe(document.documentElement, { childList: true, subtree: true });

  // A pushState navigation can change the url without a mutation the observer keeps.
  setInterval(() => { if (location.href !== dockHref) schedule(); }, 1500);

  schedule();
  // One poll at startup picks up a download queued elsewhere; it stops on its own when
  // nothing is active. Polling used to be re-armed by every tick, which on a live feed
  // meant a request to the backend a few times a second forever.
  watchProgress();
})();
