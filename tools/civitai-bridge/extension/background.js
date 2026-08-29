const DEFAULTS = { mode: "auto", studioPort: 8873, lmPort: 8188 };
// The standalone LoRA Manager tunnel (scripts/z3_lora_manager.py) listens here; probed
// as a fallback so the stored 8188 default keeps working for the ComfyUI-embedded LM.
const LM_FALLBACK_PORT = 9000;
let lmPortLive = null; // last LM port that answered this session

// A forwarded loopback port can hang instead of refusing, so every request carries a
// deadline. Probes fail fast because the page waits on them; a download kickoff or a
// recipe import runs against Civitai server-side and legitimately takes much longer.
const PROBE_TIMEOUT_MS = 2500;
const DEFAULT_TIMEOUT_MS = 15000;
const SLOW_TIMEOUT_MS = 45000;

async function settings() {
  const stored = await chrome.storage.local.get(DEFAULTS);
  return { ...DEFAULTS, ...stored };
}

function baseUrl(port) {
  return `http://127.0.0.1:${port}`;
}

async function fetchJson(url, init = {}, timeoutMs = DEFAULT_TIMEOUT_MS) {
  let res;
  try {
    res = await fetch(url, { ...init, signal: AbortSignal.timeout(timeoutMs) });
  } catch (err) {
    if (err && (err.name === "TimeoutError" || err.name === "AbortError")) {
      throw new Error(`超时 ${Math.round(timeoutMs / 1000)}s: ${url}`);
    }
    throw err;
  }
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = { detail: text.slice(0, 200) }; }
  if (!res.ok) {
    const detail = (data && (data.detail || data.error)) || res.statusText;
    const err = new Error(String(detail));
    err.status = res.status;
    throw err;
  }
  return data;
}

async function probe() {
  const s = await settings();
  const studio = `${baseUrl(s.studioPort)}/api/civitai-bridge/health`;
  const tryStudio = async () => {
    const data = await fetchJson(studio, {}, PROBE_TIMEOUT_MS);
    return { kind: "studio", port: s.studioPort, data };
  };
  const tryLmPort = async (port) => {
    const data = await fetchJson(`${baseUrl(port)}/api/lm/health-check`, {}, PROBE_TIMEOUT_MS);
    if (!data || (data.status && data.status !== "ok" && data.success === false)) {
      throw new Error("LM unhealthy");
    }
    lmPortLive = port;
    return { kind: "lm", port, data };
  };
  const tryLm = async () => {
    const ports = [...new Set([lmPortLive, s.lmPort, LM_FALLBACK_PORT].filter(Boolean))];
    let lastErr = null;
    for (const port of ports) {
      try { return await tryLmPort(port); } catch (err) { lastErr = err; }
    }
    throw lastErr || new Error("LM unreachable");
  };
  if (s.mode === "studio") return tryStudio();
  if (s.mode === "lm") return tryLm();
  try { return await tryStudio(); } catch { return tryLm(); }
}

// Every message used to re-probe, which on a busy feed meant a health check a few
// times a second per tab. Cache the answer for a few seconds; cache a failure only
// briefly so starting Studio recovers on its own instead of staying "未连接".
const PROBE_TTL_OK_MS = 12000;
const PROBE_TTL_FAIL_MS = 1500;
// A click must never be answered out of a stale cache. "health" is not in the set:
// the popup polls it every couple of seconds, and the short failure ttl already gets
// it back to 已连接 within ~1.5s of Studio starting. The popup asks for a fresh probe
// explicitly ({ fresh: true }) when it opens and after settings are saved.
const FORCE_PROBE = new Set(["download", "recipe", "downloadMissing", "recipeGenerate"]);
let probeCache = null; // { at, backend } | { at, backend: null, error }
let probeInFlight = null;

async function probeCached(force) {
  if (!force && probeCache) {
    const ttl = probeCache.backend ? PROBE_TTL_OK_MS : PROBE_TTL_FAIL_MS;
    if (Date.now() - probeCache.at < ttl) {
      if (probeCache.backend) return probeCache.backend;
      throw probeCache.error;
    }
  }
  if (probeInFlight) return probeInFlight;
  const run = probe().then(
    (backend) => { probeCache = { at: Date.now(), backend }; return backend; },
    (err) => { probeCache = { at: Date.now(), backend: null, error: err }; throw err; },
  );
  probeInFlight = run;
  run.catch(() => {}).finally(() => { if (probeInFlight === run) probeInFlight = null; });
  return run;
}

function mapOwnedFromLm(ids, payload) {
  const byId = new Map();
  for (const row of payload.results || []) byId.set(Number(row.modelId), row);
  return {
    results: ids.map((q) => {
      const mid = Number(q.modelId);
      const vid = q.versionId != null ? Number(q.versionId) : null;
      const row = byId.get(mid);
      const versions = (row && row.versions) || [];
      const hit = vid != null ? versions.find((v) => Number(v.versionId) === vid) : versions[0];
      if (hit) {
        return {
          status: "exact",
          file: hit.fileName || null,
          name: hit.name || hit.fileName || null,
          match: vid != null ? "version" : "model",
          modelId: mid,
          versionId: vid,
        };
      }
      if (versions.length) {
        return {
          status: "other_version",
          file: versions[0].fileName || null,
          name: versions[0].name || null,
          match: "model",
          modelId: mid,
          versionId: vid,
          ownedVersionIds: versions.map((v) => v.versionId),
        };
      }
      return { status: "missing", file: null, name: null, modelId: mid, versionId: vid };
    }),
  };
}

function mapJobsFromLm(queuePayload, historyPayload) {
  const queue = (queuePayload && queuePayload.queue) || [];
  const history = (historyPayload && (historyPayload.items || historyPayload.history)) || [];
  const seen = new Set();
  const jobs = [];
  const push = (item, fallbackStatus) => {
    const id = String(item.download_id || item.downloadId || "");
    if (!id || seen.has(id)) return;
    seen.add(id);
    const statusRaw = String(item.status || fallbackStatus || "unknown").toLowerCase();
    const status = ({ progress: "downloading", waiting: "queued", already: "completed", cancelled: "canceled" })[statusRaw] || statusRaw;
    const bytes = item.bytes_downloaded ?? item.bytesDownloaded;
    const total = item.total_bytes ?? item.totalBytes;
    let progress = item.progress;
    if (progress == null && bytes != null && total) progress = (Number(bytes) / Number(total)) * 100;
    jobs.push({
      downloadId: id,
      name: item.model_name || item.modelName || item.name,
      status,
      progress,
      bytesDownloaded: bytes,
      totalBytes: total,
      bytesPerSecond: item.bytes_per_second,
      active: ["queued", "downloading", "paused"].includes(status),
    });
  };
  queue.forEach((item) => push(item));
  history.forEach((item) => push(item, "completed"));
  return { jobs, active: jobs.filter((j) => j.active), stats: (queuePayload && queuePayload.stats) || {} };
}

async function studioCall(port, path, body, timeoutMs) {
  return fetchJson(`${baseUrl(port)}${path}`, body === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }, timeoutMs);
}

// Image ids that already have a recipe, read out of whichever store is answering.
// Both backends expose the source url; the id is what the content script matches on.
// The Studio route clamps page_size to 100, so a bigger number silently truncates.
// Walk pages instead, bounded so a large store cannot stall the popup.
const SAVED_PAGE_SIZE = 100;
const SAVED_MAX_PAGES = 5;

async function collectSavedImageIds(fetchPage) {
  const ids = [];
  for (let page = 1; page <= SAVED_MAX_PAGES; page++) {
    const data = await fetchPage(page);
    const items = (data && data.items) || [];
    if (!items.length) break;
    ids.push(...imageIdsFromRecipes(items));
    // Keep going only while a page comes back full. `total` is a hint, not a contract —
    // the LM-direct payload may omit it, and Number(undefined) would end the walk early.
    if (items.length < SAVED_PAGE_SIZE) break;
  }
  return ids;
}

function imageIdsFromRecipes(items) {
  const ids = [];
  for (const item of items || []) {
    const src = String(item.sourceUrl || item.source_path || item.source_url || "");
    const m = src.match(/\/images\/(\d+)/);
    if (m) ids.push(Number(m[1]));
  }
  return ids;
}

const SLOW_MESSAGES = new Set(["download", "recipe", "downloadMissing"]);

async function handle(message) {
  const slow = SLOW_MESSAGES.has(message.type) ? SLOW_TIMEOUT_MS : DEFAULT_TIMEOUT_MS;
  const backend = await probeCached(FORCE_PROBE.has(message.type) || Boolean(message.fresh));
  if (message.type === "health") {
    if (backend.kind === "studio") {
      return { ...backend.data, backend: "studio", port: backend.port };
    }
    return {
      ok: true,
      studio: false,
      backend: "lm",
      port: backend.port,
      loraManager: { ok: true, source: `127.0.0.1:${backend.port}` },
      index: { ok: true },
    };
  }
  if (backend.kind === "studio") {
    if (message.type === "owned") {
      return studioCall(backend.port, "/api/civitai-bridge/owned", { ids: message.ids || [], enrich: Boolean(message.enrich) });
    }
    if (message.type === "inspect") {
      return studioCall(backend.port, "/api/civitai-bridge/inspect", {
        url: message.url || "", imageId: message.imageId, modelId: message.modelId,
        versionId: message.versionId, site: message.site,
      });
    }
    if (message.type === "download") {
      return studioCall(backend.port, "/api/civitai-bridge/download", {
        modelId: message.modelId, versionId: message.versionId,
        site: message.site, allowNsfw: message.allowNsfw,
      }, slow);
    }
    if (message.type === "recipe") {
      return studioCall(backend.port, "/api/civitai-bridge/recipe", {
        imageId: message.imageId, url: message.url || "", site: message.site, name: message.name,
      }, slow);
    }
    if (message.type === "progress") {
      const q = message.downloadId ? `?downloadId=${encodeURIComponent(message.downloadId)}` : "";
      return fetchJson(`${baseUrl(backend.port)}/api/civitai-bridge/progress${q}`);
    }
    if (message.type === "recipes") {
      const qs = new URLSearchParams();
      if (message.pageSize) qs.set("pageSize", String(message.pageSize));
      if (message.page) qs.set("page", String(message.page));
      if (message.search) qs.set("search", String(message.search));
      const tail = qs.toString() ? `?${qs}` : "";
      return fetchJson(`${baseUrl(backend.port)}/api/civitai-bridge/recipes${tail}`);
    }
    if (message.type === "downloadMissing") {
      return studioCall(backend.port, "/api/civitai-bridge/download-missing", {
        recipeId: message.recipeId, resources: message.resources, allowNsfw: message.allowNsfw,
      }, slow);
    }
    if (message.type === "savedImages") {
      const ids = await collectSavedImageIds((page) =>
        fetchJson(`${baseUrl(backend.port)}/api/civitai-bridge/recipes?pageSize=${SAVED_PAGE_SIZE}&page=${page}`));
      return { imageIds: ids };
    }
    if (message.type === "recipeGenerate") {
      const rid = encodeURIComponent(String(message.recipeId || ""));
      return studioCall(backend.port, `/api/civitai-bridge/recipes/${rid}/generate`, {
        overrides: message.overrides,
      });
    }
  } else {
    const root = baseUrl(backend.port);
    if (message.type === "owned") {
      const ids = message.ids || [];
      const modelIds = [...new Set(ids.map((x) => x.modelId).filter(Boolean))].join(",");
      const payload = modelIds
        ? await fetchJson(`${root}/api/lm/check-models-exist?modelIds=${modelIds}`)
        : { results: [] };
      return mapOwnedFromLm(ids, payload);
    }
    if (message.type === "inspect") {
      if (message.imageId) {
        // LM-direct cannot parse the image; null = "unknown", not "ready" (0.3.0 fix).
        return { kind: "image", imageId: message.imageId, title: "Civitai image", resources: [], genParams: null, recipeReady: null };
      }
      const owned = mapOwnedFromLm(
        [{ modelId: message.modelId, versionId: message.versionId }],
        await fetchJson(`${root}/api/lm/check-models-exist?modelIds=${message.modelId || ""}`),
      );
      return { kind: "model", modelId: message.modelId, versionId: message.versionId, owned: owned.results[0] };
    }
    if (message.type === "download") {
      const downloadId = `scb_${message.modelId || 0}_${message.versionId || 0}_${Date.now()}`;
      const qs = new URLSearchParams({
        model_id: String(message.modelId || ""),
        use_default_paths: "true",
        download_id: downloadId,
      });
      if (message.versionId) qs.set("model_version_id", String(message.versionId));
      const data = await fetchJson(`${root}/api/lm/download-model-get?${qs}`, {}, slow);
      if (data && data.success === false) throw new Error(data.error || "download failed");
      const already = /already exists/i.test(String((data && data.error) || ""));
      return { ok: true, already, downloadId, modelId: message.modelId, versionId: message.versionId };
    }
    if (message.type === "recipe") {
      const imageUrl = message.url || `https://civitai.com/images/${message.imageId}`;
      const qs = new URLSearchParams({ image_url: imageUrl });
      const data = await fetchJson(`${root}/api/lm/recipes/import-from-url?${qs}`, {}, slow);
      if (data && data.error && !data.success) throw new Error(data.error);
      return { ok: true, recipe: data, title: message.name || imageUrl };
    }
    if (message.type === "progress") {
      const [queue, history] = await Promise.all([
        fetchJson(`${root}/api/lm/downloads/queue`).catch(() => ({ queue: [] })),
        fetchJson(`${root}/api/lm/downloads/history?limit=8`).catch(() => ({ items: [] })),
      ]);
      return mapJobsFromLm(queue, history);
    }
    if (message.type === "recipes") {
      const qs = new URLSearchParams({ page: String(message.page || 1), page_size: String(message.pageSize || 5) });
      if (message.search) qs.set("search", String(message.search));
      const data = await fetchJson(`${root}/api/lm/recipes?${qs}`);
      const items = ((data && data.items) || []).map((item) => ({
        id: item.id,
        title: item.title,
        sourceUrl: item.source_path,
        missingCount: (item.loras || []).filter((l) => !l.exclude && (!l.inLibrary || l.isDeleted)).length,
      }));
      return { ok: true, items, total: data && data.total };
    }
    if (message.type === "downloadMissing") {
      const queued = [];
      const skipped = [];
      const already = [];
      for (const res of message.resources || []) {
        const versionId = res.modelVersionId || res.versionId;
        if (!versionId && !res.modelId) {
          skipped.push({ name: res.modelName || res.name || "?", reason: "no version id" });
          continue;
        }
        const downloadId = `scb_${res.modelId || 0}_${versionId || 0}_${Date.now()}`;
        const qs = new URLSearchParams({ use_default_paths: "true", download_id: downloadId });
        if (res.modelId) qs.set("model_id", String(res.modelId));
        if (versionId) qs.set("model_version_id", String(versionId));
        try {
          const data = await fetchJson(`${root}/api/lm/download-model-get?${qs}`, {}, slow);
          const errText = String((data && data.error) || "");
          if (/already exists/i.test(errText)) {
            already.push({ name: res.modelName || res.name });
            continue;
          }
          if (data && data.success === false) throw new Error(data.error || "download failed");
          queued.push({ downloadId, name: res.modelName || res.name, versionId });
        } catch (err) {
          skipped.push({ name: res.modelName || res.name || "?", reason: String(err && err.message || err) });
        }
      }
      return { ok: true, queued, skipped, already };
    }
    if (message.type === "savedImages") {
      const ids = await collectSavedImageIds((page) =>
        fetchJson(`${root}/api/lm/recipes?page=${page}&page_size=${SAVED_PAGE_SIZE}`));
      return { imageIds: ids };
    }
    if (message.type === "recipeGenerate") {
      // LoRA Manager stores recipes but cannot render; generation lives in Studio.
      throw new Error("LM 直连模式不出图，请切到 Studio 后端");
    }
  }
  if (message.type === "settings") return settings();
  throw new Error(`unknown message ${message.type}`);
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  handle(message)
    .then((data) => sendResponse({ ok: true, data }))
    .catch((err) => sendResponse({ ok: false, error: String(err && err.message ? err.message : err) }));
  return true;
});
