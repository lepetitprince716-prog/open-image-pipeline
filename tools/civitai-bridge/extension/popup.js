const statusEl = document.getElementById("status");
const detailEl = document.getElementById("detail");
const jobsEl = document.getElementById("jobs");
const recipesEl = document.getElementById("recipes");
const modeEl = document.getElementById("mode");
const lmPortEl = document.getElementById("lmPort");
const studioPortEl = document.getElementById("studioPort");

const recipeState = new Map(); // recipe id -> button labels/notes, kept across repaints
let lastRecipes = [];
let lastRecipesTotal = 0;
let backendKind = null;
let studioPortLive = 8873;

function fmtBytes(n) {
  if (n == null || !Number.isFinite(Number(n))) return "";
  const v = Number(n);
  if (v < 1024 * 1024) return `${(v / 1024).toFixed(0)} KB`;
  if (v < 1024 * 1024 * 1024) return `${(v / 1024 / 1024).toFixed(1)} MB`;
  return `${(v / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

function paint(ok, text) {
  statusEl.textContent = ok ? "已连接" : "未连接";
  statusEl.className = `pill ${ok ? "on" : "off"}`;
  detailEl.textContent = text;
}

function esc(s) {
  return String(s == null ? "" : s).replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[ch]));
}

function recipeActions(r) {
  // Generation only exists on the Studio backend; LoRA Manager just stores recipes.
  if (backendKind !== "studio") {
    return r.missingCount ? `<span class="recipe-missing">缺 ${esc(r.missingCount)}</span>` : "";
  }
  const st = recipeState.get(r.id) || {};
  const fill = r.missingCount
    ? `<button type="button" class="recipe-btn" data-recipe-fill="${esc(r.id)}"${st.fillBusy ? " disabled" : ""}>${esc(st.fillLabel || `补齐 ${r.missingCount}`)}</button>`
    : "";
  const gen = `<button type="button" class="recipe-btn recipe-btn-go" data-recipe-gen="${esc(r.id)}"${st.genBusy ? " disabled" : ""}>${esc(st.genLabel || "出图")}</button>`;
  return fill + gen;
}

function recipeNote(r) {
  const st = recipeState.get(r.id) || {};
  if (!st.note) return "";
  const link = st.jobId
    ? ` <a href="http://127.0.0.1:${esc(studioPortLive)}/history/" target="_blank" rel="noreferrer">看进度</a>`
    : "";
  return `<div class="recipe-note">${esc(st.note)}${link}</div>`;
}

function paintRecipes(data) {
  const items = (data && data.items) || [];
  if (!items.length) {
    recipesEl.innerHTML = "";
    return;
  }
  lastRecipes = items;
  lastRecipesTotal = data.total ?? items.length;
  // Drop state for rows that scrolled out of the page: a recipe that leaves the top 5
  // and later comes back must not resurrect a stale label or an old jobId link.
  const live = new Set(items.map((r) => r.id));
  for (const id of [...recipeState.keys()]) {
    if (!live.has(id)) recipeState.delete(id);
  }
  recipesEl.innerHTML = `<div class="section-title">最近配方 (${esc(data.total ?? items.length)})</div>`
    + items.map((r) => `<div class="recipe-row">
        <span class="recipe-title">${esc(r.title || r.id)}</span>
        ${recipeActions(r)}
      </div>${recipeNote(r)}`).join("");
}

function renderRecipes() {
  chrome.runtime.sendMessage({ type: "recipes", pageSize: 5 }, (resp) => {
    if (chrome.runtime.lastError || !resp || !resp.ok || !resp.data) {
      recipesEl.innerHTML = "";
      return;
    }
    paintRecipes(resp.data);
  });
}

// Repaint from the rows already on screen: the 2s refresh must not wipe the label of
// a button the user just pressed, and a click must show its own result immediately.
function setRecipeState(id, patch) {
  recipeState.set(id, { ...(recipeState.get(id) || {}), ...patch });
  paintRecipes({ items: lastRecipes, total: lastRecipesTotal });
}

function startGenerate(id) {
  setRecipeState(id, { genBusy: true, genLabel: "排队中…", note: "" });
  chrome.runtime.sendMessage({ type: "recipeGenerate", recipeId: id }, (resp) => {
    if (chrome.runtime.lastError || !resp || !resp.ok) {
      const err = (chrome.runtime.lastError && chrome.runtime.lastError.message)
        || (resp && resp.error) || "出图失败";
      setRecipeState(id, { genBusy: false, genLabel: "重试", note: err, jobId: "" });
      return;
    }
    const jobId = (resp.data && resp.data.jobId) || "";
    const warnings = (resp.data && resp.data.warnings) || [];
    setRecipeState(id, {
      genBusy: false,
      genLabel: "再出一张",
      jobId,
      note: [jobId ? `已排 ${jobId}` : "已排队", ...warnings].join(" · "),
    });
  });
}

function fillMissing(id) {
  setRecipeState(id, { fillBusy: true, fillLabel: "补齐中…", note: "" });
  chrome.runtime.sendMessage({ type: "downloadMissing", recipeId: id }, (resp) => {
    if (chrome.runtime.lastError || !resp || !resp.ok) {
      const err = (chrome.runtime.lastError && chrome.runtime.lastError.message)
        || (resp && resp.error) || "补齐失败";
      setRecipeState(id, { fillBusy: false, fillLabel: "重试", note: err });
      return;
    }
    const d = resp.data || {};
    const queued = (d.queued || []).length;
    const skipped = (d.skipped || []);
    setRecipeState(id, {
      fillBusy: false,
      fillLabel: `已排 ${queued}`,
      note: skipped.length ? `跳过 ${skipped.length}：${skipped.map((s) => s.reason).join("; ")}` : "",
    });
  });
}

recipesEl.addEventListener("click", (ev) => {
  const target = ev.target instanceof Element ? ev.target : null;
  if (!target) return;
  const genBtn = target.closest("[data-recipe-gen]");
  if (genBtn) {
    startGenerate(genBtn.getAttribute("data-recipe-gen"));
    return;
  }
  const fillBtn = target.closest("[data-recipe-fill]");
  if (fillBtn) fillMissing(fillBtn.getAttribute("data-recipe-fill"));
});

// The background caches its backend probe. Opening the popup and saving settings are
// the moments that deserve a fresh one; the 2s repaint in between can read the cache.
async function refresh(fresh = false) {
  const stored = await chrome.storage.local.get({ mode: "auto", studioPort: 8873, lmPort: 8188 });
  modeEl.value = stored.mode || "auto";
  studioPortEl.value = stored.studioPort || 8873;
  lmPortEl.value = stored.lmPort || 8188;
  chrome.runtime.sendMessage({ type: "health", fresh }, (resp) => {
    if (chrome.runtime.lastError) {
      paint(false, chrome.runtime.lastError.message);
      return;
    }
    if (!resp || !resp.ok) {
      paint(false, (resp && resp.error) || "本机没有 Studio，也没有 LoRA Manager");
      return;
    }
    const d = resp.data || {};
    backendKind = d.backend === "lm" ? "lm" : "studio";
    studioPortLive = Number(stored.studioPort) || 8873;
    if (d.backend === "lm") {
      paint(true, `本机 LoRA Manager :${d.port} · 下到 E:`);
    } else {
      const n = d.index && d.index.records;
      paint(true, `Studio :${d.port || 8873}${n ? ` · 库 ${n}` : ""}`);
    }
    chrome.runtime.sendMessage({ type: "progress" }, (prog) => {
      if (!prog || !prog.ok || !prog.data || !(prog.data.jobs || []).length) {
        jobsEl.innerHTML = "";
        return;
      }
      jobsEl.innerHTML = prog.data.jobs.slice(0, 6).map((j) => {
        const pct = Math.round(Number(j.progress) || 0);
        const size = [fmtBytes(j.bytesDownloaded), fmtBytes(j.totalBytes)].filter(Boolean).join(" / ");
        return `<div class="job">
          <div class="job-name">${esc(j.name || j.downloadId)}</div>
          <div class="bar"><i style="width:${j.active ? pct : 100}%"></i></div>
          <div class="job-meta">${esc(j.status)}${j.active ? ` ${pct}%` : ""} ${esc(size)}</div>
        </div>`;
      }).join("");
    });
    renderRecipes();
  });
}

document.getElementById("save").addEventListener("click", async () => {
  await chrome.storage.local.set({
    mode: modeEl.value,
    studioPort: Number(studioPortEl.value) || 8873,
    lmPort: Number(lmPortEl.value) || 8188,
  });
  refresh(true);
});

refresh(true);
setInterval(() => { refresh(); }, 2000);
