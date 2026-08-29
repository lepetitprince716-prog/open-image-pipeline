#!/usr/bin/env python3
"""Static interactive gallery for the Z3 model index — browse + copy, no backend.

Generates a single self-contained HTML page (`z3_model_index/browser.html`) served by the existing
serve_preview.py (127.0.0.1:8770). Merges two local JSONs so nothing is duplicated:
  - z3_model_index/index.json  -> checkpoints + name-only support kinds (built by z3_model_index.py)
  - lora_catalog/z3_index.json -> the full LoRA records (lora_catalog stays the single source)

The page embeds the merged payload inline and does all filtering client-side (search / kind tabs /
base-model / NSFW toggle). LoRA & checkpoint previews load from Z3 (preview is a relative path, so
the page prefixes the host); off-LAN the <img> fails gracefully to a name chip. Per-card buttons copy
the canonical name (the exact value LoraLoader/CheckpointLoaderSimple accept) and the triggers.

Usage:
    python3 scripts/z3_model_browser.py build      # write the page, print its serve_preview URL
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from z3_comfy import Z3_HOST

ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = ROOT / "z3_model_index" / "index.json"
LORA_INDEX = ROOT / "lora_catalog" / "z3_index.json"
OUT = ROOT / "z3_model_index" / "browser.html"
PORT = (os.environ.get("LORAMAKE_PREVIEW_PORT", "8770").strip() or "8770")

# Order + display labels for the kind tabs (data keys -> label). Only kinds with items render.
KIND_LABELS = {
    "lora": "LoRA",
    "checkpoint": "Checkpoint",
    "vae": "VAE",
    "controlnet": "ControlNet",
    "upscale_models": "Upscale",
    "clip_vision": "CLIP-Vision",
    "diffusion_models": "Diffusion",
    "ipadapter": "IP-Adapter",
}


def _lora_cards() -> list[dict]:
    """Normalize full LoRA records into the shared card shape; [] if the LoRA index is missing."""
    if not LORA_INDEX.exists():
        return []
    out = []
    for r in json.loads(LORA_INDEX.read_text()):
        out.append({
            "kind": "lora",
            "canonical_name": r.get("file", ""),
            "name": r.get("name", ""),
            "base_model": r.get("base_model", "unknown"),
            "content_rating": r.get("content_rating", "unknown"),
            "preview": r.get("preview"),
            "triggers": r.get("triggers", []),
            "tags": r.get("tags", []),
            "strength": r.get("strength_default"),
        })
    return out


def _normalize_asset(it: dict) -> dict:
    """Coerce an index.json asset record (checkpoint / name-only kind) into the card shape."""
    return {
        "kind": it.get("kind", "unknown"),
        "canonical_name": it.get("canonical_name", ""),
        "name": it.get("name", ""),
        "base_model": it.get("base_model", "unknown"),
        "content_rating": it.get("content_rating", "unknown"),
        "preview": it.get("preview"),
        "triggers": [],
        "tags": it.get("tags", []),
        "strength": None,
    }


def _payload() -> dict:
    """Merge the LoRA index and the unified index into one client payload."""
    assets: dict[str, list[dict]] = {"lora": _lora_cards()}
    generated_at = None
    refresh_status = None
    if INDEX_PATH.exists():
        idx = json.loads(INDEX_PATH.read_text())
        generated_at = idx.get("generated_at")
        refresh_status = idx.get("refresh_status")
        for kind, items in (idx.get("assets") or {}).items():
            # data key must match the per-record `kind` and the JS LABELS (collection key is plural)
            key = "checkpoint" if kind == "checkpoints" else kind
            assets[key] = [_normalize_asset(it) for it in items]
    counts = {k: len(v) for k, v in assets.items() if v}
    return {
        "host": Z3_HOST.rstrip("/"),
        "generated_at": generated_at,
        "refresh_status": refresh_status,
        "counts": counts,
        "assets": {k: v for k, v in assets.items() if v},
    }


def build() -> str:
    """Write the gallery page and return its repo-relative serve_preview path."""
    payload = _payload()
    data_json = json.dumps(payload, ensure_ascii=False)
    html = _TEMPLATE.replace("__DATA__", data_json)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    rel = "/" + OUT.relative_to(ROOT).as_posix()
    print(f"http://127.0.0.1:{PORT}{rel}")
    return rel


# --------------------------------------------------------------------------- page template
# __DATA__ is replaced with the JSON payload. CSS/JS use no Python interpolation (literal braces).
_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Z3 model index</title>
<style>
  :root {
    --bg: oklch(16% 0.012 265);
    --surface: oklch(21% 0.014 265);
    --surface-2: oklch(25% 0.016 265);
    --line: oklch(32% 0.018 265);
    --text: oklch(94% 0.005 265);
    --muted: oklch(68% 0.012 265);
    --faint: oklch(54% 0.012 265);
    --accent: oklch(72% 0.15 230);
    --ok: oklch(72% 0.16 150);
    --warn: oklch(72% 0.17 55);
    --nsfw: oklch(66% 0.2 20);
    --radius: 13px;
    --gap: 14px;
    --dur: 160ms;
    --ease: cubic-bezier(0.16, 1, 0.3, 1);
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--text);
    font: 14px/1.5 ui-sans-serif, -apple-system, system-ui, sans-serif;
  }
  code, .mono { font-family: ui-monospace, "SF Mono", Menlo, monospace; }

  header {
    position: sticky; top: 0; z-index: 10;
    background: color-mix(in oklab, var(--bg) 82%, transparent);
    backdrop-filter: blur(14px) saturate(1.2);
    border-bottom: 1px solid var(--line);
    padding: 14px clamp(14px, 3vw, 30px) 10px;
  }
  .title-row { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; }
  h1 { font-size: 17px; font-weight: 650; letter-spacing: -0.01em; margin: 0; }
  .meta { font-size: 11.5px; color: var(--faint); font-family: ui-monospace, monospace; }
  .meta b { color: var(--muted); font-weight: 600; }

  .controls { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-top: 12px; }
  .search { flex: 1 1 280px; position: relative; }
  .search input {
    width: 100%; padding: 9px 12px 9px 32px; border-radius: 10px;
    border: 1px solid var(--line); background: var(--surface); color: var(--text);
    font-size: 13.5px; transition: border-color var(--dur) var(--ease);
  }
  .search input:focus { outline: none; border-color: var(--accent); }
  .search::before {
    content: ""; position: absolute; left: 11px; top: 50%; width: 13px; height: 13px;
    transform: translateY(-50%);
    background: var(--faint);
    -webkit-mask: no-repeat center/contain url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="black" stroke-width="2"><circle cx="11" cy="11" r="7"/><path d="m21 21-4.3-4.3"/></svg>');
    mask: no-repeat center/contain url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="black" stroke-width="2"><circle cx="11" cy="11" r="7"/><path d="m21 21-4.3-4.3"/></svg>');
  }
  select, .toggle {
    padding: 8px 11px; border-radius: 10px; border: 1px solid var(--line);
    background: var(--surface); color: var(--text); font-size: 13px; cursor: pointer;
  }
  .toggle { display: inline-flex; align-items: center; gap: 7px; user-select: none; }
  .toggle input { accent-color: var(--nsfw); width: 15px; height: 15px; }

  .tabs { display: flex; gap: 7px; flex-wrap: wrap; margin-top: 12px; }
  .tab {
    padding: 5px 12px; border-radius: 999px; border: 1px solid var(--line);
    background: transparent; color: var(--muted); font-size: 12.5px; cursor: pointer;
    transition: all var(--dur) var(--ease); white-space: nowrap;
  }
  .tab:hover { border-color: var(--faint); color: var(--text); }
  .tab.active { background: var(--accent); border-color: var(--accent); color: oklch(18% 0.02 265); font-weight: 600; }
  .tab .n { opacity: 0.7; font-family: ui-monospace, monospace; font-size: 11px; }
  .tab.active .n { opacity: 0.85; }

  main { padding: clamp(14px, 3vw, 26px); }
  .count-line { color: var(--faint); font-size: 12px; margin: 0 0 14px; font-family: ui-monospace, monospace; }
  .grid { display: grid; gap: var(--gap); grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); }

  .card {
    background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius);
    overflow: hidden; display: flex; flex-direction: column;
    transition: transform var(--dur) var(--ease), border-color var(--dur) var(--ease);
  }
  .card:hover { transform: translateY(-3px); border-color: var(--faint); }
  .thumb { aspect-ratio: 3 / 4; background: var(--surface-2); position: relative; overflow: hidden; }
  .thumb img { width: 100%; height: 100%; object-fit: cover; display: block; }
  .thumb.noimg { display: grid; place-items: center; aspect-ratio: 16 / 7; }
  .thumb .ph { display: none; }
  .thumb.noimg .ph { display: block; font-family: ui-monospace, monospace; font-size: 11px; color: var(--faint); }
  .rating-dot {
    position: absolute; top: 9px; right: 9px; width: 9px; height: 9px; border-radius: 50%;
    box-shadow: 0 0 0 2px color-mix(in oklab, var(--bg) 60%, transparent);
  }
  .r-sfw { background: var(--ok); } .r-unknown { background: var(--faint); } .r-nsfw { background: var(--nsfw); }

  .body { padding: 11px 12px 12px; display: flex; flex-direction: column; gap: 7px; flex: 1; }
  .chips { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }
  .chip {
    font-size: 10.5px; font-weight: 600; padding: 2px 8px; border-radius: 6px;
    text-transform: uppercase; letter-spacing: 0.03em;
  }
  .base { font-size: 10.5px; color: var(--muted); padding: 2px 7px; border: 1px solid var(--line); border-radius: 6px; }
  .nm { font-size: 13.5px; font-weight: 600; line-height: 1.3; letter-spacing: -0.01em; }
  .canon { font-size: 11px; color: var(--faint); word-break: break-all; }
  .trig { font-size: 11px; color: var(--muted); word-break: break-word;
          display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
  .str { font-size: 11px; color: var(--accent); font-family: ui-monospace, monospace; }
  .actions { display: flex; gap: 7px; margin-top: auto; padding-top: 4px; }
  .btn {
    flex: 1; padding: 6px 8px; border-radius: 8px; border: 1px solid var(--line);
    background: var(--surface-2); color: var(--text); font-size: 11.5px; cursor: pointer;
    transition: all var(--dur) var(--ease);
  }
  .btn:hover { border-color: var(--accent); color: var(--accent); }
  .btn:active { transform: scale(0.97); }
  .btn.copied { background: var(--ok); border-color: var(--ok); color: oklch(18% 0.02 265); }

  .empty { text-align: center; color: var(--faint); padding: 60px 20px; font-size: 14px; }
  .more { text-align: center; color: var(--faint); font-size: 12px; padding: 18px; font-family: ui-monospace, monospace; }
  @media (max-width: 520px) { .grid { grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); } }
</style>
</head>
<body>
<header>
  <div class="title-row">
    <h1>Z3 model index</h1>
    <span class="meta" id="meta"></span>
  </div>
  <div class="controls">
    <label class="search"><input id="q" type="search" placeholder="search name / trigger / tag / base…" autocomplete="off"></label>
    <select id="base"><option value="">all bases</option></select>
    <label class="toggle"><input id="nsfw" type="checkbox"> show NSFW</label>
  </div>
  <div class="tabs" id="tabs"></div>
</header>
<main>
  <p class="count-line" id="countline"></p>
  <div class="grid" id="grid"></div>
  <div class="more" id="more"></div>
</main>

<script type="application/json" id="data">__DATA__</script>
<script>
(function () {
  const DATA = JSON.parse(document.getElementById("data").textContent);
  const LABELS = {
    lora: "LoRA", checkpoint: "Checkpoint", vae: "VAE", controlnet: "ControlNet",
    upscale_models: "Upscale", clip_vision: "CLIP-Vision", diffusion_models: "Diffusion", ipadapter: "IP-Adapter"
  };
  const KIND_HUE = { lora: 230, checkpoint: 285, vae: 150, controlnet: 55, upscale_models: 20, clip_vision: 330, diffusion_models: 195, ipadapter: 100 };
  const MAX_RENDER = 400;
  const HOST = DATA.host || "";
  const ORDER = Object.keys(LABELS).filter(k => DATA.assets[k] && DATA.assets[k].length);
  // tag every record with its kind once (records are reused across renders)
  ORDER.forEach(k => DATA.assets[k].forEach(r => { r.kind = k; }));

  const state = { kind: ORDER[0] || "lora", q: "", base: "", nsfw: false };

  const el = id => document.getElementById(id);
  const grid = el("grid"), more = el("more"), tabsEl = el("tabs"), baseSel = el("base"), countLine = el("countline");

  // meta line
  const total = Object.values(DATA.counts).reduce((a, b) => a + b, 0);
  el("meta").innerHTML = `<b>${total}</b> assets · host <b>${HOST.replace(/^https?:\/\//, "")}</b>` +
    (DATA.generated_at ? ` · refreshed <b>${DATA.generated_at}</b>` : " · <b>no index.json yet</b>") +
    (DATA.refresh_status && DATA.refresh_status !== "ok" ? ` · status <b>${DATA.refresh_status}</b>` : "");

  // base dropdown (union of bases across all kinds)
  const bases = new Set();
  ORDER.forEach(k => DATA.assets[k].forEach(r => r.base_model && bases.add(r.base_model)));
  [...bases].sort((a, b) => a.localeCompare(b)).forEach(b => {
    const o = document.createElement("option"); o.value = b; o.textContent = b; baseSel.appendChild(o);
  });

  function visibleByNsfw(r) { return state.nsfw || r.content_rating !== "nsfw"; }

  function kindCount(k) { return DATA.assets[k].filter(visibleByNsfw).length; }

  function renderTabs() {
    tabsEl.innerHTML = "";
    ORDER.forEach(k => {
      const b = document.createElement("button");
      b.className = "tab" + (k === state.kind ? " active" : "");
      b.innerHTML = `${LABELS[k]} <span class="n">${kindCount(k)}</span>`;
      b.onclick = () => { state.kind = k; renderTabs(); render(); };
      tabsEl.appendChild(b);
    });
  }

  function matches(r) {
    if (!visibleByNsfw(r)) return false;
    if (state.base && r.base_model !== state.base) return false;
    const q = state.q.trim().toLowerCase();
    if (!q) return true;
    const hay = [r.canonical_name, r.name, r.base_model, (r.tags || []).join(" "), (r.triggers || []).join(" ")].join(" ").toLowerCase();
    if (hay.includes(q)) return true;
    return q.split(/\s+/).every(t => hay.includes(t));
  }

  function copy(btn, text) {
    navigator.clipboard.writeText(text).then(() => {
      const old = btn.textContent;
      btn.textContent = "copied ✓"; btn.classList.add("copied");
      setTimeout(() => { btn.textContent = old; btn.classList.remove("copied"); }, 1100);
    }).catch(() => { btn.textContent = "copy failed"; });
  }

  function card(r) {
    const c = document.createElement("div");
    c.className = "card";
    const hue = KIND_HUE[r.kind] ?? 230;
    const ratingClass = r.content_rating === "nsfw" ? "r-nsfw" : (r.content_rating === "sfw" ? "r-sfw" : "r-unknown");

    let thumb;
    if (r.preview) {
      thumb = `<div class="thumb"><span class="rating-dot ${ratingClass}"></span>` +
        `<img loading="lazy" decoding="async" src="${HOST}${r.preview}" alt="" ` +
        `onerror="this.closest('.thumb').classList.add('noimg');this.style.display='none'">` +
        `<span class="ph">no preview</span></div>`;
    } else {
      thumb = `<div class="thumb noimg"><span class="rating-dot ${ratingClass}"></span><span class="ph">${LABELS[r.kind] || r.kind}</span></div>`;
    }

    const trig = (r.triggers || []).join(", ");
    const strLine = (r.strength != null) ? `<div class="str">str ${r.strength}</div>` : "";
    const trigLine = trig ? `<div class="trig" title="${escapeAttr(trig)}">⚑ ${escapeHtml(trig)}</div>` : "";
    const copyTrig = trig ? `<button class="btn" data-copy="trig">copy triggers</button>` : "";

    c.innerHTML = thumb +
      `<div class="body">` +
        `<div class="chips">` +
          `<span class="chip" style="background:oklch(70% 0.14 ${hue} / 0.18);color:oklch(82% 0.12 ${hue})">${LABELS[r.kind] || r.kind}</span>` +
          (r.base_model ? `<span class="base">${escapeHtml(r.base_model)}</span>` : "") +
        `</div>` +
        `<div class="nm">${escapeHtml(r.name || r.canonical_name)}</div>` +
        `<div class="canon mono">${escapeHtml(r.canonical_name)}</div>` +
        trigLine + strLine +
        `<div class="actions">` +
          `<button class="btn" data-copy="name">copy name</button>` + copyTrig +
        `</div>` +
      `</div>`;

    c.querySelector('[data-copy="name"]').onclick = e => copy(e.target, r.canonical_name);
    const ct = c.querySelector('[data-copy="trig"]');
    if (ct) ct.onclick = e => copy(e.target, trig);
    return c;
  }

  function render() {
    const items = DATA.assets[state.kind] || [];
    const hits = items.filter(matches);
    grid.innerHTML = "";
    if (!hits.length) { grid.innerHTML = `<div class="empty">no ${LABELS[state.kind] || state.kind} matches</div>`; more.textContent = ""; countLine.textContent = "0 results"; return; }
    const frag = document.createDocumentFragment();
    hits.slice(0, MAX_RENDER).forEach(r => frag.appendChild(card(r)));
    grid.appendChild(frag);
    countLine.textContent = `${hits.length} ${LABELS[state.kind] || state.kind}${hits.length === 1 ? "" : "s"}` +
      (state.q || state.base ? " (filtered)" : "");
    more.textContent = hits.length > MAX_RENDER ? `showing first ${MAX_RENDER} of ${hits.length} — refine the search to see the rest` : "";
  }

  function escapeHtml(s) { return String(s).replace(/[&<>]/g, m => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[m])); }
  function escapeAttr(s) { return escapeHtml(s).replace(/"/g, "&quot;"); }

  let t;
  el("q").addEventListener("input", e => { clearTimeout(t); t = setTimeout(() => { state.q = e.target.value; render(); }, 140); });
  baseSel.addEventListener("change", e => { state.base = e.target.value; render(); });
  el("nsfw").addEventListener("change", e => { state.nsfw = e.target.checked; renderTabs(); render(); });

  renderTabs();
  render();
})();
</script>
</body>
</html>
"""


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Generate the Z3 model-index gallery page")
    ap.add_argument("action", nargs="?", default="build", choices=("build",))
    ap.parse_args()
    build()


if __name__ == "__main__":
    main()
