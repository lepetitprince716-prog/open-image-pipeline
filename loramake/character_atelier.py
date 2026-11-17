"""The character atelier: five moments (summon, meet, book, fill, collect) over one state file.

State lives in ``outputs/characters/<id>/atelier.json``; the design card stays the source of
identity in ``configs/character_designs/<id>.json``. GPT work (the reference sheet and its
edit passes) goes through the console job queue; BOXA work runs in this module's single
background worker so the page always has one line of status to show.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Callable

from . import character_recipes as recipes
from . import designer_brain as brain
from . import image_meta
from .config import OUTPUTS

ROOT = OUTPUTS.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import character_design as cd  # noqa: E402

CARDS = ROOT / "configs" / "character_designs"
HOME = OUTPUTS / "characters"
SCHEMA = "loramake.atelier/v1"
CARD_SCHEMA = "loramake.character-design/v1"
MOMENTS = ("summon", "meet", "book", "fill", "collect", "done")
SLOTS = ("front", "side", "book")
JOB_POLL_S = 3.0
JOB_WAIT_S = 1200.0
LOG_KEEP = 40
# When the critic seat runs on its own: never / at the decision points (the adopted sheet and
# every picked image) / on every produced image. The CLI can always review any file by hand.
CRITIC_MODES = ("off", "gates", "all")

_lock = threading.RLock()
_queue: deque[tuple[str, Callable[[], None]]] = deque()
_worker: threading.Thread | None = None
_wake = threading.Event()
_critic_queue: deque[tuple[str, Callable[[], None]]] = deque()
_critic_worker: threading.Thread | None = None
_critic_wake = threading.Event()


# ---------------------------------------------------------------- state

def home(card_id: str) -> Path:
    return HOME / card_id


def state_path(card_id: str) -> Path:
    return home(card_id) / "atelier.json"


def _empty_state(card_id: str) -> dict:
    return {
        "schema": SCHEMA, "id": card_id, "voice": "", "moment": "summon",
        "sheet": {"design": None, "versions": [], "book": None, "book_style": None,
                  "front": None, "side": None, "expressions": {}},
        "candidates": {"book": [], "front": [], "side": []},
        "artifacts": {},
        "rejected": [], "chat": [], "job": None, "error": None, "log": [], "collected": None,
    }


def load_state(card_id: str) -> dict:
    path = state_path(card_id)
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return _empty_state(card_id)
    base = _empty_state(card_id)
    base.update(body if isinstance(body, dict) else {})
    return base


def save_state(state: dict) -> None:
    path = state_path(state["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _update(card_id: str, **fields: Any) -> dict:
    with _lock:
        state = load_state(card_id)
        state.update(fields)
        save_state(state)
        return state


def _log(card_id: str, line: str) -> None:
    with _lock:
        state = load_state(card_id)
        state["log"] = (state.get("log") or [])[-(LOG_KEEP - 1):] + [f"{time.strftime('%H:%M:%S')} {line}"]
        save_state(state)


def _set_job(card_id: str, kind: str | None, detail: str = "", job_id: str | None = None) -> None:
    job = None if kind is None else {"kind": kind, "detail": detail, "job_id": job_id, "started_at": time.time()}
    _update(card_id, job=job)


def load_card(card_id: str) -> dict | None:
    try:
        card = json.loads((CARDS / f"{card_id}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return card if isinstance(card, dict) else None


def save_card(card: dict) -> Path:
    CARDS.mkdir(parents=True, exist_ok=True)
    path = CARDS / f"{card['id']}.json"
    path.write_text(json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _rel(path: Path | str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(p)


# ---------------------------------------------------------------- card revision stamps

def stamp(card_id: str, rel: str, sha8: str, critic: dict | None = None) -> dict:
    """Bind one artifact to the card revision it was rendered from: sidecar + state index.

    ``critic`` is the critic report summary (see character_critic.summary_of); it lands under
    the sidecar's ``gates.critic`` without a ``pass`` key, so gate_status() never says pass.
    """
    patch: dict[str, Any] = {"card_id": card_id, "card_sha8": sha8}
    if critic:
        patch["gates"] = {"critic": dict(critic)}
    try:
        image_meta.update_sidecar(ROOT / rel, patch)
    except OSError:
        pass  # the state index below is the record the page reads; the sidecar is a courtesy copy
    with _lock:
        state = load_state(card_id)
        old = dict((state.get("artifacts") or {}).get(rel) or {})
        entry = {**old, "card_sha8": sha8, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        if critic is not None:
            entry["critic"] = dict(critic)
        entry.setdefault("critic", None)
        state.setdefault("artifacts", {})[rel] = entry
        save_state(state)
        return entry


def annotate(state: dict, card: dict | None) -> dict:
    """Derived view for the API: the current card revision and a ``stale`` flag per artifact.

    Never persisted — staleness is always recomputed against the card on disk.
    """
    current = cd.card_sha8(card) if isinstance(card, dict) else None
    artifacts = {}
    for rel, entry in (state.get("artifacts") or {}).items():
        entry = dict(entry or {})
        entry["stale"] = bool(current) and entry.get("card_sha8") != current
        artifacts[rel] = entry
    return {**state, "card_sha8": current, "artifacts": artifacts}


# ---------------------------------------------------------------- worker

def _run_worker() -> None:
    while True:
        with _lock:
            item = _queue.popleft() if _queue else None
        if item is None:
            _wake.wait(timeout=1.0)
            _wake.clear()
            continue
        card_id, task = item
        try:
            task()
        except Exception as exc:  # the page shows the error; the worker must survive
            _update(card_id, error=f"{exc}"[:300], job=None)
            _log(card_id, "failed: " + traceback.format_exc().strip().splitlines()[-1][:200])


def enqueue(card_id: str, task: Callable[[], None]) -> None:
    global _worker
    with _lock:
        _queue.append((card_id, task))
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_run_worker, name="atelier-worker", daemon=True)
            _worker.start()
    _wake.set()


# ---------------------------------------------------------------- critic (own thread, never the GPU queue)

def critic_mode() -> str:
    raw = os.environ.get("LORAMAKE_CRITIC", "gates").strip().lower()
    if raw in ("0", "off", "no", "false"):
        return "off"
    return raw if raw in CRITIC_MODES else "gates"


def _run_critic_worker() -> None:
    while True:
        with _lock:
            item = _critic_queue.popleft() if _critic_queue else None
        if item is None:
            _critic_wake.wait(timeout=1.0)
            _critic_wake.clear()
            continue
        card_id, task = item
        _run_critic_task(card_id, task)


def _run_critic_task(card_id: str, task: Callable[[], None]) -> None:
    try:
        task()
    except Exception as exc:  # a seat hiccup is a log line, never an error on the page
        _log(card_id, f"critic skipped: {exc}"[:200])


def enqueue_critic(card_id: str, task: Callable[[], None]) -> None:
    """Queue a critic pass on the side thread; the thread starts only when there is work."""
    global _critic_worker
    with _lock:
        _critic_queue.append((card_id, task))
        if _critic_worker is None or not _critic_worker.is_alive():
            _critic_worker = threading.Thread(target=_run_critic_worker, name="atelier-critic", daemon=True)
            _critic_worker.start()
    _critic_wake.set()


def _review(card_id: str, rel: str | None, at: str) -> None:
    """Ask the critic seat about one artifact when the mode allows it (``at`` = gate | every).

    Runs on the critic thread; the image is already on the page. Failure = a log line and a
    「未核对」 badge, nothing else.
    """
    mode = critic_mode()
    if not rel or mode == "off" or (mode == "gates" and at != "gate"):
        return

    def task() -> None:
        from . import character_critic as critic
        critic.review(card_id, ROOT / rel)

    enqueue_critic(card_id, task)


# ---------------------------------------------------------------- GPT jobs

def _jobs_module():
    from app import jobs
    return jobs


def _submit_sheet_job(prompt: str, client_id: str, reference: Path | None = None) -> str:
    brief = {"tier": "best", "provider": "gpt_image", "box": "boxb", "confirm_paid_generation": True,
             "sfw": True, "n": 1, "res": recipes.SHEET_SIZE, "prompt": prompt}
    if reference is not None:
        brief["reference"] = str(reference.resolve())
    job = _jobs_module().submit(brief, client_id=client_id)
    return str(job["id"])


def _wait_job(job_id: str) -> Path:
    jobs = _jobs_module()
    deadline = time.monotonic() + JOB_WAIT_S
    while time.monotonic() < deadline:
        job = jobs.get_job(job_id) or {}
        status = job.get("status")
        if status == "done":
            images = ((job.get("result") or {}).get("images")) or []
            if not images:
                raise RuntimeError("the sheet job finished without an image")
            return Path(images[0])
        if status in ("error", "cancelled", "interrupted", "failed"):
            raise RuntimeError(f"sheet job {status}: {job.get('error') or ''}"[:200])
        time.sleep(JOB_POLL_S)
    raise RuntimeError("the sheet job is taking too long; it may still finish in the queue")


def _adopt_sheet(card_id: str, produced: Path, note: str) -> Path:
    design = home(card_id) / "design"
    design.mkdir(parents=True, exist_ok=True)
    version = len(load_state(card_id)["sheet"].get("versions") or []) + 1
    target = design / f"sheet_v{version}.png"
    shutil.copy2(produced, target)
    recipes.crop_views(target, design)
    card = load_card(card_id) or {}
    sha8 = cd.card_sha8(card)
    anchor = dict(card.get("anchor") or {})
    anchor.update({"image": _rel(target), "line": "gpt_image", "notes": [note], "card_sha8": sha8})
    card["anchor"] = anchor
    save_card(card)
    with _lock:
        state = load_state(card_id)
        state["sheet"]["design"] = _rel(target)
        state["sheet"]["versions"] = (state["sheet"].get("versions") or []) + [_rel(target)]
        state["moment"] = "meet"
        state["job"] = None
        state["error"] = None
        save_state(state)
    stamp(card_id, _rel(target), sha8)
    _log(card_id, f"sheet v{version} adopted")
    _review(card_id, _rel(target), "gate")
    return target


# ---------------------------------------------------------------- moments

def summon(sentence: str) -> dict:
    """One sentence -> (in the worker) design card -> reference sheet job.

    Returns at once with a provisional card so the page can open on her while the designer
    brain thinks; the brain's card replaces the template under the same id.
    """
    sentence = sentence.strip()
    provisional = brain.template_card(sentence)
    provisional["schema"] = CARD_SCHEMA
    card_id = provisional["id"]
    if (CARDS / f"{card_id}.json").exists():
        card_id = f"{card_id}_{uuid.uuid4().hex[:4]}"
        provisional["id"] = card_id
    save_card(provisional)
    state = _empty_state(card_id)
    state["job"] = {"kind": "summon", "detail": "设计师在想她是谁", "job_id": None, "started_at": time.time()}
    state["log"] = [f"{time.strftime('%H:%M:%S')} summoned: {sentence[:80]}"]
    save_state(state)

    def task() -> None:
        card, voice, source = brain.compose_card(sentence, schema=CARD_SCHEMA)
        card["id"] = card_id
        save_card(card)
        with _lock:
            fresh = load_state(card_id)
            fresh["voice"] = voice
            fresh["chat"] = [{"role": "her", "text": voice}]
            save_state(fresh)
        _log(card_id, f"card composed ({source})")
        prompt = recipes.sheet_prompt(card)
        (home(card_id) / "design").mkdir(parents=True, exist_ok=True)
        (home(card_id) / "design" / "sheet_prompt.txt").write_text(prompt, encoding="utf-8")
        _set_job(card_id, "summon", "正在画她的设定图")
        job_id = _submit_sheet_job(prompt, client_id=f"atelier-{card_id}-sheet-{uuid.uuid4().hex[:6]}")
        _set_job(card_id, "summon", "正在画她的设定图", job_id)
        produced = _wait_job(job_id)
        _adopt_sheet(card_id, produced, "summoned from one sentence")

    enqueue(card_id, task)
    return load_state(card_id)


def edit_sheet(card_id: str, instruction: str, summary: str = "") -> dict:
    card = load_card(card_id)
    state = load_state(card_id)
    current = state["sheet"].get("design")
    if card is None or not current:
        raise ValueError("no reference sheet to edit yet")
    prompt = recipes.edit_prompt(card, instruction)
    reference = ROOT / current

    def task() -> None:
        _set_job(card_id, "edit", summary or "在改设定图")
        job_id = _submit_sheet_job(prompt, client_id=f"atelier-{card_id}-edit-{uuid.uuid4().hex[:6]}", reference=reference)
        _set_job(card_id, "edit", summary or "在改设定图", job_id)
        produced = _wait_job(job_id)
        _adopt_sheet(card_id, produced, f"edit pass: {instruction[:120]}")

    enqueue(card_id, task)
    return load_state(card_id)


def say(card_id: str, text: str) -> dict:
    """Talk to her; a proposal with a visual consequence is offered, not applied."""
    card = load_card(card_id)
    if card is None:
        raise ValueError(f"no design card {card_id!r}")
    with _lock:
        state = load_state(card_id)
        state["chat"] = (state.get("chat") or [])[-30:] + [{"role": "you", "text": text.strip()}]
        save_state(state)
    answer = brain.reply(card, state.get("voice") or "", state["chat"], text)
    with _lock:
        state = load_state(card_id)
        turn = {"role": "her", "text": answer["say"]}
        if answer.get("proposal"):
            turn["proposal"] = answer["proposal"]
        state["chat"] = state["chat"] + [turn]
        save_state(state)
    return state


def start_book(card_id: str, seed: int = 4101, styles: tuple[dict, ...] = recipes.BOOK_STYLES[:4]) -> dict:
    card = load_card(card_id)
    state = load_state(card_id)
    design = home(card_id) / "design"
    bust = design / "bust_x4.png"
    if card is None or not state["sheet"].get("design") or not bust.is_file():
        raise ValueError("summon her first: the art book needs the reference sheet")
    rejected = set(state.get("rejected") or [])
    sha8 = cd.card_sha8(card)

    def task() -> None:
        _update(card_id, moment="book", error=None)
        for style in styles:
            _set_job(card_id, "book", f"正在请「{style['label']}」画她")
            produced = recipes.run_command(recipes.book_command(card, style, bust, seed))
            target = home(card_id) / "book" / f"{style['id']}_{seed}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(produced, target)
            with _lock:
                fresh = load_state(card_id)
                pages = [p for p in fresh["candidates"]["book"] if p.get("style") != style["id"]]
                if _rel(target) not in rejected:
                    pages.append({"style": style["id"], "label": style["label"], "path": _rel(target), "lora": style["lora"]})
                fresh["candidates"]["book"] = pages
                save_state(fresh)
            stamp(card_id, _rel(target), sha8)
            _log(card_id, f"book page {style['label']} ready")
            _review(card_id, _rel(target), "every")
        recipes.free_vram()
        _set_job(card_id, None)

    enqueue(card_id, task)
    return load_state(card_id)


def _style_for(state: dict) -> dict | None:
    style_id = state["sheet"].get("book_style")
    return next((s for s in recipes.BOOK_STYLES if s["id"] == style_id), None)


def start_fill(card_id: str, views: tuple[str, ...] = ("front", "side"), seed: int = 4101,
               expressions: bool = True) -> dict:
    """Grow the standing-art set in the chosen style: full-body views, then six expressions."""
    card = load_card(card_id)
    state = load_state(card_id)
    style = _style_for(state)
    design = home(card_id) / "design"
    if card is None or style is None:
        raise ValueError("pick an art-book page first")
    bust = design / "bust_x4.png"
    book_page = ROOT / state["sheet"]["book"] if state["sheet"].get("book") else None
    sha8 = cd.card_sha8(card)

    def task() -> None:
        _update(card_id, moment="fill", error=None)
        for view in views:
            _set_job(card_id, "fill", f"正在画她的{'正面' if view == 'front' else '侧面'}")
            padded = recipes.pad_to_canvas(design / f"{view}_full.png", design / f"{view}_pad.png")
            produced = recipes.run_command(recipes.view_command(card, view, padded, bust, style["lora"], seed))
            _set_job(card_id, "fill", "补回她的脸")
            refined = recipes.run_command(recipes.face_command(card, produced, style["lora"], seed, card_id))
            target = home(card_id) / "set" / f"{view}_{seed}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(refined, target)
            with _lock:
                fresh = load_state(card_id)
                fresh["candidates"][view] = [c for c in fresh["candidates"][view] if c.get("path") != _rel(target)]
                fresh["candidates"][view].append({"path": _rel(target), "seed": seed})
                if not fresh["sheet"].get(view):
                    fresh["sheet"][view] = _rel(target)
                save_state(fresh)
            stamp(card_id, _rel(target), sha8)
            _log(card_id, f"{view} view ready")
            _review(card_id, _rel(target), "every")
        if expressions and book_page is not None:
            for name, label, tags in recipes.EXPRESSIONS:
                _set_job(card_id, "fill", f"正在画表情「{label}」")
                produced = recipes.run_command(recipes.face_command(card, book_page, style["lora"], seed, card_id, tags))
                target = home(card_id) / "set" / f"expr_{name}.png"
                shutil.copy2(produced, target)
                with _lock:
                    fresh = load_state(card_id)
                    fresh["sheet"]["expressions"][name] = _rel(target)
                    save_state(fresh)
                stamp(card_id, _rel(target), sha8)
                _log(card_id, f"expression {label} ready")
                _review(card_id, _rel(target), "every")
        recipes.free_vram()
        with _lock:
            fresh = load_state(card_id)
            fresh["job"] = None
            fresh["moment"] = "collect"
            save_state(fresh)

    enqueue(card_id, task)
    return load_state(card_id)


def pick(card_id: str, slot: str, path: str) -> dict:
    if slot not in SLOTS:
        raise ValueError(f"unknown slot {slot!r}")
    with _lock:
        state = load_state(card_id)
        candidates = state["candidates"].get(slot) or []
        chosen = next((c for c in candidates if c.get("path") == path), None)
        if chosen is None:
            raise ValueError("that image is not among her candidates")
        state["sheet"][slot] = path
        if slot == "book":
            state["sheet"]["book_style"] = chosen.get("style")
        save_state(state)
    _log(card_id, f"picked {slot}: {Path(path).name}")
    _review(card_id, path, "gate")
    if slot == "book":
        return start_fill(card_id)
    return load_state(card_id)


def release(card_id: str, path: str) -> dict:
    with _lock:
        state = load_state(card_id)
        state["rejected"] = sorted(set(state.get("rejected") or []) | {path})
        for slot, items in state["candidates"].items():
            state["candidates"][slot] = [c for c in items if c.get("path") != path]
        for slot in SLOTS:
            if state["sheet"].get(slot) == path:
                state["sheet"][slot] = None
                if slot == "book":
                    state["sheet"]["book_style"] = None
        save_state(state)
    _log(card_id, f"released {Path(path).name}")
    return load_state(card_id)


def again(card_id: str, slot: str) -> dict:
    """Roll the slot once more with a fresh seed."""
    seed = int(time.time()) % 90000 + 1000
    if slot == "book":
        return start_book(card_id, seed=seed)
    if slot in ("front", "side"):
        return start_fill(card_id, views=(slot,), seed=seed, expressions=False)
    raise ValueError(f"unknown slot {slot!r}")


def collect(card_id: str) -> dict:
    """Matte every set image to transparent PNG and write the DNA page."""
    card = load_card(card_id)
    state = load_state(card_id)
    if card is None or not state["sheet"].get("front"):
        raise ValueError("nothing to collect yet")

    def task() -> None:
        _set_job(card_id, "collect", "正在抠图收录")
        out = home(card_id) / "collected"
        out.mkdir(parents=True, exist_ok=True)
        items = {"front": state["sheet"].get("front"), "side": state["sheet"].get("side"), "book": state["sheet"].get("book")}
        items.update({f"expr_{k}": v for k, v in (state["sheet"].get("expressions") or {}).items()})
        matted, missing = {}, []
        for name, rel in items.items():
            if not rel:
                continue
            src = ROOT / rel
            dst = out / f"{name}.png"
            if recipes.matte(src, dst):
                matted[name] = _rel(dst)
            else:
                shutil.copy2(src, dst)
                missing.append(name)
        dna = {"schema": "loramake.character-dna/v1", "id": card_id, "name_cn": card.get("name_cn"),
               "card_sha8": cd.card_sha8(card),
               "voice": state.get("voice"), "anchor": state["sheet"].get("design"), "style": state["sheet"].get("book_style"),
               "images": {k: v for k, v in items.items() if v}, "matted": matted,
               "matte_missing": missing, "palette": card.get("palette"), "signature": card.get("signature_element"),
               "collected_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        (home(card_id) / "dna.json").write_text(json.dumps(dna, ensure_ascii=False, indent=2), encoding="utf-8")
        with _lock:
            fresh = load_state(card_id)
            fresh["collected"] = {"dir": _rel(out), "matted": len(matted), "unmatted": missing}
            fresh["moment"] = "done"
            fresh["job"] = None
            save_state(fresh)
        _log(card_id, f"collected: {len(matted)} matted, {len(missing)} plain")

    enqueue(card_id, task)
    return load_state(card_id)
