#!/usr/bin/env python3
"""Produce validated NSFW LoRA pilot prompts through DeepSeek V4 Pro.

The engineering layer assembles bounded briefs from the approved guide and live
inventory.  DeepSeek writes the prompt body.  Every model response is validated,
hashed, and, at most once per failed item, sent back for a single-item repair.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INVENTORY = ROOT / "nsfw" / "nsfw_lora_inventory_v1.json"
DEFAULT_PILOT = ROOT / "nsfw" / "nsfw_lora_pilot_v1.json"
DEFAULT_GUIDE = ROOT / "nsfw" / "NSFW_PROMPT_GUIDE.md"
DEFAULT_OUT = ROOT / "nsfw" / "prompts" / "lora_test_prompts_v1.json"
DEFAULT_RUN_ROOT = ROOT / "nsfw" / "outputs" / "prompt_author"
SCHEMA_VERSION = "nsfw-prompt-registry-v1"
AUTHOR_MODEL = "deepseek-v4-pro"

RAW_FIELDS = (
    "asset_id",
    "engine",
    "positive",
    "negative",
    "triggers_used",
    "intended_effect",
)
QUALITY_BOILERPLATE = (
    "masterpiece",
    "best quality",
    "amazing quality",
    "very aesthetic",
    "absurdres",
    "newest",
    "very awa",
)
MANDATORY_NEGATIVE = (
    "child",
    "loli",
    "young",
    "baby face",
    "petite",
    "teen",
    "underage",
    "censored",
    "mosaic",
)
WORD_BANDS = {
    "light": (60, 100),
    "simple": (60, 100),
    "moderate": (80, 140),
    "heavy": (100, 180),
}
CJK_RE = re.compile(r"[\u3400-\u9fff]")
WEIGHT_RE = re.compile(r"\([^()\n]+:\s*[0-9]+(?:\.[0-9]+)?\)")
DOUBLE_PAREN_RE = re.compile(r"\(\([^()]+\)\)")
BRACE_RE = re.compile(r"[{}]")
PARTICIPANT_RE = re.compile(r"\b[1-9](?:girl|girls|boy|boys)\b", re.IGNORECASE)


class PromptAuthorError(RuntimeError):
    """Prompt authoring or validation failure."""


Invoker = Callable[[str, str, Path], tuple[int, str, str]]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PromptAuthorError(f"missing file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PromptAuthorError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PromptAuthorError(f"{path} must contain a JSON object")
    return payload


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]*", text))


def _contains_term(text: str, term: str) -> bool:
    return term.casefold() in text.casefold()


def validate_raw_entry(entry: Any, pilot_item: dict[str, Any]) -> list[str]:
    asset_id = pilot_item.get("asset_id", "unknown")
    if not isinstance(entry, dict):
        return [f"{asset_id}: output must be an object"]
    problems: list[str] = []
    missing = [field for field in RAW_FIELDS if field not in entry]
    extra = sorted(set(entry) - set(RAW_FIELDS))
    if missing:
        problems.append(f"{asset_id}: missing fields {missing}")
    if extra:
        problems.append(f"{asset_id}: unexpected fields {extra}")
    if missing:
        return problems
    for field in ("asset_id", "engine", "positive", "negative", "intended_effect"):
        if not isinstance(entry[field], str) or not entry[field].strip():
            problems.append(f"{asset_id}: {field} must be a non-empty string")
    if not isinstance(entry.get("triggers_used"), list) or not all(
        isinstance(item, str) and item.strip() for item in entry.get("triggers_used", [])
    ):
        problems.append(f"{asset_id}: triggers_used must be a string array")
    if problems:
        return problems
    if entry["asset_id"] != asset_id:
        problems.append(f"{asset_id}: returned asset_id {entry['asset_id']!r}")
    if entry["engine"] != pilot_item.get("engine"):
        problems.append(f"{asset_id}: engine must be {pilot_item.get('engine')!r}")

    positive = entry["positive"].strip()
    negative = entry["negative"].strip()
    intended = entry["intended_effect"].strip()
    for field, text in (("positive", positive), ("negative", negative), ("intended_effect", intended)):
        if CJK_RE.search(text):
            problems.append(f"{asset_id}: {field} must be English only")
    for term in QUALITY_BOILERPLATE:
        if _contains_term(positive, term):
            problems.append(f"{asset_id}: positive contains banned boilerplate {term!r}")
    if WEIGHT_RE.search(positive) or DOUBLE_PAREN_RE.search(positive) or BRACE_RE.search(positive):
        problems.append(f"{asset_id}: positive contains banned weight syntax")
    if WEIGHT_RE.search(negative) or DOUBLE_PAREN_RE.search(negative) or BRACE_RE.search(negative):
        problems.append(f"{asset_id}: negative contains banned weight syntax")
    for term in MANDATORY_NEGATIVE:
        if not _contains_term(negative, term):
            problems.append(f"{asset_id}: negative missing {term!r}")
    if not PARTICIPANT_RE.search(positive):
        problems.append(f"{asset_id}: positive must state participant count and gender")

    level = str(pilot_item.get("explicit_level", "moderate"))
    low, high = WORD_BANDS.get(level, WORD_BANDS["moderate"])
    count = word_count(positive)
    if not low <= count <= high:
        problems.append(f"{asset_id}: positive word_count {count} outside {level} band {low}-{high}")

    allowed = [str(value).strip() for value in pilot_item.get("triggers", []) if str(value).strip()]
    used = [str(value).strip() for value in entry["triggers_used"]]
    if allowed and not used:
        problems.append(f"{asset_id}: triggers_used is empty")
    allowed_folded = {value.casefold() for value in allowed}
    for trigger in used:
        if allowed and trigger.casefold() not in allowed_folded:
            problems.append(f"{asset_id}: undeclared trigger {trigger!r}")
        if not _contains_term(positive, trigger):
            problems.append(f"{asset_id}: trigger {trigger!r} not present in positive")
    if allowed and not any(_contains_term(positive, trigger) for trigger in allowed):
        problems.append(f"{asset_id}: positive contains none of the approved triggers")
    return problems


def validate_pilot(pilot: dict[str, Any], inventory: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    if pilot.get("schema_version") != "nsfw-lora-pilot-v1":
        problems.append("pilot schema_version must be nsfw-lora-pilot-v1")
    if pilot.get("gpu_submissions") != 0:
        problems.append("pilot gpu_submissions must remain 0 before author gate")
    items = pilot.get("items")
    if not isinstance(items, list) or not items:
        return problems + ["pilot items must be a non-empty list"]
    ids = [item.get("asset_id") for item in items if isinstance(item, dict)]
    if len(ids) != len(items) or len(ids) != len(set(ids)):
        problems.append("pilot asset_id values must be unique")
    inventory_ids = {item.get("asset_id") for item in inventory.get("assets", []) if isinstance(item, dict)}
    for item in items:
        if not isinstance(item, dict):
            problems.append("pilot item must be an object")
            continue
        if item.get("asset_id") not in inventory_ids:
            problems.append(f"pilot asset missing from inventory: {item.get('asset_id')}")
        if item.get("engine") != "illustrious":
            problems.append(f"{item.get('asset_id')}: first pilot engine must be illustrious")
    expected_hash = sha256_text(json.dumps(inventory, ensure_ascii=False, sort_keys=True))
    if pilot.get("inventory_sha256") != expected_hash:
        problems.append("pilot inventory_sha256 does not match inventory")
    return problems


def batch_items(items: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    if batch_size < 1:
        raise PromptAuthorError("batch_size must be positive")
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    order: list[tuple[str, str, str]] = []
    for item in items:
        key = (
            str(item.get("checkpoint", "")),
            str(item.get("engine", "")),
            str(item.get("explicit_level", "")),
        )
        if key not in groups:
            order.append(key)
        groups[key].append(item)
    batches: list[list[dict[str, Any]]] = []
    for key in order:
        group = groups[key]
        batches.extend(group[index:index + batch_size] for index in range(0, len(group), batch_size))
    return batches


def _word_band(item: dict[str, Any]) -> tuple[int, int]:
    return WORD_BANDS.get(str(item.get("explicit_level")), WORD_BANDS["moderate"])


def _brief_item(item: dict[str, Any]) -> dict[str, Any]:
    low, high = _word_band(item)
    return {
        "asset_id": item["asset_id"],
        "card_stem": item["card_stem"],
        "engine": item["engine"],
        "checkpoint": item["checkpoint"],
        "explicit_level": item["explicit_level"],
        "positive_word_band": f"{low}-{high}",
        "approved_triggers": item.get("triggers", []),
        "content_flags": item.get("content_flags", []),
        "composition_contract": {
            "participant_count_and_gender": "state explicitly in positive",
            "camera_and_framing": "choose a stable composition that exposes the LoRA's intended effect",
            "comparison_control": "write one deterministic prompt reusable across strengths 0.5/0.7/0.9 at seed 7001",
        },
    }


def approved_guide_excerpt(guide: str) -> str:
    """Keep only the approved sections needed by the first Illustrious pilot.

    The source guide hash still binds the full document.  Sending the entire guide on every
    call adds unrelated Anima/open-question context and materially increases response latency.
    """
    ranges = (
        ("## 〇、一页结论", "## 一、Scaffold 结构模板"),
        ("### 1.1 Illustrious/SDXL 线", "### 1.2 Anima 线"),
        ("## 二、密度规范", "## 三、负面提示词规范"),
        ("### 3.2 NSFW 必备负面", "## 四、引擎差异"),
        ("## 八、用户口味规则", "## 十、开放问题"),
    )
    sections: list[str] = []
    for start, end in ranges:
        begin = guide.find(start)
        finish = guide.find(end, begin + len(start)) if begin >= 0 else -1
        if begin < 0 or finish < 0:
            raise PromptAuthorError(f"approved guide section missing: {start!r} -> {end!r}")
        sections.append(guide[begin:finish].strip())
    return "\n\n---\n\n".join(sections)


def build_batch_request(items: list[dict[str, Any]], guide: str) -> str:
    output_contract = {
        "format": "Return only one JSON array. No Markdown fences, commentary, or analysis.",
        "entry_fields_exactly": list(RAW_FIELDS),
        "field_types": {
            "asset_id": "string copied exactly",
            "engine": "string copied exactly",
            "positive": "English prompt string",
            "negative": "English negative prompt string",
            "triggers_used": "array of exact strings selected from approved_triggers",
            "intended_effect": "concise English engineering description",
        },
    }
    return (
        "You are the sole prompt-body author for this NSFW LoRA GPU comparison batch.\n"
        "Follow the approved guide and each asset brief exactly. Every depicted participant is an adult.\n"
        "Do not add weight syntax or generic quality boilerplate. Preserve exact asset_id and engine values.\n"
        "Return raw JSON only — no Markdown fences, no ```json wrappers, no commentary.\n"
        "Every string in triggers_used MUST appear verbatim as a substring of positive "
        "(example: if approved trigger is doublepen, positive must contain the token doublepen, "
        "not only the phrase double penetration).\n"
        "Prefer 1-4 highest-signal approved triggers; do not dump the whole trigger list into triggers_used.\n"
        "Respect each brief's own positive_word_band exactly; briefs in one batch may differ.\n\n"
        "OUTPUT CONTRACT\n"
        + json.dumps(output_contract, ensure_ascii=False, indent=2)
        + "\n\nASSET BRIEFS\n"
        + json.dumps([_brief_item(item) for item in items], ensure_ascii=False, indent=2)
        + "\n\nAPPROVED GUIDE\n"
        + approved_guide_excerpt(guide)
        + "\n"
    )


def build_repair_request(
    item: dict[str, Any], guide: str, prior: Any, problems: list[str]
) -> str:
    low, high = _word_band(item)
    return (
        "Repair exactly one NSFW LoRA prompt entry. Every depicted participant is an adult.\n"
        "Return only one JSON object with exactly these fields: "
        + ", ".join(RAW_FIELDS)
        + ". No Markdown fences or ```json wrappers, no commentary or analysis.\n"
        "triggers_used MUST be a JSON array of exact approved trigger strings, for example "
        '["approved trigger"]. Every triggers_used string MUST appear verbatim inside positive.\n'
        "Use only 1-4 triggers that are actually present in positive. Respect positive_word_band "
        f"exactly ({low}-{high} words).\n\n"
        "VALIDATION ERRORS\n"
        + json.dumps(problems, ensure_ascii=False, indent=2)
        + "\n\nASSET BRIEF\n"
        + json.dumps(_brief_item(item), ensure_ascii=False, indent=2)
        + "\n\nPRIOR ENTRY\n"
        + json.dumps(prior, ensure_ascii=False, indent=2)
        + "\n\nAPPROVED GUIDE\n"
        + approved_guide_excerpt(guide)
        + "\n"
    )


def _strip_markdown_fence(text: str) -> str:
    """Drop a single outer ``` / ```json fence if the model wraps JSON anyway."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _json_slice(text: str, *, expected: type) -> str | None:
    """Slice the outermost JSON container out of prose, e.g. 'Here is the JSON: {...}'."""
    open_char, close_char = ("[", "]") if expected is list else ("{", "}")
    begin = text.find(open_char)
    finish = text.rfind(close_char)
    if begin < 0 or finish <= begin:
        return None
    return text[begin:finish + 1]


def parse_exact_json(text: str, *, expected: type) -> Any:
    stripped = _strip_markdown_fence(text)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        candidate = _json_slice(stripped, expected=expected)
        if candidate is None:
            raise PromptAuthorError(f"model stdout is not exact JSON: {exc}") from exc
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            raise PromptAuthorError(f"model stdout is not exact JSON: {exc}") from exc
    if not isinstance(payload, expected):
        raise PromptAuthorError(f"model JSON must be {expected.__name__}, got {type(payload).__name__}")
    return payload


def make_mcall_invoker(mcall_bin: str, model: str, timeout: int, max_tokens: int) -> Invoker:
    def invoke(prompt: str, call_id: str, run_dir: Path) -> tuple[int, str, str]:
        request_path = run_dir / f"{call_id}.request.txt"
        stdout_path = run_dir / f"{call_id}.stdout.json"
        stderr_path = run_dir / f"{call_id}.stderr.log"
        atomic_write(request_path, prompt)
        result = subprocess.run(
            [
                mcall_bin,
                "deepseek",
                "-f",
                str(request_path),
                "-m",
                model,
                "--timeout",
                str(timeout),
                "--",
                "--max-tokens",
                str(max_tokens),
            ],
            capture_output=True,
            text=True,
        )
        atomic_write(stdout_path, result.stdout)
        atomic_write(stderr_path, result.stderr)
        return result.returncode, result.stdout, result.stderr

    return invoke


def _entry_index(raw: list[Any]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    indexed: dict[str, dict[str, Any]] = {}
    problems: list[str] = []
    for index, value in enumerate(raw):
        if not isinstance(value, dict) or not isinstance(value.get("asset_id"), str):
            problems.append(f"batch entry {index} has no string asset_id")
            continue
        asset_id = value["asset_id"]
        if asset_id in indexed:
            problems.append(f"batch contains duplicate asset_id {asset_id}")
            continue
        indexed[asset_id] = value
    return indexed, problems


def _registry_entry(
    item: dict[str, Any], raw: dict[str, Any] | None, *, batch_id: str,
    input_sha256: str, output_sha256: str, problems: list[str], repair_attempted: bool,
    created_at: str,
) -> dict[str, Any]:
    selected = raw or {
        "asset_id": item["asset_id"],
        "engine": item["engine"],
        "positive": "",
        "negative": "",
        "triggers_used": [],
        "intended_effect": "",
    }
    prompt_fields = {field: selected.get(field) for field in RAW_FIELDS}
    return {
        **prompt_fields,
        "card_stem": item["card_stem"],
        "checkpoint": item["checkpoint"],
        "explicit_level": item["explicit_level"],
        "strengths": item["strengths"],
        "seed": item["seed"],
        "status": "pass" if not problems else "HOLD",
        "validation_errors": problems,
        "repair_attempted": repair_attempted,
        "author_model": AUTHOR_MODEL,
        "batch_id": batch_id,
        "input_sha256": input_sha256,
        "output_sha256": output_sha256,
        "prompt_sha256": sha256_text(canonical_json(prompt_fields)),
        "created_at": created_at,
    }


def author_prompts(
    pilot: dict[str, Any], guide: str, invoke: Invoker, run_dir: Path, *,
    batch_size: int = 5, selected_items: list[dict[str, Any]] | None = None,
    model: str = AUTHOR_MODEL,
) -> dict[str, Any]:
    if model != AUTHOR_MODEL:
        raise PromptAuthorError(f"author model must be pinned to {AUTHOR_MODEL}")
    items = selected_items if selected_items is not None else list(pilot["items"])
    run_dir.mkdir(parents=True, exist_ok=True)
    generated_at = utc_now()
    registry: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    for batch_number, batch in enumerate(batch_items(items, batch_size), start=1):
        batch_id = f"batch-{batch_number:03d}"
        request = build_batch_request(batch, guide)
        request_sha = sha256_text(request)
        rc, stdout, stderr = invoke(request, batch_id, run_dir)
        calls.append({
            "call_id": batch_id,
            "kind": "batch",
            "asset_ids": [item["asset_id"] for item in batch],
            "returncode": rc,
            "input_sha256": request_sha,
            "output_sha256": sha256_text(stdout),
            "stderr_sha256": sha256_text(stderr),
        })
        indexed: dict[str, dict[str, Any]] = {}
        batch_problems: list[str] = []
        if rc != 0:
            batch_problems.append(f"{batch_id}: model call returncode {rc}")
        else:
            try:
                parsed = parse_exact_json(stdout, expected=list)
                indexed, parse_problems = _entry_index(parsed)
                batch_problems.extend(parse_problems)
            except PromptAuthorError as exc:
                batch_problems.append(f"{batch_id}: {exc}")

        for item in batch:
            asset_id = item["asset_id"]
            candidate = indexed.get(asset_id)
            problems = list(batch_problems)
            if candidate is None:
                problems.append(f"{asset_id}: missing from batch response")
            else:
                problems.extend(validate_raw_entry(candidate, item))
            repaired = False
            final_output_sha = sha256_text(stdout)
            final_input_sha = request_sha
            if problems:
                repaired = True
                repair_id = f"repair-{asset_id}"
                repair_request = build_repair_request(item, guide, candidate, problems)
                repair_sha = sha256_text(repair_request)
                repair_rc, repair_stdout, repair_stderr = invoke(repair_request, repair_id, run_dir)
                calls.append({
                    "call_id": repair_id,
                    "kind": "repair",
                    "asset_ids": [asset_id],
                    "returncode": repair_rc,
                    "input_sha256": repair_sha,
                    "output_sha256": sha256_text(repair_stdout),
                    "stderr_sha256": sha256_text(repair_stderr),
                })
                final_input_sha = repair_sha
                final_output_sha = sha256_text(repair_stdout)
                if repair_rc != 0:
                    problems = [f"{asset_id}: repair call returncode {repair_rc}"]
                else:
                    try:
                        repaired_candidate = parse_exact_json(repair_stdout, expected=dict)
                        repair_problems = validate_raw_entry(repaired_candidate, item)
                        candidate = repaired_candidate
                        problems = repair_problems
                    except PromptAuthorError as exc:
                        problems = [f"{asset_id}: repair {exc}"]
            registry.append(_registry_entry(
                item,
                candidate,
                batch_id=batch_id,
                input_sha256=final_input_sha,
                output_sha256=final_output_sha,
                problems=problems,
                repair_attempted=repaired,
                created_at=generated_at,
            ))
    registry.sort(key=lambda entry: next(
        index for index, item in enumerate(items) if item["asset_id"] == entry["asset_id"]
    ))
    pass_count = sum(entry["status"] == "pass" for entry in registry)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "complete" if len(items) == len(pilot["items"]) else "partial",
        "author_model": AUTHOR_MODEL,
        "guide_sha256": sha256_text(guide),
        "pilot_sha256": sha256_text(canonical_json(pilot)),
        "gpu_submissions": 0,
        "summary": {
            "requested": len(items),
            "passed": pass_count,
            "hold": len(items) - pass_count,
            "model_calls": len(calls),
            "repair_calls": sum(call["kind"] == "repair" for call in calls),
        },
        "calls": calls,
        "entries": registry,
    }


def validate_registry(
    payload: dict[str, Any], pilot: dict[str, Any], *, require_complete: bool = False,
    require_all_pass: bool = False,
) -> list[str]:
    problems: list[str] = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version must be {SCHEMA_VERSION}")
    if payload.get("author_model") != AUTHOR_MODEL:
        problems.append(f"author_model must be {AUTHOR_MODEL}")
    if payload.get("gpu_submissions") != 0:
        problems.append("gpu_submissions must be 0")
    if require_complete and payload.get("status") != "complete":
        problems.append("registry status is not complete")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        return problems + ["entries must be a list"]
    by_id = {item["asset_id"]: item for item in pilot.get("items", []) if isinstance(item, dict)}
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            problems.append(f"entries[{index}] must be an object")
            continue
        asset_id = entry.get("asset_id")
        if asset_id in seen:
            problems.append(f"duplicate registry asset_id {asset_id}")
        seen.add(str(asset_id))
        item = by_id.get(asset_id)
        if item is None:
            problems.append(f"registry asset_id not in pilot: {asset_id}")
            continue
        raw = {field: entry.get(field) for field in RAW_FIELDS}
        current = validate_raw_entry(raw, item)
        status = entry.get("status")
        stored = entry.get("validation_errors")
        if status == "pass" and (current or stored):
            problems.append(f"{asset_id}: pass entry has validation errors")
        if status == "HOLD" and not stored:
            problems.append(f"{asset_id}: HOLD entry must record validation_errors")
        if status not in {"pass", "HOLD"}:
            problems.append(f"{asset_id}: invalid status {status!r}")
        prompt_fields = {field: entry.get(field) for field in RAW_FIELDS}
        if entry.get("prompt_sha256") != sha256_text(canonical_json(prompt_fields)):
            problems.append(f"{asset_id}: prompt_sha256 mismatch")
        if entry.get("author_model") != AUTHOR_MODEL:
            problems.append(f"{asset_id}: author_model mismatch")
    if require_complete and seen != set(by_id):
        problems.append(f"registry pilot coverage {len(seen)}/{len(by_id)}")
    if require_all_pass:
        held = [entry.get("asset_id") for entry in entries if isinstance(entry, dict) and entry.get("status") != "pass"]
        if held:
            problems.append(f"GPU author gate has {len(held)} HOLD item(s): {held}")
    summary = payload.get("summary", {})
    if isinstance(summary, dict):
        passed = sum(entry.get("status") == "pass" for entry in entries if isinstance(entry, dict))
        held = sum(entry.get("status") == "HOLD" for entry in entries if isinstance(entry, dict))
        if summary.get("passed") != passed or summary.get("hold") != held:
            problems.append("summary passed/hold counts do not match entries")
    else:
        problems.append("summary must be an object")
    return problems


def select_items(
    pilot: dict[str, Any], *, only_level: str | None = None, limit: int | None = None
) -> list[dict[str, Any]]:
    items = list(pilot["items"])
    if only_level:
        items = [item for item in items if item.get("explicit_level") == only_level]
    if limit is not None:
        items = items[:limit]
    if not items:
        raise PromptAuthorError("item selection is empty")
    return items


def cmd_author(args: argparse.Namespace) -> None:
    inventory = load_json(args.inventory)
    pilot = load_json(args.pilot)
    guide = args.guide.read_text(encoding="utf-8")
    pilot_problems = validate_pilot(pilot, inventory)
    if pilot_problems:
        raise PromptAuthorError("pilot gate failed: " + "; ".join(pilot_problems))
    selected = select_items(pilot, only_level=args.only_level, limit=args.limit)
    run_dir = args.run_dir or (DEFAULT_RUN_ROOT / datetime.now().strftime("%Y%m%dT%H%M%S"))
    invoker = make_mcall_invoker(args.mcall_bin, args.model, args.timeout, args.max_tokens)
    payload = author_prompts(
        pilot,
        guide,
        invoker,
        run_dir,
        batch_size=args.batch_size,
        selected_items=selected,
        model=args.model,
    )
    problems = validate_registry(payload, pilot, require_complete=len(selected) == len(pilot["items"]))
    if problems:
        raise PromptAuthorError("registry gate failed: " + "; ".join(problems))
    atomic_write(args.out, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {args.out}")
    print(f"run logs: {run_dir}")


def cmd_validate(args: argparse.Namespace) -> None:
    payload = load_json(args.registry)
    pilot = load_json(args.pilot)
    problems = validate_registry(
        payload,
        pilot,
        require_complete=args.require_complete,
        require_all_pass=args.require_all_pass,
    )
    if problems:
        for problem in problems:
            print(f"FAIL {problem}", file=sys.stderr)
        raise SystemExit(2)
    print(f"OK — {len(payload['entries'])} prompt entries valid")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    author = sub.add_parser("author", help="call DeepSeek V4 Pro and build prompt registry")
    author.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    author.add_argument("--pilot", type=Path, default=DEFAULT_PILOT)
    author.add_argument("--guide", type=Path, default=DEFAULT_GUIDE)
    author.add_argument("--out", type=Path, default=DEFAULT_OUT)
    author.add_argument("--run-dir", type=Path)
    author.add_argument("--batch-size", type=int, default=5)
    author.add_argument("--only-level", choices=sorted(WORD_BANDS))
    author.add_argument("--limit", type=int)
    author.add_argument("--model", default=AUTHOR_MODEL)
    author.add_argument("--mcall-bin", default="mcall")
    author.add_argument("--timeout", type=int, default=180)
    author.add_argument("--max-tokens", type=int, default=2048)
    author.set_defaults(func=cmd_author)

    validate = sub.add_parser("validate", help="validate a stored prompt registry")
    validate.add_argument("registry", type=Path, nargs="?", default=DEFAULT_OUT)
    validate.add_argument("--pilot", type=Path, default=DEFAULT_PILOT)
    validate.add_argument("--require-complete", action="store_true")
    validate.add_argument("--require-all-pass", action="store_true")
    validate.set_defaults(func=cmd_validate)

    args = parser.parse_args()
    try:
        args.func(args)
    except PromptAuthorError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
