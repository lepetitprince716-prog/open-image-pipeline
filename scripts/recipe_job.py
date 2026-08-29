#!/usr/bin/env python3
"""Recipe quick-call surface: list / apply / complete saved LoRA-Manager recipes.

The model-callable entry for the recipe loop. Recipes live in the LoRA Manager store
on Z3 (saved by the Studio Civitai Bridge extension); this CLI reads them back, turns
one into a gen brief (app/recipe_brief.py owns the mapping), queues missing LoRAs
through the normal download path, mirrors LM records into the local management
library (recipes/library/, nsfw/recipes/library/), and runs the full loop in one
call by handing the brief to scripts/gen_job.py — the single validation + execution
path the Studio app also uses.

Two invocation forms, one result envelope (schema loramake.recipe-job/v1, always the
last stdout JSON):

  recipe_job.py list [--search TEXT] [--page N] [--page-size N]
  recipe_job.py show RECIPE_ID
  recipe_job.py find-missing RECIPE_ID | --image URL_OR_ID
  recipe_job.py download-missing RECIPE_ID [--wait] [--timeout-s N]
  recipe_job.py to-brief RECIPE_ID [--overrides JSON] [--recipe-file PATH]
  recipe_job.py generate RECIPE_ID [--overrides JSON] [--download-missing]
                         [--allow-partial-fill] [--json-out PATH]
  recipe_job.py sync
  recipe_job.py JOB.json | -          # same actions as {"action": ..., ...} JSON
  recipe_job.py --fields              # vocabulary reference

Exit codes: 0 ok · 2 invalid job/brief (a recipe LoRA Manager does not have counts as
invalid) · 3 LoRA Manager unreachable · 4 partial (some downloads skipped) · generate
propagates gen_job.py's exit code.

A failed envelope's data carries reason=unreachable|not_found|invalid, which is what
the exit code is derived from.

generate --download-missing stops on a failed fill rather than rendering without the
missing LoRAs; --allow-partial-fill downgrades that back to a warning.

Degraded (LM unreachable): to-brief --recipe-file and find-missing --image still work
and set degraded.lm=true; network actions exit 3. The Civitai token never appears in
argv, output, or logs.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents[:5]):
        if (candidate / "loramake" / "__init__.py").is_file():
            return candidate
    raise RuntimeError("could not locate repo root containing loramake/__init__.py")


ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import brief as brief_mod  # noqa: E402
from app import recipe_brief  # noqa: E402
from app import recipe_store  # noqa: E402
from app import civitai_bridge as bridge_mod  # noqa: E402

SCHEMA = "loramake.recipe-job/v1"
GEN_JOB = ROOT / "scripts" / "gen_job.py"
ACTIONS = ("list", "show", "find-missing", "download-missing", "to-brief", "generate", "sync")
EXIT_OK, EXIT_INVALID, EXIT_LM_DOWN, EXIT_PARTIAL = 0, 2, 3, 4
WAIT_POLL_S = 5.0

# Bridge failures carry a `reason`; anything without one is a bad request, not a
# transport problem. Matching on the message text instead would read
# "recipe not found ... unreachable" as an outage.
FAILURE_EXITS = {"unreachable": EXIT_LM_DOWN, "not_found": EXIT_INVALID, "invalid": EXIT_INVALID}
TRUE_STRINGS = frozenset({"true", "1", "yes", "on"})


def _exit_for_failure(failure: dict | None) -> int:
    return FAILURE_EXITS.get(str((failure or {}).get("reason") or ""), EXIT_INVALID)


def _as_bool(value: Any) -> bool:
    """JSON jobs arrive from models and hand-written files, where "false" is common
    enough that bool() on the raw value would open a gate the caller asked to close.
    Anything unrecognized reads as False: these flags guard downloads, so the safe
    answer to an ambiguous value is the closed one."""
    if isinstance(value, str):
        return value.strip().lower() in TRUE_STRINGS
    return bool(value)


def _bridge() -> bridge_mod.CivitaiBridge:
    return bridge_mod.get_bridge()


def _emit(action: str, ok: bool, data: Any, *, warnings: list[str] | None = None,
          degraded_lm: bool = False, gen_result: Any = None, exit_code: int = EXIT_OK) -> int:
    envelope = {
        "schema": SCHEMA,
        "action": action,
        "ok": ok,
        "data": data,
        "warnings": warnings or [],
        "degraded": {"lm": degraded_lm},
    }
    if gen_result is not None:
        envelope["gen_result"] = gen_result
    print(json.dumps(envelope, ensure_ascii=False, indent=2))
    return exit_code


def _print_fields() -> None:
    print("actions   :", ", ".join(ACTIONS))
    print("list      : {\"action\":\"list\", \"search\"?, \"page\"?, \"page_size\"?}")
    print("show      : {\"action\":\"show\", \"id\"}")
    print("find-missing    : {\"action\":\"find-missing\", \"id\"} or {\"image\": \"<civitai url|id>\"}")
    print("download-missing: {\"action\":\"download-missing\", \"id\", \"wait\"?, \"timeout_s\"?}")
    print("to-brief  : {\"action\":\"to-brief\", \"id\", \"overrides\"?} or {\"recipe_file\": PATH}")
    print("generate  : {\"action\":\"generate\", \"id\", \"overrides\"?, \"download_missing\"?,"
          " \"allow_partial_fill\"?, \"json_out\"?}")
    print("            # a failed download_missing stops the run unless allow_partial_fill is true")
    print("sync      : {\"action\":\"sync\"}  # mirror LM store into recipes/library/")
    print("overrides : any gen brief field (gen_job.py --fields), null deletes a derived field")
    print("\nExample: {\"action\":\"generate\", \"id\":\"<recipe-id>\", \"overrides\":{\"tier\":\"draft\",\"n\":1}}")


def _load_recipe_dto(job: dict) -> tuple[dict | None, list[str], bool, dict | None]:
    """-> (recipe_dto, warnings, degraded_lm, failure). failure is None or
    {"error": str, "reason": "unreachable" | "not_found" | "invalid"}."""
    warnings: list[str] = []
    recipe_file = job.get("recipe_file")
    if recipe_file:
        try:
            raw = json.loads(Path(recipe_file).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return None, warnings, False, {"error": f"recipe_file unreadable: {exc}",
                                           "reason": "invalid"}
        # Accept a raw LM item, a normalized DTO, or a local library record.
        if isinstance(raw, dict) and isinstance(raw.get("recipe"), dict):
            raw = raw["recipe"]
        dto = raw if isinstance(raw, dict) and raw.get("genParams") else recipe_brief.recipe_from_lm_item(raw)
        return dto, warnings, True, None
    recipe_id = str(job.get("id") or "")
    if not recipe_id:
        return None, warnings, False, {"error": "id or recipe_file required", "reason": "invalid"}
    local = recipe_store.load_record(recipe_id)
    detail = _bridge().recipe_detail(recipe_id)
    if detail.get("ok"):
        return detail["recipe"], warnings, False, None
    if local and isinstance(local.get("recipe"), dict):
        # A local mirror answers whichever way the read failed, so say which one it was.
        why = ("recipe missing from LoRA Manager" if detail.get("reason") == "not_found"
               else "LoRA Manager unreachable")
        warnings.append(f"{why}; using the local library snapshot")
        return local["recipe"], warnings, True, None
    return None, warnings, False, {
        "error": str(detail.get("error") or "recipe unavailable"),
        "reason": str(detail.get("reason") or "unreachable"),
    }


def _act_list(job: dict) -> int:
    result = _bridge().recipes(
        page=int(job.get("page") or 1),
        page_size=int(job.get("page_size") or 20),
        search=str(job.get("search") or ""),
    )
    if not result.get("ok"):
        return _emit("list", False, result, degraded_lm=True, exit_code=EXIT_LM_DOWN)
    return _emit("list", True, result)


def _act_show(job: dict) -> int:
    dto, warnings, degraded, failure = _load_recipe_dto(job)
    if dto is None:
        return _emit("show", False, failure, exit_code=_exit_for_failure(failure))
    data: dict[str, Any] = {"recipe": dto}
    local = recipe_store.enrichment(str(dto.get("id") or ""))
    if local:
        data["local"] = local
    return _emit("show", True, data, warnings=warnings, degraded_lm=degraded)


def _missing_from_image_recipe(recipe: dict) -> list[dict]:
    """civitai_recipe.get_recipe(..., local=True) loras absent from the local catalog.
    Shape contract: recipe["loras"] entries carry local={"present": bool, ...}."""
    return [
        lora for lora in recipe.get("loras") or []
        if not (lora.get("local") or {}).get("present")
    ]


def _act_find_missing(job: dict) -> int:
    image = job.get("image")
    if image and not job.get("id") and not job.get("recipe_file"):
        # Un-saved Civitai image: the richer read-only extractor already cross-checks
        # the local catalog; borrow it as a library (SCRIPTS_INDEX borrowing policy).
        import civitai_recipe  # noqa: PLC0415 — scripts/ sibling, resolved via sys.path

        try:
            recipe = civitai_recipe.get_recipe(str(image), local=True)
        except Exception as exc:  # noqa: BLE001 — network/parse errors reported, not raised
            return _emit("find-missing", False, {"error": str(exc)}, exit_code=EXIT_LM_DOWN)
        if recipe.get("error"):
            return _emit("find-missing", False, {"image": image, "error": recipe["error"]},
                         exit_code=EXIT_LM_DOWN)
        data = {
            "image": image,
            "missing": _missing_from_image_recipe(recipe),
            "unresolved": recipe.get("unresolved") or [],
        }
        return _emit("find-missing", True, data, degraded_lm=True)
    dto, warnings, degraded, failure = _load_recipe_dto(job)
    if dto is None:
        return _emit("find-missing", False, failure, exit_code=_exit_for_failure(failure))
    return _emit(
        "find-missing", True,
        {"id": dto.get("id"), "missing": recipe_brief.missing_loras(dto)},
        warnings=warnings, degraded_lm=degraded,
    )


def _act_download_missing(job: dict) -> int:
    recipe_id = str(job.get("id") or "")
    if not recipe_id:
        return _emit("download-missing", False, {"error": "id required"}, exit_code=EXIT_INVALID)
    result = _bridge().download_missing(recipe_id=recipe_id, allow_nsfw=_as_bool(job.get("allow_nsfw")))
    if not result.get("ok"):
        code = _exit_for_failure(result)
        return _emit("download-missing", False, result, degraded_lm=(code == EXIT_LM_DOWN),
                     exit_code=code)
    if job.get("wait") and result.get("queued"):
        result["wait"] = _wait_for_downloads(
            [item.get("downloadId") for item in result["queued"] if item.get("downloadId")],
            timeout_s=float(job.get("timeout_s") or 900),
        )
    exit_code = EXIT_PARTIAL if result.get("skipped") else EXIT_OK
    return _emit("download-missing", True, result, exit_code=exit_code)


def _wait_for_downloads(download_ids: list[str], *, timeout_s: float) -> dict[str, Any]:
    pending = set(download_ids)
    deadline = time.time() + timeout_s
    statuses: dict[str, str] = {}
    while pending and time.time() < deadline:
        progress = _bridge().progress()
        for dl_job in progress.get("jobs") or []:
            did = dl_job.get("downloadId")
            if did in pending and dl_job.get("status") in {"completed", "failed", "canceled"}:
                statuses[did] = dl_job["status"]
                pending.discard(did)
        if pending:
            time.sleep(WAIT_POLL_S)
    for did in pending:
        statuses[did] = "timeout"
    return {"statuses": statuses, "timedOut": sorted(pending)}


def _act_to_brief(job: dict) -> int:
    dto, warnings, degraded, failure = _load_recipe_dto(job)
    if dto is None:
        return _emit("to-brief", False, failure, exit_code=_exit_for_failure(failure))
    job_brief, brief_warnings = recipe_brief.brief_from_recipe(dto, job.get("overrides"))
    problems = brief_mod.validate(job_brief)
    data = {
        "brief": job_brief,
        "problems": problems,
        "missing": recipe_brief.missing_loras(dto),
    }
    if problems:
        return _emit("to-brief", False, data, warnings=warnings + brief_warnings,
                     degraded_lm=degraded, exit_code=EXIT_INVALID)
    return _emit("to-brief", True, data, warnings=warnings + brief_warnings, degraded_lm=degraded)


def _act_generate(job: dict) -> int:
    dto, warnings, degraded, failure = _load_recipe_dto(job)
    if dto is None:
        return _emit("generate", False, failure, exit_code=_exit_for_failure(failure))
    if job.get("download_missing"):
        fill = _bridge().download_missing(recipe_id=str(dto.get("id") or ""),
                                          allow_nsfw=_as_bool(job.get("allow_nsfw")))
        if not fill.get("ok") and not job.get("allow_partial_fill"):
            # Rendering now would silently drop the LoRAs the recipe is built on.
            code = _exit_for_failure(fill)
            warnings.append("download-missing failed; pass allow_partial_fill to render anyway")
            return _emit("generate", False, {"fill": fill, "error": fill.get("error")},
                         warnings=warnings, degraded_lm=degraded or code == EXIT_LM_DOWN,
                         exit_code=code)
        if fill.get("ok") and fill.get("queued"):
            wait = _wait_for_downloads(
                [item.get("downloadId") for item in fill["queued"] if item.get("downloadId")],
                timeout_s=float(job.get("timeout_s") or 900),
            )
            fill["wait"] = wait
            if not wait.get("timedOut"):
                # Re-read so the just-landed LoRAs count as in-library for the brief.
                _bridge().invalidate()
                refreshed, _, _, _ = _load_recipe_dto({"id": dto.get("id")})
                dto = refreshed or dto
        if fill.get("ok"):
            warnings.append(f"download-missing: {len(fill.get('queued') or [])} queued, "
                            f"{len(fill.get('skipped') or [])} skipped")
        else:
            warnings.append("download-missing failed, rendering without it: "
                            f"{fill.get('error') or 'unknown error'}")
    job_brief, brief_warnings = recipe_brief.brief_from_recipe(dto, job.get("overrides"))
    problems = brief_mod.validate(job_brief)
    if problems:
        return _emit("generate", False, {"brief": job_brief, "problems": problems},
                     warnings=warnings + brief_warnings, degraded_lm=degraded, exit_code=EXIT_INVALID)
    if job.get("json_out"):
        json_out = Path(str(job["json_out"])).expanduser()
        json_out.parent.mkdir(parents=True, exist_ok=True)
        keep_result = True
    else:
        fd, tmp = tempfile.mkstemp(prefix="recipe_job_", suffix=".json")
        import os
        os.close(fd)
        json_out = Path(tmp)
        keep_result = False
    with tempfile.NamedTemporaryFile("w", suffix=".json", prefix="recipe_brief_",
                                     delete=False, encoding="utf-8") as handle:
        json.dump(job_brief, handle, ensure_ascii=False)
        brief_path = Path(handle.name)
    try:
        rc = subprocess.run(
            [sys.executable, str(GEN_JOB), str(brief_path), "--json-out", str(json_out)]
        ).returncode
    finally:
        brief_path.unlink(missing_ok=True)
    gen_result: Any = None
    if json_out.is_file():
        try:
            gen_result = json.loads(json_out.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            gen_result = None
        if not keep_result:
            json_out.unlink(missing_ok=True)
    recipe_id = str(dto.get("id") or "")
    if recipe_id:
        images = (gen_result or {}).get("images") if isinstance(gen_result, dict) else None
        try:
            recipe_store.upsert_from_lm(dto)
        except ValueError:
            pass
        recipe_store.append_run(recipe_id, {
            "via": "recipe_job",
            "rc": rc,
            "briefTier": job_brief.get("tier"),
            "images": images if isinstance(images, list) else None,
        })
    ok = rc == 0
    return _emit("generate", ok, {"brief": job_brief, "rc": rc},
                 warnings=warnings + brief_warnings, degraded_lm=degraded,
                 gen_result=gen_result, exit_code=rc)


def _act_sync(job: dict) -> int:
    result = _bridge().sync_recipes()
    if not result.get("ok"):
        return _emit("sync", False, result, degraded_lm=True, exit_code=EXIT_LM_DOWN)
    return _emit("sync", True, result)


_HANDLERS = {
    "list": _act_list,
    "show": _act_show,
    "find-missing": _act_find_missing,
    "download-missing": _act_download_missing,
    "to-brief": _act_to_brief,
    "generate": _act_generate,
    "sync": _act_sync,
}


def _run_job(job: object) -> int:
    if not isinstance(job, dict):
        print("job must be a JSON object", file=sys.stderr)
        return EXIT_INVALID
    action = str(job.get("action") or "")
    if not action and job.get("image"):
        action = "find-missing"
    handler = _HANDLERS.get(action)
    if handler is None:
        print(f"action must be one of: {', '.join(ACTIONS)}", file=sys.stderr)
        return EXIT_INVALID
    return handler(job)


def _parse_overrides(raw: str | None) -> dict | None:
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        sys.exit(f"--overrides is not valid JSON: {exc}")
    if not isinstance(data, dict):
        sys.exit("--overrides must be a JSON object")
    return data


def _subcommand_job(args: argparse.Namespace) -> dict:
    job: dict[str, Any] = {"action": args.action}
    for attr, key in (
        ("recipe_id", "id"), ("search", "search"), ("page", "page"),
        ("page_size", "page_size"), ("image", "image"), ("recipe_file", "recipe_file"),
        ("wait", "wait"), ("timeout_s", "timeout_s"), ("download_missing", "download_missing"),
        ("allow_nsfw", "allow_nsfw"), ("allow_partial_fill", "allow_partial_fill"),
        ("json_out", "json_out"),
    ):
        value = getattr(args, attr, None)
        if value not in (None, False, ""):
            job[key] = value
    overrides = _parse_overrides(getattr(args, "overrides", None))
    if overrides is not None:
        job["overrides"] = overrides
    return job


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--fields":
        _print_fields()
        return
    # JSON form: a file path or '-' as the sole positional (the gen_job.py pattern).
    if argv and argv[0] not in ACTIONS and not argv[0].startswith("-"):
        if len(argv) != 1:
            sys.exit(f"unknown action {argv[0]!r}; run --fields for the vocabulary")
        raw = sys.stdin.read() if argv[0] == "-" else Path(argv[0]).read_text(encoding="utf-8")
        try:
            job = json.loads(raw)
        except json.JSONDecodeError as exc:
            sys.exit(f"job is not valid JSON: {exc}")
        raise SystemExit(_run_job(job))
    if argv and argv[0] == "-":
        try:
            job = json.loads(sys.stdin.read())
        except json.JSONDecodeError as exc:
            sys.exit(f"job is not valid JSON: {exc}")
        raise SystemExit(_run_job(job))

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="action", required=True)
    p = sub.add_parser("list", help="list saved recipes (LM store)")
    p.add_argument("--search", default="")
    p.add_argument("--page", type=int, default=1)
    p.add_argument("--page-size", dest="page_size", type=int, default=20)
    p = sub.add_parser("show", help="one recipe + local enrichment")
    p.add_argument("recipe_id")
    p = sub.add_parser("find-missing", help="LoRAs a recipe needs but Z3 lacks")
    p.add_argument("recipe_id", nargs="?")
    p.add_argument("--image", help="un-saved civitai image url/id (offline catalog check)")
    p = sub.add_parser("download-missing", help="queue every missing LoRA of a recipe")
    p.add_argument("recipe_id")
    p.add_argument("--wait", action="store_true")
    p.add_argument("--timeout-s", dest="timeout_s", type=float, default=900)
    p.add_argument("--allow-nsfw", dest="allow_nsfw", action="store_true",
                   help="let NSFW-rated LoRAs through the download gate")
    p = sub.add_parser("to-brief", help="recipe -> validated gen brief (no side effects)")
    p.add_argument("recipe_id", nargs="?")
    p.add_argument("--overrides", help="JSON object of brief fields")
    p.add_argument("--recipe-file", dest="recipe_file", help="offline: local DTO/record/LM item JSON")
    p = sub.add_parser("generate", help="full loop: recipe -> brief -> gen_job.py")
    p.add_argument("recipe_id")
    p.add_argument("--overrides", help="JSON object of brief fields")
    p.add_argument("--download-missing", dest="download_missing", action="store_true")
    p.add_argument("--allow-partial-fill", dest="allow_partial_fill", action="store_true",
                   help="with --download-missing: render even if the fill fails")
    p.add_argument("--allow-nsfw", dest="allow_nsfw", action="store_true",
                   help="with --download-missing: let NSFW-rated LoRAs through the gate")
    p.add_argument("--timeout-s", dest="timeout_s", type=float, default=900)
    p.add_argument("--json-out", dest="json_out", help="keep gen_job's result JSON at PATH")
    sub.add_parser("sync", help="mirror the LM store into the local library")
    args = ap.parse_args(argv)
    raise SystemExit(_run_job(_subcommand_job(args)))


if __name__ == "__main__":
    main()
