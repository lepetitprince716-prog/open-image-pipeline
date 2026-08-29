#!/usr/bin/env python3
"""Forge (Stable Diffusion WebUI Forge) txt2img engine — direct gen + metadata repro.

Two modes on the Z3 Forge deployment (D:\\Forge\\forge, A1111 kernel, port 7862
reached through the 127.0.0.1:17862 loopback tunnel — docs/forge_line.md):

  direct   --prompt ...            plain txt2img with LoRA / hires / ADetailer knobs.
  repro    --source image.png      replay a Forge/A1111 image from its embedded
                                   ``parameters`` metadata: full field mapping
                                   (loramake/forge_payload.py), hash-matched assets,
                                   then a metadata round-trip diff + CCIP/grayscale
                                   similarity report.

Fail-closed: asset resolution and name preflights run against the live /sdapi/v1
enums and reject the run before any GPU work. PLAN-only by default; --run executes.
Output -> outputs/<character>/by_model/<DRIVING_MODEL>/forge/ with the A1111
``parameters`` chunk re-embedded plus the loramake meta sidecar (gallery + the
GEN_FILES_OUT manifest come from save_image_with_meta).

Usage:
  python3 scripts/forge_gen.py --prompt "1girl, rooftop at dusk" --lora mylora:0.8 --run
  python3 scripts/forge_gen.py --source civitai_original.png --run
  python3 scripts/forge_gen.py --source original.png --strict --json /tmp/payload.json
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OUTPUT_CHARACTER, by_model_dir  # noqa: E402  (bootstraps repo root on sys.path)
from loramake import forge, forge_payload  # noqa: E402
from loramake.forge_endpoints import resolve_forge_host  # noqa: E402
import image_recipe  # noqa: E402  (sibling; A1111 metadata parsing + imgutils bridge)

MODEL = os.environ.get("DRIVING_MODEL", "claude").strip().lower() or "claude"
DEFAULT_NEG = "lowres, worst quality, bad quality, jpeg artifacts, signature, watermark"
DEFAULT_SAMPLER = "DPM++ 2M"
DEFAULT_SCHEDULER = "karras"
DEFAULT_RES = "832x1216"
DEFAULT_HR_UPSCALER = "4x_foolhardy_Remacri"
DEFAULT_ADETAILER_MODEL = "face_yolov8n.pt"
REPORT_SCHEMA = "loramake.forge-repro-report/v1"
KERNEL_COMMIT = "dfdcbab6"  # the deployed Forge build the repro reference bands were measured on
HONESTY = (
    '"100%" means every material metadata field honored + hash-matched assets + the same '
    "Forge kernel; bit-identity across machines is limited by xformers nondeterminism, "
    "GPU/torch numeric drift, ADetailer version deltas, and unobtainable private embeddings. "
    "Reference band (same kernel, cross-machine): grayscale correlation ~0.75-0.79, face CCIP ~0.90."
)
MISSING_ASSET_HINT = (
    "fetch onto Z3 with the macloramanager line (LoRA Manager :9000 -> E:) or junction into "
    "D:\\Forge\\forge models, then POST /sdapi/v1/refresh-checkpoints + /refresh-loras; newly "
    "junctioned CHECKPOINTS additionally need a Forge restart (docs/forge_line.md)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forge txt2img engine (direct + metadata repro)")
    parser.add_argument("--prompt", help="direct mode: positive prompt")
    parser.add_argument("--neg", default=None, help=f"negative prompt (direct default: {DEFAULT_NEG!r})")
    parser.add_argument("--source", help="repro mode: PNG whose embedded A1111 parameters to replay")
    parser.add_argument("--n", type=int, default=1, help="images per run (n_iter; repro seed only holds for image 0)")
    parser.add_argument("--seed", type=int, help="seed (direct: random if omitted; repro: overrides the source seed)")
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--cfg", type=float, default=5.0)
    parser.add_argument("--res", default=DEFAULT_RES, help=f"WxH (default {DEFAULT_RES})")
    parser.add_argument("--sampler", default=DEFAULT_SAMPLER)
    parser.add_argument("--scheduler", default=DEFAULT_SCHEDULER)
    parser.add_argument("--ckpt", help="checkpoint title/stem (direct mode; omitted = server's loaded checkpoint)")
    parser.add_argument("--vae", help="VAE name (direct mode; omitted = server default)")
    parser.add_argument("--clip-skip", type=int, help="CLIP skip (direct mode)")
    parser.add_argument("--lora", action="append", default=[], metavar="NAME[:WEIGHT]",
                        help="add <lora:NAME:WEIGHT> to the prompt (repeatable; validated vs /loras)")
    parser.add_argument("--trigger", action="append", default=[],
                        help="text appended to the prompt (LoRA trigger words; repeatable)")
    parser.add_argument("--hires", action="store_true", help="direct mode: enable hires fix")
    parser.add_argument("--hr-scale", type=float, default=1.5)
    parser.add_argument("--hr-upscaler", default=DEFAULT_HR_UPSCALER)
    parser.add_argument("--hr-steps", type=int, default=20)
    parser.add_argument("--hr-denoise", type=float, default=0.4)
    parser.add_argument("--hr-cfg", type=float)
    parser.add_argument("--adetailer", nargs="?", const=DEFAULT_ADETAILER_MODEL, default=None,
                        metavar="MODEL", help=f"direct mode: enable ADetailer (default {DEFAULT_ADETAILER_MODEL})")
    parser.add_argument("--ad-conf", type=float, default=0.3)
    parser.add_argument("--ad-denoise", type=float, default=0.4)
    parser.add_argument("--param-override", action="append", default=[], metavar="KEY=VALUE",
                        help="post-mapping payload tweak (JSON value when parseable; recorded in the report)")
    parser.add_argument("--strict", action="store_true",
                        help="repro: unmapped material metadata keys become fatal")
    parser.add_argument("--allow-missing-embedding", action="store_true",
                        help="repro: downgrade a missing textual-inversion embedding from fatal to warning")
    parser.add_argument("--repro-report", help="repro report path (default <out-dir>/forge_repro_<stamp>.json)")
    parser.add_argument("--out-dir", help="output directory (default outputs/<char>/by_model/<model>/forge)")
    parser.add_argument("--host", help="Forge endpoint (default FORGE_HOST env, else the 17862 loopback)")
    parser.add_argument("--timeout", type=float, help="request budget in seconds (default FORGE_TIMEOUT / 1800)")
    parser.add_argument("--json", help="write the final payload + resolution notes to this path")
    parser.add_argument("--run", action="store_true", help="execute (default: PLAN only)")
    args = parser.parse_args()
    if bool(args.prompt) == bool(args.source):
        parser.error("exactly one of --prompt (direct) or --source (repro) is required")
    if args.n < 1 or args.n > 16:
        parser.error("--n must be in 1..16")
    return args


def parse_lora_flag(spec: str) -> tuple[str, float]:
    name, sep, weight = spec.rpartition(":")
    if sep and name and weight.replace(".", "", 1).isdigit():
        return name, float(weight)
    return spec, 1.0


def read_source_parsed(path: str) -> dict:
    """The authoritative metadata: the PNG ``parameters`` chunk, EXIF fallback."""
    source = Path(path)
    if not source.is_file():
        sys.exit(f"ERROR: source image not found: {path}")
    text = None
    try:
        chunks = image_recipe.decode_text_chunks(image_recipe.read_png_chunks(str(source)))
        text = chunks.get("parameters")
    except Exception:
        pass
    if not text:
        text = image_recipe.read_exif_parameters(str(source))
    if not text:
        sys.exit("ERROR: source carries no A1111 'parameters' metadata (PNG chunk or EXIF). "
                 "The embedded chunk is the authoritative recipe — a stripped re-upload cannot be replayed.")
    parsed = image_recipe.parse_a1111_parameters(text)
    if not parsed.get("parameters"):
        sys.exit("ERROR: source metadata has no parseable parameter line (Steps/Sampler/Seed...).")
    return parsed


def build_direct(args: argparse.Namespace) -> forge_payload.MappedPayload:
    width, _, height = str(args.res).partition("x")
    if not (width.isdigit() and height.isdigit()):
        sys.exit(f"ERROR: --res must be WxH, got {args.res!r}")
    prompt = args.prompt.strip()
    assets: dict = {"checkpoint": None, "vae": None, "loras": [], "embeddings": []}
    existing_tags = {t["name"] for t in forge_payload.parse_lora_tags(prompt)}
    for spec in args.lora:
        name, weight = parse_lora_flag(spec)
        if name not in existing_tags:
            prompt += f", <lora:{name}:{weight:g}>"
        assets["loras"].append({"name": name, "weight": weight, "short_hash": None, "in_prompt": True})
    payload: dict = {
        "prompt": prompt,
        "negative_prompt": args.neg if args.neg is not None else DEFAULT_NEG,
        "steps": args.steps,
        "cfg_scale": args.cfg,
        "width": int(width),
        "height": int(height),
        "sampler_name": args.sampler,
        "scheduler": args.scheduler,
        "send_images": True,
        "save_images": False,
    }
    override: dict = {}
    if args.clip_skip is not None:
        override["CLIP_stop_at_last_layers"] = args.clip_skip
    if args.vae:
        assets["vae"] = {"name": args.vae, "short_hash": None}
    if args.ckpt:
        assets["checkpoint"] = {"name": args.ckpt, "short_hash": None}
    if args.hires:
        payload.update({
            "enable_hr": True,
            "hr_scale": args.hr_scale,
            "hr_upscaler": args.hr_upscaler,
            "hr_second_pass_steps": args.hr_steps,
            "denoising_strength": args.hr_denoise,
            # Omitting this crashes Forge's hires path with a NoneType error.
            "hr_additional_modules": ["Use same choices"],
        })
        if args.hr_cfg is not None:
            payload["hr_cfg"] = args.hr_cfg
    if args.adetailer is not None:
        payload["alwayson_scripts"] = {"adetailer": {"args": [
            True, args.adetailer, "None", args.ad_conf, 4, 4, args.ad_denoise, True, 32,
        ]}}
    if override:
        payload["override_settings"] = override
        payload["override_settings_restore_afterwards"] = False
    return forge_payload.MappedPayload(payload=payload, assets=assets, warnings=[],
                                       unmapped={}, report_only={})


def apply_overrides(mapped: forge_payload.MappedPayload, args: argparse.Namespace) -> None:
    payload = mapped.payload
    payload["n_iter"] = args.n
    payload["batch_size"] = 1
    if args.seed is not None:
        payload["seed"] = args.seed
    elif payload.get("seed") is None:
        payload["seed"] = random.randint(0, 2**32 - 1)  # explicit so the produced metadata is complete
    if args.trigger:
        payload["prompt"] = payload["prompt"].rstrip(", ") + ", " + ", ".join(args.trigger)
    for item in args.param_override:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            sys.exit(f"ERROR: --param-override needs KEY=VALUE, got {item!r}")
        try:
            payload[key.strip()] = json.loads(value)
        except json.JSONDecodeError:
            payload[key.strip()] = value
        mapped.warnings.append(f"param override applied: {key.strip()}={value!r}")


def _suggest(name: str, candidates: list[str], limit: int = 3) -> str:
    close = difflib.get_close_matches(name.lower(), [c.lower() for c in candidates], n=limit, cutoff=0.4)
    hits = [c for c in candidates if c.lower() in close]
    return f"; closest: {', '.join(hits[:limit])}" if hits else ""


def resolve_assets(host: str, mapped: forge_payload.MappedPayload,
                   args: argparse.Namespace) -> tuple[list[str], list[str]]:
    """Match asset stubs against the live enums; mutate the payload with matched names.

    Returns (problems, notes). Problems are fail-closed; an unavailable enum
    endpoint degrades to a note, never a false rejection.
    """
    problems: list[str] = []
    notes: list[str] = []
    payload = mapped.payload
    assets = mapped.assets

    checkpoint = assets.get("checkpoint")
    if checkpoint:
        models = forge.sd_models(host)
        entry, basis = forge_payload.match_checkpoint(checkpoint.get("name"),
                                                      checkpoint.get("short_hash"), models)
        if entry:
            payload.setdefault("override_settings", {})["sd_model_checkpoint"] = entry["title"]
            payload["override_settings_restore_afterwards"] = False
            checkpoint["matched"] = {"title": entry.get("title"), "basis": basis}
            notes.append(f"checkpoint {checkpoint.get('name') or checkpoint.get('short_hash')!r} "
                         f"-> {entry.get('title')!r} (matched by {basis})")
        elif not models:
            fallback = checkpoint.get("name")
            if fallback:
                payload.setdefault("override_settings", {})["sd_model_checkpoint"] = fallback
                payload["override_settings_restore_afterwards"] = False
                notes.append(f"checkpoint enum unavailable — sending name {fallback!r} unverified")
            else:
                problems.append("checkpoint is hash-only and /sd-models is unreachable; cannot resolve")
        else:
            titles = [str(m.get("title") or "") for m in models]
            problems.append(
                f"checkpoint not found on Forge: {checkpoint.get('name')!r} "
                f"(hash {checkpoint.get('short_hash')}){_suggest(str(checkpoint.get('name')), titles)} "
                f"— {MISSING_ASSET_HINT}")

    vae = assets.get("vae")
    if vae:
        vaes = forge.sd_vaes(host)
        names = [str(v.get("model_name") or "") for v in vaes]
        hit = next((n for n in names if n.lower() == str(vae["name"]).lower()), None)
        if hit is None and vaes:
            stem = forge_payload._stem(vae["name"])
            hit = next((n for n in names if forge_payload._stem(n) == stem), None)
        if hit:
            payload.setdefault("override_settings", {})["sd_vae"] = hit
            payload["override_settings_restore_afterwards"] = False
            notes.append(f"VAE {vae['name']!r} -> {hit!r}")
        elif not vaes:
            payload.setdefault("override_settings", {})["sd_vae"] = vae["name"]
            payload["override_settings_restore_afterwards"] = False
            notes.append(f"VAE enum unavailable — sending {vae['name']!r} unverified")
        else:
            problems.append(f"VAE not found on Forge: {vae['name']!r}"
                            f"{_suggest(str(vae['name']), names)} — {MISSING_ASSET_HINT}")

    lora_entries = forge.loras(host)
    for stub in assets.get("loras") or []:
        entry, basis = forge_payload.match_lora(stub["name"], lora_entries)
        if entry:
            stub["matched"] = {"name": entry.get("name"), "basis": basis}
            notes.append(f"lora {stub['name']!r} -> {entry.get('name')!r} (matched by {basis})")
        elif not lora_entries:
            notes.append(f"lora enum unavailable — {stub['name']!r} unverified")
        else:
            names = [str(e.get("name") or "") for e in lora_entries]
            problems.append(f"LoRA not found on Forge: {stub['name']!r}"
                            f"{_suggest(stub['name'], names)} — {MISSING_ASSET_HINT}")

    embed_names = forge.embedding_names(host)
    for stub in assets.get("embeddings") or []:
        if embed_names is None:
            notes.append(f"embedding enum unavailable — {stub['name']!r} unverified")
        elif stub["name"] not in embed_names:
            message = (f"embedding not found on Forge: {stub['name']!r} — it would render as literal "
                       f"prompt text; {MISSING_ASSET_HINT}")
            if args.allow_missing_embedding:
                mapped.warnings.append(message + " (allowed by --allow-missing-embedding)")
            else:
                problems.append(message + " (or pass --allow-missing-embedding)")
        else:
            notes.append(f"embedding {stub['name']!r} present")

    sampler = payload.get("sampler_name")
    known_samplers = forge.sampler_names(host)
    if sampler and known_samplers is not None and sampler not in known_samplers:
        problems.append(f"sampler not known to Forge: {sampler!r}"
                        f"{_suggest(str(sampler), sorted(known_samplers))}")
    scheduler = payload.get("scheduler")
    known_schedulers = forge.scheduler_names(host)
    if scheduler and known_schedulers is not None:
        lowered = {s.lower() for s in known_schedulers}
        if str(scheduler).lower() not in lowered:
            problems.append(f"scheduler not known to Forge: {scheduler!r}"
                            f"{_suggest(str(scheduler), sorted(known_schedulers))}")
    elif scheduler and known_schedulers is None:
        notes.append("scheduler enum unavailable — sending unverified")
    hr_upscaler = payload.get("hr_upscaler")
    known_upscalers = forge.upscaler_names(host)
    if hr_upscaler and known_upscalers is not None and hr_upscaler not in known_upscalers:
        problems.append(f"hires upscaler not found on Forge: {hr_upscaler!r}"
                        f"{_suggest(str(hr_upscaler), sorted(known_upscalers))} — {MISSING_ASSET_HINT}")

    if "alwayson_scripts" in payload:
        available = forge.adetailer_available(host)
        if available is False:
            problems.append("ADetailer script is not installed on this Forge; cannot honor the "
                            "source's ADetailer pass")
        elif available is None:
            notes.append("scripts enum unavailable — ADetailer presence unverified")
        ad_models = forge.adetailer_model_names(host)
        wanted = payload["alwayson_scripts"]["adetailer"]["args"][1]
        if ad_models is not None and wanted not in ad_models:
            problems.append(f"ADetailer model not found: {wanted!r}"
                            f"{_suggest(str(wanted), sorted(ad_models))}")
    return problems, notes


def print_plan(host: str, mode: str, mapped: forge_payload.MappedPayload,
               problems: list[str], notes: list[str], reachable: bool) -> None:
    print(f"[forge] mode={mode} host={host}" + ("" if reachable else "  (UNREACHABLE — plan only)"))
    print(json.dumps(mapped.payload, ensure_ascii=False, indent=2))
    for note in notes:
        print(f"  [asset] {note}")
    for warning in mapped.warnings:
        print(f"  [warn] {warning}")
    for key, value in mapped.unmapped.items():
        print(f"  [unmapped] {key}: {value}")
    if mapped.report_only:
        print(f"  [provenance] {json.dumps(mapped.report_only, ensure_ascii=False)}")
    for problem in problems:
        print(f"  [PREFLIGHT-FAIL] {problem}")


def read_produced_parsed(path: Path) -> dict:
    try:
        chunks = image_recipe.decode_text_chunks(image_recipe.read_png_chunks(str(path)))
        return image_recipe.parse_a1111_parameters(chunks.get("parameters") or "")
    except Exception:
        return {}


def verify_repro(source: str, source_parsed: dict, saved: list[Path],
                 mapped: forge_payload.MappedPayload, report_path: Path) -> None:
    entries = []
    for path in saved:
        produced_parsed = read_produced_parsed(path)
        rows = forge_payload.diff_parameters(source_parsed, produced_parsed)
        failing = [r for r in rows if r["status"] in ("DIFF", "SOURCE_ONLY") and not r["benign"]]
        ccip = image_recipe.run_bridge("ccip", source, str(path))
        gray = image_recipe.run_bridge("graycorr", source, str(path))
        entries.append({
            "path": str(path),
            "fields": rows,
            "non_benign_mismatches": len(failing),
            "ccip": ccip,
            "graycorr": gray,
        })
        verdict = "PASS" if not failing else f"FAIL ({len(failing)} field(s))"
        corr = gray.get("correlation") if isinstance(gray, dict) and gray.get("ok") else None
        ccip_diff = ccip.get("difference") if isinstance(ccip, dict) and ccip.get("ok") else None
        print(f"  [repro] {path.name}: fields {verdict}"
              + (f", graycorr {corr}" if corr is not None else ", graycorr unavailable")
              + (f", ccip diff {ccip_diff}" if ccip_diff is not None else ""))
        for row in failing:
            print(f"      {row['status']}: {row['field']}: source={row['source']!r} "
                  f"produced={row['produced']!r}")
    report = {
        "schema": REPORT_SCHEMA,
        "source": source,
        "kernel_commit": KERNEL_COMMIT,
        "honesty": HONESTY,
        "warnings": mapped.warnings,
        "unmapped": mapped.unmapped,
        "provenance": mapped.report_only,
        "assets": mapped.assets,
        "produced": entries,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [repro] report -> {report_path}")


def main() -> None:
    args = parse_args()
    mode = "repro" if args.source else "direct"
    host = (args.host or resolve_forge_host("z3")).rstrip("/")

    if mode == "repro":
        source_parsed = read_source_parsed(args.source)
        mapped = forge_payload.payload_from_params(source_parsed)
    else:
        source_parsed = {}
        mapped = build_direct(args)
    apply_overrides(mapped, args)

    reachable, ping_msg = forge.ping(host)
    problems: list[str] = []
    notes: list[str] = []
    if reachable:
        problems, notes = resolve_assets(host, mapped, args)
    if args.strict and mapped.unmapped:
        problems.append("unmapped metadata keys under --strict: "
                        + ", ".join(sorted(mapped.unmapped)))

    print_plan(host, mode, mapped, problems, notes, reachable)
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"payload": mapped.payload, "assets": mapped.assets, "warnings": mapped.warnings,
             "unmapped": mapped.unmapped, "problems": problems, "notes": notes},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  [json] -> {args.json}")
    if problems:
        sys.exit(2)
    if not args.run:
        print("PLAN only — add --run to execute.")
        return
    if not reachable:
        sys.exit(f"ERROR: {ping_msg}")
    print(f"  {ping_msg}")

    out_dir = Path(args.out_dir) if args.out_dir else by_model_dir(OUTPUT_CHARACTER, "forge", model=MODEL)
    started = time.monotonic()
    try:
        images, info = forge.txt2img(host, mapped.payload, timeout=args.timeout)
    except forge.ForgeError as exc:
        sys.exit(f"ERROR: {exc}")
    elapsed = round(time.monotonic() - started, 1)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    stem = f"forge_{stamp}_s{mapped.payload.get('seed')}"
    meta_base = {
        "engine": "forge",
        "kernel_commit": KERNEL_COMMIT,
        "host": host,
        "mode": mode,
        "source": args.source,
        "character": OUTPUT_CHARACTER,
        "driving_model": MODEL,
        "payload": {k: v for k, v in mapped.payload.items() if k not in ("send_images", "save_images")},
        "assets": mapped.assets,
    }
    saved = forge.save_results(images, info, out_dir=out_dir, stem=stem, meta_base=meta_base)
    print(f"  [done] {len(saved)} image(s) in {elapsed}s")
    for path in saved:
        print(f"  {path}")
    if mode == "repro" and saved:
        report_path = (Path(args.repro_report) if args.repro_report
                       else out_dir / f"forge_repro_{stamp}.json")
        verify_repro(args.source, source_parsed, saved, mapped, report_path)


if __name__ == "__main__":
    main()
