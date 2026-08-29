"""Offline, deterministic contracts for the GPTArt generation/review line.

This module deliberately has no command-line entry point and never starts a
generation process.  ``compile_generation_argv`` returns an argv for a caller
to inspect or execute, while ``build_packet`` only reads already generated
files (and optionally invokes explicitly requested, read-only machine probes).
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PLAN_VERSION = "gptart-plan-v1"
PACKET_VERSION = "gptart-packet-v1"
REVIEW_VERSION = "gptart-review-v1"
ROLES = ("anchor", "composition_rupture", "chromatic_rupture", "kinetic_rupture")
ROLE_AXES = {
    "composition_rupture": "composition",
    "chromatic_rupture": "color",
    "kinetic_rupture": "pose",
}
HARD_GATES = ("anatomy", "identity", "intent_fidelity", "physical_logic")
ART_AXES = (
    "visual_hierarchy", "composition_tension", "light_color_causality", "medium_coherence",
    "specificity", "narrative_presence", "composition_invention", "color_invention",
    "pose_invention", "risk_taken", "risk_resolved", "surprise", "memorability",
    "authorial_specificity",
)
SLOP_FLAGS = (
    "digit_anatomy_error", "default_centered_portrait", "plastic_skin", "decorative_noise", "meaningless_glow",
    "random_accessories", "fake_text", "overprocessed_hdr", "conflicting_styles",
    "background_fill", "prompt_literal_collage", "batch_repetition",
)
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_DIRECTION_KEYS = ("intent", "composition", "lighting", "color_mood")
_RESULT_KEYS = ("intent", "box", "rc", "new_files", "survivors", "dropped", "note", "contact_sheet")
_REVIEW_KEYS = {
    "schema_version", "status", "reviewer_path", "reviewer_actor", "packet_path", "packet_sha256",
    "contact_sheet_read_verified", "full_resolution_indices", "candidates", "batch", "top3",
    "no_recommendation", "created_at",
}
_REVIEW_CANDIDATE_KEYS = {
    "index", "hard_gates", "art_axes", "intentional_distortion", "distortion_supports_thesis",
    "slop_flags", "decisive_observation", "largest_risk", "recommended_delta", "preserve_next_round",
}
_TOP3_KEYS = {"position", "index", "why", "risk", "next_delta"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _errors_for_plan(plan: Any, plan_path: Path) -> list[str]:
    errors: list[str] = []
    if not isinstance(plan, dict):
        return ["plan must be an object"]
    allowed_plan_keys = {"schema_version", "run_id", "subject", "user_goal", "protected_invariants",
                         "concepts", "review_contract"}
    for key in sorted(set(plan) - allowed_plan_keys):
        errors.append(f"unknown plan field: {key}")
    if plan.get("schema_version") != PLAN_VERSION:
        errors.append(f"schema_version must be {PLAN_VERSION}")
    run_id = plan.get("run_id")
    if not isinstance(run_id, str) or not _SAFE_RUN_ID.fullmatch(run_id) or ".." in run_id:
        errors.append("run_id must be path-safe")
    for key in ("subject", "user_goal", "protected_invariants", "concepts", "review_contract"):
        if key not in plan:
            errors.append(f"missing plan field: {key}")
    subject = plan.get("subject")
    if not ((_nonempty(subject)) or (isinstance(subject, dict) and bool(subject))):
        errors.append("subject must be a non-empty string or non-empty object")
    if not _nonempty(plan.get("user_goal")):
        errors.append("user_goal must be a non-empty string")
    if not isinstance(plan.get("protected_invariants"), list) or not all(
        _nonempty(v) for v in plan.get("protected_invariants", [])
    ):
        errors.append("protected_invariants must be a list of non-empty strings")
    review_contract = plan.get("review_contract")
    if not isinstance(review_contract, dict):
        errors.append("review_contract must be an object")
    else:
        if set(review_contract) != {"hard_gates", "art_axes"}:
            errors.append("review_contract must contain only hard_gates and art_axes")
        if review_contract.get("hard_gates") != list(HARD_GATES):
            errors.append("review_contract.hard_gates must exactly match the GPTArt contract")
        if review_contract.get("art_axes") != list(ART_AXES):
            errors.append("review_contract.art_axes must exactly match the GPTArt contract")
    concepts = plan.get("concepts")
    if not isinstance(concepts, list):
        errors.append("concepts must be a list")
        return errors
    seen_ids: set[str] = set()
    seen_roles: list[str] = []
    for idx, concept in enumerate(concepts):
        prefix = f"concepts[{idx}]"
        if not isinstance(concept, dict):
            errors.append(f"{prefix} must be an object")
            continue
        allowed_concept_keys = {"concept_id", "role", "visual_thesis", "creative_rupture", "supporting_axes",
                                "protected_invariants", "direction_source", "direction_intent", "argv",
                                "preserve", "avoid", "candidate_count"}
        for key in sorted(set(concept) - allowed_concept_keys):
            errors.append(f"{prefix} unknown field: {key}")
        concept_id = concept.get("concept_id")
        if not isinstance(concept_id, str) or not _SAFE_RUN_ID.fullmatch(concept_id) or ".." in concept_id:
            errors.append(f"{prefix}.concept_id must be path-safe")
        elif concept_id in seen_ids:
            errors.append(f"duplicate concept_id: {concept_id}")
        else:
            seen_ids.add(concept_id)
        role = concept.get("role")
        if role not in ROLES:
            errors.append(f"{prefix}.role must be one of {ROLES}")
        else:
            seen_roles.append(role)
        if not _nonempty(concept.get("visual_thesis")):
            errors.append(f"{prefix}.visual_thesis must be non-empty")
        rupture = concept.get("creative_rupture")
        if not isinstance(rupture, dict):
            errors.append(f"{prefix}.creative_rupture must be an object")
        else:
            if set(rupture) != {"primary_axis", "device", "purpose"}:
                errors.append(f"{prefix}.creative_rupture contains unknown or missing fields")
            if rupture.get("primary_axis") not in ("composition", "color", "pose"):
                errors.append(f"{prefix}.creative_rupture.primary_axis is invalid")
            expected_axis = ROLE_AXES.get(role)
            if expected_axis is not None and rupture.get("primary_axis") != expected_axis:
                errors.append(f"{prefix}.creative_rupture.primary_axis must be {expected_axis} for role {role}")
            for key in ("device", "purpose"):
                if not _nonempty(rupture.get(key)):
                    errors.append(f"{prefix}.creative_rupture.{key} must be non-empty")
        if "protected_invariants" in concept:
            invariants = concept["protected_invariants"]
            if not isinstance(invariants, list) or not all(_nonempty(v) for v in invariants):
                errors.append(f"{prefix}.protected_invariants must be a list of non-empty strings")
        if "supporting_axes" in concept:
            axes = concept["supporting_axes"]
            if (not isinstance(axes, dict) or not axes or set(axes) - {"composition", "color", "pose"}
                    or not all(_nonempty(v) for v in axes.values())):
                errors.append(f"{prefix}.supporting_axes must be a non-empty axis-to-string object")
        source = concept.get("direction_source")
        direction_intent = concept.get("direction_intent")
        if not isinstance(source, str) or not source:
            errors.append(f"{prefix}.direction_source must be a string")
        elif source == "freeform":
            if not isinstance(direction_intent, dict):
                errors.append(f"{prefix}.direction_intent must be an object for freeform")
            else:
                allowed_direction_keys = {"provenance", *_DIRECTION_KEYS}
                if set(direction_intent) - allowed_direction_keys:
                    errors.append(f"{prefix}.direction_intent contains unknown fields")
                if not _nonempty(direction_intent.get("provenance")):
                    errors.append(f"{prefix}.direction_intent.provenance must be non-empty")
                if not _nonempty(direction_intent.get("intent")):
                    errors.append(f"{prefix}.direction_intent.intent must be non-empty")
        else:
            card = _direction_card(source)
            if card is None:
                errors.append(f"{prefix}.direction_source card not found: {source}")
            elif not isinstance(direction_intent, dict):
                errors.append(f"{prefix}.direction_intent must be an object")
            else:
                expected = {"intent": card.get("intent"), **{
                    k: (card.get("slots") or {}).get(k) for k in _DIRECTION_KEYS[1:]
                }}
                if direction_intent != expected:
                    errors.append(f"{prefix}.direction_intent must exactly match card intent/slots")
        argv = concept.get("argv")
        count = concept.get("candidate_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            errors.append(f"{prefix}.candidate_count must be a positive integer")
        if not isinstance(argv, list) or not all(isinstance(v, str) for v in argv):
            errors.append(f"{prefix}.argv must be a list of strings")
        else:
            expected_gen = str(Path(__file__).resolve().parent / "gen.py")
            if len(argv) < 3 or argv[:3] != [sys.executable, expected_gen, "artistic"]:
                errors.append(f"{prefix}.argv must start with [sys.executable, repo/scripts/gen.py, artistic]")
            forbidden = {"--run", "--json-out", "--contact-sheet-out", "--no-wait"}
            if forbidden.intersection(argv):
                errors.append(f"{prefix}.argv contains forbidden router flag")
            try:
                separator = argv.index("--")
            except ValueError:
                separator = len(argv)
            router = argv[:separator]
            if argv.count("--sfw") != 1 or router.count("--sfw") != 1:
                errors.append(f"{prefix}.argv must contain --sfw exactly once in the router segment")
            if argv.count("--n") != 1 or router.count("--n") != 1:
                errors.append(f"{prefix}.argv must contain --n exactly once in the router segment")
            try:
                n_pos = router.index("--n")
                n_value = int(router[n_pos + 1])
            except (ValueError, TypeError, IndexError):
                n_value = None
                errors.append(f"{prefix}.argv must contain --n <candidate_count>")
            if n_value is not None and n_value != count:
                errors.append(f"{prefix}.argv --n must equal candidate_count")
        for key in ("preserve", "avoid"):
            value = concept.get(key)
            if not isinstance(value, list) or not all(_nonempty(v) for v in value):
                errors.append(f"{prefix}.{key} must be a list of non-empty strings")
    if sorted(seen_roles) != sorted(ROLES):
        errors.append("concept roles must contain each of the four roles exactly once")
    if len(concepts) != 4:
        errors.append("concepts must contain exactly four entries")
    return errors


def _direction_card(slug: str) -> dict[str, Any] | None:
    try:
        mod = _import_local("art_directions")
        result = mod.get(slug)
        return result if isinstance(result, dict) else None
    except (ImportError, OSError, json.JSONDecodeError):
        return None


def validate_plan(plan: dict, plan_path: Path) -> list[str]:
    """Return semantic plan errors; this function never raises for bad input."""
    return _errors_for_plan(plan, Path(plan_path))


def _import_local(name: str):
    """Import a sibling script in both direct-script and ``scripts.*`` package styles."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name != name:
            raise
        return importlib.import_module(f"scripts.{name}")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON {path}: {exc}") from exc


def _plan_and_concept(plan_path: Path, concept: dict) -> tuple[dict, dict]:
    plan = _read_json(plan_path)
    errors = validate_plan(plan, plan_path)
    if errors:
        raise ValueError("invalid plan: " + "; ".join(errors))
    concept_id = concept.get("concept_id") if isinstance(concept, dict) else None
    matches = [c for c in plan["concepts"] if c.get("concept_id") == concept_id]
    if len(matches) != 1 or matches[0] != concept:
        raise ValueError("concept does not belong to the validated plan")
    return plan, matches[0]


def compile_generation_argv(plan_path: Path, concept: dict, *, execute: bool) -> list[str]:
    """Compile one safe ``gen.py artistic`` argv without executing it."""
    plan_path = Path(plan_path).resolve()
    _, selected = _plan_and_concept(plan_path, concept)
    argv = list(selected["argv"])
    concept_id = selected["concept_id"]
    insertion = [
        "--json-out", str(plan_path.parent / f"generation_result.{concept_id}.json"),
        "--contact-sheet-out", str(plan_path.parent / f"contact_sheet.{concept_id}.png"),
    ]
    if execute:
        targets = (Path(insertion[1]), Path(insertion[3]))
        existing = [str(path) for path in targets if os.path.lexists(path)]
        if existing:
            raise ValueError("generation output already exists: " + ", ".join(existing))
        insertion.append("--run")
    try:
        separator = argv.index("--")
    except ValueError:
        separator = len(argv)
    return argv[:separator] + insertion + argv[separator:]


def _resolve_candidate(raw: Any, result_path: Path) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"survivor path must be a non-empty string in {result_path}")
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = result_path.parent / p
    return p.resolve()


def _sidecar_for(image: Path) -> Path | None:
    candidates = (Path(str(image) + ".meta.json"), image.with_suffix(".meta.json"), image.with_suffix(".json"))
    return next((p for p in candidates if p.is_file()), None)


def _find_reference(plan: dict, concept: dict) -> str | None:
    subject = plan.get("subject")
    if isinstance(subject, dict):
        for key in ("reference_image", "ref", "identity_reference"):
            if _nonempty(subject.get(key)):
                return subject[key]
    argv = concept.get("argv", [])
    if "--ref" in argv:
        i = argv.index("--ref")
        if i + 1 < len(argv) and _nonempty(argv[i + 1]):
            return argv[i + 1]
    return None


def _probe_statuses(image: Path, plan: dict, concept: dict, *, probe_machine: bool) -> tuple[str, str, str]:
    reference = _find_reference(plan, concept)
    if not probe_machine:
        return "unverified", "unverified", "not_applicable" if reference is None else "unverified"
    try:
        qc = _import_local("qc_score")
        sfw = _import_local("sfw_gate")
        consistency = _import_local("consistency_check")
        qc_data = qc.qc_one(getattr(qc, "DEFAULT_VENV_PY", ""), str(image))
        qc_status = "unverified" if qc_data is None else ("pass" if not qc.flags_for(qc_data, None) else "fail")
        rating = sfw.wd14_rating(str(image))
        verdict = sfw.image_ok(rating)
        sfw_status = "unverified" if verdict is None else ("pass" if verdict else "fail")
        if reference is None:
            ccip_status = "not_applicable"
        else:
            ccip_data = consistency.ccip_one(getattr(consistency, "DEFAULT_VENV_PY", ""), reference, str(image))
            same_character = ccip_data.get("same_character") if isinstance(ccip_data, dict) else None
            ccip_status = ("pass" if same_character else "fail") if isinstance(same_character, bool) else "unverified"
        return qc_status, sfw_status, ccip_status
    except Exception:
        # Probe availability is deliberately fail-open and honest: no exception
        # can turn an unverified local model into a passing gate.
        return "unverified", "unverified", "not_applicable" if reference is None else "unverified"


def _write_exclusive(path: Path, data: bytes) -> tuple[int, int]:
    """Write a new file and return its device/inode ownership token."""
    ownership: tuple[int, int] | None = None
    try:
        with path.open("xb") as fh:
            stat = os.fstat(fh.fileno())
            ownership = (stat.st_dev, stat.st_ino)
            fh.write(data)
            fh.flush()
            return ownership
    except Exception:
        if ownership is not None:
            _unlink_if_owned(path, ownership)
        raise


def _unlink_if_owned(path: Path, ownership: tuple[int, int]) -> None:
    try:
        stat = path.stat()
        if (stat.st_dev, stat.st_ino) == ownership:
            path.unlink()
    except FileNotFoundError:
        return


def build_packet(plan_path: Path, result_paths: dict[str, Path], output_dir: Path,
                 *, probe_machine: bool = False) -> dict:
    """Build a stable packet from explicit per-concept generation result files."""
    plan_path = Path(plan_path)
    output_dir = Path(output_dir)
    plan = _read_json(plan_path)
    errors = validate_plan(plan, plan_path)
    if errors:
        raise ValueError("invalid plan: " + "; ".join(errors))
    if not output_dir.is_dir():
        raise ValueError(f"output_dir must already exist: {output_dir}")
    packet_path = output_dir / "packet.json"
    sheet_path = output_dir / "contact_sheet.png"
    if packet_path.exists() or sheet_path.exists():
        raise ValueError("packet.json/contact_sheet.png already exists")
    expected_ids = [c["concept_id"] for c in plan["concepts"]]
    if set(result_paths) != set(expected_ids) or len(result_paths) != len(expected_ids):
        raise ValueError("result_paths concept ids must exactly match plan concepts")
    resolved_result_paths = {concept_id: Path(result_paths[concept_id]).resolve() for concept_id in expected_ids}
    if len(set(resolved_result_paths.values())) != len(expected_ids):
        raise ValueError("result paths must resolve to four distinct files")

    summaries: dict[str, dict] = {}
    concepts = {concept["concept_id"]: concept for concept in plan["concepts"]}
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for concept_id in expected_ids:
        result_path = resolved_result_paths[concept_id]
        result = _read_json(result_path)
        if not isinstance(result, dict) or any(k not in result for k in _RESULT_KEYS):
            raise ValueError(f"result {result_path} must contain generation summary fields")
        unknown_result_keys = set(result) - (set(_RESULT_KEYS) | {"best"})
        if unknown_result_keys:
            raise ValueError(f"unknown result fields in {result_path}: {sorted(unknown_result_keys)}")
        if result.get("intent") != "artistic":
            raise ValueError(f"result {result_path}.intent must be artistic")
        if not _nonempty(result.get("box")):
            raise ValueError(f"result {result_path}.box must be non-empty")
        if isinstance(result.get("rc"), bool) or result.get("rc") != 0:
            raise ValueError(f"result {result_path}.rc must be integer zero")
        for key in ("new_files", "survivors"):
            values = result.get(key)
            if not isinstance(values, list) or not all(_nonempty(value) for value in values):
                raise ValueError(f"result {result_path}.{key} must be a list of non-empty strings")
        resolved_new_files = {_resolve_candidate(raw, result_path) for raw in result["new_files"]}
        resolved_survivors = [_resolve_candidate(raw, result_path) for raw in result["survivors"]]
        if not set(resolved_survivors).issubset(resolved_new_files):
            raise ValueError(f"result {result_path}.survivors must be a subset of new_files")
        if not isinstance(result.get("dropped"), dict):
            raise ValueError(f"result {result_path}.dropped must be an object")
        if result.get("note") is not None and not isinstance(result.get("note"), str):
            raise ValueError(f"result {result_path}.note must be a string or null")

        arm_sheet: Path | None = None
        arm_sheet_sha256: str | None = None
        if result["survivors"]:
            raw_sheet = result.get("contact_sheet")
            if not _nonempty(raw_sheet):
                raise ValueError(f"result {result_path}.contact_sheet is required when survivors exist")
            arm_sheet = Path(raw_sheet).expanduser()
            if not arm_sheet.is_absolute():
                arm_sheet = result_path.parent / arm_sheet
            arm_sheet = arm_sheet.resolve()
            if not arm_sheet.is_file():
                raise ValueError(f"per-arm contact sheet is missing: {arm_sheet}")
            arm_sheet_sha256 = _sha256(arm_sheet)
        summaries[concept_id] = result
        for image in resolved_survivors:
            if not image.is_file():
                raise ValueError(f"survivor image is missing: {image}")
            if image in seen:
                raise ValueError(f"duplicate resolved survivor path: {image}")
            seen.add(image)
            sidecar = _sidecar_for(image)
            rows.append({
                "concept_id": concept_id,
                "concept": concepts[concept_id],
                "image": image,
                "image_sha256": _sha256(image),
                "sidecar": sidecar,
                "sidecar_sha256": _sha256(sidecar) if sidecar else None,
                "result_path": result_path,
                "arm_sheet": arm_sheet,
                "arm_sheet_sha256": arm_sheet_sha256,
            })
    if not rows:
        raise ValueError("all concepts have empty survivors; packet cannot be built")

    candidates: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        qc_status, sfw_status, ccip_status = _probe_statuses(
            row["image"], plan, row["concept"], probe_machine=probe_machine
        )
        candidates.append({
            "index": index,
            "concept_id": row["concept_id"],
            "path": str(row["image"]),
            "sha256": row["image_sha256"],
            "sidecar": str(row["sidecar"]) if row["sidecar"] else None,
            "sidecar_sha256": row["sidecar_sha256"],
            "survivor": True,
            "generation_result": str(row["result_path"]),
            "qc": qc_status,
            "sfw": sfw_status,
            "ccip": ccip_status,
            "per_arm_contact_sheet": str(row["arm_sheet"]),
            "per_arm_contact_sheet_sha256": row["arm_sheet_sha256"],
        })

    # All source paths and hashes are prepared before any final output exists.
    temporary: Path | None = None
    try:
        contact_sheet = _import_local("contact_sheet")
        with tempfile.NamedTemporaryFile(prefix="gptart-sheet-", suffix=".png", dir=output_dir,
                                         delete=False) as tmp:
            temporary = Path(tmp.name)
        contact_sheet.build_sheet([str(row["image"]) for row in rows], str(temporary))
        sheet_bytes = temporary.read_bytes()
        packet = {
            "schema_version": PACKET_VERSION,
            "run_id": plan["run_id"],
            "plan_sha256": _sha256(plan_path),
            "contact_sheet": str(sheet_path.resolve()),
            "contact_sheet_sha256": hashlib.sha256(sheet_bytes).hexdigest(),
            "generation_results": summaries,
            "candidates": candidates,
        }
        packet_bytes = json.dumps(packet, ensure_ascii=False, indent=2).encode("utf-8")
    except Exception as exc:
        raise ValueError(f"could not build combined contact sheet: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    sheet_ownership: tuple[int, int] | None = None
    try:
        sheet_ownership = _write_exclusive(sheet_path, sheet_bytes)
        _write_exclusive(packet_path, packet_bytes)
    except Exception as exc:
        if sheet_ownership is not None:
            _unlink_if_owned(sheet_path, sheet_ownership)
        raise ValueError(f"could not exclusively create packet outputs: {exc}") from exc
    return packet


def make_review_template(packet: dict, packet_path: Path) -> dict:
    """Create an explicitly unread, complete-coverage review skeleton."""
    if not isinstance(packet, dict) or packet.get("schema_version") != PACKET_VERSION:
        raise ValueError("packet must be a gptart-packet-v1 object")
    packet_path = Path(packet_path).resolve()
    if not packet_path.is_file():
        raise ValueError(f"packet_path does not exist: {packet_path}")
    disk_packet = _read_json(packet_path)
    if disk_packet != packet:
        raise ValueError("packet on disk does not exactly match the supplied packet")
    candidates = []
    for source in packet.get("candidates", []):
        candidates.append({
            "index": source.get("index"),
            "hard_gates": {key: "uncertain" for key in HARD_GATES},
            "art_axes": {key: "uncertain" for key in ART_AXES},
            "intentional_distortion": "uncertain",
            "distortion_supports_thesis": "uncertain",
            "slop_flags": [],
            "decisive_observation": "",
            "largest_risk": "",
            "recommended_delta": "",
            "preserve_next_round": "",
        })
    return {
        "schema_version": REVIEW_VERSION,
        "status": "template",
        "reviewer_path": "gptart",
        "reviewer_actor": "codex-vision",
        "packet_path": str(packet_path),
        "packet_sha256": _sha256(packet_path),
        "contact_sheet_read_verified": False,
        "full_resolution_indices": [],
        "candidates": candidates,
        "batch": {
            "concept_difference": "",
            "batch_homogeneity": "",
            "effective_rupture_axis": "",
            "experiment_to_continue": "",
        },
        "top3": [],
        "no_recommendation": "",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _hash_matches(review: dict, packet: dict) -> list[str]:
    errors: list[str] = []
    packet_value = review.get("packet_path")
    packet_path = Path(packet_value) if isinstance(packet_value, str) and packet_value else Path(".")
    if not packet_path.is_file():
        errors.append("packet_path does not exist")
    else:
        try:
            if _read_json(packet_path) != packet:
                errors.append("packet on disk does not match supplied packet")
        except ValueError:
            errors.append("packet_path is not readable JSON")
        if review.get("packet_sha256") != _sha256(packet_path):
            errors.append("packet_sha256 does not match packet_path")
    contact = packet.get("contact_sheet")
    if not isinstance(contact, str) or not Path(contact).is_file():
        errors.append("packet contact_sheet is missing")
    elif packet.get("contact_sheet_sha256") != _sha256(Path(contact)):
        errors.append("contact_sheet_sha256 drifted")
    for row in packet.get("candidates", []):
        image = Path(row.get("path", ""))
        if not image.is_file():
            errors.append(f"candidate image missing: {image}")
        elif row.get("sha256") != _sha256(image):
            errors.append(f"candidate image hash drifted: {image}")
        sidecar = row.get("sidecar")
        if sidecar:
            sidecar_path = Path(sidecar)
            if not sidecar_path.is_file() or row.get("sidecar_sha256") != _sha256(sidecar_path):
                errors.append(f"candidate sidecar hash drifted: {sidecar_path}")
        elif row.get("sidecar_sha256") is not None:
            errors.append(f"candidate sidecar hash exists without a path: {row.get('index')}")
        arm_sheet = row.get("per_arm_contact_sheet")
        if not isinstance(arm_sheet, str) or not Path(arm_sheet).is_file():
            errors.append(f"per-arm contact sheet is missing: {arm_sheet}")
        elif row.get("per_arm_contact_sheet_sha256") != _sha256(Path(arm_sheet)):
            errors.append(f"per-arm contact sheet hash drifted: {arm_sheet}")
    return errors


def validate_review(packet: dict, review: dict) -> list[str]:
    """Return structural, hash and cross-record review errors."""
    errors: list[str] = []
    if not isinstance(packet, dict) or packet.get("schema_version") != PACKET_VERSION:
        return ["packet must be a gptart-packet-v1 object"]
    if not isinstance(review, dict) or review.get("schema_version") != REVIEW_VERSION:
        return ["review must be a gptart-review-v1 object"]
    if set(review) != _REVIEW_KEYS:
        errors.append("review top-level fields must exactly match the contract")
    if review.get("reviewer_path") != "gptart":
        errors.append("reviewer_path must be gptart")
    if review.get("reviewer_actor") != "codex-vision":
        errors.append("reviewer_actor must be codex-vision")
    if not _nonempty(review.get("created_at")):
        errors.append("created_at must be non-empty")
    status = review.get("status")
    if status not in ("template", "complete"):
        errors.append("status must be template or complete")
    expected_indices = [row.get("index") for row in packet.get("candidates", [])]
    valid_expected_indices = all(isinstance(index, int) and not isinstance(index, bool)
                                 for index in expected_indices)
    if not valid_expected_indices or len(set(expected_indices)) != len(expected_indices):
        errors.append("packet candidate indices are not unique")
    rows = review.get("candidates")
    review_indices = [row.get("index") for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
    valid_review_indices = all(isinstance(index, int) and not isinstance(index, bool) for index in review_indices)
    if (not isinstance(rows, list) or len(review_indices) != len(rows) or not valid_expected_indices
            or not valid_review_indices
            or set(review_indices) != set(expected_indices) or len(set(review_indices)) != len(review_indices)):
        errors.append("review candidates must cover packet candidate indices exactly once")
    review_rows = rows if isinstance(rows, list) else []
    seen_review: set[Any] = set()
    for row in review_rows:
        if not isinstance(row, dict):
            errors.append("review candidate must be an object")
            continue
        index = row.get("index")
        if set(row) != _REVIEW_CANDIDATE_KEYS:
            errors.append(f"candidate fields must exactly match the contract: {index!r}")
        if not isinstance(index, int) or isinstance(index, bool):
            errors.append(f"review candidate index must be an integer: {index!r}")
        else:
            if index in seen_review:
                errors.append(f"duplicate review candidate index: {index}")
            seen_review.add(index)
        gates = row.get("hard_gates")
        if not isinstance(gates, dict) or set(gates) != set(HARD_GATES):
            errors.append(f"candidate {index} hard_gates are incomplete")
        else:
            for key, value in gates.items():
                if value not in ("pass", "fail", "uncertain"):
                    errors.append(f"candidate {index} hard gate {key} has invalid enum")
        axes = row.get("art_axes")
        if not isinstance(axes, dict) or set(axes) != set(ART_AXES):
            errors.append(f"candidate {index} art_axes are incomplete")
        elif any(value not in ("strong", "adequate", "weak", "uncertain") for value in axes.values()):
            errors.append(f"candidate {index} art axis has invalid enum")
        for key in ("intentional_distortion", "distortion_supports_thesis"):
            if row.get(key) not in ("yes", "no", "uncertain"):
                errors.append(f"candidate {index} {key} has invalid enum")
        slop = row.get("slop_flags")
        if (not isinstance(slop, list) or not all(isinstance(value, str) for value in slop)
                or any(value not in SLOP_FLAGS for value in slop)):
            errors.append(f"candidate {index!r} slop_flags has invalid enum")
        elif len(set(slop)) != len(slop):
            errors.append(f"candidate {index!r} slop_flags must be unique")
        if status == "complete":
            if (isinstance(slop, list) and "digit_anatomy_error" in slop
                    and (not isinstance(gates, dict) or gates.get("anatomy") != "fail")):
                errors.append(
                    f"candidate {index} digit_anatomy_error requires hard_gates.anatomy=fail"
                )
            for key in ("decisive_observation", "largest_risk", "recommended_delta", "preserve_next_round"):
                if not _nonempty(row.get(key)):
                    errors.append(f"candidate {index} {key} must be non-empty for complete review")
            if isinstance(row.get("recommended_delta"), list):
                errors.append(f"candidate {index} recommended_delta must be one string")
            intentional = row.get("intentional_distortion")
            supports = row.get("distortion_supports_thesis")
            if intentional == "yes" and supports == "uncertain":
                errors.append(f"candidate {index} intentional distortion support cannot remain uncertain")
            if supports == "yes" and intentional != "yes":
                errors.append(f"candidate {index} distortion contradiction: support=yes without intentional=yes")
            if (intentional == "yes" and supports == "yes" and isinstance(gates, dict)
                    and any(value == "fail" for value in gates.values())):
                errors.append(f"candidate {index} distortion contradiction: support=yes with a hard-gate fail")
        elif status == "template":
            if isinstance(gates, dict) and any(value != "uncertain" for value in gates.values()):
                errors.append(f"template candidate {index} hard gates must all be uncertain")
            if isinstance(axes, dict) and any(value != "uncertain" for value in axes.values()):
                errors.append(f"template candidate {index} art axes must all be uncertain")
            if row.get("intentional_distortion") != "uncertain" or row.get("distortion_supports_thesis") != "uncertain":
                errors.append(f"template candidate {index} distortion fields must be uncertain")
            if row.get("slop_flags") != []:
                errors.append(f"template candidate {index} slop_flags must be empty")
            if any(row.get(key) != "" for key in (
                "decisive_observation", "largest_risk", "recommended_delta", "preserve_next_round"
            )):
                errors.append(f"template candidate {index} text fields must be empty")

    full_res = review.get("full_resolution_indices")
    if (not isinstance(full_res, list) or not all(isinstance(index, int) and not isinstance(index, bool)
                                                 for index in full_res)
            or len(set(full_res)) != len(full_res)):
        errors.append("full_resolution_indices must be a unique list")
        full_res = []
    elif any(index not in expected_indices for index in full_res):
        errors.append("full_resolution_indices contains an unknown candidate")
    sheet_verified = review.get("contact_sheet_read_verified")
    if status == "template":
        if sheet_verified is not False or full_res:
            errors.append("template review must have contact_sheet_read_verified=false and no full-resolution indices")
        if review.get("top3") != []:
            errors.append("template review top3 must be empty")
        if review.get("no_recommendation") != "":
            errors.append("template review no_recommendation must be empty")
        batch = review.get("batch")
        batch_keys = {"concept_difference", "batch_homogeneity", "effective_rupture_axis",
                      "experiment_to_continue"}
        if not isinstance(batch, dict) or set(batch) != batch_keys or any(value != "" for value in batch.values()):
            errors.append("template review batch text fields must be empty")
    elif sheet_verified is not True:
        errors.append("complete review must set contact_sheet_read_verified=true")
    if status == "complete":
        errors.extend(_hash_matches(review, packet))

    if status == "complete":
        batch = review.get("batch")
        batch_keys = {"concept_difference", "batch_homogeneity", "effective_rupture_axis",
                      "experiment_to_continue"}
        if not isinstance(batch, dict) or set(batch) != batch_keys:
            errors.append("complete review requires batch diagnosis")
        else:
            for key in ("concept_difference", "batch_homogeneity", "effective_rupture_axis", "experiment_to_continue"):
                if not _nonempty(batch.get(key)):
                    errors.append(f"complete review batch.{key} must be non-empty")

    top3 = review.get("top3")
    if not isinstance(top3, list):
        errors.append("top3 must be a list")
        top3 = []
    slots: set[str] = set()
    top_indices: set[Any] = set()
    for item in top3:
        if not isinstance(item, dict):
            errors.append("top3 item must be an object")
            continue
        if set(item) != _TOP3_KEYS:
            errors.append("top3 fields must exactly match the contract")
        slot = item.get("position")
        if slot not in ("complete", "breakthrough", "frontier"):
            errors.append("top3 position is invalid")
        elif slot in slots:
            errors.append(f"top3 position repeated: {slot}")
        else:
            slots.add(slot)
        index = item.get("index")
        if not isinstance(index, int) or isinstance(index, bool):
            errors.append("top3 index must be an integer")
            continue
        if index in top_indices:
            errors.append(f"top3 index repeated: {index}")
        top_indices.add(index)
        if index not in expected_indices:
            errors.append(f"top3 index does not exist: {index}")
        if index not in full_res:
            errors.append(f"top3 index was not read at full resolution: {index}")
        match = next((row for row in review_rows if isinstance(row, dict) and row.get("index") == index), None)
        if match:
            gates = match.get("hard_gates", {})
            if isinstance(gates, dict):
                if any(value == "fail" for value in gates.values()):
                    errors.append(f"hard-fail candidate cannot enter top3: {index}")
                if slot in ("complete", "breakthrough") and any(value != "pass" for value in gates.values()):
                    errors.append(f"top3 {slot} candidate {index} hard gates must all pass")
        if status == "complete":
            for key in ("why", "risk", "next_delta"):
                if not _nonempty(item.get(key)):
                    errors.append(f"top3 {key} must be non-empty for complete review")
    no_recommendation = review.get("no_recommendation")
    if top3 and no_recommendation != "":
        errors.append("no_recommendation must be empty when top3 is non-empty")
    if status == "complete" and not top3 and not _nonempty(no_recommendation):
        errors.append("empty top3 requires a non-empty no_recommendation")
    return errors


def render_review_summary(review: dict) -> str:
    """Render only the five user-facing review headings; never crown or mutate files."""
    by_position = {item.get("position"): item for item in review.get("top3", [])
                   if isinstance(item, dict)}
    lines: list[str] = []
    for heading, key in (("COMPLETE", "complete"), ("BREAKTHROUGH", "breakthrough"), ("FRONTIER", "frontier")):
        lines.append(heading)
        item = by_position.get(key)
        if item:
            lines.append(f"- index {item.get('index')}: {item.get('why', '')}")
            if item.get("risk"):
                lines.append(f"  risk: {item['risk']}")
            if item.get("next_delta"):
                lines.append(f"  next: {item['next_delta']}")
        else:
            lines.append("- none")
    batch = review.get("batch") or {}
    lines.append("BATCH DIAGNOSIS")
    for key in ("concept_difference", "batch_homogeneity", "effective_rupture_axis", "experiment_to_continue"):
        value = batch.get(key)
        if value:
            lines.append(f"- {key}: {value}")
    if review.get("no_recommendation"):
        lines.append(f"- {review['no_recommendation']}")
    lines.append("NEXT ROUND")
    for row in review.get("candidates", []):
        if isinstance(row, dict) and row.get("recommended_delta"):
            lines.append(f"- index {row.get('index')}: {row['recommended_delta']}")
    return "\n".join(lines)
