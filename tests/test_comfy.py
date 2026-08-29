"""Focused tests for shared ComfyUI cleanup and loader-asset preflight helpers."""
from __future__ import annotations

import json
import os
import stat
from types import SimpleNamespace

import pytest

from loramake import comfy


def test_free_vram_uses_real_curl_returncode_and_http_fail_mode(monkeypatch):
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout="204")

    monkeypatch.setattr(comfy.subprocess, "run", run)
    assert comfy.free_vram("http://box.invalid:8188") is True
    argv, kwargs = calls[0]
    assert argv[:3] == ["curl", "-sS", "--fail"]
    assert argv[argv.index("--noproxy") + 1] == "box.invalid"
    assert argv[argv.index("--write-out") + 1] == "%{http_code}"
    assert argv[-1] == "@-"
    assert "http://box.invalid:8188/free" in argv
    assert kwargs["text"] is True

    monkeypatch.setattr(
        comfy.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=22, stdout="500")
    )
    assert comfy.free_vram() is False
    monkeypatch.setattr(
        comfy.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="302")
    )
    assert comfy.free_vram() is False


def test_free_vram_returns_false_when_curl_cannot_start(monkeypatch):
    def fail(*_args, **_kwargs):
        raise OSError("curl unavailable")

    monkeypatch.setattr(comfy.subprocess, "run", fail)
    assert comfy.free_vram() is False


def test_preflight_assets_checks_unet_clip_and_vae_with_backward_compatible_defaults(monkeypatch):
    present = {
        ("UNETLoader", "unet_name"): {"unet.safetensors"},
        ("LoraLoader", "lora_name"): set(),
        ("CLIPLoader", "clip_name"): {"clip.safetensors"},
        ("VAELoader", "vae_name"): {"vae.safetensors"},
    }
    monkeypatch.setattr(comfy, "object_info_enum",
                    lambda node, field, host=None: present.get((node, field), set()))

    comfy.preflight_assets(
        "krea2",
        "unet.safetensors",
        [],
        clips=["clip.safetensors"],
        vaes=["vae.safetensors"],
    )
    # Old positional/keyword surface remains valid for existing generation scripts.
    comfy.preflight_assets("krea2", "unet.safetensors", [])

    for loader, kwargs in (
        ("CLIPLoader.clip_name", {"clips": ["missing-clip.safetensors"]}),
        ("VAELoader.vae_name", {"vaes": ["missing-vae.safetensors"]}),
    ):
        with pytest.raises(SystemExit) as exc:
            comfy.preflight_assets("krea2", "unet.safetensors", [], **kwargs)
        assert loader in str(exc.value)


def test_save_image_with_meta_exclusive_mode_never_replaces_existing_artifacts(tmp_path):
    image = tmp_path / "same.png"
    image.write_bytes(b"competitor")
    with pytest.raises(FileExistsError):
        comfy.save_image_with_meta(image, b"ours", {"seed": 1}, overwrite=False)
    assert image.read_bytes() == b"competitor"

    image.unlink()
    sidecar = tmp_path / "same.png.meta.json"
    sidecar.write_text("competitor metadata", encoding="utf-8")
    with pytest.raises(FileExistsError):
        comfy.save_image_with_meta(image, b"ours", {"seed": 1}, overwrite=False)
    assert not image.exists()  # our partially published PNG is rolled back
    assert sidecar.read_text(encoding="utf-8") == "competitor metadata"


def test_emit_event_is_private_and_reports_prompt_id(tmp_path, monkeypatch):
    events = tmp_path / "attempt.jsonl"
    monkeypatch.setenv("GEN_EVENTS_OUT", str(events))
    seen = []
    record = comfy.emit_event("prompt_submitted", sink=seen.append, prompt_id="ours")
    saved = json.loads(events.read_text(encoding="utf-8"))
    assert saved["prompt_id"] == "ours" == record["prompt_id"]
    assert saved["observed_at"].endswith("Z")
    assert seen == [record]
    if hasattr(os, "fchmod"):
        assert stat.S_IMODE(events.stat().st_mode) == 0o600


def test_event_and_generated_file_manifests_work_without_fchmod(tmp_path, monkeypatch):
    events = tmp_path / "attempt.jsonl"
    files = tmp_path / "files.txt"
    image = tmp_path / "image.png"
    monkeypatch.setenv("GEN_EVENTS_OUT", str(events))
    monkeypatch.setenv("GEN_FILES_OUT", str(files))
    monkeypatch.delattr(comfy.os, "fchmod", raising=False)

    comfy.emit_event("prompt_submitted", prompt_id="windows-compatible")
    comfy.report_generated_file(image)

    assert json.loads(events.read_text(encoding="utf-8"))["prompt_id"] == "windows-compatible"
    assert files.read_text(encoding="utf-8").strip() == str(image.resolve())


def test_cancel_prompt_deletes_only_exact_pending_and_preserves_foreign(monkeypatch):
    state = {"running": ["foreign-running"], "pending": ["ours", "foreign-pending"]}
    posts = []

    def request(_host, endpoint, *, payload=None, timeout=30.0):
        if payload is None:
            return {"queue_running": [[0, p] for p in state["running"]],
                    "queue_pending": [[0, p] for p in state["pending"]]}
        posts.append((endpoint, payload))
        if endpoint == "/queue":
            state["pending"] = [p for p in state["pending"] if p not in payload["delete"]]
        return {}

    monkeypatch.setattr(comfy, "_request_json", request)
    outcome = comfy.cancel_prompt("http://box", "ours", timeout=1)
    assert outcome["ok"] is True
    assert posts == [("/queue", {"delete": ["ours"]})]
    assert state == {"running": ["foreign-running"], "pending": ["foreign-pending"]}


def test_cancel_prompt_interrupts_only_when_ours_is_unique_running(monkeypatch):
    state = {"running": ["ours"], "pending": ["foreign-pending"]}
    posts = []

    def request(_host, endpoint, *, payload=None, timeout=30.0):
        if payload is None:
            return {"queue_running": [[0, p] for p in state["running"]],
                    "queue_pending": [[0, p] for p in state["pending"]]}
        posts.append((endpoint, payload))
        if endpoint == "/interrupt":
            state["running"] = []
        return {}

    monkeypatch.setattr(comfy, "_request_json", request)
    outcome = comfy.cancel_prompt("http://box", "ours", timeout=1)
    assert outcome["ok"] is True
    assert posts == [("/interrupt", {})]
    assert state["pending"] == ["foreign-pending"]

    state["running"] = ["ours", "foreign-running"]
    posts.clear()
    outcome = comfy.cancel_prompt("http://box", "ours", timeout=1)
    assert outcome["ok"] is False and outcome["action"] == "unsafe_running"
    assert posts == []


def test_classify_gen_error_buckets():
    assert comfy.classify_gen_error("torch.OutOfMemoryError", "CUDA out of memory") == "oom"
    assert comfy.classify_gen_error("ValueError", "value 'x.safetensors' not in list") == "model_missing"
    assert comfy.classify_gen_error("invalid_prompt", "Cannot execute: node does not exist") == "node_missing"
    assert comfy.classify_gen_error("RuntimeError", "tensor shape mismatch") == "other"


def test_execution_error_detail_extracts_node_and_exception():
    item = {
        "status": {
            "status_str": "error",
            "messages": [
                ["execution_start", {"prompt_id": "p1"}],
                ["execution_error", {
                    "node_id": "12", "node_type": "KSampler",
                    "exception_type": "torch.OutOfMemoryError",
                    "exception_message": "CUDA out of memory. Tried to allocate 2.0 GiB",
                    "traceback": ["  File a.py, line 1\n", "  File b.py, line 2\n", "boom\n"],
                }],
            ],
        }
    }
    detail = comfy._execution_error_detail(item)
    assert detail["node_id"] == "12"
    assert detail["node_type"] == "KSampler"
    assert detail["error_class"] == "oom"
    assert "CUDA out of memory" in detail["exception_message"]
    assert detail["traceback_tail"].endswith("boom\n")


def test_execution_error_detail_none_without_error_message():
    assert comfy._execution_error_detail({"status": {"messages": [["execution_start", {}]]}}) is None
    assert comfy._execution_error_detail({}) is None


def test_submit_error_detail_parses_comfyui_validation_body():
    body = json.dumps({
        "error": {"type": "prompt_outputs_failed_validation", "message": "Prompt outputs failed validation"},
        "node_errors": {
            "4": {"class_type": "CheckpointLoaderSimple",
                  "errors": [{"message": "Value not in list: ckpt_name: 'nope.safetensors' not in list"}]},
        },
    })
    detail = comfy._submit_error_detail(body)
    assert detail["error_class"] == "model_missing"
    assert "node 4 (CheckpointLoaderSimple)" in detail["exception_message"]
    assert comfy._submit_error_detail("not json at all") is None
    assert comfy._submit_error_detail(json.dumps({"prompt_id": "fine"})) is None


def test_execution_cached_reuses_local_paths(tmp_path, monkeypatch):
    png = tmp_path / "hit.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 2000)
    pid = "cached-pid"

    def fake_curl(argv, **kwargs):
        if any(str(a).endswith("/prompt") for a in argv):
            return json.dumps({"prompt_id": pid})
        if any("/history/" in str(a) for a in argv):
            return json.dumps({
                pid: {
                    "outputs": {},
                    "status": {
                        "completed": True,
                        "status_str": "success",
                        "messages": [["execution_cached", {}]],
                    },
                }
            })
        raise AssertionError(argv)

    monkeypatch.setattr(comfy, "curl", fake_curl)
    monkeypatch.delenv("COMFY_WS_WAIT", raising=False)
    fetched = comfy.submit_and_fetch({"1": {}}, cached_paths=[str(png)])
    assert fetched is not None
    assert fetched[0][1]["source"] == "local_cached_output"
    assert fetched[0][0] == png.read_bytes()


def test_ws_completion_disabled_by_default(monkeypatch):
    monkeypatch.delenv("COMFY_WS_WAIT", raising=False)
    monkeypatch.delenv("GEN_EVENTS_OUT", raising=False)
    monkeypatch.delenv("LORAMAKE_ATTEMPT_ID", raising=False)
    assert comfy._ws_completion_enabled() is False
    monkeypatch.setenv("COMFY_WS_WAIT", "1")
    assert comfy._ws_completion_enabled() is True


def _enum_payload(names: list[str]) -> str:
    return json.dumps({"Loader": {"input": {"required": {"name": [names, {}]}}}})


@pytest.fixture()
def enum_cache(tmp_path, monkeypatch):
    """Isolated enum caches: fresh disk path, empty in-process cache and provenance set."""
    path = tmp_path / "enum_cache.json"
    monkeypatch.setenv("LORAMAKE_ENUM_CACHE", "1")
    monkeypatch.setenv("LORAMAKE_ENUM_CACHE_PATH", str(path))
    monkeypatch.delenv("LORAMAKE_ENUM_CACHE_TTL", raising=False)
    comfy._ENUM_CACHE.clear()
    comfy._ENUM_FROM_DISK.clear()
    yield path
    comfy._ENUM_CACHE.clear()
    comfy._ENUM_FROM_DISK.clear()


def test_object_info_enum_serves_from_disk_across_processes(enum_cache, monkeypatch):
    monkeypatch.setattr(comfy, "curl", lambda _args: _enum_payload(["a.safetensors"]))
    assert comfy.object_info_enum("Loader", "name") == {"a.safetensors"}
    assert enum_cache.is_file()

    # Simulate a fresh process: empty L1, and any live probe would now fail.
    comfy._ENUM_CACHE.clear()
    comfy._ENUM_FROM_DISK.clear()
    monkeypatch.setattr(comfy, "curl", lambda _args: (_ for _ in ()).throw(AssertionError("no live probe")))
    assert comfy.object_info_enum("Loader", "name") == {"a.safetensors"}
    key = (comfy.Z3_HOST.rstrip("/"), "Loader", "name")
    assert key in comfy._ENUM_FROM_DISK


def test_object_info_enum_disk_ttl_and_disable(enum_cache, monkeypatch):
    monkeypatch.setattr(comfy, "curl", lambda _args: _enum_payload(["a.safetensors"]))
    comfy.object_info_enum("Loader", "name")
    comfy._ENUM_CACHE.clear()

    monkeypatch.setenv("LORAMAKE_ENUM_CACHE_TTL", "0")
    monkeypatch.setattr(comfy, "curl", lambda _args: _enum_payload(["fresh.safetensors"]))
    assert comfy.object_info_enum("Loader", "name") == {"fresh.safetensors"}

    comfy._ENUM_CACHE.clear()
    monkeypatch.delenv("LORAMAKE_ENUM_CACHE_TTL", raising=False)
    monkeypatch.setenv("LORAMAKE_ENUM_CACHE", "0")
    monkeypatch.setattr(comfy, "curl", lambda _args: _enum_payload(["live.safetensors"]))
    assert comfy.object_info_enum("Loader", "name") == {"live.safetensors"}


def test_suggest_assets_prefers_basename_containment_then_fuzzy():
    known = {
        "Illustrious\\style\\HueSpark1llust.safetensors",
        "Anima\\style\\GothicNeonAnima.safetensors",
        "4x_foolhardy_Remacri.pth",
    }
    assert comfy.suggest_assets("GothicNeonAnima.safetensors", known) == [
        "Anima\\style\\GothicNeonAnima.safetensors"
    ]
    assert comfy.suggest_assets("foolhardy_Remacri.pth", known) == ["4x_foolhardy_Remacri.pth"]
    assert comfy.suggest_assets("HueSpark1lust.safetensors", known) == [
        "Illustrious\\style\\HueSpark1llust.safetensors"
    ]
    assert comfy.suggest_assets("totally-unrelated.bin", known) == []


def test_preflight_reverifies_cached_miss_live_before_rejecting(enum_cache, monkeypatch):
    box = comfy.Z3_HOST.rstrip("/")
    stale = {"schema": "loramake.object-info-cache/v1", "entries": {
        f"{box}|CheckpointLoaderSimple|ckpt_name": {
            "names": ["old.ckpt"], "observed_at": __import__("time").time(),
        },
        f"{box}|LoraLoader|lora_name": {
            "names": ["ok_lora.safetensors"], "observed_at": __import__("time").time(),
        },
    }}
    enum_cache.write_text(json.dumps(stale), encoding="utf-8")

    def live(argv):
        url = argv[-1]
        if "CheckpointLoaderSimple" in url:
            return json.dumps({"CheckpointLoaderSimple": {"input": {"required": {
                "ckpt_name": [["old.ckpt", "new.ckpt"], {}]}}}})
        if "LoraLoader" in url:
            return json.dumps({"LoraLoader": {"input": {"required": {
                "lora_name": [["ok_lora.safetensors"], {}]}}}})
        raise AssertionError(url)

    monkeypatch.setattr(comfy, "curl", live)
    # The cached view is missing new.ckpt; the live re-verify must rescue the run.
    comfy.preflight_assets("sdxl", "new.ckpt", ["ok_lora.safetensors"])


def test_preflight_rejection_names_closest_candidates(monkeypatch):
    enums = {
        ("CheckpointLoaderSimple", "ckpt_name"): {"Illustrious\\base model\\ilustmix_v111.safetensors"},
        ("LoraLoader", "lora_name"): {"ok_lora.safetensors"},
    }
    monkeypatch.setattr(comfy, "object_info_enum",
                        lambda node, field, host=None: enums[(node, field)])
    with pytest.raises(SystemExit) as exc:
        comfy.preflight_assets("sdxl", "ilustmix_v111.safetensors", [])
    message = str(exc.value)
    assert "closest:" in message
    assert "Illustrious\\base model\\ilustmix_v111.safetensors" in message
