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


def test_submit_and_fetch_cancels_remote_prompt_on_timeout(monkeypatch, capsys, tmp_path):
    pid = "timeout-pid"
    cancelled = []
    events = tmp_path / "attempt.jsonl"
    monkeypatch.setenv("GEN_EVENTS_OUT", str(events))
    monkeypatch.delenv("COMFY_WS_WAIT", raising=False)
    monkeypatch.delenv("LORAMAKE_ATTEMPT_ID", raising=False)

    def fake_curl(args, data=None, binary=False, **_kwargs):
        joined = " ".join(str(a) for a in args)
        if "/prompt" in joined:
            return json.dumps({"prompt_id": pid})
        if "/history/" in joined:
            return "{}"
        raise AssertionError(args)

    def fake_cancel(host, prompt_id, **kwargs):
        cancelled.append((host, prompt_id, kwargs))
        return {"ok": True, "action": "interrupt_unique_running", "prompt_id": prompt_id,
                "reason": ""}

    monkeypatch.setattr(comfy, "curl", fake_curl)
    monkeypatch.setattr(comfy, "cancel_prompt", fake_cancel)
    assert comfy.submit_and_fetch({"1": {}}, label="slow", timeout=0.0) is None
    assert cancelled[0][1] == pid
    err = capsys.readouterr().err
    assert "cancel pid=timeout-pid action=interrupt_unique_running ok=True" in err
    record = json.loads(events.read_text(encoding="utf-8").splitlines()[-1])
    assert record["event"] == "prompt_timeout"
    assert record["prompt_id"] == pid
    assert record["cancel_ok"] is True
    assert record["cancel_action"] == "interrupt_unique_running"


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


GRAPH_A = {"6": {"class_type": "CLIPTextEncode", "inputs": {"text": "red hair"}}}
GRAPH_B = {"6": {"class_type": "CLIPTextEncode", "inputs": {"text": "blue hair"}}}


def _history_entry(pid: str, graph: dict, *filenames: str) -> dict:
    """A completed /history entry; no filenames = every node was served from the execution cache
    and ComfyUI 0.27.1 listed nothing (prompt submitted without a client_id)."""
    images = [{"filename": name, "subfolder": "", "type": "output"} for name in filenames]
    return {"prompt": [0, pid, graph, {}, ["9"]],
            "outputs": {"9": {"images": images}} if images else {},
            "status": {"completed": True, "status_str": "success",
                       "messages": [["execution_cached", {"nodes": ["6", "9"], "prompt_id": pid}]]}}


def _cached_rerun_box(monkeypatch, listing: dict, files: dict[str, bytes]) -> list[str]:
    """Serve /prompt, the rerun's own /history entry, the /history listing and /view; return the
    filenames /view was asked for. A file missing from ``files`` answers like a 404 (empty body)."""
    viewed: list[str] = []

    def fake_curl(argv, **_kwargs):
        url = next(str(a) for a in argv if str(a).startswith("http"))
        if url.endswith("/prompt"):
            return json.dumps({"prompt_id": "rerun"})
        if url.endswith("/history/rerun"):
            return json.dumps({"rerun": listing["rerun"]})
        if url.endswith(f"/history?max_items={comfy._CACHED_LOOKBACK}"):
            return json.dumps(listing)
        if url.endswith("/view"):
            viewed.append(next(str(a)[len("filename="):] for a in argv if str(a).startswith("filename=")))
            return files.get(viewed[-1], b"")
        raise AssertionError(argv)

    monkeypatch.setattr(comfy, "curl", fake_curl)
    monkeypatch.delenv("COMFY_WS_WAIT", raising=False)
    monkeypatch.delenv("LORAMAKE_FETCH_PREVIEW", raising=False)
    return viewed


def test_execution_cached_fetches_the_earlier_output_of_the_same_graph(monkeypatch):
    # Rerun of B, fully cached on BOXB. A shares B's seed and file stem, and its render is newer;
    # only the entry whose graph equals B's may supply the pixels.
    png_a = b"\x89PNG\r\n\x1a\n" + b"A" * 2000 + b"\x00\x00\x00\x00IEND\xaeB`\x82"
    png_b = b"\x89PNG\r\n\x1a\n" + b"B" * 2000 + b"\x00\x00\x00\x00IEND\xaeB`\x82"
    listing = {
        "run_b": _history_entry("run_b", GRAPH_B, "boxa_quality_00512_.png"),
        "run_a": _history_entry("run_a", GRAPH_A, "boxa_quality_00513_.png"),
        "cached_again": _history_entry("cached_again", GRAPH_B),
        "rerun": _history_entry("rerun", GRAPH_B),
    }
    viewed = _cached_rerun_box(monkeypatch, listing, {"boxa_quality_00512_.png": png_b,
                                                      "boxa_quality_00513_.png": png_a})
    events: list[dict] = []

    fetched = comfy.submit_and_fetch(GRAPH_B, first_only=False, event_sink=events.append)

    assert viewed == ["boxa_quality_00512_.png"]
    assert [raw for raw, _item in fetched] == [png_b]
    assert events[-1]["event"] == "prompt_completed" and events[-1]["cached"] is True


@pytest.mark.parametrize("earlier", [
    pytest.param({"run_a": _history_entry("run_a", GRAPH_A, "boxa_quality_00513_.png")}, id="no-earlier-run"),
    pytest.param({"run_b": _history_entry("run_b", GRAPH_B, "boxa_quality_00512_.png")}, id="file-cleaned"),
])
def test_execution_cached_fails_loudly_without_the_earlier_output(monkeypatch, capsys, earlier):
    listing = {**earlier, "rerun": _history_entry("rerun", GRAPH_B)}
    _cached_rerun_box(monkeypatch, listing, {})  # every /view answers like a 404
    events: list[dict] = []

    assert comfy.submit_and_fetch(GRAPH_B, event_sink=events.append) is None

    assert "ERROR [cached_without_output]" in capsys.readouterr().out
    assert events[-1]["event"] == "prompt_error"
    assert events[-1]["error_class"] == "cached_without_output"
    assert "re-run with a changed input" in events[-1]["exception_message"]


def _fake_box(pid: str, view_calls: list, body: bytes):
    def fake_curl(argv, **kwargs):
        if any(str(a).endswith("/prompt") for a in argv):
            return json.dumps({"prompt_id": pid})
        if any("/history/" in str(a) for a in argv):
            image = {"filename": "q_00001_.png", "subfolder": "", "type": "output"}
            return json.dumps({pid: {"outputs": {"9": {"images": [image]}}}})
        if any(str(a).endswith("/view") for a in argv):
            view_calls.append([str(a) for a in argv])
            return body
        raise AssertionError(argv)
    return fake_curl


def _webp_bytes() -> bytes:
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.effect_noise((64, 64), 60).convert("RGB").save(buf, format="WEBP", quality=80)
    return buf.getvalue()


def test_preview_fetch_pulls_webp_and_hands_back_a_png_with_its_origin(monkeypatch):
    calls: list = []
    monkeypatch.setattr(comfy, "curl", _fake_box("pv-pid", calls, _webp_bytes()))
    monkeypatch.delenv("COMFY_WS_WAIT", raising=False)
    monkeypatch.setenv("LORAMAKE_FETCH_PREVIEW", "boxa")

    fetched = comfy.submit_and_fetch({"1": {}}, first_only=False, allow_preview=True)

    assert any("preview=webp;80" in a for a in calls[0])
    raw, item = fetched[0]
    assert comfy.looks_like_png(raw)
    assert item["fetched_preview"] == "webp;80" and item["box"] == "boxa"
    assert item["filename"] == "q_00001_.png"


@pytest.mark.parametrize("flag", ["", "0", "off"])
def test_preview_fetch_off_keeps_the_full_png(monkeypatch, flag):
    calls: list = []
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 2000 + b"\x00\x00\x00\x00IEND\xaeB`\x82"
    monkeypatch.setattr(comfy, "curl", _fake_box("png-pid", calls, png))
    monkeypatch.delenv("COMFY_WS_WAIT", raising=False)
    monkeypatch.setenv("LORAMAKE_FETCH_PREVIEW", flag)

    fetched = comfy.submit_and_fetch({"1": {}}, first_only=False, allow_preview=True)

    assert not any("preview=" in a for a in calls[0])
    assert fetched[0][0] == png and "fetched_preview" not in fetched[0][1]


def test_callers_that_do_not_record_the_original_always_get_the_full_png(monkeypatch):
    calls: list = []
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 2000 + b"\x00\x00\x00\x00IEND\xaeB`\x82"
    monkeypatch.setattr(comfy, "curl", _fake_box("full-pid", calls, png))
    monkeypatch.delenv("COMFY_WS_WAIT", raising=False)
    monkeypatch.setenv("LORAMAKE_FETCH_PREVIEW", "boxa")

    fetched = comfy.submit_and_fetch({"1": {}}, first_only=False)

    assert not any("preview=" in a for a in calls[0])
    assert fetched[0][0] == png and "fetched_preview" not in fetched[0][1]


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
    key = (comfy.BOXA_HOST.rstrip("/"), "Loader", "name")
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
    box = comfy.BOXA_HOST.rstrip("/")
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


@pytest.mark.parametrize("args, host, expected", [
    (["http://100.64.0.254:8388/object_info/KSampler"], None, "100.64.0.254"),
    (["-X", "POST", "http://boxb.invalid:8388/prompt", "-d", "@-"], None, "boxb.invalid"),
    (["-G", "http://boxb.invalid:8388/view", "--data-urlencode", "filename=a.png"], None, "boxb.invalid"),
    (["-o", "/dev/null"], None, comfy.NP),
    (["http://boxa.invalid:8188/history/x"], "http://box.invalid:8388", "box.invalid"),
])
def test_curl_bypasses_the_proxy_for_the_box_it_talks_to(monkeypatch, args, host, expected):
    """Every transport call (submit / history / view / object_info) must bypass the ambient proxy
    for ITS box, not only BOXA's: a BOXB run under HTTPS_PROXY used to lose every call after upload."""
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:3128")
    monkeypatch.setattr(comfy, "_CF_TOKEN_FILE", None)
    seen = []

    def run(argv, **kwargs):
        seen.append(argv)
        return SimpleNamespace(returncode=0, stdout="{}")

    monkeypatch.setattr(comfy.subprocess, "run", run)
    comfy.curl(args, host=host)
    argv = seen[0]
    assert argv[argv.index("--noproxy") + 1] == expected


def test_node_present_and_object_info_probe_bypass_the_proxy_for_their_host(monkeypatch):
    monkeypatch.setattr(comfy, "_CF_TOKEN_FILE", None)
    seen = []

    def run(argv, **kwargs):
        seen.append(argv)
        return SimpleNamespace(returncode=0, stdout=json.dumps({"KSampler": {"input": {"required": {}}}}))

    monkeypatch.setattr(comfy.subprocess, "run", run)
    assert comfy.node_present("KSampler", host="http://100.64.0.254:8388") is True
    assert comfy.object_info_enum("KSampler", "sampler_name", host="http://100.64.0.254:8388", refresh=True) == set()
    for argv in seen:
        assert argv[argv.index("--noproxy") + 1] == "100.64.0.254"
