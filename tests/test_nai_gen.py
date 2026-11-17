"""NovelAI txt2img client: model resolution, centers, payload shape, offline CLI.

All tests are offline; the only network path (generate) is exercised with a
stubbed urlopen so error-hint mapping stays pinned without a token.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from scripts import nai_gen

ROOT = Path(__file__).resolve().parent.parent
NAI_GEN = ROOT / "scripts" / "nai_gen.py"


def _args(**overrides):
    """Minimal argparse namespace stand-in for build_payload."""
    base = dict(
        width=832,
        height=1216,
        scale=5.0,
        sampler="k_euler_ancestral",
        steps=28,
        seed=1234,
        cfg_rescale=0.0,
        noise_schedule="karras",
        uc_preset=3,
        quality_toggle=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# --- model resolution -----------------------------------------------------


def test_resolve_model_alias():
    assert nai_gen.resolve_model("v4.5-full") == "nai-diffusion-4-5-full"
    assert nai_gen.resolve_model("v5-curated") == "nai-diffusion-5-curated"


def test_resolve_model_raw_id_passthrough():
    # Unknown-but-well-formed ids pass so new models work before the alias
    # table catches up.
    assert nai_gen.resolve_model("nai-diffusion-9-future") == "nai-diffusion-9-future"


def test_resolve_model_rejects_garbage():
    with pytest.raises(SystemExit, match="unknown model"):
        nai_gen.resolve_model("sdxl-turbo")


@pytest.mark.parametrize(
    "model_id,expected",
    [
        ("nai-diffusion-4-full", True),
        ("nai-diffusion-5-full", True),
        ("nai-diffusion-3", False),
        ("nai-diffusion-furry-3", False),
    ],
)
def test_is_v4_family(model_id, expected):
    assert nai_gen.is_v4_family(model_id) is expected


# --- centers ----------------------------------------------------------------


def test_parse_center_ok():
    assert nai_gen.parse_center("0.3,0.5") == {"x": 0.3, "y": 0.5}


@pytest.mark.parametrize("text", ["0.3", "a,b", "0.1,0.2,0.3", ""])
def test_parse_center_rejects_bad_shape(text):
    with pytest.raises(SystemExit, match="bad --char-pos"):
        nai_gen.parse_center(text)


@pytest.mark.parametrize("text", ["-0.1,0.5", "1.1,0.5", "0.5,2"])
def test_parse_center_rejects_out_of_range(text):
    with pytest.raises(SystemExit, match="out of range"):
        nai_gen.parse_center(text)


def test_validate_center_rejects_missing_keys():
    with pytest.raises(SystemExit, match="bad center"):
        nai_gen.validate_center({"x": 0.3}, "job character 1")


# --- build_characters -------------------------------------------------------


def test_build_characters_pairs_positions_by_order():
    args = SimpleNamespace(
        char=["girl, silver hair", "girl, black hair"],
        char_pos=["0.3,0.5", "0.7,0.5"],
    )
    chars = nai_gen.build_characters(args)
    assert chars == [
        {"prompt": "girl, silver hair", "center": {"x": 0.3, "y": 0.5}},
        {"prompt": "girl, black hair", "center": {"x": 0.7, "y": 0.5}},
    ]


def test_build_characters_allows_fewer_positions():
    args = SimpleNamespace(char=["a", "b"], char_pos=["0.3,0.5"])
    chars = nai_gen.build_characters(args)
    assert "center" in chars[0] and "center" not in chars[1]


# --- load_job ----------------------------------------------------------------


def test_load_job_direct(tmp_path):
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"base": "1girl, cafe"}), encoding="utf-8")
    assert nai_gen.load_job(job)["base"] == "1girl, cafe"


def test_load_job_unwraps_single_batch_entry(tmp_path):
    job = tmp_path / "batch.json"
    job.write_text(json.dumps({"B1": {"base": "x"}, "note": "meta"}), encoding="utf-8")
    assert nai_gen.load_job(job) == {"base": "x"}


def test_load_job_rejects_multi_entry_batch(tmp_path):
    job = tmp_path / "batch.json"
    job.write_text(json.dumps({"B1": {"base": "x"}, "B2": {"base": "y"}}), encoding="utf-8")
    with pytest.raises(SystemExit, match="multiple entries"):
        nai_gen.load_job(job)


def test_load_job_rejects_missing_base(tmp_path):
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"prompt": "no base key"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="missing keys"):
        nai_gen.load_job(job)


# --- build_payload ------------------------------------------------------------


def test_payload_v4_with_centers_uses_coords():
    characters = [
        {"prompt": "girl a", "center": {"x": 0.3, "y": 0.5}},
        {"prompt": "girl b", "center": {"x": 0.7, "y": 0.5}},
    ]
    payload = nai_gen.build_payload(
        "nai-diffusion-4-5-full", "2girls, rooftop", characters, "lowres", _args()
    )
    params = payload["parameters"]
    v4 = params["v4_prompt"]
    assert v4["use_coords"] is True
    assert v4["caption"]["base_caption"] == "2girls, rooftop"
    assert [c["centers"] for c in v4["caption"]["char_captions"]] == [
        [{"x": 0.3, "y": 0.5}],
        [{"x": 0.7, "y": 0.5}],
    ]
    # Negative captions always carry empty text at the default center.
    neg = params["v4_negative_prompt"]["caption"]
    assert neg["base_caption"] == "lowres"
    assert neg["char_captions"] == [
        {"char_caption": "", "centers": [{"x": 0.5, "y": 0.5}]},
        {"char_caption": "", "centers": [{"x": 0.5, "y": 0.5}]},
    ]


def test_payload_v4_without_centers_defaults_half():
    characters = [{"prompt": "girl a"}]
    payload = nai_gen.build_payload(
        "nai-diffusion-4-5-full", "1girl", characters, "neg", _args()
    )
    v4 = payload["parameters"]["v4_prompt"]
    assert v4["use_coords"] is False
    assert v4["caption"]["char_captions"][0]["centers"] == [{"x": 0.5, "y": 0.5}]


def test_payload_mixed_placement_rejected():
    characters = [
        {"prompt": "girl a", "center": {"x": 0.3, "y": 0.5}},
        {"prompt": "girl b"},
    ]
    with pytest.raises(SystemExit, match="every character a center or none"):
        nai_gen.build_payload(
            "nai-diffusion-4-5-full", "2girls", characters, "neg", _args()
        )


def test_payload_v3_has_no_v4_prompt():
    payload = nai_gen.build_payload(
        "nai-diffusion-3", "1girl", [{"prompt": "girl"}], "neg", _args()
    )
    assert "v4_prompt" not in payload["parameters"]
    assert payload["model"] == "nai-diffusion-3"


def test_payload_core_params_echoed():
    payload = nai_gen.build_payload(
        "nai-diffusion-5-full", "1girl", [], "neg", _args(seed=42, steps=20)
    )
    params = payload["parameters"]
    assert params["seed"] == 42
    assert params["steps"] == 20
    assert params["negative_prompt"] == "neg"
    assert payload["action"] == "generate"


# --- save / manifest ----------------------------------------------------------


def _zip_bytes(names=("a.png", "b.png")):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name in names:
            zf.writestr(name, b"png-bytes-" + name.encode())
    return buf.getvalue()


def test_save_response_unzips_multi_image(tmp_path):
    saved = nai_gen.save_response(_zip_bytes(), tmp_path, "20260827_120000")
    assert len(saved) == 2
    assert all(p.read_bytes().startswith(b"png-bytes-") for p in saved)


def test_save_response_raw_body(tmp_path):
    saved = nai_gen.save_response(b"rawpng", tmp_path, "20260827_120000")
    assert saved == [tmp_path / "nai_20260827_120000.png"]


def test_unique_path_increments_on_clash(tmp_path):
    first = tmp_path / "nai_x_0.png"
    first.write_bytes(b"old")
    path = nai_gen.unique_path(tmp_path, "nai_x_0")
    assert path == tmp_path / "nai_x_0-1.png"


def test_append_manifest_omits_negative_from_parameters(tmp_path):
    payload = nai_gen.build_payload("nai-diffusion-3", "1girl", [], "the-negative", _args())
    saved = [tmp_path / "a.png"]
    nai_gen.append_manifest(tmp_path, payload, saved)
    record = json.loads((tmp_path / "manifest.jsonl").read_text(encoding="utf-8"))
    assert record["negative"] == "the-negative"
    assert "negative_prompt" not in record["parameters"]
    assert record["files"] == [str(tmp_path / "a.png")]



def test_apply_img2img_sets_action_and_params():
    payload = nai_gen.build_payload("nai-diffusion-5-full", "1girl", [], "neg", _args())
    out = nai_gen.apply_img2img(payload, "b64", 0.4, 0.0)
    assert out["action"] == "img2img"
    assert out["parameters"]["image"] == "b64"
    assert out["parameters"]["strength"] == 0.4
    assert out["parameters"]["noise"] == 0.0
    assert out["parameters"]["extra_noise_seed"] == 1234
    assert payload["action"] == "generate"  # original untouched
    assert "image" not in payload["parameters"]


def test_append_manifest_drops_img2img_base64_keeps_source(tmp_path):
    payload = nai_gen.build_payload("nai-diffusion-5-full", "1girl", [], "neg", _args())
    payload = nai_gen.apply_img2img(payload, "A" * 5000, 0.4, 0.0)
    nai_gen.append_manifest(tmp_path, payload, [], image_source="draft.png")
    record = json.loads((tmp_path / "manifest.jsonl").read_text(encoding="utf-8"))
    assert "image" not in record["parameters"]
    assert record["parameters"]["strength"] == 0.4
    assert record["image_source"].endswith("draft.png")


# --- generate error hints -------------------------------------------------------


def _http_error(code):
    return HTTPError(
        url=nai_gen.API_URL,
        code=code,
        msg="err",
        hdrs=None,  # type: ignore[arg-type]
        fp=io.BytesIO(b'{"detail":"boom"}'),
    )


@pytest.mark.parametrize(
    "code,hint",
    [(401, "token invalid"), (402, "insufficient Anlas"), (429, "rate limited")],
)
def test_generate_maps_http_errors_to_hints(monkeypatch, tmp_path, code, hint):
    def _raise(*_args, **_kwargs):
        raise _http_error(code)

    monkeypatch.setattr(nai_gen.urllib.request, "urlopen", _raise)
    payload = nai_gen.build_payload("nai-diffusion-3", "1girl", [], "neg", _args())
    with pytest.raises(SystemExit, match=hint):
        nai_gen.generate(payload, "token", tmp_path)


# --- CLI dry-run (offline end-to-end) --------------------------------------------


def test_cli_dry_run_prints_payload_without_token():
    proc = subprocess.run(
        [sys.executable, str(NAI_GEN), "--prompt", "1girl, cafe", "--dry-run"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["model"] == "nai-diffusion-4-5-full"  # default alias
    assert payload["input"] == "1girl, cafe"
    assert payload["parameters"]["width"] == 832  # portrait default


def test_cli_dry_run_img2img_elides_source(tmp_path):
    src = tmp_path / "draft.png"
    src.write_bytes(b"not decoded in dry-run")
    proc = subprocess.run(
        [sys.executable, str(NAI_GEN), "--prompt", "1girl", "--model", "v5-full",
         "--image", str(src), "--strength", "0.4", "--width", "1280",
         "--height", "1856", "--anlas-ledger", "none", "--dry-run"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["action"] == "img2img"
    assert payload["parameters"]["image"] == f"<unencoded image: {src}>"
    assert payload["parameters"]["width"] == 1280


def test_cli_img2img_rejects_non_64_size(tmp_path):
    src = tmp_path / "draft.png"
    src.write_bytes(b"x")
    proc = subprocess.run(
        [sys.executable, str(NAI_GEN), "--prompt", "1girl", "--image", str(src),
         "--width", "1000", "--height", "1856", "--dry-run"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode != 0
    assert "multiples of 64" in proc.stderr


def test_cli_requires_prompt_or_job():
    proc = subprocess.run(
        [sys.executable, str(NAI_GEN), "--dry-run"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode != 0
    assert "--prompt or --job is required" in proc.stderr


# --- vibe transfer ---------------------------------------------------------


class _FakeResponse:
    def __init__(self, body):
        self.body = body

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def _vibe_args(**overrides):
    base = dict(
        vibe=None,
        vibe_strength=[nai_gen.DEFAULT_VIBE_STRENGTH],
        vibe_info=[nai_gen.DEFAULT_VIBE_INFO],
        vibe_cache="unused",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _reference(tmp_path, name="style_01.jpg", data=b"jpeg-bytes"):
    path = tmp_path / name
    path.write_bytes(data)
    return path


def test_vibe_cache_path_keys_on_model_and_ie(tmp_path):
    a = nai_gen.vibe_cache_path(tmp_path, b"img", 1.0, "nai-diffusion-4-5-full")
    b = nai_gen.vibe_cache_path(tmp_path, b"img", 1.0, "nai-diffusion-5-full")
    c = nai_gen.vibe_cache_path(tmp_path, b"img", 0.5, "nai-diffusion-4-5-full")
    assert a != b != c and a != c
    assert a.name.endswith("_ie1.00_nai-diffusion-4-5-full.bin")


def test_vibe_cache_path_sanitizes_model_component(tmp_path):
    # resolve_model passes raw nai-diffusion-* ids through, so a separator in
    # the id must not move the cache file outside the cache dir.
    hostile = nai_gen.vibe_cache_path(
        tmp_path, b"img", 1.0, "nai-diffusion-4/../../evil"
    )
    assert hostile.parent == tmp_path
    assert hostile.resolve().is_relative_to(tmp_path.resolve())
    assert ".." not in hostile.resolve().parts


def test_vibe_cache_path_distinct_after_sanitizing(tmp_path):
    a = nai_gen.vibe_cache_path(tmp_path, b"img", 1.0, "nai-diffusion-9:x")
    b = nai_gen.vibe_cache_path(tmp_path, b"img", 1.0, "nai-diffusion-9-x")
    assert a.parent == b.parent == tmp_path
    assert a != b


def test_elide_bulk_keeps_offline_placeholders():
    payload = {
        "model": "nai-diffusion-4-5-full",
        "input": "p",
        "parameters": {
            "negative_prompt": "n",
            "reference_image_multiple": [
                "A" * 100,
                "<unencoded vibe: /refs/style.jpg>",
            ],
        },
    }
    out = nai_gen.elide_bulk(payload)
    assert out["parameters"]["reference_image_multiple"] == [
        "<base64 100 chars>",
        "<unencoded vibe: /refs/style.jpg>",
    ]
    # The original payload must stay untouched for the real request.
    assert payload["parameters"]["reference_image_multiple"][0] == "A" * 100


def test_resolve_vibe_encodes_and_caches(monkeypatch, tmp_path):
    sent = {}

    def _urlopen(request, timeout=None):
        sent["url"] = request.full_url
        sent["body"] = json.loads(request.data)
        return _FakeResponse(b"vibe-block")

    monkeypatch.setattr(nai_gen.urllib.request, "urlopen", _urlopen)
    ref = _reference(tmp_path)
    cache = tmp_path / "vibes"
    record = nai_gen.resolve_vibe(ref, 1.0, "nai-diffusion-4-5-full", "tok", cache)

    assert sent["url"] == nai_gen.ENCODE_VIBE_URL
    assert sent["body"]["information_extracted"] == 1.0
    assert sent["body"]["model"] == "nai-diffusion-4-5-full"
    # Content-derived seeds are what make the cache correct; never pin them.
    assert "focus_seed" not in sent["body"]
    assert "info_extract_seed" not in sent["body"]
    assert record["block"] == b"vibe-block"
    assert record["cached"] is False
    assert Path(record["cache"]).read_bytes() == b"vibe-block"


def test_resolve_vibe_reuses_cache_without_network(monkeypatch, tmp_path):
    def _boom(*_args, **_kwargs):
        raise AssertionError("cache hit must not touch the network")

    ref = _reference(tmp_path)
    cache = tmp_path / "vibes"
    cache.mkdir()
    path = nai_gen.vibe_cache_path(
        cache, ref.read_bytes(), 1.0, "nai-diffusion-4-5-full"
    )
    path.write_bytes(b"already-here")

    monkeypatch.setattr(nai_gen.urllib.request, "urlopen", _boom)
    record = nai_gen.resolve_vibe(ref, 1.0, "nai-diffusion-4-5-full", "tok", cache)
    assert record["block"] == b"already-here"
    assert record["cached"] is True


def test_resolve_vibe_offline_leaves_block_unencoded(monkeypatch, tmp_path):
    def _boom(*_args, **_kwargs):
        raise AssertionError("offline resolve must not touch the network")

    monkeypatch.setattr(nai_gen.urllib.request, "urlopen", _boom)
    record = nai_gen.resolve_vibe(
        _reference(tmp_path), 1.0, "nai-diffusion-4-5-full", None,
        tmp_path / "vibes", offline=True,
    )
    assert record["block"] is None
    assert record["cached"] is False


def test_resolve_vibe_rejects_missing_file(tmp_path):
    with pytest.raises(SystemExit, match="vibe reference not found"):
        nai_gen.resolve_vibe(
            tmp_path / "nope.jpg", 1.0, "nai-diffusion-4-5-full", "tok", tmp_path
        )


@pytest.mark.parametrize("info", [-0.1, 1.5])
def test_resolve_vibe_rejects_info_out_of_range(tmp_path, info):
    with pytest.raises(SystemExit, match="out of range"):
        nai_gen.resolve_vibe(
            _reference(tmp_path), info, "nai-diffusion-4-5-full", "tok", tmp_path
        )


def test_spread_vibe_values_broadcasts_single():
    assert nai_gen.spread_vibe_values([0.6], 3, "--vibe-strength") == [0.6, 0.6, 0.6]


def test_spread_vibe_values_rejects_partial_list():
    with pytest.raises(SystemExit, match="one value or one per --vibe"):
        nai_gen.spread_vibe_values([0.6, 0.4], 3, "--vibe-strength")


def test_build_vibes_warns_when_strengths_exceed_advice(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        nai_gen, "resolve_vibe",
        lambda path, info, model, token, cache, offline=False: {
            "source": str(path), "cache": "c", "information_extracted": info,
            "model": model, "block": b"x", "cached": True,
        },
    )
    args = _vibe_args(vibe=["a.jpg", "b.jpg"], vibe_strength=[0.8])
    vibes = nai_gen.build_vibes(args, "nai-diffusion-4-5-full", "tok")
    assert [v["strength"] for v in vibes] == [0.8, 0.8]
    assert "strengths sum to 1.60" in capsys.readouterr().err


def test_vibe_parameters_emits_parallel_arrays():
    vibes = [
        {"block": b"aa", "strength": 0.6, "information_extracted": 1.0, "source": "a"},
        {"block": b"bb", "strength": 0.3, "information_extracted": 0.5, "source": "b"},
    ]
    params = nai_gen.vibe_parameters(vibes)
    assert params["reference_strength_multiple"] == [0.6, 0.3]
    assert params["reference_information_extracted_multiple"] == [1.0, 0.5]
    assert params["reference_image_multiple"] == ["YWE=", "YmI="]


def test_vibe_parameters_empty_adds_nothing():
    assert nai_gen.vibe_parameters([]) == {}


def test_build_payload_injects_vibe_arrays():
    vibes = [
        {"block": b"aa", "strength": 0.6, "information_extracted": 1.0, "source": "a"}
    ]
    payload = nai_gen.build_payload(
        "nai-diffusion-4-5-full", "1girl", [], "neg", _args(), vibes
    )
    params = payload["parameters"]
    assert params["reference_image_multiple"] == ["YWE="]
    assert params["reference_information_extracted_multiple"] == [1.0]


def test_manifest_records_vibe_provenance_not_base64(tmp_path):
    vibes = [{
        "block": b"aa", "strength": 0.6, "information_extracted": 1.0,
        "source": "docs/style_01.jpg", "cache": "outputs/nai/vibes/abc_ie1.00_m.bin",
        "model": "nai-diffusion-4-5-full", "cached": False,
    }]
    payload = nai_gen.build_payload(
        "nai-diffusion-4-5-full", "1girl", [], "neg", _args(), vibes
    )
    nai_gen.append_manifest(tmp_path, payload, [tmp_path / "a.png"], vibes)
    record = json.loads((tmp_path / "manifest.jsonl").read_text(encoding="utf-8"))
    assert "reference_image_multiple" not in record["parameters"]
    assert record["parameters"]["reference_strength_multiple"] == [0.6]
    assert record["vibes"] == [{
        "source": "docs/style_01.jpg",
        "cache": "outputs/nai/vibes/abc_ie1.00_m.bin",
        "strength": 0.6,
        "information_extracted": 1.0,
        "model": "nai-diffusion-4-5-full",
    }]


def test_elide_bulk_replaces_base64_with_size_marker():
    payload = {"parameters": {"reference_image_multiple": ["YWE=", "YmJi"], "seed": 1}}
    out = nai_gen.elide_bulk(payload)
    assert out["parameters"]["reference_image_multiple"] == [
        "<base64 4 chars>", "<base64 4 chars>"
    ]
    assert out["parameters"]["seed"] == 1
    # The original payload keeps the real bytes for the request.
    assert payload["parameters"]["reference_image_multiple"][0] == "YWE="


def test_cli_dry_run_with_uncached_vibe_stays_offline(tmp_path):
    ref = _reference(tmp_path)
    proc = subprocess.run(
        [
            sys.executable, str(NAI_GEN), "--prompt", "1girl", "--dry-run",
            "--vibe", str(ref), "--vibe-cache", str(tmp_path / "vibes"),
        ],
        capture_output=True, text=True, timeout=30,
        env={**os.environ, "NAI_ACCESS_TOKEN": ""},
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    params = payload["parameters"]
    # One placeholder entry, not a real encode; the placeholder passes through
    # unmeasured (a size marker here would report the placeholder's length).
    assert len(params["reference_image_multiple"]) == 1
    assert params["reference_image_multiple"][0].startswith("<unencoded vibe:")
    assert params["reference_strength_multiple"] == [nai_gen.DEFAULT_VIBE_STRENGTH]
    assert not (tmp_path / "vibes").exists()


# --- Precise Reference ------------------------------------------------------


def _png(path, size=(400, 300)):
    from PIL import Image

    Image.new("RGB", size, (200, 30, 30)).save(path)
    return path


def test_reference_parameters_emits_five_parallel_arrays():
    refs = [
        {"image": b"aa", "type": "character", "strength": 1.0, "fidelity": 0.5, "source": "a"},
        {"image": b"bb", "type": "style", "strength": 0.6, "fidelity": 1.0, "source": "b"},
    ]
    params = nai_gen.reference_parameters(refs)
    assert params["director_reference_images"] == ["YWE=", "YmI="]
    assert params["director_reference_strength_values"] == [1.0, 0.6]
    # Fidelity travels inverted as the secondary strength.
    assert params["director_reference_secondary_strength_values"] == [0.5, 0.0]
    assert params["director_reference_information_extracted"] == [1.0, 1.0]
    assert params["director_reference_descriptions"] == [
        {"caption": {"base_caption": "character", "char_captions": []}, "legacy_uc": False},
        {"caption": {"base_caption": "style", "char_captions": []}, "legacy_uc": False},
    ]


def test_reference_parameters_empty_adds_nothing():
    assert nai_gen.reference_parameters([]) == {}


def test_estimate_cost_adds_flat_charge_per_reference():
    free = nai_gen.estimate_generation_cost("nai-diffusion-4-5-full", 832, 1216, 28)
    assert free == 0
    assert (
        nai_gen.estimate_generation_cost("nai-diffusion-4-5-full", 832, 1216, 28, 0, 2)
        == 2 * nai_gen.DIRECTOR_REFERENCE_ANLAS
    )


# --- Anlas price vs the NovelAI website -----------------------------------------
# Generate button readings on novelai.net/image (Opus account, 2026-09-30); no
# image was generated. Readings marked "user" were taken by the user the same day.

MEASURED_ONE_IMAGE = [
    # model, width, height, steps, Anlas on the button
    ("nai-diffusion-5-curated", 1536, 1024, 23, 39),  # user; Large landscape
    ("nai-diffusion-5-curated", 1472, 1472, 23, 54),  # user; Large square
    ("nai-diffusion-5-curated", 1088, 1920, 23, 51),  # user; Wallpaper portrait
    ("nai-diffusion-5-curated", 1920, 1088, 23, 51),  # user; Wallpaper landscape
    ("nai-diffusion-5-curated", 832, 1216, 29, 30),
    ("nai-diffusion-5-curated", 832, 1216, 40, 41),
    ("nai-diffusion-5-curated", 832, 1216, 50, 50),
    ("nai-diffusion-5-curated", 1536, 1024, 28, 45),
    ("nai-diffusion-5-curated", 1536, 1024, 29, 47),
    ("nai-diffusion-5-curated", 512, 768, 50, 20),
    ("nai-diffusion-5-curated", 1088, 1024, 23, 29),
    ("nai-diffusion-5-curated", 256, 256, 50, 5),
    ("nai-diffusion-5-curated", 64, 64, 29, 2),
    ("nai-diffusion-5-full", 1536, 1024, 23, 39),
    ("nai-diffusion-4-5-curated", 1536, 1024, 23, 26),
    ("nai-diffusion-4-5-curated", 1536, 1024, 50, 50),
    ("nai-diffusion-4-5-full", 1536, 1024, 23, 26),
    ("nai-diffusion-4-curated-preview", 1536, 1024, 23, 26),
    ("nai-diffusion-4-full", 1536, 1024, 23, 26),
    ("nai-diffusion-3", 1536, 1024, 23, 26),
    ("nai-diffusion-furry-3", 1536, 1024, 23, 26),
]

MEASURED_FREE = [
    # V5 Curated, one image: width, height, steps
    (832, 1216, 28),
    (832, 1216, 23),  # user; Normal sizes
    (1216, 832, 23),  # user
    (1024, 1024, 23),  # user
    (512, 768, 23),  # user; Small sizes
    (768, 512, 23),  # user
    (640, 640, 23),  # user
    (512, 768, 28),
    (960, 1088, 23),  # not a named size, still under 1024*1024 pixels
]

MEASURED_MULTI_IMAGE = [
    # V5 Curated, 832x1216: steps, images, Anlas on the button
    (23, 2, 26),  # free canvas: the site waives one image
    (23, 3, 52),
    (23, 4, 78),
    (29, 2, 60),  # paid canvas: every image is billed
]


@pytest.mark.parametrize("model_id,width,height,steps,site", MEASURED_ONE_IMAGE)
def test_estimate_matches_the_site_for_one_image(model_id, width, height, steps, site):
    assert nai_gen.estimate_generation_cost(model_id, width, height, steps) == site


@pytest.mark.parametrize("width,height,steps", MEASURED_FREE)
def test_measured_free_canvases_are_free(width, height, steps):
    model_id = "nai-diffusion-5-curated"
    assert nai_gen.is_opus_free_eligible(model_id, width, height, steps) is True
    assert nai_gen.estimate_generation_cost(model_id, width, height, steps) == 0


@pytest.mark.parametrize("steps,n_samples,site", MEASURED_MULTI_IMAGE)
def test_estimate_never_undercounts_multi_image_requests(steps, n_samples, site):
    estimate = nai_gen.estimate_generation_cost(
        "nai-diffusion-5-curated", 832, 1216, steps, n_samples=n_samples
    )
    assert estimate >= site


@pytest.mark.parametrize(
    "overrides",
    [
        {"width": 1088, "height": 1024},  # one row of pixels over 1024*1024
        {"steps": 29},
        {"n_samples": 2},
        {"base_image": True},
    ],
)
def test_opus_free_tier_boundaries(overrides):
    request = dict(width=832, height=1216, steps=28)
    request.update(overrides)
    assert nai_gen.is_opus_free_eligible("nai-diffusion-5-full", **request) is False
    assert nai_gen.estimate_generation_cost("nai-diffusion-5-full", **request) > 0


def test_unmeasured_model_family_is_priced_like_v5():
    assert nai_gen.estimate_generation_cost("nai-diffusion-9-future", 1536, 1024, 23) == 39


def test_cli_dry_run_charges_img2img_at_a_free_size(tmp_path):
    src = tmp_path / "draft.png"
    src.write_bytes(b"not decoded in dry-run")
    proc = subprocess.run(
        [sys.executable, str(NAI_GEN), "--prompt", "1girl", "--model", "v5-full",
         "--image", str(src), "--anlas-ledger", "none", "--dry-run"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    # 832x1216 at 28 steps is free as txt2img; with a base image it pays the V5 price.
    assert "dry-run estimate: 30.0" in proc.stderr


def test_letterbox_reference_fits_canvas_with_black_padding(tmp_path):
    from PIL import Image

    raw = _png(tmp_path / "wide.png", (400, 100)).read_bytes()
    out = Image.open(io.BytesIO(nai_gen.letterbox_reference(raw)))
    assert out.size == nai_gen.REFERENCE_CANVAS
    assert out.getpixel((0, 0)) == (0, 0, 0)  # padding
    assert out.getpixel((512, 768)) == (200, 30, 30)  # centred content


def _ref_args(**overrides):
    base = dict(
        char_ref=None, char_ref_type=None, char_ref_strength=None,
        char_ref_fidelity=None, vibe=None, character_references=[],
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_build_references_broadcasts_sliders(tmp_path):
    a, b = _png(tmp_path / "a.png"), _png(tmp_path / "b.png")
    refs = nai_gen.build_references(
        _ref_args(char_ref=[str(a), str(b)], char_ref_fidelity=[0.8]),
        "nai-diffusion-4-5-full",
    )
    assert [r["source"] for r in refs] == [str(a), str(b)]
    assert [r["fidelity"] for r in refs] == [0.8, 0.8]
    assert all(r["strength"] == nai_gen.DEFAULT_REFERENCE_STRENGTH for r in refs)
    assert all(r["type"] == "character" for r in refs)


def test_build_references_reads_job_entries_first(tmp_path):
    a, b = _png(tmp_path / "a.png"), _png(tmp_path / "b.png")
    refs = nai_gen.build_references(
        _ref_args(
            character_references=[{"image": str(a), "type": "style", "strength": 0.4}],
            char_ref=[str(b)],
        ),
        "nai-diffusion-4-5-curated",
    )
    assert [(r["source"], r["type"], r["strength"]) for r in refs] == [
        (str(a), "style", 0.4),
        (str(b), "character", nai_gen.DEFAULT_REFERENCE_STRENGTH),
    ]


def test_build_references_rejects_other_model_families(tmp_path):
    a = _png(tmp_path / "a.png")
    with pytest.raises(SystemExit, match="V4.5"):
        nai_gen.build_references(_ref_args(char_ref=[str(a)]), "nai-diffusion-5-full")


def test_build_references_rejects_vibe_in_same_request(tmp_path):
    a = _png(tmp_path / "a.png")
    with pytest.raises(SystemExit, match="Vibe Transfer"):
        nai_gen.build_references(
            _ref_args(char_ref=[str(a)], vibe=["x.png"]), "nai-diffusion-4-5-full"
        )


@pytest.mark.parametrize(
    "overrides, hint",
    [
        ({"char_ref_type": ["portrait"]}, "unknown"),
        ({"char_ref_strength": [1.5]}, "out of range"),
        ({"char_ref_fidelity": [-2.0]}, "out of range"),
    ],
)
def test_build_references_rejects_bad_sliders(tmp_path, overrides, hint):
    a = _png(tmp_path / "a.png")
    with pytest.raises(SystemExit, match=hint):
        nai_gen.build_references(
            _ref_args(char_ref=[str(a)], **overrides), "nai-diffusion-4-5-full"
        )


def test_build_references_rejects_missing_file(tmp_path):
    with pytest.raises(SystemExit, match="not found"):
        nai_gen.build_references(
            _ref_args(char_ref=[str(tmp_path / "nope.png")]), "nai-diffusion-4-5-full"
        )


def test_build_references_without_inputs_is_free_for_any_model():
    assert nai_gen.build_references(_ref_args(), "nai-diffusion-5-full") == []


def test_build_payload_injects_reference_arrays():
    refs = [{"image": b"aa", "type": "character", "strength": 1.0, "fidelity": 0.5, "source": "a"}]
    payload = nai_gen.build_payload(
        "nai-diffusion-4-5-full", "1girl", [], "neg", _args(), None, refs
    )
    params = payload["parameters"]
    assert params["director_reference_images"] == ["YWE="]
    assert params["director_reference_secondary_strength_values"] == [0.5]


def test_elide_bulk_marks_director_images():
    payload = {
        "parameters": {"director_reference_images": ["Q" * 40], "width": 832}
    }
    elided = nai_gen.elide_bulk(payload)
    assert elided["parameters"]["director_reference_images"] == ["<base64 40 chars>"]
    assert elided["parameters"]["width"] == 832


def test_append_manifest_records_reference_provenance(tmp_path):
    payload = {
        "model": "nai-diffusion-4-5-full",
        "input": "1girl",
        "parameters": {"negative_prompt": "neg", "director_reference_images": ["Q" * 40]},
    }
    refs = [{"image": b"aa", "type": "character", "strength": 1.0, "fidelity": 0.5, "source": "a.png"}]
    nai_gen.append_manifest(tmp_path, payload, [tmp_path / "x.png"], None, refs)
    record = json.loads((tmp_path / "manifest.jsonl").read_text().strip())
    assert "director_reference_images" not in record["parameters"]
    assert record["character_references"] == [
        {"source": "a.png", "type": "character", "strength": 1.0, "fidelity": 0.5}
    ]


def test_cli_dry_run_with_char_ref_estimates_reference_cost(tmp_path):
    a = _png(tmp_path / "anchor.png")
    proc = subprocess.run(
        [
            sys.executable, str(NAI_GEN), "--prompt", "1girl, upper body",
            "--char-ref", str(a), "--char-ref-fidelity", "0.7", "--dry-run",
            "--anlas-ledger", "none",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    params = payload["parameters"]
    assert params["director_reference_images"][0].startswith("<base64 ")
    assert params["director_reference_secondary_strength_values"] == [0.3]
    assert "dry-run estimate: 5.0" in proc.stderr
    assert "1 precise reference(s): +5 Anlas" in proc.stderr


def test_cli_char_ref_rejects_v5_before_spending(tmp_path):
    a = _png(tmp_path / "anchor.png")
    proc = subprocess.run(
        [
            sys.executable, str(NAI_GEN), "--prompt", "1girl", "--model", "v5-full",
            "--char-ref", str(a), "--dry-run", "--anlas-ledger", "none",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode != 0
    assert "V4.5" in proc.stderr
