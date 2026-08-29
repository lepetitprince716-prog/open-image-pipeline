"""Live smoke against the real Z3 Forge deployment.

Opt-in only: needs LORAMAKE_FORGE_LIVE=1 AND a reachable Forge (17862 tunnel up,
service running). One 512x512 8-step render — cheap, but still GPU work.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from loramake import forge  # noqa: E402
from loramake.forge_endpoints import resolve_forge_host  # noqa: E402

pytestmark = pytest.mark.skipif(
    os.environ.get("LORAMAKE_FORGE_LIVE") != "1",
    reason="live Forge smoke is opt-in: set LORAMAKE_FORGE_LIVE=1",
)


def _host() -> str:
    host = resolve_forge_host("z3")
    ok, message = forge.ping(host)
    if not ok:
        pytest.skip(message)
    return host


def test_enums_answer():
    host = _host()
    assert forge.sd_models(host), "sd-models enum came back empty"
    assert forge.sampler_names(host), "samplers enum came back empty"


def test_txt2img_parameters_chunk_roundtrips(tmp_path):
    host = _host()
    payload = {
        "prompt": "1girl, portrait, simple background",
        "negative_prompt": "lowres",
        "steps": 8,
        "cfg_scale": 4.0,
        "width": 512,
        "height": 512,
        "seed": 42,
        "sampler_name": "Euler a",
        "n_iter": 1,
        "batch_size": 1,
        "send_images": True,
        "save_images": False,
    }
    images, info = forge.txt2img(host, payload, timeout=600)
    assert images and info.get("infotexts")
    saved = forge.save_results(images, info, out_dir=tmp_path, stem="live_smoke",
                               meta_base={"engine": "forge", "mode": "live_smoke"})
    import image_recipe  # noqa: PLC0415
    chunks = image_recipe.decode_text_chunks(image_recipe.read_png_chunks(str(saved[0])))
    parsed = image_recipe.parse_a1111_parameters(chunks["parameters"])
    assert parsed["parameters"]["Seed"] == "42"
    assert parsed["parameters"]["Steps"] == "8"
