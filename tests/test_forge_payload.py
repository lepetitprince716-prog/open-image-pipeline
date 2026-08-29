"""Offline unit tests for the A1111-metadata -> Forge payload mapping."""
from __future__ import annotations

from loramake import forge_payload


def _parsed(params: dict, prompt: str = "1girl, city", negative: str = "lowres") -> dict:
    return {"prompt": prompt, "negative_prompt": negative, "parameters": params}


BASE_PARAMS = {
    "Steps": "28",
    "Sampler": "DPM++ 3M SDE",
    "Schedule type": "Karras",
    "CFG scale": "4",
    "Seed": "2541964279",
    "Size": "832x1216",
}


def test_core_scalar_fields_map_to_payload():
    mapped = forge_payload.payload_from_params(_parsed(dict(BASE_PARAMS)))
    payload = mapped.payload
    assert payload["prompt"] == "1girl, city"
    assert payload["negative_prompt"] == "lowres"
    assert payload["steps"] == 28
    assert payload["sampler_name"] == "DPM++ 3M SDE"
    assert payload["scheduler"] == "karras"
    assert payload["cfg_scale"] == 4.0
    assert payload["seed"] == 2541964279
    assert (payload["width"], payload["height"]) == (832, 1216)
    assert payload["send_images"] is True and payload["save_images"] is False
    assert mapped.unmapped == {}


def test_legacy_sampler_suffix_splits_into_scheduler():
    params = dict(BASE_PARAMS)
    del params["Schedule type"]
    params["Sampler"] = "DPM++ 2M Karras"
    payload = forge_payload.payload_from_params(_parsed(params)).payload
    assert payload["sampler_name"] == "DPM++ 2M"
    assert payload["scheduler"] == "karras"


def test_automatic_schedule_type_is_omitted():
    params = dict(BASE_PARAMS, **{"Schedule type": "Automatic"})
    payload = forge_payload.payload_from_params(_parsed(params)).payload
    assert "scheduler" not in payload


def test_clip_skip_and_ensd_go_through_override_settings():
    params = dict(BASE_PARAMS, **{"Clip skip": "2", "ENSD": "31337"})
    payload = forge_payload.payload_from_params(_parsed(params)).payload
    assert payload["override_settings"]["CLIP_stop_at_last_layers"] == 2
    assert payload["override_settings"]["eta_noise_seed_delta"] == 31337
    assert payload["override_settings_restore_afterwards"] is False


def test_discard_penultimate_sigma_maps_to_override():
    params = dict(BASE_PARAMS, **{"Discard penultimate sigma": "True"})
    mapped = forge_payload.payload_from_params(_parsed(params))
    assert mapped.payload["override_settings"]["always_discard_next_to_last_sigma"] is True
    assert "Discard penultimate sigma" not in mapped.unmapped


def test_hires_enables_and_injects_additional_modules():
    params = dict(BASE_PARAMS, **{
        "Denoising strength": "0.41",
        "Hires upscale": "1.5",
        "Hires upscaler": "4x_foolhardy_Remacri",
        "Hires steps": "20",
        "Hires CFG Scale": "4",
    })
    payload = forge_payload.payload_from_params(_parsed(params)).payload
    assert payload["enable_hr"] is True
    # Omitting this crashes Forge's hires path with a NoneType error.
    assert payload["hr_additional_modules"] == ["Use same choices"]
    assert payload["hr_scale"] == 1.5
    assert payload["hr_upscaler"] == "4x_foolhardy_Remacri"
    assert payload["hr_second_pass_steps"] == 20
    assert payload["hr_cfg"] == 4.0
    assert payload["denoising_strength"] == 0.41


def test_hires_resize_maps_to_resize_xy():
    params = dict(BASE_PARAMS, **{"Hires resize": "1248x1824"})
    payload = forge_payload.payload_from_params(_parsed(params)).payload
    assert payload["enable_hr"] is True
    assert (payload["hr_resize_x"], payload["hr_resize_y"]) == (1248, 1824)


def test_hires_without_denoise_warns():
    params = dict(BASE_PARAMS, **{"Hires upscale": "2.0"})
    mapped = forge_payload.payload_from_params(_parsed(params))
    assert any("Denoising strength" in w for w in mapped.warnings)


def test_adetailer_flat_args_shape_and_never_script_name():
    params = dict(BASE_PARAMS, **{
        "ADetailer model": "face_yolov8n.pt",
        "ADetailer confidence": "0.3",
        "ADetailer dilate erode": "4",
        "ADetailer mask blur": "4",
        "ADetailer denoising strength": "0.4",
        "ADetailer inpaint only masked": "True",
        "ADetailer inpaint padding": "32",
        "ADetailer version": "24.9.0",
    })
    mapped = forge_payload.payload_from_params(_parsed(params))
    payload = mapped.payload
    assert "script_name" not in payload  # script_name path is HTTP 422 on the deployed build
    args = payload["alwayson_scripts"]["adetailer"]["args"]
    assert args == [True, "face_yolov8n.pt", "None", 0.3, 4, 4, 0.4, True, 32]
    assert mapped.report_only["ADetailer version"] == "24.9.0"
    assert "ADetailer version" not in mapped.unmapped


def test_adetailer_defaults_fill_missing_fields():
    params = dict(BASE_PARAMS, **{"ADetailer model": "face_yolov8n.pt"})
    payload = forge_payload.payload_from_params(_parsed(params)).payload
    assert payload["alwayson_scripts"]["adetailer"]["args"] == [
        True, "face_yolov8n.pt", "None", 0.3, 4, 4, 0.4, True, 32]


def test_model_and_hashes_become_checkpoint_stub():
    params = dict(BASE_PARAMS, **{
        "Model": "wai-illustrious-v16.0",
        "Model hash": "ab12cd34ef",
    })
    assets = forge_payload.payload_from_params(_parsed(params)).assets
    assert assets["checkpoint"] == {"name": "wai-illustrious-v16.0", "short_hash": "ab12cd34ef"}


def test_lora_hashes_and_inline_tags_merge():
    params = dict(BASE_PARAMS, **{"Lora hashes": '"styleA: aaaa1111bbbb, extraB: cccc2222dddd"'})
    prompt = "1girl, <lora:styleA:0.8>"
    mapped = forge_payload.payload_from_params(_parsed(params, prompt=prompt))
    loras = {entry["name"]: entry for entry in mapped.assets["loras"]}
    assert loras["styleA"]["weight"] == 0.8
    assert loras["styleA"]["short_hash"] == "aaaa1111bbbb"
    assert loras["styleA"]["in_prompt"] is True
    assert loras["extraB"]["in_prompt"] is False
    assert any("extraB" in w for w in mapped.warnings)          # hashed but untagged
    assert any("trigger words" in w for w in mapped.warnings)   # trigger-injection gotcha


def test_ti_hashes_become_embedding_stubs():
    params = dict(BASE_PARAMS, **{"TI hashes": '"IllusP0s: 6d07a1f0, IllusN3g: 8c4b2a91"'})
    assets = forge_payload.payload_from_params(_parsed(params)).assets
    names = {e["name"] for e in assets["embeddings"]}
    assert names == {"IllusP0s", "IllusN3g"}


def test_unknown_keys_land_in_unmapped():
    params = dict(BASE_PARAMS, **{"NGMS": "2", "Downcast alphas_cumprod": "True"})
    mapped = forge_payload.payload_from_params(_parsed(params))
    assert set(mapped.unmapped) == {"NGMS", "Downcast alphas_cumprod"}


def test_version_is_report_only_not_unmapped():
    params = dict(BASE_PARAMS, **{"Version": "f2.0.1v1.10.1-previous-669-gdfdcbab6"})
    mapped = forge_payload.payload_from_params(_parsed(params))
    assert mapped.report_only["Version"].endswith("gdfdcbab6")
    assert "Version" not in mapped.unmapped


def test_split_legacy_sampler_table():
    assert forge_payload.split_legacy_sampler("DPM++ 2M Karras") == ("DPM++ 2M", "karras")
    assert forge_payload.split_legacy_sampler("DPM++ 2M SDE Exponential") == ("DPM++ 2M SDE", "exponential")
    assert forge_payload.split_legacy_sampler("Euler a") == ("Euler a", None)


def test_normalize_scheduler():
    assert forge_payload.normalize_scheduler("Karras") == "karras"
    assert forge_payload.normalize_scheduler("SGM Uniform") == "sgm_uniform"
    assert forge_payload.normalize_scheduler("Align Your Steps GITS") == "align_your_steps_GITS"


MODELS = [
    {"title": "wai-illustrious-v16.0.safetensors [ab12cd34ef]",
     "model_name": "wai-illustrious-v16.0", "filename": "D:\\models\\wai-illustrious-v16.0.safetensors",
     "hash": "ab12cd34ef", "sha256": "ab12cd34ef" + "0" * 54},
    {"title": "other.safetensors [1122334455]", "model_name": "other",
     "filename": "D:\\models\\other.safetensors", "hash": "1122334455", "sha256": "1122334455" + "f" * 54},
]


def test_match_checkpoint_prefers_hash_over_name():
    entry, basis = forge_payload.match_checkpoint("other", "ab12cd34ef", MODELS)
    assert entry["model_name"] == "wai-illustrious-v16.0"
    assert basis == "short_hash"


def test_match_checkpoint_name_fallback_ignores_title_hash_suffix():
    entry, basis = forge_payload.match_checkpoint("WAI-Illustrious-V16.0.safetensors", None, MODELS)
    assert entry["model_name"] == "wai-illustrious-v16.0"
    assert basis == "name"


def test_match_checkpoint_miss_returns_none():
    entry, basis = forge_payload.match_checkpoint("absent", "ffffffffff", MODELS)
    assert entry is None and basis == ""


def test_match_lora_by_alias_then_stem():
    entries = [{"name": "styleA-v2", "alias": "styleA", "path": "D:\\loras\\styleA-v2.safetensors"}]
    assert forge_payload.match_lora("styleA", entries)[1] == "name"
    assert forge_payload.match_lora("STYLEA-V2", entries)[1] == "name"
    assert forge_payload.match_lora("styleA-v2.safetensors", entries)[1] == "stem"
    assert forge_payload.match_lora("nope", entries) == (None, "")


def test_diff_parameters_statuses_and_benign_version():
    source = _parsed(dict(BASE_PARAMS, Version="1.10.1"))
    produced = _parsed(dict(BASE_PARAMS, **{"Version": "f2.0.1", "Model hash": "ab12cd34ef"}))
    rows = {r["field"]: r for r in forge_payload.diff_parameters(source, produced)}
    assert rows["Prompt"]["status"] == "MATCH"
    assert rows["Version"]["status"] == "DIFF" and rows["Version"]["benign"] is True
    assert rows["Model hash"]["status"] == "PRODUCED_ONLY"
    assert rows["Steps"]["status"] == "MATCH"


def test_diff_model_title_is_benign_when_hash_matches():
    source = _parsed({"Model": "WAI-Nsfw-Illustrious-16", "Model hash": "a5f58eb1c3"})
    produced = _parsed({"Model": "wai-illustrious-v16.0", "Model hash": "a5f58eb1c3"})
    rows = {r["field"]: r for r in forge_payload.diff_parameters(source, produced)}
    assert rows["Model"]["status"] == "DIFF" and rows["Model"]["benign"] is True
    mismatch = _parsed({"Model": "other", "Model hash": "1122334455"})
    rows = {r["field"]: r for r in forge_payload.diff_parameters(source, mismatch)}
    assert rows["Model"]["benign"] is False


def test_diff_parameters_numeric_normalization():
    source = _parsed({"CFG scale": "4"})
    produced = _parsed({"CFG scale": "4.0"})
    rows = {r["field"]: r for r in forge_payload.diff_parameters(source, produced)}
    assert rows["CFG scale"]["status"] == "MATCH"


def test_diff_parameters_hash_map_ordering_is_normalized():
    source = _parsed({"Lora hashes": '"a: 111, b: 222"'})
    produced = _parsed({"Lora hashes": '"b: 222, a: 111"'})
    rows = {r["field"]: r for r in forge_payload.diff_parameters(source, produced)}
    assert rows["Lora hashes"]["status"] == "MATCH"
