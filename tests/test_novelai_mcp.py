"""Offline tests for the standalone ``novelai`` MCP server.

Nothing here touches the network or needs a token: every case stops at or before
the free-tier guard, and generation itself is a subprocess these tests never spawn.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import novelai_mcp as nm  # noqa: E402


def collect() -> dict:
    collector = nm._Collector()
    nm.register(collector)
    return collector.tools


def test_registers_exactly_the_three_tools():
    assert sorted(collect()) == ["nai_budget", "nai_generate", "nai_models"]


def test_every_tool_has_a_docstring_the_model_can_route_on():
    for name, fn in collect().items():
        assert (fn.__doc__ or "").strip(), f"{name} has no docstring"


def test_models_lists_aliases_and_marks_the_default():
    info = nm.tool_models()
    assert info["default"] == "v5-full"
    assert info["aliases"]["v5-full"] == "nai-diffusion-5-full"
    assert info["free_tier"]["max_steps"] == 28


def test_dry_run_resolves_without_network_or_token(monkeypatch):
    monkeypatch.delenv("NAI_ACCESS_TOKEN", raising=False)
    out = nm.tool_generate("1girl, solo", dry_run=True)
    assert out["dry_run"] is True
    assert out["model"] == "nai-diffusion-5-full"
    assert out["size"] == [832, 1216]
    assert out["anlas_total"] == 0
    assert out["free_tier"] is True
    assert out["job"]["base"] == "1girl, solo"


def test_free_tier_refuses_a_paid_size_and_names_the_cost():
    with pytest.raises(nm.NaiError) as exc:
        nm.tool_generate("x", width=1024, height=1536, dry_run=True)
    assert "Anlas" in str(exc.value)
    assert "paid=true" in str(exc.value)


def test_free_tier_refuses_more_than_28_steps():
    with pytest.raises(nm.NaiError):
        nm.tool_generate("x", steps=40, dry_run=True)


def test_paid_true_allows_the_same_request():
    out = nm.tool_generate("x", width=1024, height=1536, steps=40, paid=True, dry_run=True)
    assert out["free_tier"] is False
    assert out["anlas_per_image"] > 0


def test_mixed_character_centers_are_refused_before_spawning():
    with pytest.raises(nm.NaiError) as exc:
        nm.tool_generate(
            "x",
            characters=[{"prompt": "a", "center": {"x": 0.3, "y": 0.5}}, {"prompt": "b"}],
            dry_run=True,
        )
    assert "center" in str(exc.value)


def test_characters_with_no_centers_are_fine():
    out = nm.tool_generate("x", characters=[{"prompt": "a"}, {"prompt": "b"}], dry_run=True)
    assert len(out["job"]["characters"]) == 2
    assert all("center" not in c for c in out["job"]["characters"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prompt": ""},
        {"prompt": "x", "model": "v9-ultra"},
        {"prompt": "x", "sampler": "nope"},
        {"prompt": "x", "noise_schedule": "nope"},
        {"prompt": "x", "uc_preset": 9},
        {"prompt": "x", "steps": 0},
        {"prompt": "x", "count": 0},
        {"prompt": "x", "count": nm.MAX_COUNT + 1},
        {"prompt": "x", "width": 100, "height": 100, "paid": True},
        {"prompt": "x", "characters": [{"center": {"x": 0.1, "y": 0.1}}]},
        {"prompt": "x", "characters": [{"prompt": "a", "center": {"x": 2.0, "y": 0.5}}]},
    ],
)
def test_bad_input_raises_naierror_not_systemexit(kwargs):
    kwargs.setdefault("dry_run", True)
    with pytest.raises(nm.NaiError):
        nm.tool_generate(**kwargs)


def test_raw_model_ids_pass_through_for_models_newer_than_the_table():
    out = nm.tool_generate("x", model="nai-diffusion-6-full", paid=True, dry_run=True)
    assert out["model"] == "nai-diffusion-6-full"


def test_a_model_outside_the_free_prefix_table_is_treated_as_paid():
    """Fail-closed: a model NovelAI ships after this table was written is not
    assumed free. Extending ``nai_gen.ANLAS_FREE_MODEL_PREFIXES`` is what makes
    a new family free, not its bare id passing ``resolve_model``."""
    with pytest.raises(nm.NaiError):
        nm.tool_generate("x", model="nai-diffusion-6-full", dry_run=True)


def test_ledger_path_is_absolute_so_an_arbitrary_cwd_cannot_split_it(monkeypatch):
    monkeypatch.delenv("NAI_ANLAS_LEDGER", raising=False)
    assert nm._ledger_path().is_absolute()


def test_ledger_can_be_disabled(monkeypatch):
    monkeypatch.setenv("NAI_ANLAS_LEDGER", "none")
    assert nm._ledger_path() is None
    assert nm.tool_budget()["ledger"] is None


def test_budget_reports_zero_when_no_ledger_file_exists(tmp_path, monkeypatch):
    monkeypatch.setenv("NAI_ANLAS_LEDGER", str(tmp_path / "absent.jsonl"))
    out = nm.tool_budget()
    assert out["exists"] is False
    assert out["used"] == 0.0
    assert out["remaining"] == out["budget"]


def test_budget_totals_a_written_ledger(tmp_path, monkeypatch):
    ledger = tmp_path / "anlas.jsonl"
    day = "2026-09-18"
    ledger.write_text(
        json.dumps({"date": day, "id": "a", "kind": "generation", "cost": 9.0,
                    "status": "reserved"}) + "\n"
        + json.dumps({"date": day, "kind": "resolve", "ref": "a", "status": "charged",
                      "cost": 9.0}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("NAI_ANLAS_LEDGER", str(ledger))
    assert nm.tool_budget(date=day)["used"] == 9.0


def test_registered_tool_returns_json_text():
    payload = json.loads(collect()["nai_models"]())
    assert payload["default"] == "v5-full"


def mcpserver():
    """The real SDK ships in the ``agent`` extra, which the CI dev lane does not
    install. Where it is absent these cases skip; where the server actually runs,
    they are the only cover for schema generation."""
    module = pytest.importorskip(
        "mcp.server.mcpserver", reason="mcp SDK lives in the 'agent' extra"
    )
    return module.MCPServer


def real_schema(tool_name: str) -> dict:
    """The input schema a real MCP client sees. ``_Collector`` never builds one,
    so this is the only check that the signatures survive schema generation."""
    import asyncio

    MCPServer = mcpserver()

    server = MCPServer("novelai")
    nm.register(server)
    tool = next(t for t in asyncio.run(server.list_tools()) if t.name == tool_name)
    return tool.input_schema


def test_the_real_server_accepts_every_signature():
    import asyncio

    MCPServer = mcpserver()

    server = MCPServer("novelai")
    nm.register(server)
    names = sorted(t.name for t in asyncio.run(server.list_tools()))
    assert names == ["nai_budget", "nai_generate", "nai_models"]


def test_character_shape_reaches_the_model_as_named_fields():
    """A bare ``list[dict]`` serializes to an object with no properties, leaving
    the model to guess 'prompt' and 'center' from prose. Keep the TypedDicts."""
    schema = real_schema("nai_generate")
    character = schema["$defs"]["Character"]
    assert character["required"] == ["prompt"]
    assert set(character["properties"]) == {"prompt", "center"}
    center = schema["$defs"]["Center"]
    assert sorted(center["required"]) == ["x", "y"]


def test_size_is_advertised_as_an_enum():
    size = real_schema("nai_generate")["properties"]["size"]
    assert sorted(size["enum"]) == ["landscape", "portrait", "square"]


def test_prompt_is_the_only_required_argument():
    assert real_schema("nai_generate")["required"] == ["prompt"]
