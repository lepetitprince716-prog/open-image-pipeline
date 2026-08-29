import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import json

import pytest

import model_quirks


TEST_CARD = {
    "arch": "test architecture",
    "style": "painterly",
    "prompt_style": "natural language",
    "recipe": "8 steps",
    "speed": "5 seconds",
    "res_range": "1024-1536",
    "vram": "snapshot only",
    "engine": "test_engine.py",
    "strengths": ["clean", "sharp"],
    "body_prior": "adult",
    "fabric_prior": "linen",
    "exposure": "balanced",
    "artifact": "none",
    "pose_control": "openpose",
    "lora_strength_ok": "0.6-0.8",
    "best_shots": ["close", "full"],
    "avoid_shots": ["low"],
    "no_freeu": True,
    "neg_add": "bad anatomy, lowres",
    "ckpt": "test.safetensors",
    "note": "verified note",
    "verified": "2026-07-05",
    "evidence": "outputs/_analysis/test/REPORT.md",
    "portrait_recipe": {"verified": "one sample", "rules": ["explicit clothing"]},
}


def _patch_cards(monkeypatch, bases=None):
    if bases is None:
        bases = {
            "testbase": dict(TEST_CARD),
            "plainbase": {"style": "plain", "best_shots": [], "no_freeu": False},
        }
    monkeypatch.setattr(model_quirks, "_load", lambda: {"bases": bases})


def _run_main(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["model_quirks.py", *argv])
    return model_quirks.main()


def test_cli_list_prints_exact_rows_and_returns_none(monkeypatch, capsys):
    _patch_cards(monkeypatch)

    ret = _run_main(monkeypatch, ["list"])

    expected = (
        f"{'testbase':<40} {'painterly':<18} best=close, full\n"
        f"{'plainbase':<40} {'plain':<18} best=-\n"
    )
    assert ret is None
    assert capsys.readouterr().out == expected


def test_cli_list_propagates_load_failure_as_system_exit(monkeypatch):
    monkeypatch.setattr(model_quirks, "_load", lambda: (_ for _ in ()).throw(SystemExit("missing quirks data: test")))

    with pytest.raises(SystemExit) as exc:
        _run_main(monkeypatch, ["list"])

    assert exc.value.code == "missing quirks data: test"


def test_cli_lookup_prints_exact_human_format(monkeypatch, capsys):
    _patch_cards(monkeypatch, {"testbase": dict(TEST_CARD)})

    ret = _run_main(monkeypatch, ["lookup", "testbase"])

    expected = (
        "# testbase\n"
        "  arch             test architecture\n"
        "  style            painterly\n"
        "  prompt_style     natural language\n"
        "  recipe           8 steps\n"
        "  speed            5 seconds\n"
        "  res_range        1024-1536\n"
        "  vram             snapshot only\n"
        "  engine           test_engine.py\n"
        "  strengths        clean, sharp\n"
        "  body_prior       adult\n"
        "  fabric_prior     linen\n"
        "  exposure         balanced\n"
        "  artifact         none\n"
        "  pose_control     openpose\n"
        "  lora_strength_ok 0.6-0.8\n"
        "  best_shots       close, full\n"
        "  avoid_shots      low\n"
        "  no_freeu         True\n"
        "  neg_add          bad anatomy, lowres\n"
        "  ckpt             test.safetensors\n"
        "  note             verified note\n"
        "  verified         2026-07-05\n"
        "  evidence         outputs/_analysis/test/REPORT.md\n"
        '  portrait_recipe  {"rules": ["explicit clothing"], "verified": "one sample"}\n'
    )
    assert ret is None
    assert capsys.readouterr().out == expected


def test_cli_lookup_json_prints_full_injected_card(monkeypatch, capsys):
    _patch_cards(monkeypatch, {"testbase": dict(TEST_CARD)})

    ret = _run_main(monkeypatch, ["lookup", "testbase", "--json"])

    expected_card = {"base": "testbase", **TEST_CARD}
    assert ret is None
    assert capsys.readouterr().out == json.dumps(expected_card, ensure_ascii=False, indent=2) + "\n"


def test_cli_neg_add_prints_value_and_missing_key_as_blank_line(monkeypatch, capsys):
    _patch_cards(monkeypatch)

    ret_value = _run_main(monkeypatch, ["neg-add", "testbase"])
    out_value = capsys.readouterr().out
    ret_blank = _run_main(monkeypatch, ["neg-add", "plainbase"])
    out_blank = capsys.readouterr().out

    assert ret_value is None
    assert out_value == "bad anatomy, lowres\n"
    assert ret_blank is None
    assert out_blank == "\n"


def test_cli_flags_prints_no_freeu_or_blank_line(monkeypatch, capsys):
    _patch_cards(monkeypatch)

    ret_flag = _run_main(monkeypatch, ["flags", "testbase"])
    out_flag = capsys.readouterr().out
    ret_blank = _run_main(monkeypatch, ["flags", "plainbase"])
    out_blank = capsys.readouterr().out

    assert ret_flag is None
    assert out_flag == "--no-freeu\n"
    assert ret_blank is None
    assert out_blank == "\n"


@pytest.mark.parametrize("argv", [["lookup", "missing"], ["neg-add", "missing"], ["flags", "missing"]])
def test_cli_lookup_neg_add_and_flags_not_found_exit_with_string_code(monkeypatch, argv):
    _patch_cards(monkeypatch, {"known": {"style": "known"}})

    with pytest.raises(SystemExit) as exc:
        _run_main(monkeypatch, argv)

    assert isinstance(exc.value.code, str)
    assert "no quirks card matches 'missing'" in exc.value.code


def test_cli_list_with_empty_bases_prints_nothing_and_exits_zero(monkeypatch, capsys):
    # Scope: this pins the graceful-empty / no-not-found contract only — that an empty
    # {"bases": {}} does NOT raise SystemExit the way lookup/neg-add/flags do via _require()
    # on a miss, and stdout stays empty. It is vacuous for the loop-body's row-formatting
    # logic (an empty bases dict never enters the loop, so a mutated loop body would still
    # pass here); that formatting is pinned separately by
    # test_cli_list_prints_exact_rows_and_returns_none, which exercises non-empty bases.
    monkeypatch.setattr(model_quirks, "_load", lambda: {"bases": {}})

    ret = _run_main(monkeypatch, ["list"])

    # _run_main calls model_quirks.main() directly (no pytest.raises), so an unexpected
    # SystemExit would already abort this test with an error rather than merely failing an
    # assertion below — ret is None confirms cmd_list returned normally in the exit-0 sense.
    assert ret is None
    assert capsys.readouterr().out == ""


def test_cli_without_subcommand_exits_with_argparse_code_two(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["model_quirks.py"])

    with pytest.raises(SystemExit) as exc:
        model_quirks.main()

    assert exc.value.code == 2


VIDEO_CARD = {
    "arch": "test video architecture",
    "engine": "test_video_engine.py",
    "frame_law": "17k+5 at 24 fps",
    "sampling": "res_multistep + simple, steps=20",
    "native_canvas": "768 px short edge",
    # Real quirks are prose sentences; the fixture has to be long enough to cross the
    # inline/one-per-line threshold or it would pin the wrong branch.
    "quirks": [
        "Turn and contact actions mirror left/right, so do not rely on left/right wording.",
        "The anchor image must not pre-complete any beat of the shot list.",
    ],
    "verified": "2026-08-13",
}


def _patch_video(monkeypatch, videos=None, bases=None):
    if videos is None:
        videos = {"testvideo": dict(VIDEO_CARD)}
    monkeypatch.setattr(
        model_quirks, "_load", lambda: {"bases": bases or {}, "video_models": videos}
    )


def test_cli_lookup_video_reads_the_video_section_not_bases(monkeypatch, capsys):
    # A name present in both sections must resolve to the video card under --video,
    # otherwise the two namespaces silently collide.
    _patch_video(monkeypatch, bases={"testvideo": dict(TEST_CARD)})

    _run_main(monkeypatch, ["lookup", "testvideo", "--video"])

    out = capsys.readouterr().out
    assert out.startswith("# testvideo\n")
    assert "test video architecture" in out
    assert "painterly" not in out


def test_cli_lookup_video_breaks_long_lists_onto_their_own_lines(monkeypatch, capsys):
    _patch_video(monkeypatch)

    _run_main(monkeypatch, ["lookup", "testvideo", "--video"])

    out = capsys.readouterr().out
    assert "  quirks\n" in out
    assert "    - The anchor image must not pre-complete any beat of the shot list.\n" in out


def test_cli_lookup_without_video_flag_does_not_see_video_models(monkeypatch):
    _patch_video(monkeypatch)

    with pytest.raises(SystemExit) as exc:
        _run_main(monkeypatch, ["lookup", "testvideo"])

    assert "quirks card" in str(exc.value.code)


def test_cli_list_appends_video_rows_after_bases(monkeypatch, capsys):
    _patch_video(monkeypatch, bases={"plainbase": {"style": "plain", "best_shots": []}})

    _run_main(monkeypatch, ["list"])

    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("plainbase")
    assert lines[1] == f"{'testvideo':<40} {'video':<18} 768 px short edge"


def test_validate_rejects_a_video_card_carrying_base_only_fields(monkeypatch):
    bad = dict(VIDEO_CARD)
    bad["neg_add"] = "bad anatomy"
    _patch_video(monkeypatch, videos={"testvideo": bad})
    monkeypatch.setattr(model_quirks, "_load",
                        lambda: {"version": 1, "bases": {},
                                 "video_models": {"testvideo": bad}})

    with pytest.raises(SystemExit) as exc:
        _run_main(monkeypatch, ["validate"])

    assert "1 validation error" in str(exc.value.code)


def test_validate_rejects_a_video_card_missing_a_required_field(monkeypatch):
    incomplete = {k: v for k, v in VIDEO_CARD.items() if k != "frame_law"}
    monkeypatch.setattr(model_quirks, "_load",
                        lambda: {"version": 1, "bases": {},
                                 "video_models": {"testvideo": incomplete}})

    with pytest.raises(SystemExit) as exc:
        _run_main(monkeypatch, ["validate"])

    assert "1 validation error" in str(exc.value.code)
