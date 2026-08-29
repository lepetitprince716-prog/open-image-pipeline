from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import project_lines


def _registry(tmp_path: Path) -> Path:
    root = tmp_path / "consumer"
    root.mkdir()
    (root / "run.py").write_text("print('ok')\n", encoding="utf-8")
    (root / "project.json").write_text("{}\n", encoding="utf-8")
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "loramake-image-lines-v1",
                "lines": [
                    {
                        "id": "demo",
                        "name": "Demo",
                        "root": str(root),
                        "runner": "run.py",
                        "config": "project.json",
                        "output_root": "outputs",
                        "isolation": "external",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_registry_resolves_external_owned_paths(tmp_path: Path) -> None:
    data = project_lines.load_registry(_registry(tmp_path))
    status = project_lines.line_status(project_lines.get_line(data, "demo"))

    assert status["ready"] is True
    assert status["isolation"] == "external"
    assert status["paths"]["output_root"].endswith("consumer/outputs")


def test_registry_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = _registry(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["lines"].append(dict(data["lines"][0]))
    path.write_text(json.dumps(data), encoding="utf-8")

    try:
        project_lines.load_registry(path)
    except project_lines.RegistryError as exc:
        assert "duplicate line id" in str(exc)
    else:
        raise AssertionError("duplicate ids should fail")
