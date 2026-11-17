#!/usr/bin/env python3
"""Validate the repository's CI reliability policy with only the stdlib.

Every workflow under .github/workflows/ gets the universal rules: no
ubuntu-latest, every external action pinned to a 40-hex commit SHA, no
`pip install` or `npx`, a per-job `runs-on: ubuntu-24.04` and
`timeout-minutes`, and a top-level least-privilege `permissions:` block.

ci.yml additionally gets the by-name rules that lock the path-gated job graph:
a closed job set, the always-on `changes` job (repository invariants, then lane
selection through tools/ci/changed_lanes.py), every other job depending on
`changes` and gated on exactly its lane output, and the aggregate `ci-gate`
that turns selected-and-succeeded / unselected-and-skipped into the one
required check by calling tools/ci/ci_gate.py and nothing else.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


WORKFLOWS_DIR = Path(".github/workflows")
GATED_WORKFLOW = "ci.yml"

ACTION_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
JOB_HEADER_RE = re.compile(r"^  ([A-Za-z0-9_-]+):\s*$")
STEP_HEADER_RE = re.compile(r"^      -\s")
USES_RE = re.compile(r"^\s*(?:-\s*)?uses:\s*([^\s#]+)")
GUARD_FLAG = "LORAMAKE_CI_BLOCK_NETWORK"
GUARD_ARMED_RE = re.compile(rf'^      {GUARD_FLAG}:\s*"1"\s*$', re.MULTILINE)
GUARD_DISARMED_RE = re.compile(rf'^\s+{GUARD_FLAG}:\s*"0"\s*$', re.MULTILINE)
SETUP_PYTHON_USES = "uses: actions/setup-python@"

LANES = ("python", "web", "chat", "rust", "comfyui")
JOB_LANE = {
    "python": "python",
    "web": "web",
    "web-served-api": "web",
    "chat": "chat",
    "rust-rdbg": "rust",
    "comfyui-harness": "comfyui",
}
SELECTOR_JOB = "changes"
GATE_JOB = "ci-gate"
ALLOWED_JOBS = frozenset(JOB_LANE) | {SELECTOR_JOB, GATE_JOB}
# Jobs that arm the Python network guard at job level; the setup-python exemption
# rule is scoped to them.
GUARDED_JOBS = ("python", "web-served-api")
SELECTOR_PERMISSIONS = frozenset({"contents: read", "pull-requests: read"})
SELECTOR_IF = "    if: ${{ github.event_name != 'pull_request' || github.event.pull_request.draft == false }}"
SELECTOR_REQUIRED = (
    "check_workflow.py",
    "check_nsfw_containment.py",
    "nsfw/scripts/sot_lint.py",
    "check_doc_links.py",
    "changed_lanes.py",
    # Compares two revisions; the bare form would compare the index to a clean
    # working tree and could never fail.
    'git diff --check "$from" "$to"',
    "uv sync --locked",
    "scripts/art_directions.py validate",
    "scripts/gen_strategy.py validate",
    "scripts/script_registry.py validate",
    "tests/test_repo_hygiene.py",
    "tests/test_mirror_manifest_policy.py",
    "tests/test_nsfw_containment_policy.py",
)
GATE_IF_RE = re.compile(r"^    if: \$\{\{ always\(\)( && .*)? \}\}$", re.MULTILINE)
# The verdict lives in a tested script, not in the gate's shell. The gate may check the
# repository out and call it; any other `run:` would be verdict logic nothing executes.
GATE_RUN = "python3 tools/ci/ci_gate.py"
GATE_RUN_RE = re.compile(r"^\s+(?:-\s+)?run:\s*(.*?)\s*$", re.MULTILINE)
GATE_CHECKOUT = "uses: actions/checkout@"
PULL_REQUEST_TYPES = frozenset({"opened", "synchronize", "reopened", "ready_for_review"})
WEB_REQUIRED = (
    "pnpm -C app-web install --frozen-lockfile",
    "pnpm -C app-web exec ajv compile",
    "pnpm -C app-web typecheck",
    "pnpm -C app-web test",
    "pnpm -C app-web build",
)


def _job_blocks(lines: list[str]) -> dict[str, list[str]]:
    try:
        start = next(index for index, line in enumerate(lines) if line.rstrip() == "jobs:") + 1
    except StopIteration:
        return {}

    blocks: dict[str, list[str]] = {}
    current: str | None = None
    for line in lines[start:]:
        match = JOB_HEADER_RE.match(line)
        if match:
            current = match.group(1)
            blocks[current] = [line]
        elif current is not None:
            blocks[current].append(line)
    return blocks


def _job_preamble(block_lines: list[str]) -> str:
    """Everything a job declares before its first step, i.e. its job-level keys."""
    preamble: list[str] = []
    for line in block_lines:
        if line.rstrip() == "    steps:":
            break
        preamble.append(line)
    return "\n".join(preamble)


def _job_steps(block_lines: list[str]) -> list[list[str]]:
    """Split a job block into one chunk per step."""
    steps: list[list[str]] = []
    for line in block_lines:
        if STEP_HEADER_RE.match(line):
            steps.append([line])
        elif steps:
            steps[-1].append(line)
    return steps


def _job_needs(preamble: str) -> list[str] | None:
    """The job's `needs:` as a list, or None when it declares none."""
    match = re.search(r"^    needs:\s*(.+?)\s*$", preamble, re.MULTILINE)
    if not match:
        return None
    value = match.group(1)
    if value.startswith("[") and value.endswith("]"):
        return [item.strip() for item in value[1:-1].split(",") if item.strip()]
    return [value]


def _job_if_lines(preamble: str) -> list[str]:
    return [line for line in preamble.splitlines() if line.startswith("    if:")]


def _job_permissions(preamble: str) -> set[str] | None:
    """The job-level `permissions:` scopes, or None when the job declares none."""
    lines = preamble.splitlines()
    for index, line in enumerate(lines):
        if line.rstrip() != "    permissions:":
            continue
        scopes: set[str] = set()
        for entry in lines[index + 1 :]:
            if not entry.startswith("      "):
                break
            scopes.add(entry.strip())
        return scopes
    return None


def _pull_request_types(text: str) -> set[str] | None:
    """The `on.pull_request.types` list, an empty set when absent, None without the trigger."""
    trigger = re.search(r"^  pull_request:\s*\n((?:    .*\n?)*)", text, re.MULTILINE)
    if not trigger:
        return None
    types = re.search(r"^    types:\s*\[([^\]]*)\]\s*$", trigger.group(1), re.MULTILINE)
    if not types:
        return set()
    return {item.strip() for item in types.group(1).split(",") if item.strip()}


def _universal_problems(text: str, lines: list[str], blocks: dict[str, list[str]]) -> list[str]:
    problems: list[str] = []
    if "ubuntu-latest" in text:
        problems.append("runner image must be pinned to ubuntu-24.04, not ubuntu-latest")
    if not re.search(r"^permissions:\s*\n\s{2}contents:\s*read\s*$", text, re.MULTILINE):
        problems.append("workflow must declare least-privilege `permissions: contents: read`")
    if re.search(r"(?:python\s+-m\s+)?pip\s+install", text):
        problems.append("workflow must install Python dependencies from uv.lock, not floating pip ranges")
    if re.search(r"\bnpx\b", text):
        problems.append("workflow must install Node tools from pnpm-lock.yaml, not resolve them with npx")

    for line_number, line in enumerate(lines, start=1):
        match = USES_RE.match(line)
        if not match:
            continue
        target = match.group(1).strip("'\"")
        if target.startswith("./"):
            continue
        if "@" not in target:
            problems.append(f"line {line_number}: external action has no immutable ref: {target}")
            continue
        action, ref = target.rsplit("@", 1)
        if not ACTION_SHA_RE.fullmatch(ref):
            problems.append(f"line {line_number}: {action} must be pinned to a 40-character commit SHA")

    if not blocks:
        problems.append("workflow has no parseable jobs")
        return problems

    for job_name, block_lines in blocks.items():
        block = "\n".join(block_lines)
        if not re.search(r"^    runs-on:\s*ubuntu-24\.04\s*$", block, re.MULTILINE):
            problems.append(f"job {job_name!r} must run on ubuntu-24.04")
        if not re.search(r"^    timeout-minutes:\s*[1-9][0-9]*\s*$", block, re.MULTILINE):
            problems.append(f"job {job_name!r} must set timeout-minutes")
    return problems


def _selector_problems(blocks: dict[str, list[str]]) -> list[str]:
    problems: list[str] = []
    block_lines = blocks.get(SELECTOR_JOB, [])
    if not block_lines:
        return problems
    block = "\n".join(block_lines)
    preamble = _job_preamble(block_lines)

    if_lines = _job_if_lines(preamble)
    if if_lines != [SELECTOR_IF]:
        problems.append(
            f"job {SELECTOR_JOB!r} must use the draft guard `{SELECTOR_IF.strip()}` as its only job-level if"
        )
    if "    outputs:" not in preamble:
        problems.append(f"job {SELECTOR_JOB!r} must declare outputs for every lane")
    else:
        for lane in LANES:
            pattern = rf"^      {lane}:\s*\$\{{\{{ steps\.[A-Za-z0-9_-]+\.outputs\.{lane} \}}\}}\s*$"
            if not re.search(pattern, preamble, re.MULTILINE):
                problems.append(f"job {SELECTOR_JOB!r} must export the lane output {lane!r} from its selection step")
    permissions = _job_permissions(preamble)
    if permissions is None:
        problems.append(f"job {SELECTOR_JOB!r} must declare `permissions: contents: read, pull-requests: read`")
    elif permissions != SELECTOR_PERMISSIONS:
        problems.append(
            f"job {SELECTOR_JOB!r} permissions must be exactly contents: read and pull-requests: read; "
            f"observed {sorted(permissions)}"
        )
    for needle in SELECTOR_REQUIRED:
        if needle not in block:
            problems.append(f"job {SELECTOR_JOB!r} must run {needle}")
    return problems


def _gated_job_problems(blocks: dict[str, list[str]]) -> list[str]:
    problems: list[str] = []
    for job_name, lane in JOB_LANE.items():
        block_lines = blocks.get(job_name)
        if block_lines is None:
            problems.append(f"workflow must define job {job_name!r}")
            continue
        preamble = _job_preamble(block_lines)
        needs = _job_needs(preamble)
        if needs is None or SELECTOR_JOB not in needs:
            problems.append(f"job {job_name!r} must depend on {SELECTOR_JOB!r} through `needs:`")
        expected_if = (
            "    if: ${{ (github.event_name == 'schedule' || github.event_name == 'workflow_dispatch') "
            f"&& needs.{SELECTOR_JOB}.outputs.{lane} == 'true' }}}}"
        )
        if_lines = _job_if_lines(preamble)
        if expected_if not in if_lines:
            problems.append(f"job {job_name!r} must be gated on its lane with `{expected_if.strip()}`")
        for line in if_lines:
            if line != expected_if:
                problems.append(f"job {job_name!r} may not carry another job-level if: {line.strip()!r}")
    for job_name, block_lines in blocks.items():
        if job_name == SELECTOR_JOB:
            continue
        if _job_permissions(_job_preamble(block_lines)) is not None:
            problems.append(
                f"job {job_name!r} may not declare job-level permissions; only {SELECTOR_JOB!r} reads pull requests"
            )
    return problems


def _guard_problems(blocks: dict[str, list[str]]) -> list[str]:
    problems: list[str] = []
    for job_name in GUARDED_JOBS:
        block_lines = blocks.get(job_name, [])
        block = "\n".join(block_lines)
        if GUARD_FLAG not in block or "tools/ci_network_guard" not in block:
            problems.append(f"job {job_name!r} must enable the Python non-loopback network guard")
        if "check_pytest_report.py" not in block:
            problems.append(f"job {job_name!r} must enforce the pytest skip policy")

        # The guard must be armed once, at job level, so no step can forget to inherit it.
        preamble = _job_preamble(block_lines)
        if not GUARD_ARMED_RE.search(preamble):
            problems.append(
                f"job {job_name!r} must arm the network guard for every step with job-level "
                f'`{GUARD_FLAG}: "1"`'
            )
        if not re.search(r"^      PYTHONPATH:.*tools/ci_network_guard\s*$", preamble, re.MULTILINE):
            problems.append(
                f"job {job_name!r} must put tools/ci_network_guard on the job-level PYTHONPATH"
            )

        # Exactly one sanctioned exemption: provisioning the interpreter. actions/setup-python
        # downloads CPython and upgrades pip from PyPI on a tool-cache miss, so an armed guard
        # fails the job before a single test runs. Nothing else may opt out — a test step that
        # disarmed the guard would silently regain the network the guard exists to deny.
        for step in _job_steps(block_lines):
            step_text = "\n".join(step)
            provisions_python = SETUP_PYTHON_USES in step_text
            disarms_guard = bool(GUARD_DISARMED_RE.search(step_text))
            if provisions_python and not disarms_guard:
                problems.append(
                    f"job {job_name!r} must disarm the network guard on its actions/setup-python "
                    f'step with step-level `{GUARD_FLAG}: "0"`; on a hosted tool-cache miss the '
                    "interpreter installer reaches PyPI and an armed guard fails the whole job"
                )
            if disarms_guard and not provisions_python:
                problems.append(
                    f"job {job_name!r} may only disarm the network guard on actions/setup-python; "
                    f"found an exemption on step {step[0].strip()!r}"
                )
            if GUARD_FLAG in step_text and not disarms_guard:
                problems.append(
                    f"job {job_name!r} may only override {GUARD_FLAG} to \"0\" on "
                    f"actions/setup-python; found another override on step {step[0].strip()!r}"
                )
    return problems


def _lane_body_problems(blocks: dict[str, list[str]]) -> list[str]:
    problems: list[str] = []
    for job_name in ("python", "web-served-api", SELECTOR_JOB):
        block = "\n".join(blocks.get(job_name, []))
        if "uv sync --locked" not in block:
            problems.append(f"job {job_name!r} must install from uv.lock with `uv sync --locked`")

    web = "\n".join(blocks.get("web", []))
    if WEB_REQUIRED[0] not in web:
        problems.append(
            "job 'web' must install Node dependencies from pnpm-lock.yaml with `--frozen-lockfile`"
        )
    for needle in WEB_REQUIRED[1:]:
        if needle not in web:
            problems.append(f"job 'web' must run `{needle}`")

    chat = "\n".join(blocks.get("chat", []))
    if "pnpm -C app-chat install --frozen-lockfile" not in chat:
        problems.append(
            "job 'chat' must install Node dependencies from pnpm-lock.yaml with `--frozen-lockfile`"
        )
    if "pnpm -C app-chat typecheck" not in chat:
        problems.append("job 'chat' must typecheck app-chat")
    if "pnpm -C app-chat test" not in chat:
        problems.append("job 'chat' must test app-chat")

    python_block = "\n".join(blocks.get("python", []))
    if "ffmpeg" not in python_block or "ffprobe" not in python_block:
        problems.append("job 'python' must provision ffmpeg and ffprobe for video integration coverage")
    if "uv pip check" not in python_block:
        problems.append("job 'python' must verify the synchronized environment with `uv pip check`")
    if "scripts/contract_drift.py --check" not in python_block:
        problems.append("job 'python' must retain the independent endpoint contract drift gate")
    if "for attempt in {1..10}" not in python_block:
        problems.append("job 'python' must repeat the known-regression stability loop ten times")

    web_api = "\n".join(blocks.get("web-served-api", []))
    if "--expected-skips 0" not in web_api:
        problems.append("job 'web-served-api' must require zero skips against the real export")
    return problems


def _gate_problems(blocks: dict[str, list[str]]) -> list[str]:
    problems: list[str] = []
    gate_lines = blocks.get(GATE_JOB, [])
    gate = "\n".join(gate_lines)
    if not gate:
        problems.append("workflow must define the aggregate ci-gate job")
        return problems
    preamble = _job_preamble(gate_lines)
    if not GATE_IF_RE.search(preamble):
        problems.append(
            "ci-gate must use `if: ${{ always() ... }}` so failed dependencies cannot skip it"
        )
    needs_match = re.search(r"^    needs:\s*\[([^]]+)\]\s*$", gate, re.MULTILINE)
    if not needs_match:
        problems.append("ci-gate must declare its complete dependency list inline")
    else:
        observed_needs = {item.strip() for item in needs_match.group(1).split(",")}
        expected_needs = set(blocks) - {GATE_JOB}
        if observed_needs != expected_needs:
            problems.append(
                "ci-gate dependencies do not match all real jobs: "
                f"expected {sorted(expected_needs)}, observed {sorted(observed_needs)}"
            )
    if GATE_CHECKOUT not in gate:
        problems.append("ci-gate must check out the repository so tools/ci/ci_gate.py is on disk")
    commands = GATE_RUN_RE.findall(gate)
    if commands != [GATE_RUN]:
        problems.append(
            f"ci-gate must reach its verdict with exactly one `run: {GATE_RUN}` and no other shell "
            f"logic; observed {commands}"
        )
    if f"needs.{SELECTOR_JOB}.result" not in gate:
        problems.append(f"ci-gate must read needs.{SELECTOR_JOB}.result")
    for lane in LANES:
        if f"needs.{SELECTOR_JOB}.outputs.{lane}" not in gate:
            problems.append(f"ci-gate must read needs.{SELECTOR_JOB}.outputs.{lane}")
    for job_name in blocks:
        if job_name in (GATE_JOB, SELECTOR_JOB):
            continue
        if f"needs.{job_name}.result" not in gate:
            problems.append(f"ci-gate must read needs.{job_name}.result")
    return problems


def _gated_workflow_problems(text: str, blocks: dict[str, list[str]]) -> list[str]:
    problems: list[str] = []
    if "  schedule:" not in text or "  workflow_dispatch:" not in text:
        problems.append("workflow must keep scheduled and manual drift-detection triggers")
    types = _pull_request_types(text)
    if types is None:
        problems.append("workflow must trigger on pull_request")
    elif types != PULL_REQUEST_TYPES:
        problems.append(
            f"on.pull_request.types must be exactly {sorted(PULL_REQUEST_TYPES)}; observed {sorted(types)}"
        )
    if not blocks:
        return problems

    for job_name in sorted(set(blocks) - ALLOWED_JOBS):
        problems.append(
            f"job {job_name!r} is not part of the locked job graph; known jobs: {sorted(ALLOWED_JOBS)}"
        )
    if SELECTOR_JOB not in blocks:
        problems.append(f"workflow must define the always-on {SELECTOR_JOB!r} job")
    problems.extend(_selector_problems(blocks))
    problems.extend(_gated_job_problems(blocks))
    problems.extend(_guard_problems(blocks))
    problems.extend(_lane_body_problems(blocks))
    problems.extend(_gate_problems(blocks))
    return problems


def check_workflow(path: Path) -> list[str]:
    """Universal rules for any workflow file; the job-graph rules as well for ci.yml."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    blocks = _job_blocks(lines)
    problems = _universal_problems(text, lines, blocks)
    if path.name == GATED_WORKFLOW:
        problems.extend(_gated_workflow_problems(text, blocks))
    return problems


def workflow_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.iterdir() if path.suffix in (".yml", ".yaml"))


def check_all(directory: Path) -> list[str]:
    """Every workflow under `directory`, each problem prefixed with its file name."""
    paths = workflow_files(directory)
    if not paths:
        return [f"no workflow files found under {directory}"]
    problems: list[str] = []
    for path in paths:
        problems.extend(f"{path.name}: {problem}" for problem in check_workflow(path))
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "workflow",
        nargs="*",
        type=Path,
        help="workflow files to check (default: every .yml/.yaml under --workflows-dir)",
    )
    parser.add_argument("--workflows-dir", type=Path, default=WORKFLOWS_DIR)
    args = parser.parse_args(argv)

    targets = list(args.workflow) or workflow_files(args.workflows_dir)
    if not targets:
        print(f"CI_POLICY_ERROR: no workflow files found under {args.workflows_dir}")
        return 1
    failed = False
    for path in targets:
        problems = check_workflow(path)
        for problem in problems:
            print(f"CI_POLICY_ERROR: {path}: {problem}")
        if problems:
            failed = True
        else:
            print(f"CI policy OK: {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
