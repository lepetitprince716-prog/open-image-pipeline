#!/usr/bin/env python3
"""Validate the repository's CI reliability policy with only the stdlib."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ACTION_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
JOB_HEADER_RE = re.compile(r"^  ([A-Za-z0-9_-]+):\s*$")
STEP_HEADER_RE = re.compile(r"^      -\s")
USES_RE = re.compile(r"^\s*(?:-\s*)?uses:\s*([^\s#]+)")
GUARD_FLAG = "LORAMAKE_CI_BLOCK_NETWORK"
GUARD_ARMED_RE = re.compile(rf'^      {GUARD_FLAG}:\s*"1"\s*$', re.MULTILINE)
GUARD_DISARMED_RE = re.compile(rf'^\s+{GUARD_FLAG}:\s*"0"\s*$', re.MULTILINE)
SETUP_PYTHON_USES = "uses: actions/setup-python@"


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


def check_workflow(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    problems: list[str] = []

    if "ubuntu-latest" in text:
        problems.append("runner image must be pinned to ubuntu-24.04, not ubuntu-latest")
    if "  schedule:" not in text or "  workflow_dispatch:" not in text:
        problems.append("workflow must keep scheduled and manual drift-detection triggers")
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

    blocks = _job_blocks(lines)
    if not blocks:
        problems.append("workflow has no parseable jobs")
        return problems

    for job_name, block_lines in blocks.items():
        block = "\n".join(block_lines)
        if not re.search(r"^    runs-on:\s*ubuntu-24\.04\s*$", block, re.MULTILINE):
            problems.append(f"job {job_name!r} must run on ubuntu-24.04")
        if not re.search(r"^    timeout-minutes:\s*[1-9][0-9]*\s*$", block, re.MULTILINE):
            problems.append(f"job {job_name!r} must set timeout-minutes")

    for job_name in ("python", "web-served-api", "validate"):
        block = "\n".join(blocks.get(job_name, []))
        if "uv sync --locked" not in block:
            problems.append(f"job {job_name!r} must install from uv.lock with `uv sync --locked`")

    for job_name in ("schemas", "web"):
        block = "\n".join(blocks.get(job_name, []))
        if "pnpm -C app-web install --frozen-lockfile" not in block:
            problems.append(
                f"job {job_name!r} must install Node dependencies from pnpm-lock.yaml "
                "with `--frozen-lockfile`"
            )

    for job_name in ("python", "web-served-api"):
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

    python_block = "\n".join(blocks.get("python", []))
    if "ffmpeg" not in python_block or "ffprobe" not in python_block:
        problems.append("job 'python' must provision ffmpeg and ffprobe for video integration coverage")
    if "uv pip check" not in python_block:
        problems.append("job 'python' must verify the synchronized environment with `uv pip check`")
    if "scripts/contract_drift.py --check" not in python_block:
        problems.append("job 'python' must retain the independent endpoint contract drift gate")

    web_api = "\n".join(blocks.get("web-served-api", []))
    if "--expected-skips 0" not in web_api:
        problems.append("job 'web-served-api' must require zero skips against the real export")

    schemas = "\n".join(blocks.get("schemas", []))
    if "pnpm -C app-web exec ajv compile" not in schemas:
        problems.append("job 'schemas' must use the AJV compiler pinned by app-web/pnpm-lock.yaml")

    gate = "\n".join(blocks.get("ci-gate", []))
    if not gate:
        problems.append("workflow must define the aggregate ci-gate job")
    elif "if: ${{ always() }}" not in gate:
        problems.append("ci-gate must use `if: ${{ always() }}` so failed dependencies cannot skip it")
    else:
        needs_match = re.search(r"^    needs:\s*\[([^]]+)\]\s*$", gate, re.MULTILINE)
        if not needs_match:
            problems.append("ci-gate must declare its complete dependency list inline")
        else:
            observed_needs = {item.strip() for item in needs_match.group(1).split(",")}
            expected_needs = set(blocks) - {"ci-gate"}
            if observed_needs != expected_needs:
                problems.append(
                    "ci-gate dependencies do not match all real jobs: "
                    f"expected {sorted(expected_needs)}, observed {sorted(observed_needs)}"
                )

    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("workflow", nargs="?", type=Path, default=Path(".github/workflows/ci.yml"))
    args = parser.parse_args()
    problems = check_workflow(args.workflow)
    if problems:
        for problem in problems:
            print(f"CI_POLICY_ERROR: {problem}")
        return 1
    print(f"CI policy OK: {args.workflow}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
