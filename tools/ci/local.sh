#!/usr/bin/env bash
# Run the hosted CI contract on this Mac. Does not use GitHub Actions minutes.
#
#   tools/ci/local.sh           # default: policy + python + validate (what drawing PRs need)
#   tools/ci/local.sh all       # also rust, web, schemas, comfyui harness
#   tools/ci/local.sh python    # one lane
#   CI_FULL=1 tools/ci/local.sh # include the 10x stability loop
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 TZ=UTC
export LORAMAKE_CI_BLOCK_NETWORK="${LORAMAKE_CI_BLOCK_NETWORK:-1}"
export PYTHONPATH="${ROOT}/tools/ci_network_guard${PYTHONPATH:+:$PYTHONPATH}"

UV=(uv)
if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is required (https://docs.astral.sh/uv/)" >&2
  exit 2
fi

run_uv() {
  # Installs may hit the network; tests stay behind the guard.
  LORAMAKE_CI_BLOCK_NETWORK=0 "${UV[@]}" "$@"
}

lane="${1:-default}"

policy() {
  python3 tools/ci/check_workflow.py
  git diff --check
}

python_lane() {
  if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
    echo "warn: ffmpeg/ffprobe missing; video tests may skip" >&2
  fi
  run_uv sync --locked --extra dev
  run_uv lock --check
  run_uv pip check
  run_uv run --locked --extra dev ruff check scripts/ app/ tests/ loramake/ tools/ci/
  mkdir -p .ci-artifacts
  run_uv run --locked --extra dev python -m pytest -q -ra --durations=25 \
    --junitxml=.ci-artifacts/pytest.xml
  run_uv run --locked --extra dev python tools/ci/check_pytest_report.py .ci-artifacts/pytest.xml
  run_uv run --locked --extra dev python scripts/contract_drift.py --check
  if [[ "${CI_FULL:-0}" == "1" ]]; then
    for attempt in {1..10}; do
      echo "stability attempt ${attempt}/10"
      run_uv run --locked --extra dev python -m pytest -q --maxfail=1 --disable-warnings \
        tests/test_app_m1.py::test_cancel_endpoint \
        tests/test_app_m1.py::test_connect_leaves_a_real_database_in_wal \
        tests/test_app_jobs_attempts.py::test_cancel_terminates_the_whole_local_process_group_within_two_seconds \
        tests/test_quality_gen_composition.py::test_composition_literal_text_allowed_with_no_wildcard
    done
  fi
}

validate() {
  run_uv sync --locked --extra dev
  run_uv run --locked --extra dev python scripts/art_directions.py validate
  run_uv run --locked --extra dev python scripts/gen_strategy.py validate
  run_uv run --locked --extra dev python scripts/script_registry.py validate
}

rust_lane() {
  cargo fmt --manifest-path tools/rdbg/Cargo.toml --all -- --check
  cargo clippy --locked --manifest-path tools/rdbg/Cargo.toml --all-targets -- -D warnings
  cargo test --locked --manifest-path tools/rdbg/Cargo.toml --all-targets
}

schemas() {
  pnpm -C app-web install --frozen-lockfile
  pnpm -C app-web exec ajv compile \
    --spec=draft2020 --strict=true -c ajv-formats \
    -s ../scripts/schema/anima_step2a_page_index_probe_v1.schema.json \
    -s ../scripts/schema/anima_step2a_page_inventory_v1.schema.json \
    -s ../scripts/schema/anima_step2a_intake_manifest_v1.schema.json \
    -s ../scripts/schema/anima_step2a_removal_ledger_v1.schema.json \
    -s ../scripts/schema/script_registry_v1.schema.json
}

web() {
  pnpm -C app-web install --frozen-lockfile
  pnpm -C app-web typecheck
  pnpm -C app-web build
}

web_served() {
  [[ -d app-web/out ]] || { echo "error: run the web lane first (need app-web/out)" >&2; exit 1; }
  REQUIRE_WEB_EXPORT=1 run_uv run --locked --extra dev python -m pytest tests/test_app_m2.py -q -ra \
    --junitxml=.ci-artifacts/web-served-api.xml
  run_uv run --locked --extra dev python tools/ci/check_pytest_report.py \
    .ci-artifacts/web-served-api.xml --expected-skips 0
}

comfyui() {
  GOVERNANCE_CHECK_MODE=head zsh comfyui-z3/scripts/run-all-checks.zsh
}

case "$lane" in
  default)
    policy
    python_lane
    validate
    ;;
  all)
    policy
    python_lane
    validate
    rust_lane
    schemas
    web
    web_served
    comfyui
    ;;
  policy) policy ;;
  python) python_lane ;;
  validate) validate ;;
  rust) rust_lane ;;
  schemas) schemas ;;
  web) web ;;
  web-served) web_served ;;
  comfyui) comfyui ;;
  *)
    echo "usage: $0 [default|all|policy|python|validate|rust|schemas|web|web-served|comfyui]" >&2
    exit 2
    ;;
esac

echo "local CI OK ($lane)"
