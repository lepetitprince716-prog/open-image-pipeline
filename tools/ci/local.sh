#!/usr/bin/env bash
# Run the hosted CI contract on this Mac. Does not use GitHub Actions minutes.
#
#   tools/ci/local.sh              # default: policy + python + validate (what drawing PRs need)
#   tools/ci/local.sh changed      # policy, then only the lanes the working-tree diff selects
#                                  # (same rule table as hosted CI: tools/ci/changed_lanes.py)
#   tools/ci/local.sh all          # every lane: policy, python, validate, rust, web, web-served,
#                                  # chat, comfyui
#   tools/ci/local.sh python       # one lane; also policy, validate, rust, web, web-served, chat,
#                                  # comfyui, schemas (only the AJV compile step of the web lane)
#   CI_FULL=1 tools/ci/local.sh    # include the 10x stability loop
#   CI_MATCH_HOSTED=1 tools/ci/local.sh  # sync the venv with the hosted interpreter (3.14.6)
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

run_uv_install() {
  # Dependency synchronization may hit the network; test commands inherit the armed guard.
  LORAMAKE_CI_BLOCK_NETWORK=0 "${UV[@]}" "$@"
}

run_uv() {
  "${UV[@]}" "$@"
}

uv_sync() {
  local args=(sync --locked --extra dev)
  if [[ "${CI_MATCH_HOSTED:-0}" == "1" ]]; then
    args+=(--python 3.14.6)
  fi
  run_uv_install "${args[@]}"
}

lane="${1:-default}"

policy() {
  python3 tools/ci/check_workflow.py
  python3 tools/ci/check_nsfw_containment.py
  python3 tools/ci/check_doc_links.py
  git diff --check
}

python_lane() {
  if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
    echo "warn: ffmpeg/ffprobe missing; video tests may skip" >&2
  fi
  uv_sync
  run_uv lock --check
  run_uv pip check
  run_uv run --locked --extra dev ruff check scripts/ app/ tests/ loramake/ tools/ci/
  mkdir -p .ci-artifacts
  run_uv run --locked --extra dev python -m pytest -q -ra --durations=25 \
    --junitxml=.ci-artifacts/pytest.xml
  run_uv run --locked --extra dev python tools/ci/check_pytest_report.py \
    .ci-artifacts/pytest.xml --allow-missing-expected
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
  uv_sync
  run_uv run --locked --extra dev python scripts/art_directions.py validate
  run_uv run --locked --extra dev python scripts/gen_strategy.py validate
  run_uv run --locked --extra dev python scripts/script_registry.py validate
  # The tree-wide invariants hosted CI runs in its always-on `changes` job.
  run_uv run --locked --extra dev python -m pytest -q -p no:cacheprovider \
    tests/test_repo_hygiene.py \
    tests/test_mirror_manifest_policy.py \
    tests/test_nsfw_containment_policy.py
}

rust_lane() {
  cargo fmt --manifest-path tools/rdbg/Cargo.toml --all -- --check
  cargo clippy --locked --manifest-path tools/rdbg/Cargo.toml --all-targets -- -D warnings
  cargo test --locked --manifest-path tools/rdbg/Cargo.toml --all-targets
}

ajv_compile() {
  pnpm -C app-web exec ajv compile \
    --spec=draft2020 --strict=true -c ajv-formats \
    -s ../scripts/schema/anima_step2a_page_index_probe_v1.schema.json \
    -s ../scripts/schema/anima_step2a_page_inventory_v1.schema.json \
    -s ../scripts/schema/anima_step2a_intake_manifest_v1.schema.json \
    -s ../scripts/schema/anima_step2a_removal_ledger_v1.schema.json \
    -s ../scripts/schema/script_registry_v1.schema.json
}

schemas() {
  pnpm -C app-web install --frozen-lockfile
  ajv_compile
}

web() {
  pnpm -C app-web install --frozen-lockfile
  ajv_compile
  pnpm -C app-web typecheck
  pnpm -C app-web test
  pnpm -C app-web build
}

web_served() {
  [[ -d app-web/out ]] || { echo "error: run the web lane first (need app-web/out)" >&2; exit 1; }
  mkdir -p .ci-artifacts
  REQUIRE_WEB_EXPORT=1 run_uv run --locked --extra dev python -m pytest tests/test_app_m2.py -q -ra \
    --junitxml=.ci-artifacts/web-served-api.xml
  run_uv run --locked --extra dev python tools/ci/check_pytest_report.py \
    .ci-artifacts/web-served-api.xml --expected-skips 0
}

chat() {
  pnpm -C app-chat install --frozen-lockfile
  pnpm -C app-chat typecheck
  pnpm -C app-chat test
}

comfyui() {
  GOVERNANCE_CHECK_MODE=head zsh comfyui-boxa/scripts/run-all-checks.zsh
}

changed_paths() {
  # Everything that differs from main: committed on this branch, staged, unstaged, untracked.
  # A rename in `git status` reads "old -> new"; keep the new name.
  local base
  if ! base="$(git merge-base origin/main HEAD 2>/dev/null || git merge-base main HEAD)"; then
    echo "error: no merge base with origin/main or main" >&2
    return 2
  fi
  {
    git diff --name-only "$base" HEAD
    git status --porcelain --untracked-files=all | cut -c4- | sed -E 's/^.* -> //'
  } | sort -u
}

changed() {
  policy
  local assignments
  assignments="$(changed_paths | python3 tools/ci/changed_lanes.py --paths-from - --format shell --explain)"
  # The selector prints only LANE_<NAME>=true|false lines in shell format; refuse anything else.
  if grep -Ev '^LANE_[A-Z]+=(true|false)$' <<<"$assignments" | grep -q .; then
    echo "error: unexpected selector output:" >&2
    echo "$assignments" >&2
    exit 2
  fi
  eval "$assignments"
  local ran=0
  if [[ "${LANE_PYTHON:-false}" == "true" ]]; then
    python_lane
    validate
    ran=1
  fi
  if [[ "${LANE_WEB:-false}" == "true" ]]; then
    web
    web_served
    ran=1
  fi
  if [[ "${LANE_CHAT:-false}" == "true" ]]; then
    chat
    ran=1
  fi
  if [[ "${LANE_RUST:-false}" == "true" ]]; then
    rust_lane
    ran=1
  fi
  if [[ "${LANE_COMFYUI:-false}" == "true" ]]; then
    comfyui
    ran=1
  fi
  if [[ "$ran" == "0" ]]; then
    echo "no lane selected by the changed paths; policy only"
  fi
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
    web
    web_served
    chat
    comfyui
    ;;
  changed) changed ;;
  policy) policy ;;
  python) python_lane ;;
  validate) validate ;;
  rust) rust_lane ;;
  web) web ;;
  web-served) web_served ;;
  chat) chat ;;
  comfyui) comfyui ;;
  schemas) schemas ;;
  *)
    echo "usage: $0 [default|changed|all|policy|python|validate|rust|web|web-served|chat|comfyui|schemas]" >&2
    exit 2
    ;;
esac

echo "local CI OK ($lane)"
