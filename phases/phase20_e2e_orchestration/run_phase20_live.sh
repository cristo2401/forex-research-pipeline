#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_root"

python './phases/phase20_e2e_orchestration/run_phase20_orchestrator.py' \
  --config './phases/phase20_e2e_orchestration/config_phase20.json' \
  live-h1 "$@"
