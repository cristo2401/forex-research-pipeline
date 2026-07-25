#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_root"

python ./phases/phase21_top_engine/run_phase21_top_engine_cli.py \
  --config ./phases/phase21_top_engine/config_phase21.json \
  paper-live "$@"
