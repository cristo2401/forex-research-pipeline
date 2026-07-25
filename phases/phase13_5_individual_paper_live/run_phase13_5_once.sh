#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_root"

python "./phases/phase13_5_individual_paper_live/run_phase13_5_paper_live.py" \
  --config "./phases/phase13_5_individual_paper_live/config_phase13_5.json" \
  --once 1 --loop 0 "$@"
