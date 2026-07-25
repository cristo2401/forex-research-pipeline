#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_root"

python "./phases/phase18_5_dynamic_leverage/run_phase18_5_dynamic_leverage_ctrader.py" \
  --out_dir "./phases/phase18_5_dynamic_leverage/output_phase18_5_final_main_v3c" \
  --run_phase18_2 1 --skip_if_exists 0 --policy model_ctde_robust --cost_mult 1 \
  --max_gross_leverage 6.0 --target_gross_leverage 3.0 --min_target_mult 0.2 --max_target_mult 2.2 \
  --loss_w_turnover 0.19 --cost_per_turnover 0.00009 --turnover_smooth_alpha 0.30 \
  --exec_rebalance_deadband 0.24 --exec_rebalance_cooldown_bars 0 --exec_min_turnover 0.22 \
  --gate_turnover_ratio_max 0.85 --robust_score_turn_penalty 1.0 --robust_score_dd_penalty 0.28 \
  --robust_score_trade_penalty 0.35 --robust_score_rebalance_penalty 0.20 \
  --robust_alpha_candidates "0.15,0.25,0.35,0.45" \
  --account_equity_usd 3000 --account_leverage_cap 10000 --max_margin_utilization 0.35 \
  --leverage_tiers "1000000:500,5000000:200,inf:100" --extra_cost_per_turnover 0.00002 --slippage_per_turnover 0.00002 \
  --shield_dd_limit -0.08 --shield_risk_mult 0.55 --min_exec_scale 0.05
