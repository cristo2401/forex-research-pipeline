# Phase 18.5 - Dynamic Leverage (cTrader-aware)

`phase18.5` adds a cTrader-like execution layer on top of `phase18.2` CTDE outputs.

## Goal

- Reuse `phase18.2` as base model.
- Simulate dynamic leverage/margin behavior closer to cTrader:
  - leverage tiers by USD exposure,
  - margin utilization cap,
  - additional execution friction (extra cost + slippage),
  - drawdown shield scaling.

## Main script

- `run_phase18_5_dynamic_leverage_ctrader.py`

## Run

```bash
source .venv/bin/activate

python "phases/phase18_5_dynamic_leverage/run_phase18_5_dynamic_leverage_ctrader.py" \
  --run_phase18_2 1 \
  --policy model_ctde_robust \
  --cost_mult 1 \
  --max_gross_leverage 3.5 \
  --target_gross_leverage 1.8 \
  --leverage_tiers "1000000:500,5000000:200,inf:100" \
  --max_margin_utilization 0.35 \
  --account_equity_usd 3000
```

## Outputs

In `out_dir` (default: `output_phase18_5_dynamic_ctrader_v1`):

- `phase18_5_summary.json`
- `phase18_5_metrics_comparison.csv`
- `phase18_5_ctrader_execution_steps.parquet`
- `phase18_5_ctrader_execution_steps.csv`
- `phase18_5_equity_compare.png`
- `phase18_5_margin_utilization.png`
- `phase18_5.log`

## Notes

- This phase is paper/backtest only. No live orders are sent.
- Tiers are configurable because broker settings can differ.
- The script also records official cTrader docs used to define execution assumptions.
