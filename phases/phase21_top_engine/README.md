# Phase21 - Top Engine CLI (E2E)

CLI robusto para ejecutar el motor portfolio top (`phase16 -> phase18.2 -> phase18.5`) con:
- export de checkpoint inferible,
- backtest reciente 90/120 días en modo frozen,
- servicio paper-live H1 con guardrails.

## Comandos

```bash
source .venv/bin/activate

# 1) Preparar checkpoint congelado
python ./phases/phase21_top_engine/run_phase21_top_engine_cli.py \
  --config ./phases/phase21_top_engine/config_phase21.json \
  prepare-checkpoint --data_source ctrader

# 2) Backtest reciente (sobre run ya preparado)
python ./phases/phase21_top_engine/run_phase21_top_engine_cli.py \
  --config ./phases/phase21_top_engine/config_phase21.json \
  backtest-recent --run_dir <RUN_DIR_PHASE21> --days 90,120 --data_source ctrader

# 3) Full-run (prepare + backtests)
python ./phases/phase21_top_engine/run_phase21_top_engine_cli.py \
  --config ./phases/phase21_top_engine/config_phase21.json \
  full-run --days 90,120 --data_source ctrader

# 4) Paper-live H1
python ./phases/phase21_top_engine/run_phase21_top_engine_cli.py \
  --config ./phases/phase21_top_engine/config_phase21.json \
  paper-live --run_dir <RUN_DIR_PHASE21>

# 5) Estado consolidado
python ./phases/phase21_top_engine/run_phase21_top_engine_cli.py \
  --config ./phases/phase21_top_engine/config_phase21.json \
  status --run_dir <RUN_DIR_PHASE21>
```

## Artefactos principales

Cada corrida usa:
- `phase21_manifest.json`
- `phase21_status.json`
- `phase21_alerts.jsonl`
- `phase21_guardrails.json`
- `phase21_snapshot.json`
- `links_to_child_runs.json`
- `checkpoint/phase21_top_engine_checkpoint.pt`

Y, para backtests:
- `backtest_90d/summary.json`
- `backtest_120d/summary.json`

Y, para live:
- `paper_live/state.json`
- `paper_live/decisions.parquet`
