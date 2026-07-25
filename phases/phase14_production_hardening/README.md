# Phase14 - Hardening Produccion

Objetivo: endurecer la estrategia para reducir sobreoperacion y validar robustez con costos duros.

Incluye:
- Reduccion de turnover con politica anti-overtrading (histeresis + min hold + limite trades/hora)
- Revalidacion con costos `x1/x2/x3`
- Walk-forward extendido por fold temporal + metricas por regimen (`cluster_id`)
- Kill switch y limites de riesgo:
  - drawdown diario maximo
  - max trades por hora
  - max leverage
- Runner E2E robusto (`phase13 -> phase14`) con reintentos, logs y alertas en JSONL

## Scripts

- `run_phase14_1_hardening.py`
  - Entrada: una corrida de `phase13` (`stage05_phase12_predictions.parquet`)
  - Salida: artefactos de hardening + validacion robusta

- `run_phase14_2_robust_executor.py`
  - Orquesta `phase13` y luego `phase14` con retries y alertas.

## Uso rapido

### 1) Hardening sobre una corrida phase13 ya existente

```bash
python "phases/phase14_production_hardening/run_phase14_1_hardening.py" \
  --phase13_run_dir "./phases/phase13_pipeline_cli/output_runs/phase13_full_gbpusd_m1_phase1_dedup_leaksafe_v1" \
  --strict_only 1 \
  --cost_pips 1.2
```

### 2) Runner E2E robusto (phase13 + phase14)

```bash
python "phases/phase14_production_hardening/run_phase14_2_robust_executor.py" \
  --input_m1 "./phases/phase13_pipeline_cli/gbpusd_20000530_20251205_dedup_from_phase1.parquet" \
  --symbol GBPUSD \
  --strict_temporal_eval 1 \
  --priors_source windows_only \
  --max_retries 3
```

## Artefactos principales (run_phase14_1)

- `phase14_summary.json`
- `phase14_param_search.csv`
- `phase14_cost_validation_full.csv`
- `phase14_cost_validation_test.csv`
- `phase14_walkforward_folds.csv`
- `phase14_walkforward_regime_metrics.csv`
- `phase14_recent_period_metrics.csv`
- `phase14_policy_strict_predictions.parquet`
- `phase14_equity_compare_x1.png`
- `phase14.log`

## Alertas y seguridad

- `run_phase14_2_robust_executor.py` escribe alertas en `phase14_alerts.jsonl`
- Este flujo es paper-only; no ejecuta ordenes reales.
