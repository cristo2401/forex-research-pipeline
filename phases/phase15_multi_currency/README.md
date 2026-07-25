# Phase 15 - Multi Divisa Benchmark 10->7

Orquestador batch leak-safe para evaluar 10 pares FX con `phase13 -> phase14`, aplicar costos conservadores por símbolo y seleccionar 7 finalistas.

## Comando principal

```bash
python phase15/run_phase15_multi_fx.py \
  --input_root ./data/parquet_all \
  --out_dir ./phases/phase15_multi_currency/output_runs/phase15_multi_YYYYMMDD \
  --strict_mode 1
```

## Canasta default (10)

`EURUSD, GBPUSD, USDJPY, AUDUSD, NZDUSD, USDCAD, USDCHF, EURJPY, EURGBP, GBPJPY`

## Costos base default (pips)

- EURUSD 0.9
- GBPUSD 1.1
- USDJPY 0.9
- AUDUSD 1.0
- NZDUSD 1.2
- USDCAD 1.1
- USDCHF 1.1
- EURJPY 1.2
- EURGBP 1.0
- GBPJPY 1.5

## Ranking y selección

- Métrica primaria: `hardened_test_metrics_nooverlap.x1.sharpe_ann`
- Desempates: `x2.sharpe_ann` y `x1.max_drawdown` (menos negativo mejor)
- Filtro de calidad: `x1.sharpe_ann > 0` y `x1.cum_net_return > 0`
- Guardrail informativo: `x2.sharpe_ann > 0` y `x2.cum_net_return > 0`
- Si faltan pares para 7: completa por mejor `x1.sharpe_ann` con `borderline=true`

## Artefactos

- `phase15_symbol_results.csv`
- `phase15_ranking_top10.csv`
- `phase15_selected7.csv`
- `phase15_failures.json`
- `phase15_summary.json`
- `phase15_manifest.json`
- `phase13_runs/` y `phase14_runs/` por símbolo
- `logs/` con stdout/stderr por etapa

## Seguridad

- Solo paper, sin órdenes reales
- Verificación obligatoria de leak audit por símbolo (`all_eval_ts_gt_prior_max=true`)
- Validación estricta con `strict_mode=1` usa:
  - `strict_temporal_eval=1`
  - `strict_only=1`
  - `purged_gap_bars=24`
  - `eval_stride=24`
