# Phase 13.5 - Paper Live Individual (Auto Lookback)

Orquestador robusto para arrancar paper-live individual desde hoy, con auto-probe de lookback mínimo.

## Flujo

1. Fetch M1 amplio (`max_probe_hours`) desde cTrader demo (o modo file).
2. Probe automático de lookback en múltiples ventanas (`probe_hours_csv`).
3. Corre `phase13` en cada ventana y evalúa:
   - `status == SUCCESS`
   - `n_segments >= min_segments`
   - `stage05` con filas
   - señal fresca: `age_h_vs_latest_m1 <= freshness_max_hours`
4. Selecciona lookback mínimo que pasa + `safety_buffer_hours`.
5. Arranca `phase19` paper-live individual con ese lookback.
   - Si no hay candidato fresco, puede arrancar en modo degradado conservador (`allow_degraded_start=1`).

## Comandos

One-shot (recomendado para validación inicial):

```bash
./phases/phase13_5_individual_paper_live/run_phase13_5_once.sh
```

Loop H1 continuo:

```bash
./phases/phase13_5_individual_paper_live/run_phase13_5_live.sh
```

CLI directo:

```bash
source .venv/bin/activate
python "./phases/phase13_5_individual_paper_live/run_phase13_5_paper_live.py" \
  --config "./phases/phase13_5_individual_paper_live/config_phase13_5.json"
```

## Salidas

Cada corrida crea:

- `config.json`
- `phase13_5.log`
- `data/<symbol>_probe_m1.parquet`
- `probe_results.csv`
- `probe_results.json`
- `phase13_probe_runs/`
- `live_service/` (salidas de phase19)
- `phase13_5_summary.json`

## Seguridad

- `paper_only`.
- No envía órdenes reales.
- Reintentos en fetch y phase13.
- Si no hay frescura suficiente, fuerza inicio degradado (si está habilitado) con límites de riesgo más bajos.
