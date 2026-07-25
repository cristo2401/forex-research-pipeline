# Phase 19 - Paper Live cTrader (IC Markets Demo)

Esta phase deja un servicio paper-live en H1:

1. Obtiene M1 desde cTrader Open API (demo).
2. Ejecuta `phase13` con modelos congelados.
3. Extrae la señal más reciente (`lev_phase12_2_frozen`).
4. Aplica risk-engine local (cap leverage, suavizado, kill-switch diario).
5. Guarda estado, decisiones, alertas y logs.

No envía órdenes reales. Es paper/shadow con datos de broker demo.

## Archivos

- `run_phase19_ctrader_oauth.py`: helper OAuth (URL + canje de `auth_code` a token).
- `run_phase19_fetch_m1_once.py`: fetch one-shot de barras M1.
- `run_phase19_paper_live.py`: daemon principal H1.
- `phase19_utils.py`: utilidades `.env`, JSON seguro, alertas.

## Setup

```bash
source .venv/bin/activate

python -m pip install ctrader-open-api service_identity
```

Copia `.env.example` a `.env` y completa `CTRADER_CLIENT_ID` / `CTRADER_CLIENT_SECRET`.
Define también `CTRADER_REDIRECT_URI` y asegúrate que coincida **exactamente** con el redirect configurado en la app de cTrader (incluyendo slash final), por ejemplo: `http://localhost/`.

## 1) OAuth token (una vez)

```bash
python "phases/phase19_ctrader_paper_live/run_phase19_ctrader_oauth.py" --print_only
```

Abre la URL, autoriza, toma `code` del redirect y canjea:

```bash
python "phases/phase19_ctrader_paper_live/run_phase19_ctrader_oauth.py" --auth_code "<CODE>"
```

## 2) Probar fetch M1

```bash
python "phases/phase19_ctrader_paper_live/run_phase19_fetch_m1_once.py" \
  --symbol GBPUSD \
  --env_file "./phases/phase19_ctrader_paper_live/.env" \
  --output "./phases/phase19_ctrader_paper_live/test_gbpusd_m1.parquet" \
  --demo 1
```

## 3) Correr daemon paper-live

One-shot (1 ciclo):

```bash
python "phases/phase19_ctrader_paper_live/run_phase19_paper_live.py" \
  --symbol GBPUSD \
  --env_file "./phases/phase19_ctrader_paper_live/.env" \
  --out_dir "./phases/phase19_ctrader_paper_live/output_phase19_live_v1" \
  --data_source ctrader \
  --once 1
```

Loop continuo alineado a H1:

```bash
python "phases/phase19_ctrader_paper_live/run_phase19_paper_live.py" \
  --symbol GBPUSD \
  --env_file "./phases/phase19_ctrader_paper_live/.env" \
  --out_dir "./phases/phase19_ctrader_paper_live/output_phase19_live_v1" \
  --data_source ctrader \
  --loop 1 \
  --align_hour 1 \
  --hour_delay_sec 20
```

## Salidas clave

En `out_dir`:

- `status.json`: estado actual del daemon.
- `state.json`: estado persistente paper (equity, lev, day pnl).
- `paper_decisions.csv`: historial de decisiones y PnL realizado.
- `alerts.jsonl`: alertas y errores.
- `phase19.log`: log de ejecución.
- `phase13_runs/`: runs completos por ciclo.

## Modo Guardrail-Aware (Phase20)

`run_phase19_paper_live.py` ahora soporta guardrails externos (sin romper compatibilidad):

- `--phase20_guardrail_enable 1`
- `--phase20_guardrail_file <path>`
- `--phase20_guardrail_stale_hours 36`
- `--phase20_degraded_max_abs_lev 0.30`
- `--phase20_degraded_max_delta_lev_per_step 0.10`

Si el guardrail está `FAILED`, falta, o está stale, el motor entra en `DEGRADED` y reduce límites efectivos de leverage/cambio.
