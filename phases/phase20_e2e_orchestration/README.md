# Phase20 - CLI Robusto E2E + Paper Live

Orquesta dos procesos separados:

1. `live-h1`: daemon paper-live (reutiliza phase19).
2. `nightly-hardening`: corrida nocturna strict (phase13 + sanity + phase14) y emite guardrails.
3. `status`: health consolidado (live + nightly + guardrails).

## Config

Archivo principal:

- `config_phase20.json`

v1 soporta solo `GBPUSD`.

## Uso

### Live H1

```bash
./phases/phase20_e2e_orchestration/run_phase20_live.sh
```

One-shot file-mode:

```bash
./phases/phase20_e2e_orchestration/run_phase20_live.sh \
  --once 1 --loop 0 --data_source file \
  --input_m1_file "./phases/phase13_pipeline_cli/sample_gbpusd_m1_2024q1.parquet"
```

### Nightly hardening

```bash
./phases/phase20_e2e_orchestration/run_phase20_nightly.sh
```

### Status

```bash
python "./phases/phase20_e2e_orchestration/run_phase20_orchestrator.py" \
  --config "./phases/phase20_e2e_orchestration/config_phase20.json" \
  status
```

## Salidas

Root por defecto:

- `./phases/phase20_e2e_orchestration/output_runs`

Archivos clave:

- `nightly_runs/<run_id>/phase20_manifest.json`
- `nightly_runs/<run_id>/phase20_status.json`
- `nightly_runs/<run_id>/phase20_alerts.jsonl`
- `nightly_runs/<run_id>/phase20_live_guardrails.json`
- `nightly_runs/<run_id>/phase20_status_snapshot.json`
- `shared/phase20_live_guardrails.json` (consumido por live)
- `live_service/` (estado y decisiones de phase19)
- `phase20_status_snapshot.json` (consolidado)

## Seguridad

- `paper_only=true`
- No envio de ordenes reales.
- Guardrail-aware: si guardrail nightly falla o esta stale (>36h), `live-h1` entra en `DEGRADED`.
