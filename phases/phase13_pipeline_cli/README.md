# Phase13 Pipeline CLI

Orquestador experimental de inferencia con modelos congelados:

`M1 crudo -> H1 -> segmentos -> embeddings+clusters -> decision -> leverage RL (paper)`

## Requisitos

- Entorno virtual activo con las dependencias de `requirements.txt`.
- Bundles y modelos de las fases previas disponibles localmente. No se incluyen en
  el repositorio público; este comando no los entrena ni descarga automáticamente.
- Symbol v1: `GBPUSD`
- Input: CSV o Parquet con columnas minimas:
  - `timestamp` (o `dt`), `open`, `high`, `low`, `close`
  - `volume` es opcional

## Uso

```bash
python "phases/phase13_pipeline_cli/run_phase13_pipeline.py" \
  --input_m1 "/ruta/al/input_m1.parquet" \
  --symbol GBPUSD \
  --cost_pips 1.2 \
  --mode infer_frozen \
  --out_dir "./phases/phase13_pipeline_cli/output_runs"
```

## Modos

- `infer_frozen` (default): usa bundles/modelos congelados.
- `retrain_full`: el nombre está reservado, pero **v1 ejecuta inferencia congelada y
  no reentrena los modelos**. El runner deja constancia de esta limitación en el
  manifiesto. No es una ruta de reentrenamiento completo.

## Costos

- `--cost_pips` (default `1.2`): costo base en pips aplicado en stage04/stage05 y reportado en `phase13_summary.json` + `phase13_leak_audit.json`.

## Salidas por corrida

- `manifest.json`
- `status.json`
- `stage01_h1.parquet`
- `stage02_segments_catalog.parquet`
- `stage02_segments_long.parquet`
- `stage03_embeddings.npz`
- `stage03_cluster_assignments.parquet`
- `stage04_phase10_predictions.parquet`
- `stage05_phase12_predictions.parquet`
- `phase13_summary.json`
- `phase13_equity.png`
- `phase13.log`

## Seguridad

- `paper_only=true`
- No envio de ordenes reales
- No uso de credenciales broker
