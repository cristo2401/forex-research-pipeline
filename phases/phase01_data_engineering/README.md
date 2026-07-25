# Fase 1 — Data engineering M1 → H1 (Dataset_H1_v1)

Este paquete crea un **dataset H1** (desde tus parquets M1) con:
- Resample OHLC correcto (open=primero, high=max, low=min, close=último)
- Manejo de gaps **sin inventar velas**: agrega flags (`n_m1`, `is_complete`, `gap_hours`)
- Alineación temporal (UTC recomendado)
- `manifest.json` con rangos, schema y checksums
- `data_quality_report.md` con auditoría e integridad
- `calendar_master.parquet` (índice horario maestro) + `availability_masks.parquet` (máscaras por asset)

> **Input:** `./data/parquet_all`
> **Output recomendado:** `./data/Dataset_H1_v1`

---

## 1) Requisitos
```bash
python3 -V
python3 -c "import pandas, pyarrow; print('ok')"
```

---

## 2) Ejecución (MVP: solo FX)
```bash
python3 build_h1_dataset.py \
  --input_dir ./data/parquet_all \
  --output_dir ./data/Dataset_H1_v1 \
  --timezone UTC \
  --fx_only 1 \
  --n_jobs 6
```

### Parámetros útiles
- `--fx_only 1`: filtra a pares FX (6 letras, base/quote en códigos ISO de FX; excluye XAU/XAG/SPX/WTI…)
- `--n_jobs`: paraleliza por archivo (assets independientes)
- `--force 1`: rehace outputs existentes
- `--no_calendar 1`: si no quieres generar calendar/masks (más rápido)

---

## 3) Estructura de salida
```
Dataset_H1_v1/
  parquet_h1/
    eurusd.parquet
    gbpusd.parquet
    ...
  calendar_master.parquet
  availability_masks.parquet
  manifest.json
  data_quality_report.md
  logs/
    build.log
```

---

## 4) Columnas del H1 (mínimas)
- `timestamp` (UTC, datetime)
- `open`, `high`, `low`, `close` (float)
- `volume` (si existe en M1, se suma)
- `n_m1` (int): cuántas velas M1 aportaron a esa hora
- `is_complete` (bool): `n_m1 == 60`
- `gap_hours` (int): horas desde la última barra existente
- `is_gap` (bool): `gap_hours > 1`

---

## 5) Integrity checks (qué valida)
- monotonicidad temporal y duplicados
- OHLC coherente (high>=max(open,close) y low<=min(open,close))
- `n_m1` en [1..60] (o NaN si la hora no existe)
- gap detection consistente

---

Si quieres, el siguiente paso es **Fase 2 (TokenSpec)** usando este H1 como base.
