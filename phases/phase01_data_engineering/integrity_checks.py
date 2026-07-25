#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import logging
import re
import sys
from pathlib import Path
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm
from fx_universe import normalize_symbol_from_filename

# Configuración de Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("IntegrityChecker")

@dataclass
class IntegrityResult:
    file: str
    ok: bool
    issues: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

@dataclass(frozen=True)
class TimeframeSpec:
    raw: str
    minutes: int
    folder_tag: str

def parse_timeframe(raw: str) -> TimeframeSpec:
    token = re.sub(r"\s+", "", raw.strip().lower())
    unit_aliases = {
        "m": "m",
        "min": "m",
        "mins": "m",
        "minute": "m",
        "minutes": "m",
        "minuto": "m",
        "minutos": "m",
        "h": "h",
        "hr": "h",
        "hrs": "h",
        "hour": "h",
        "hours": "h",
        "hora": "h",
        "horas": "h",
    }
    m = re.fullmatch(r"(\d+)([a-z]+)", token)
    if not m:
        raise ValueError(f"Formato de timeframe inválido: '{raw}'. Ejemplos: 10min, 20m, 1h, 4h.")

    n = int(m.group(1))
    unit = unit_aliases.get(m.group(2))
    if n <= 0 or unit is None:
        raise ValueError(f"Timeframe inválido: '{raw}'.")

    minutes = n if unit == "m" else n * 60
    if minutes % 60 == 0:
        folder_tag = f"h{minutes // 60}"
    else:
        folder_tag = f"m{minutes}"
    return TimeframeSpec(raw=raw, minutes=minutes, folder_tag=folder_tag)

def parse_symbols_filter(raw: str | None) -> set[str] | None:
    if raw is None:
        return None
    token = raw.strip()
    if token == "":
        return None

    items: list[str]
    if token.startswith("[") and token.endswith("]"):
        try:
            parsed = ast.literal_eval(token)
        except Exception:
            parsed = token
        if isinstance(parsed, (list, tuple, set)):
            items = [str(x) for x in parsed]
        else:
            items = [str(parsed)]
    else:
        items = re.split(r"[,;\s]+", token)

    symbols = {
        normalize_symbol_from_filename(s.strip())
        for s in items
        if s.strip()
    }
    return symbols or None

# -----------------------------
# Core de Validación
# -----------------------------

def check_one(path: Path, expected_m1: int, eps: float = 1e-7) -> IntegrityResult:
    """
    Realiza una auditoría profunda de un archivo H1 utilizando vectorización.
    """
    try:
        # Cargamos el archivo. En H1 el tamaño es manejable en RAM.
        df = pq.read_table(str(path)).to_pandas()
    except Exception as e:
        return IntegrityResult(path.name, False, [f"Error de lectura: {e}"])

    issues = []

    # 1. Verificación de Columnas Esenciales
    required = ["timestamp", "open", "high", "low", "close", "n_m1"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return IntegrityResult(path.name, False, [f"Faltan columnas: {missing}"])

    # 2. Integridad Temporal
    ts = pd.to_datetime(df["timestamp"])
    if ts.isna().any():
        issues.append(f"NaT en timestamp: {ts.isna().sum()}")

    if not ts.is_monotonic_increasing:
        issues.append("No es monotónico creciente")

    dups = ts.duplicated().sum()
    if dups > 0:
        issues.append(f"Timestamps duplicados: {dups}")

    # 3. Lógica de Precios (OHLC)
    # Usamos NumPy para máxima velocidad
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values

    # Precios <= 0 (Flash crashes o errores de sensor/data)
    if (o <= 0).any() or (c <= 0).any() or (l <= 0).any():
        issues.append("Detectados precios cero o negativos")

    # Coherencia OHLC: High debe ser el máximo y Low el mínimo (con epsilon)
    bad_h = (h < np.maximum(o, c) - eps)
    bad_l = (l > np.minimum(o, c) + eps)
    bad_hl = (h < l)

    total_bad_ohlc = np.logical_or.reduce((bad_h, bad_l, bad_hl)).sum()
    if total_bad_ohlc > 0:
        issues.append(f"Filas con OHLC incoherente: {total_bad_ohlc}")

    # 4. Validaciones de Ingeniería de Datos (Fase 1)
    n_m1 = df["n_m1"].values
    bad_n = (n_m1 < 1) | (n_m1 > expected_m1)
    if bad_n.any():
        issues.append(f"n_m1 fuera de rango [1,{expected_m1}]: {bad_n.sum()}")

    if "gap_bars" in df.columns:
        gaps = pd.to_numeric(df["gap_bars"], errors="coerce").fillna(1).values
        if (gaps < 1).any():
            issues.append(f"gap_bars < 1 detectado: {(gaps < 1).sum()}")
    elif "gap_hours" in df.columns:
        gaps = pd.to_numeric(df["gap_hours"], errors="coerce").fillna(1).values
        if (gaps <= 0).any():
            issues.append(f"gap_hours <= 0 detectado: {(gaps <= 0).sum()}")
    else:
        issues.append("No existe columna gap_bars ni gap_hours")
        gaps = np.array([np.nan], dtype=float)

    # 5. Estadísticas de Salud
    stats = {
        "rows": len(df),
        "completeness": (n_m1 == expected_m1).mean() * 100,
        "avg_gap": gaps.mean()
    }

    return IntegrityResult(path.name, len(issues) == 0, issues, stats)

# -----------------------------
# Main y Reporte
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Auditor de Integridad para dataset resampleado desde M1")
    ap.add_argument("--dataset_dir", required=True, help="Ruta base del dataset")
    ap.add_argument("--timeframe", default="1h", help="Ejemplos: 10min, 20m, 1h, 4h")
    ap.add_argument("--symbols", default="", help="Opcional. Lista separada por coma/espacio. Ej: EURUSD,GBPUSD")
    ap.add_argument("--detailed", action="store_true", help="Muestra todos los problemas")
    args = ap.parse_args()

    try:
        timeframe = parse_timeframe(args.timeframe)
        selected_symbols = parse_symbols_filter(args.symbols)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    ds_path = Path(args.dataset_dir) / f"parquet_{timeframe.folder_tag}"
    if not ds_path.exists():
        logger.error(f"Error: No existe la carpeta {ds_path}")
        sys.exit(1)

    files = sorted(list(ds_path.glob("*.parquet")))
    if selected_symbols is not None:
        files = [f for f in files if normalize_symbol_from_filename(f.name) in selected_symbols]
        logger.info("Filtro de símbolos: %s", ",".join(sorted(selected_symbols)))
    else:
        logger.info("Filtro de símbolos: ALL")

    if not files:
        logger.error("No hay archivos para auditar con los filtros actuales.")
        sys.exit(1)

    logger.info(f"Iniciando auditoría de {len(files)} archivos...\n")

    results = []
    for f in tqdm(files, desc="Check", unit="file"):
        results.append(check_one(f, expected_m1=timeframe.minutes))

    # Resumen Final
    print("\n" + "="*50)
    print("RESUMEN DE INTEGRIDAD")
    print("="*50)

    fails = [r for r in results if not r.ok]

    if not fails:
        print("✅ TODOS LOS ARCHIVOS PASARON LA PRUEBA.")
    else:
        print(f"❌ SE ENCONTRARON PROBLEMAS EN {len(fails)} ARCHIVOS:")
        for r in fails:
            print(f"\n[!] {r.file}")
            for issue in r.issues:
                print(f"    - {issue}")

    print("="*50)

    if fails:
        sys.exit(1)

if __name__ == "__main__":
    main()
