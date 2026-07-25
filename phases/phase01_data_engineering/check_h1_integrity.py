#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

# Configuración de Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("IntegrityChecker")

@dataclass
class IntegrityResult:
    file: str
    ok: bool
    issues: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

def check_one(path: Path, eps: float = 1e-7) -> IntegrityResult:
    try:
        df = pq.read_table(str(path)).to_pandas()
    except Exception as e:
        return IntegrityResult(path.name, False, [f"Error de lectura: {e}"])

    issues = []
    required = ["timestamp", "open", "high", "low", "close", "n_m1", "gap_hours"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return IntegrityResult(path.name, False, [f"Faltan columnas: {missing}"])

    ts = pd.to_datetime(df["timestamp"])
    if ts.isna().any():
        issues.append(f"NaT en timestamp: {ts.isna().sum()}")
    if not ts.is_monotonic_increasing:
        issues.append("No es monotónico creciente")
    dups = ts.duplicated().sum()
    if dups > 0:
        issues.append(f"Timestamps duplicados: {dups}")

    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    if (o <= 0).any() or (c <= 0).any() or (l <= 0).any():
        issues.append("Detectados precios cero o negativos")

    # Coherencia OHLC con tolerancia epsilon
    bad_h = (h < np.maximum(o, c) - eps)
    bad_l = (l > np.minimum(o, c) + eps)
    bad_hl = (h < l)

    total_bad_ohlc = np.logical_or.reduce((bad_h, bad_l, bad_hl)).sum()
    if total_bad_ohlc > 0:
        issues.append(f"Filas con OHLC incoherente: {total_bad_ohlc}")

    n_m1 = df["n_m1"].values
    bad_n = (n_m1 < 1) | (n_m1 > 60)
    if bad_n.any():
        issues.append(f"n_m1 fuera de rango [1,60]: {bad_n.sum()}")

    gaps = df["gap_hours"].values
    if (gaps < 1).any():
        issues.append(f"gap_hours < 1 detectado: {(gaps < 1).sum()}")

    stats = {
        "rows": len(df),
        "completeness": (n_m1 == 60).mean() * 100,
        "avg_gap": gaps.mean()
    }
    return IntegrityResult(path.name, len(issues) == 0, issues, stats)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--detailed", action="store_true")
    args = ap.parse_args()

    ds_path = Path(args.dataset_dir) / "parquet_h1"
    if not ds_path.exists():
        logger.error(f"No existe la carpeta {ds_path}")
        sys.exit(1)

    files = sorted(list(ds_path.glob("*.parquet")))
    results = [check_one(f) for f in tqdm(files, desc="Auditoría H1")]

    print("\n" + "="*50)
    print("RESUMEN DE INTEGRIDAD")
    print("="*50)

    fails = [r for r in results if not r.ok]
    if not fails:
        print("✅ TODOS LOS ARCHIVOS PASARON LA PRUEBA.")
    else:
        print(f"❌ PROBLEMAS EN {len(fails)} ARCHIVOS:")
        for r in fails:
            print(f"\n[!] {r.file}: {r.issues}")
    print("="*50)
    if fails: sys.exit(1)

if __name__ == "__main__":
    main()