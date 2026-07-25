#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import hashlib
import logging
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from fx_universe import normalize_symbol_from_filename, is_fx_symbol
from timestamp_utils import detect_timestamp_column, coerce_timestamp, to_utc_naive

# Optional progress bars (no hard dependency)
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    def tqdm(x, **kwargs):  # type: ignore
        return x

# -----------------------------
# Helpers
# -----------------------------

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def pick_ohlc_columns(cols: list[str]) -> dict[str, str]:
    # Try common patterns
    lower = {c.lower(): c for c in cols}
    mapping: dict[str, str] = {}
    for k in ["open", "high", "low", "close", "volume"]:
        if k in lower:
            mapping[k] = lower[k]

    # Alternative names
    alt = {
        "open": ["o"],
        "high": ["h"],
        "low": ["l"],
        "close": ["c"],
        "volume": ["v", "tick_volume", "tickvol", "vol"],
    }
    for k, alts in alt.items():
        if k not in mapping:
            for a in alts:
                if a in lower:
                    mapping[k] = lower[a]
                    break

    # OHLC required
    for req in ["open", "high", "low", "close"]:
        if req not in mapping:
            raise ValueError(f"Missing required column '{req}'. Available columns: {cols}")
    return mapping

@dataclass
class AssetBuildStats:
    symbol: str
    input_file: str
    output_file: str
    input_rows: int
    output_rows: int
    start: str
    end: str
    pct_complete_hours: float
    n_gaps_total: int
    n_weekend_gaps: int
    n_data_gaps: int
    n_ohlc_invalid: int
    had_duplicates: bool
    was_unsorted: bool
    sha256: str

# -----------------------------
# Timeframe parsing
# -----------------------------

@dataclass(frozen=True)
class TimeframeSpec:
    raw: str
    minutes: int
    pandas_freq: str
    display_label: str
    folder_tag: str

def parse_optional_utc_datetime(raw: str | None, *, end_of_day_if_date: bool = False) -> pd.Timestamp | None:
    if raw is None:
        return None

    token = raw.strip()
    if token == "":
        return None

    has_time = bool(re.search(r"\d{1,2}:\d{2}", token))
    try:
        ts = pd.to_datetime(token, dayfirst=True, utc=True)
    except Exception as e:
        raise ValueError(f"Fecha inválida '{raw}'. Usa por ejemplo 01/01/2024 o 01/01/2024 13:30.") from e

    ts = ts.tz_convert("UTC").tz_localize(None)
    if end_of_day_if_date and not has_time:
        ts = ts + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return ts

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

def parse_timeframe(raw: str) -> TimeframeSpec:
    token = raw.strip().lower()
    token = re.sub(r"\s+", "", token)

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
    if unit is None:
        raise ValueError(f"Unidad de timeframe no soportada en '{raw}'. Usa minutos u horas.")
    if n <= 0:
        raise ValueError(f"El timeframe debe ser > 0: '{raw}'.")

    minutes = n if unit == "m" else n * 60
    pandas_freq = f"{minutes}min"

    if minutes % 60 == 0:
        h = minutes // 60
        display_label = f"{h}H"
        folder_tag = f"h{h}"
    else:
        display_label = f"{minutes}min"
        folder_tag = f"m{minutes}"

    return TimeframeSpec(
        raw=raw,
        minutes=minutes,
        pandas_freq=pandas_freq,
        display_label=display_label,
        folder_tag=folder_tag,
    )

# -----------------------------
# Streaming resample M1 -> timeframe (vectorized groupby per row-group)
# -----------------------------

def aggregate_m1_to_h1_streaming(
    input_path: Path,
    timezone_name: str = "UTC",
    make_timestamp_naive_utc: bool = True,
    eps_ohlc: float = 1e-7,
    timeframe_minutes: int = 60,
    timeframe_floor_rule: str = "60min",
    start_ts: pd.Timestamp | None = None,
    end_ts: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Memory-safe aggregation using Parquet row groups.
    Ahora potenciado con Micro-Momentum, VWAP, Volatilidad, ER y Delta Volume.
    """
    pf = pq.ParquetFile(str(input_path))
    cols = pf.schema_arrow.names

    ts_col = detect_timestamp_column(cols)
    ohlc = pick_ohlc_columns(cols)

    open_col = ohlc["open"]
    high_col = ohlc["high"]
    low_col = ohlc["low"]
    close_col = ohlc["close"]
    vol_col = ohlc.get("volume", None)

    use_cols = [ts_col, open_col, high_col, low_col, close_col]
    if vol_col is not None:
        use_cols.append(vol_col)

    total_rows = 0
    had_duplicates = False
    was_unsorted = False
    n_ohlc_invalid_m1 = 0

    # bucket_ts -> [open, high, low, close, vol_sum, n, open_ts, close_ts,
    #             sum_pv, sum_delta_v, sum_abs_diff, sum_c, sum_c2]
    agg: dict[np.datetime64, list] = {}
    last_ts = None

    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg, columns=use_cols)
        df = table.to_pandas()
        total_rows += len(df)

        ts = coerce_timestamp(df[ts_col], assume_utc=True)
        if ts.isna().any():
            df = df.loc[~ts.isna()].copy()
            ts = ts.loc[~ts.isna()]
        if len(df) == 0:
            continue

        if make_timestamp_naive_utc:
            ts = to_utc_naive(ts)
        else:
            ts = pd.to_datetime(ts, utc=True)

        if last_ts is not None and ts.iloc[0] < last_ts:
            was_unsorted = True

        if not ts.is_monotonic_increasing:
            was_unsorted = True
            order = np.argsort(ts.to_numpy())
            df = df.iloc[order].reset_index(drop=True)
            ts = ts.iloc[order].reset_index(drop=True)

        if ts.duplicated().any():
            had_duplicates = True
            keep = ~ts.duplicated(keep="last")
            df = df.loc[keep].copy()
            ts = ts.loc[keep]
            df.reset_index(drop=True, inplace=True)
            ts.reset_index(drop=True, inplace=True)

        if start_ts is not None or end_ts is not None:
            mask = pd.Series(True, index=df.index)
            if start_ts is not None:
                mask &= ts >= start_ts
            if end_ts is not None:
                mask &= ts <= end_ts
            if not mask.all():
                df = df.loc[mask].copy()
                ts = ts.loc[mask]
                if len(df) == 0:
                    continue
                df.reset_index(drop=True, inplace=True)
                ts.reset_index(drop=True, inplace=True)

        last_ts = ts.iloc[-1]

        # Use float64 to avoid catastrophic cancellation in second-moment stats.
        df[open_col] = pd.to_numeric(df[open_col], errors="coerce").astype(np.float64)
        df[high_col] = pd.to_numeric(df[high_col], errors="coerce").astype(np.float64)
        df[low_col] = pd.to_numeric(df[low_col], errors="coerce").astype(np.float64)
        df[close_col] = pd.to_numeric(df[close_col], errors="coerce").astype(np.float64)
        if vol_col is not None:
            df[vol_col] = pd.to_numeric(df[vol_col], errors="coerce").fillna(0.0).astype(np.float64)

        o = df[open_col].to_numpy(dtype=np.float64)
        h = df[high_col].to_numpy(dtype=np.float64)
        l = df[low_col].to_numpy(dtype=np.float64)
        c = df[close_col].to_numpy(dtype=np.float64)

        # Validar OHLC
        invalid = (h < np.maximum(o, c) - eps_ohlc) | (l > np.minimum(o, c) + eps_ohlc) | (h < l - eps_ohlc)
        n_ohlc_invalid_m1 += int(invalid.sum())

        # --- PRE-CÁLCULOS ESTADÍSTICOS EN M1 ---
        # 1. VWAP (Precio Típico * Volumen)
        df['typ_p'] = (h + l + c) / 3.0

        if vol_col is not None:
            v = df[vol_col].to_numpy()
            df['pv'] = df['typ_p'] * v
            # Aproximación del Delta de Volumen (Signo de la vela * Volumen)
            df['delta_v'] = v * np.sign(c - o)
        else:
            df['pv'] = 0.0
            df['delta_v'] = 0.0

        # 2. Eficiencia (Ruido = suma de cambios absolutos minuto a minuto)
        df['abs_diff'] = df[close_col].diff().abs().fillna(0.0)

        # 3. Volatilidad (Suma de C y C^2 para la Varianza)
        df['c2'] = c ** 2

        ts_hour = pd.to_datetime(ts).dt.floor(timeframe_floor_rule)
        if make_timestamp_naive_utc:
            ts_hour = to_utc_naive(ts_hour)

        df["ts_hour"] = ts_hour.to_numpy()
        df["_ts"] = pd.to_datetime(ts).to_numpy()

        # Vectorized GroupBy
        g = df.groupby("ts_hour", sort=False)

        open_s = g[open_col].first()
        high_s = g[high_col].max()
        low_s = g[low_col].min()
        close_s = g[close_col].last()
        n_s = g.size()
        open_ts_s = g["_ts"].first()
        close_ts_s = g["_ts"].last()
        vol_s = g[vol_col].sum() if vol_col is not None else None

        # Sumatorias para estadísticos
        pv_s = g['pv'].sum()
        delta_v_s = g['delta_v'].sum()
        abs_diff_s = g['abs_diff'].sum()
        sum_c_s = g[close_col].sum()
        sum_c2_s = g['c2'].sum()

        buckets = open_s.index.to_numpy(dtype="datetime64[ns]")
        opens = open_s.to_numpy()
        highs = high_s.to_numpy()
        lows = low_s.to_numpy()
        closes = close_s.to_numpy()
        ns = n_s.to_numpy(dtype=np.int64)
        open_tss = open_ts_s.to_numpy(dtype="datetime64[ns]")
        close_tss = close_ts_s.to_numpy(dtype="datetime64[ns]")

        vols = vol_s.to_numpy() if vol_s is not None else None
        pvs = pv_s.to_numpy()
        delta_vs = delta_v_s.to_numpy()
        abs_diffs = abs_diff_s.to_numpy()
        sum_cs = sum_c_s.to_numpy()
        sum_c2s = sum_c2_s.to_numpy()

        for i, hour in enumerate(buckets):
            o_v = float(opens[i]); h_v = float(highs[i]); l_v = float(lows[i]); c_v = float(closes[i]); n_v = int(ns[i])
            ot_v = open_tss[i]; ct_v = close_tss[i]

            v_v = float(vols[i]) if vols is not None else 0.0
            pv_v = float(pvs[i])
            dv_v = float(delta_vs[i])
            ad_v = float(abs_diffs[i])
            sc_v = float(sum_cs[i])
            sc2_v = float(sum_c2s[i])

            if hour not in agg:
                agg[hour] = [o_v, h_v, l_v, c_v, v_v, n_v, ot_v, ct_v, pv_v, dv_v, ad_v, sc_v, sc2_v]
            else:
                p = agg[hour]
                if ot_v < p[6]:
                    p[0] = o_v; p[6] = ot_v
                if ct_v > p[7]:
                    p[3] = c_v; p[7] = ct_v
                p[1] = max(p[1], h_v)
                p[2] = min(p[2], l_v)
                p[5] += n_v
                p[4] += v_v
                p[8] += pv_v
                p[9] += dv_v
                p[10] += ad_v
                p[11] += sc_v
                p[12] += sc2_v
                agg[hour] = p

    if not agg:
        raise RuntimeError(f"No data aggregated from {input_path}")

    buckets_sorted = np.array(sorted(agg.keys()))
    rows = [agg[h] for h in buckets_sorted]

    col_names = [
        "open", "high", "low", "close", "volume", "n_m1", "_open_ts", "_close_ts",
        "_sum_pv", "delta_vol", "_sum_abs_diff", "_sum_c", "_sum_c2"
    ]
    df_h1 = pd.DataFrame(rows, columns=col_names)
    df_h1.insert(0, "timestamp", pd.to_datetime(buckets_sorted))
    df_h1.drop(columns=["_open_ts", "_close_ts"], inplace=True)

    # --- CÁLCULO FINAL DE ARSENAL ESTADÍSTICO ---

    # 1. Fuerza de Cierre (Manejo de Dojis / división por cero)
    hl_range = df_h1["high"] - df_h1["low"]
    df_h1["close_strength"] = np.where(hl_range > 0, (df_h1["close"] - df_h1["low"]) / hl_range, 0.5)

    # 2. VWAP y Distancia
    df_h1["vwap"] = np.where(df_h1["volume"] > 0, df_h1["_sum_pv"] / df_h1["volume"], df_h1["close"])
    df_h1["vwap_distance"] = np.where(df_h1["vwap"] > 0, ((df_h1["close"] - df_h1["vwap"]) / df_h1["vwap"]) * 100.0, 0.0)

    # 3. Ratio de Eficiencia (Señal / Ruido)
    signal = (df_h1["close"] - df_h1["open"]).abs()
    noise = df_h1["_sum_abs_diff"]
    df_h1["efficiency_ratio"] = np.where(noise > 0, signal / noise, 0.0)

    # 4. Volatilidad Interna (Desviación estándar de los cierres M1)
    mean_c = df_h1["_sum_c"] / df_h1["n_m1"]
    mean_c2 = df_h1["_sum_c2"] / df_h1["n_m1"]
    variance = mean_c2 - (mean_c ** 2)

    # Clamp only negative floating-point noise; keep genuinely small positive variance.
    variance = np.where(variance < 0.0, 0.0, variance)

    # Calcular raíz, asegurando que si n_m1 == 1, la volatilidad sea estrictamente 0
    df_h1["volatility"] = np.where(df_h1["n_m1"] > 1, np.sqrt(variance), 0.0)

    # Limpieza de columnas temporales
    df_h1.drop(columns=["_sum_pv", "_sum_abs_diff", "_sum_c", "_sum_c2"], inplace=True)

    # --- LÓGICA DE GAPS ---
    df_h1["is_complete"] = df_h1["n_m1"] == int(timeframe_minutes)
    dt_minutes = df_h1["timestamp"].diff().dt.total_seconds().div(60.0)
    df_h1["gap_bars"] = (dt_minutes / float(timeframe_minutes)).round().fillna(1).astype(int)
    # Backward compatibility: keep historical column name.
    df_h1["gap_hours"] = dt_minutes.div(60.0).fillna(float(timeframe_minutes) / 60.0)
    df_h1["is_gap"] = df_h1["gap_bars"] > 1

    prev_ts = df_h1["timestamp"].shift(1)
    cur_ts = df_h1["timestamp"]
    gap_hours = df_h1["gap_hours"]

    is_weekend_gap = (
        df_h1["is_gap"]
        & gap_hours.between(30, 90)
        & (prev_ts.dt.weekday.isin([4, 5]))
        & (cur_ts.dt.weekday == 6)
    )
    df_h1["is_weekend_gap"] = is_weekend_gap.fillna(False)
    df_h1["is_data_gap"] = df_h1["is_gap"] & (~df_h1["is_weekend_gap"])

    invalid_tf = (df_h1["high"] < df_h1[["open", "close"]].max(axis=1) - eps_ohlc) | \
                 (df_h1["low"]  > df_h1[["open", "close"]].min(axis=1) + eps_ohlc) | \
                 (df_h1["high"] < df_h1["low"] - eps_ohlc)
    n_ohlc_invalid_tf = int(invalid_tf.sum())

    audit = dict(
        input_rows=int(total_rows),
        output_rows=int(len(df_h1)),
        start=str(df_h1["timestamp"].iloc[0]),
        end=str(df_h1["timestamp"].iloc[-1]),
        pct_complete_hours=float(df_h1["is_complete"].mean() * 100.0),
        n_gaps_total=int(df_h1["is_gap"].sum()),
        n_weekend_gaps=int(df_h1["is_weekend_gap"].sum()),
        n_data_gaps=int(df_h1["is_data_gap"].sum()),
        n_ohlc_invalid_m1=int(n_ohlc_invalid_m1),
        n_ohlc_invalid_h1=int(n_ohlc_invalid_tf),
        had_duplicates=bool(had_duplicates),
        was_unsorted=bool(was_unsorted),
        ts_col=ts_col,
        ohlc_map=ohlc,
        eps_ohlc=float(eps_ohlc),
    )

    return df_h1, audit

# -----------------------------
# Writing outputs
# -----------------------------

def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, str(path), compression="zstd")

def build_calendar_and_masks_optimized(
    stats: list[AssetBuildStats],
    out_dir: Path,
    logger: logging.Logger,
    calendar_freq: str,
) -> int:
    starts = [pd.to_datetime(s.start) for s in stats if s.start]
    ends = [pd.to_datetime(s.end) for s in stats if s.end]
    if not starts:
        return 0

    full_idx = pd.date_range(start=min(starts), end=max(ends), freq=calendar_freq)
    total_periods = int(len(full_idx))

    calendar = pd.DataFrame({"timestamp": full_idx})
    write_parquet(calendar, out_dir / "calendar_master.parquet")

    idx = pd.Index(full_idx)
    masks = pd.DataFrame({"timestamp": full_idx})

    for s in tqdm(stats, desc="Building masks", total=len(stats)):
        ts_existing = pq.read_table(s.output_file, columns=["timestamp"]).to_pandas()["timestamp"]
        ts_existing = pd.to_datetime(ts_existing)
        pos = idx.get_indexer(ts_existing)
        ok = pos >= 0
        arr = np.zeros(len(idx), dtype=bool)
        arr[pos[ok]] = True
        masks[s.symbol] = arr

    write_parquet(masks, out_dir / "availability_masks.parquet")
    logger.info("Calendar+Masks written. total_periods_in_range=%d", total_periods)
    return total_periods

def write_data_quality_report(stats: list[AssetBuildStats], out_path: Path, timeframe_label: str, timeframe_minutes: int) -> None:
    lines: list[str] = []
    lines.append(f"# Data Quality Report — Dataset_{timeframe_label}_v1")
    lines.append("")
    lines.append(f"Generado: {now_utc_iso()} (UTC)")
    lines.append("")
    lines.append("## Resumen por asset")
    lines.append("")
    lines.append(f"| symbol | start | end | M1 rows | {timeframe_label} rows | % complete {timeframe_label} | gaps total | weekend gaps | data gaps | OHLC invalid ({timeframe_label}) | dup | unsorted | sha256 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for s in stats:
        lines.append(
            f"| {s.symbol} | {s.start} | {s.end} | {s.input_rows:,} | {s.output_rows:,} | "
            f"{s.pct_complete_hours:.2f}% | {s.n_gaps_total:,} | {s.n_weekend_gaps:,} | {s.n_data_gaps:,} | "
            f"{s.n_ohlc_invalid:,} | {int(s.had_duplicates)} | {int(s.was_unsorted)} | `{s.sha256[:12]}…` |"
        )
    lines.append("")
    lines.append("## Notas")
    lines.append(f"- `% complete {timeframe_label}` = porcentaje de velas con `n_m1 == {timeframe_minutes}`.")
    lines.append("- `gaps total` = número de veces que `gap_bars > 1` entre velas existentes.")
    lines.append("- `weekend gaps` = gaps detectados como cierre normal FX (heurístico Fri→Sun).")
    lines.append("- `data gaps` = gaps que NO parecen fin de semana (potencial issue de data).")
    lines.append(f"- `OHLC invalid ({timeframe_label})` usa tolerancia eps (por defecto 1e-7).")
    out_path.write_text("\n".join(lines), encoding="utf-8")

def build_manifest(stats: list[AssetBuildStats], out_dir: Path, meta: dict, timeframe_label: str) -> None:
    d = {
        "dataset_name": f"Dataset_{timeframe_label}_v1",
        "created_at_utc": now_utc_iso(),
        **meta,
        "assets": {s.symbol: asdict(s) for s in stats},
    }
    (out_dir / "manifest.json").write_text(json.dumps(d, indent=2), encoding="utf-8")

# -----------------------------
# Parallel build
# -----------------------------

def _build_one(args):
    input_path, out_dir, timezone_name, force, eps_ohlc, timeframe, start_ts, end_ts = args
    sym = normalize_symbol_from_filename(input_path.name)
    out_path = out_dir / f"parquet_{timeframe.folder_tag}" / f"{sym}.parquet"

    should_rebuild = bool(force or start_ts is not None or end_ts is not None)
    if out_path.exists() and not should_rebuild:
        sha = sha256_file(out_path)
        t = pq.read_table(str(out_path)).to_pandas()
        if "is_complete" not in t.columns and "n_m1" in t.columns:
            t["is_complete"] = t["n_m1"] == timeframe.minutes
        if "is_gap" not in t.columns:
            if "gap_bars" in t.columns:
                t["is_gap"] = t["gap_bars"] > 1
            elif "gap_hours" in t.columns:
                bars = (pd.to_numeric(t["gap_hours"], errors="coerce") * 60.0 / float(timeframe.minutes)).round().fillna(1)
                t["is_gap"] = bars.astype(int) > 1
            else:
                t["is_gap"] = False
        if "is_weekend_gap" not in t.columns:
            t["is_weekend_gap"] = False
        if "is_data_gap" not in t.columns:
            t["is_data_gap"] = t["is_gap"] & (~t["is_weekend_gap"])
        pct_complete = float(t["is_complete"].mean() * 100.0) if len(t) else 0.0
        return sym, out_path, None, AssetBuildStats(
            symbol=sym,
            input_file=str(input_path),
            output_file=str(out_path),
            input_rows=-1,
            output_rows=int(len(t)),
            start=str(t["timestamp"].iloc[0]) if len(t) else "",
            end=str(t["timestamp"].iloc[-1]) if len(t) else "",
            pct_complete_hours=pct_complete,
            n_gaps_total=int(t["is_gap"].sum()) if len(t) else 0,
            n_weekend_gaps=int(t["is_weekend_gap"].sum()) if len(t) else 0,
            n_data_gaps=int(t["is_data_gap"].sum()) if len(t) else 0,
            n_ohlc_invalid=0,
            had_duplicates=False,
            was_unsorted=False,
            sha256=sha,
        )

    try:
        df_h1, audit = aggregate_m1_to_h1_streaming(
            input_path,
            timezone_name=timezone_name,
            eps_ohlc=eps_ohlc,
            timeframe_minutes=timeframe.minutes,
            timeframe_floor_rule=timeframe.pandas_freq,
            start_ts=start_ts,
            end_ts=end_ts,
        )
    except RuntimeError as e:
        if "No data aggregated from" in str(e):
            return sym, out_path, {"skipped": "no_data_in_range"}, None
        raise

    write_parquet(df_h1, out_path)
    sha = sha256_file(out_path)

    stats = AssetBuildStats(
        symbol=sym,
        input_file=str(input_path),
        output_file=str(out_path),
        input_rows=int(audit["input_rows"]),
        output_rows=int(audit["output_rows"]),
        start=audit["start"],
        end=audit["end"],
        pct_complete_hours=float(audit["pct_complete_hours"]),
        n_gaps_total=int(audit["n_gaps_total"]),
        n_weekend_gaps=int(audit["n_weekend_gaps"]),
        n_data_gaps=int(audit["n_data_gaps"]),
        n_ohlc_invalid=int(audit["n_ohlc_invalid_h1"]),
        had_duplicates=bool(audit["had_duplicates"]),
        was_unsorted=bool(audit["was_unsorted"]),
        sha256=sha,
    )
    return sym, out_path, audit, stats

def _setup_logger(out_dir: Path) -> logging.Logger:
    logger = logging.getLogger("build_dataset")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(out_dir / "logs" / "build.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, type=str)
    ap.add_argument("--output_dir", required=True, type=str)
    ap.add_argument("--timezone", default="UTC", type=str)
    ap.add_argument("--fx_only", default=1, type=int, help="1=solo FX (recomendado). 0=todo")
    ap.add_argument("--n_jobs", default=4, type=int)
    ap.add_argument("--force", default=0, type=int)
    ap.add_argument("--no_calendar", default=0, type=int)
    ap.add_argument("--eps_ohlc", default=1e-7, type=float)
    ap.add_argument("--timeframe", default="1h", type=str, help="Ejemplos: 10min, 20m, 1h, 4h, 1hora, 4horas")
    ap.add_argument("--start_date", default="", type=str, help="Opcional. Ejemplos: 01/01/2024 o 2024-01-01")
    ap.add_argument("--end_date", default="", type=str, help="Opcional. Ejemplos: 02/02/2024 o 2024-02-02")
    ap.add_argument("--symbols", default="", type=str, help="Opcional. Lista de símbolos separada por coma/espacio. Ej: EURUSD,GBPUSD")
    args = ap.parse_args()

    timeframe = parse_timeframe(args.timeframe)
    start_ts = parse_optional_utc_datetime(args.start_date, end_of_day_if_date=False)
    end_ts = parse_optional_utc_datetime(args.end_date, end_of_day_if_date=True)
    selected_symbols = parse_symbols_filter(args.symbols)
    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        raise SystemExit(f"Rango inválido: start_date ({start_ts}) > end_date ({end_ts}).")

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = _setup_logger(out_dir)
    logger.info("Starting build_h1_dataset.py")
    logger.info("Input dir: %s", in_dir)
    logger.info("Output dir: %s", out_dir)
    logger.info("timeframe=%s (%s, %d min)", args.timeframe, timeframe.display_label, timeframe.minutes)
    if start_ts is not None or end_ts is not None:
        logger.info("date_range_utc=[%s .. %s]", start_ts if start_ts is not None else "-inf", end_ts if end_ts is not None else "+inf")
    else:
        logger.info("date_range_utc=FULL")
    if selected_symbols is not None:
        logger.info("symbols_filter=%s", ",".join(sorted(selected_symbols)))
    else:
        logger.info("symbols_filter=ALL")
    logger.info("fx_only=%s n_jobs=%d force=%s no_calendar=%s eps_ohlc=%g",
                bool(args.fx_only), args.n_jobs, bool(args.force), bool(args.no_calendar), args.eps_ohlc)

    files = sorted([p for p in in_dir.rglob("*.parquet") if p.is_file()])
    if args.fx_only:
        files = [p for p in files if is_fx_symbol(normalize_symbol_from_filename(p.name))]
    if selected_symbols is not None:
        files = [p for p in files if normalize_symbol_from_filename(p.name) in selected_symbols]

    if not files:
        if selected_symbols is not None:
            raise SystemExit(f"No parquet files found after filtering. symbols={sorted(selected_symbols)}")
        raise SystemExit("No parquet files found after filtering.")

    from multiprocessing import Pool
    force = bool(args.force)
    task_args = [(p, out_dir, args.timezone, force, float(args.eps_ohlc), timeframe, start_ts, end_ts) for p in files]

    stats: list[AssetBuildStats] = []

    logger.info("Processing %d assets...", len(task_args))

    if args.n_jobs <= 1:
        for sym, out_path, audit, st in tqdm(map(_build_one, task_args), total=len(task_args), desc="Assets"):
            if st is None:
                logger.warning("Skipped %s (no data in selected date range)", sym)
                continue
            stats.append(st)
            logger.info("Done %s -> %s (%s rows=%s)", sym, out_path.name, timeframe.display_label, st.output_rows)
    else:
        with Pool(processes=args.n_jobs) as pool:
            it = pool.imap_unordered(_build_one, task_args)
            for sym, out_path, audit, st in tqdm(it, total=len(task_args), desc="Assets"):
                if st is None:
                    logger.warning("Skipped %s (no data in selected date range)", sym)
                    continue
                stats.append(st)
                logger.info("Done %s -> %s (%s rows=%s)", sym, out_path.name, timeframe.display_label, st.output_rows)

    stats.sort(key=lambda x: x.symbol)

    write_data_quality_report(stats, out_dir / "data_quality_report.md", timeframe.display_label, timeframe.minutes)

    total_periods_in_range = 0
    if args.no_calendar == 0:
        total_periods_in_range = build_calendar_and_masks_optimized(stats, out_dir, logger, timeframe.pandas_freq)

    global_completeness = float(np.mean([s.pct_complete_hours for s in stats])) if stats else 0.0

    build_manifest(
        stats,
        out_dir,
        meta=dict(
            source_dir=str(in_dir),
            timezone=args.timezone,
            resample_rule=timeframe.pandas_freq,
            timeframe_label=timeframe.display_label,
            timeframe_minutes=timeframe.minutes,
            date_range_start_utc=(start_ts.isoformat() if start_ts is not None else None),
            date_range_end_utc=(end_ts.isoformat() if end_ts is not None else None),
            symbols_filter=(sorted(selected_symbols) if selected_symbols is not None else None),
            fx_only=bool(args.fx_only),
            market_type=("FX" if bool(args.fx_only) else "MIXED"),
            eps_ohlc=float(args.eps_ohlc),
            total_periods_in_range=int(total_periods_in_range),
            global_completeness=float(global_completeness),
            notes=f"{timeframe.display_label} built from M1 with streaming row-group processing + groupby aggregation. Weekend gaps labeled. Execution model: decide at close(t), enter open(t+1).",
        ),
        timeframe_label=timeframe.display_label,
    )

    logger.info("OK: built %s dataset in %s", timeframe.display_label, out_dir)
    logger.info("- assets: %d", len(stats))
    logger.info("- manifest: %s", out_dir / "manifest.json")
    logger.info("- report: %s", out_dir / "data_quality_report.md")

if __name__ == "__main__":
    main()
