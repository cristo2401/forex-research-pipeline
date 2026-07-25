#!/usr/bin/env python3
from __future__ import annotations
import pandas as pd
import numpy as np
import re
import logging

logger = logging.getLogger("TimestampUtils")

# Prioridad de nombres de columnas para evitar falsos positivos
PRIMARY_CANDIDATES = [
    "timestamp", "time", "datetime", "date", "dt", "ts", "time_utc"
]

def detect_timestamp_column(columns: list[str]) -> str:
    """
    Detecta la columna de tiempo usando regex y prioridades.
    """
    cols_lower = [c.lower() for c in columns]

    # 1. Búsqueda por coincidencia exacta (Case-insensitive)
    for cand in PRIMARY_CANDIDATES:
        if cand in cols_lower:
            return columns[cols_lower.index(cand)]

    # 2. Búsqueda por patrones comunes (ej. 'Time (UTC)', 'open_time')
    for c in columns:
        cl = c.lower()
        if re.search(r'^(time|date|dt|ts|timestamp)', cl) or \
           re.search(r'(time|date|dt|ts|timestamp)$', cl):
            return c

    raise ValueError(f"No se detectó columna de tiempo en: {columns}")

def _parse_numeric_datetime(arr: np.ndarray) -> pd.DatetimeIndex:
    """
    Parsea formatos YYYYMMDDHHMM(SS) usando aritmética en lugar de strings
    para ganar velocidad (hasta 10x más rápido).
    """
    # Si el número empieza con 20xx y tiene 12-14 dígitos
    # Ejemplo: 202310271530
    m = np.nanmedian(arr)

    if 190001010000 <= m <= 21001231235959:
        # Convertimos a string solo si es estrictamente necesario para pd.to_datetime
        # pero con un formato explícito para evitar inferencia lenta.
        s = arr.astype(np.int64).astype(str)
        if len(s[0]) == 12:
            return pd.to_datetime(s, format="%Y%m%d%H%M", utc=True)
        elif len(s[0]) == 14:
            return pd.to_datetime(s, format="%Y%m%d%H%M%S", utc=True)

    # Heurística por magnitud para Epochs
    # s < 1e10, ms < 1e13, us < 1e16, ns < 1e19
    if m < 1e10:
        unit = 's'
    elif m < 1e13:
        unit = 'ms'
    elif m < 1e16:
        unit = 'us'
    else:
        unit = 'ns'

    return pd.to_datetime(arr, unit=unit, utc=True)

def coerce_timestamp(series: pd.Series, assume_utc: bool = True) -> pd.Series:
    """
    Convierte cualquier serie a datetime64[ns] UTC de forma ultra-robusta.
    """
    if series.empty:
        return series

    # 1. Ya es datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        if series.dt.tz is None and assume_utc:
            return series.dt.tz_localize("UTC")
        return series.dt.tz_convert("UTC") if series.dt.tz else series

    # 2. Es numérico (Epoch o YYYYMMDD...)
    if pd.api.types.is_numeric_dtype(series):
        arr = series.values
        ts_index = _parse_numeric_datetime(arr)
        return pd.Series(ts_index, index=series.index)

    # 3. Es String / Object
    # Usamos cache=True para que si hay timestamps repetidos (común en M1) sea instantáneo
    return pd.to_datetime(series, utc=True, errors="coerce", cache=True)

def to_utc_naive(series: pd.Series) -> pd.Series:
    """
    Estandariza a UTC 'naive'.
    Muchos modelos de ML y Parquet prefieren no tener info de zona horaria
    pero asumir que es UTC para evitar conflictos de offsets.
    """
    if not pd.api.types.is_datetime64_any_dtype(series):
        series = coerce_timestamp(series)

    if series.dt.tz is not None:
        series = series.dt.tz_convert("UTC").dt.tz_localize(None)
    return series