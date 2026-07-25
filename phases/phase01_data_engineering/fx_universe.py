#!/usr/bin/env python3
from __future__ import annotations
import re
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Set

# Configuración de Logging para auditoría
logger = logging.getLogger("FX_Universe")

# --- Jerarquía de Monedas ---

# Monedas con máxima liquidez (G8)
MAJORS_CCY: Set[str] = {"EUR", "GBP", "AUD", "NZD", "USD", "CAD", "CHF", "JPY"}

# Monedas G10 + Estables (Usualmente forman los 'Minors')
G10_CCY: Set[str] = MAJORS_CCY | {"SEK", "NOK", "DKK"}

# Monedas Emergentes (Exotics)
EM_CCY: Set[str] = {"PLN", "CZK", "HUF", "TRY", "MXN", "ZAR", "HKD", "SGD", "ILS", "BRL", "CLP"}

# Universo Total FX
FX_CCY: Set[str] = G10_CCY | EM_CCY

# Exclusiones explícitas (Evita que XAUUSD o BTCUSD pasen como FX)
NON_FX_BASES: Set[str] = {
    "XAU", "XAG", "XPT", "XPD", "WTI", "BCO", "NGAS", # Commodities
    "SPX", "NSX", "UDX", "AUX", "FRX", "GRX", "HKX", "UKX", "ETX", # Índices
    "BTC", "ETH", "SOL", "USDT" # Crypto
}

@dataclass(frozen=True)
class SymbolInfo:
    raw: str
    symbol: str   # e.g., "EURUSD"
    base: str     # e.g., "EUR"
    quote: str    # e.g., "USD"
    category: str # 'major', 'minor', 'exotic', 'commodity_index', 'unknown'
    is_fx: bool
    is_usd_cross: bool # Útil para modelos que trackean el índice DXY

# --- Lógica de Procesamiento ---

def normalize_symbol_from_filename(filename: str) -> str:
    """
    Extrae el símbolo de 6 letras de un nombre de archivo, manejando rutas y extensiones.
    e.g. '/data/m1/eurusd_2023.parquet' -> 'EURUSD'
    """
    stem = Path(filename).stem.upper()

    # Regex mejorada: Busca 6 letras seguidas o 3+3 separadas por [./-_]
    match = re.search(r'([A-Z]{3})[./\-_]?([A-Z]{3})', stem)
    if match:
        return f"{match.group(1)}{match.group(2)}"

    # Fallback: limpiar todo lo que no sea letra y tomar las primeras 6
    clean = re.sub(r'[^A-Z]', '', stem)
    return clean[:6] if len(clean) >= 6 else stem

def get_symbol_metadata(raw_sym: str) -> SymbolInfo:
    """
    Realiza la disección completa del símbolo y su clasificación de mercado.
    """
    clean_sym = normalize_symbol_from_filename(raw_sym)

    if len(clean_sym) != 6:
        return SymbolInfo(raw_sym, clean_sym, "", "", "unknown", False, False)

    base, quote = clean_sym[:3], clean_sym[3:]

    # 1. Filtro de Exclusiones
    if base in NON_FX_BASES or quote in NON_FX_BASES:
        return SymbolInfo(raw_sym, clean_sym, base, quote, "commodity_index", False, False)

    # 2. Verificación de Universo FX
    is_fx = (base in FX_CCY) and (quote in FX_CCY)
    is_usd_cross = is_fx and (base == "USD" or quote == "USD")

    # 3. Categorización Granular
    category = "unknown"
    if is_fx:
        if base in MAJORS_CCY and quote in MAJORS_CCY:
            category = "major"
        elif base in G10_CCY and quote in G10_CCY:
            category = "minor"
        elif base in EM_CCY or quote in EM_CCY:
            category = "exotic"

    return SymbolInfo(
        raw=raw_sym,
        symbol=clean_sym,
        base=base,
        quote=quote,
        category=category,
        is_fx=is_fx,
        is_usd_cross=is_usd_cross
    )

def is_fx_symbol(sym: str) -> bool:
    """Helper de alta velocidad para filtrado inicial."""
    meta = get_symbol_metadata(sym)
    return meta.is_fx