"""Run the real phase-1 CLI on synthetic data and verify its OHLCV contract.

This is a data-engineering smoke check, not a strategy/backtest evaluation.
Only temporary local files are used. No broker connection or GPU is required.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    timestamps = pd.date_range("2024-01-08", periods=240, freq="min")
    # Leave one partial hour and one entirely missing hour.
    timestamps = timestamps[(timestamps.hour != 2) & ~((timestamps.hour == 1) & (timestamps.minute == 15))]
    n = len(timestamps)
    opening = 1.10 + np.arange(n, dtype=float) * 1e-5
    source = pd.DataFrame({
        "timestamp": timestamps,
        "open": opening,
        "high": opening + 3e-5,
        "low": opening - 2e-5,
        "close": opening + 1e-5,
        "volume": np.ones(n),
    })
    with tempfile.TemporaryDirectory(prefix="fx_synthetic_") as scratch:
        root = Path(scratch)
        inputs, outputs = root / "input", root / "output"
        inputs.mkdir()
        # A row group boundary deliberately cuts across an hourly bucket.
        source.to_parquet(inputs / "EURUSD.parquet", index=False, row_group_size=37)
        subprocess.run([
            sys.executable,
            str(ROOT / "phases/phase01_data_engineering/build_h1_dataset.py"),
            "--input_dir", str(inputs), "--output_dir", str(outputs),
            "--n_jobs", "1", "--timeframe", "1h", "--timezone", "UTC",
        ], check=True)
        actual = pd.read_parquet(outputs / "parquet_h1/EURUSD.parquet")
        expected = source.set_index("timestamp").resample("1h").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna(subset=["open"]).reset_index()
        pd.testing.assert_series_equal(actual["timestamp"], expected["timestamp"], check_dtype=False)
        for column in ("open", "high", "low", "close", "volume"):
            np.testing.assert_allclose(actual[column], expected[column], rtol=1e-10, atol=1e-12)
        np.testing.assert_array_equal(actual["n_m1"], [60, 59, 60])
        np.testing.assert_array_equal(actual["is_complete"], [True, False, True])
        np.testing.assert_array_equal(actual["is_gap"], [False, False, True])
        np.testing.assert_array_equal(actual["is_data_gap"], [False, False, True])
        np.testing.assert_array_equal(actual["is_weekend_gap"], [False, False, False])
        with (outputs / "manifest.json").open(encoding="utf-8") as handle:
            json.load(handle)
        if not (outputs / "data_quality_report.md").is_file():
            raise RuntimeError("Missing quality report")
        print(actual[["timestamp", "n_m1", "is_complete", "is_data_gap"]].to_string(index=False))
    print("PASS: synthetic M1 -> H1 checks")


if __name__ == "__main__":
    main()
