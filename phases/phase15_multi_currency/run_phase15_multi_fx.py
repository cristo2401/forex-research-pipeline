#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SYMBOLS: Tuple[str, ...] = (
    'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD',
    'USDCAD', 'USDCHF', 'EURJPY', 'EURGBP', 'GBPJPY',
)

DEFAULT_COST_PIPS: Dict[str, float] = {
    'EURUSD': 0.9,
    'GBPUSD': 1.1,
    'USDJPY': 0.9,
    'AUDUSD': 1.0,
    'NZDUSD': 1.2,
    'USDCAD': 1.1,
    'USDCHF': 1.1,
    'EURJPY': 1.2,
    'EURGBP': 1.0,
    'GBPJPY': 1.5,
}


@dataclass
class Phase15Config:
    input_root: str
    out_dir: str
    strict_mode: bool = True
    symbols: Tuple[str, ...] = DEFAULT_SYMBOLS
    n_select: int = 7
    priors_source: str = 'windows_only'
    purged_gap_bars: int = 24
    eval_stride: int = 24
    seed: int = 42
    fail_fast: bool = False

    phase13_script: str = str(ROOT / 'phases/phase13_pipeline_cli/run_phase13_pipeline.py')
    phase14_script: str = str(ROOT / 'phases/phase14_production_hardening/run_phase14_1_hardening.py')
    phase13_sanity_script: str = str(ROOT / 'phases/phase13_pipeline_cli/run_phase13_sanity_checks.py')


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def now_tag() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')


def safe_json_write(path: Path, obj: dict):
    path.write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def parse_symbols(raw: str) -> Tuple[str, ...]:
    out: List[str] = []
    for x in (raw or '').split(','):
        s = x.strip().upper()
        if not s:
            continue
        if len(s) != 6:
            raise ValueError(f'Invalid symbol={s!r}; expected 6 letters (e.g., EURUSD).')
        out.append(s)
    if not out:
        raise ValueError('No symbols provided.')
    # stable unique
    seen = set()
    uniq: List[str] = []
    for s in out:
        if s not in seen:
            uniq.append(s)
            seen.add(s)
    return tuple(uniq)


def detect_timestamp_col(columns: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in columns}
    for c in ['timestamp', 'dt', 'datetime', 'time', 'date']:
        if c in lower_map:
            return lower_map[c]
    return None


def prepare_dedup_input(input_file: Path, out_file: Path) -> Dict[str, int]:
    df = pd.read_parquet(input_file).copy()
    ts_col = detect_timestamp_col(df.columns.tolist())
    if ts_col is None:
        raise ValueError(f'Missing timestamp-like column in {input_file.name}')

    df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce', utc=True)
    bad_ts = int(df[ts_col].isna().sum())
    if bad_ts > 0:
        raise ValueError(f'Invalid timestamps in {input_file.name}: {bad_ts}')

    rows_in = int(len(df))
    dup_in = int(df[ts_col].duplicated().sum())
    df = df.sort_values(ts_col).drop_duplicates(subset=[ts_col], keep='last').reset_index(drop=True)
    rows_out = int(len(df))

    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_file, index=False)
    return {
        'rows_in': rows_in,
        'rows_out': rows_out,
        'duplicates_removed': int(rows_in - rows_out),
        'duplicates_detected': dup_in,
    }


def choose_input_file(input_root: Path, symbol: str) -> Path:
    pats = sorted(input_root.glob(f'{symbol.lower()}_*.parquet'))
    if len(pats) != 1:
        if len(pats) == 0:
            raise FileNotFoundError(f'No input parquet found for {symbol} in {input_root}')
        raise RuntimeError(f'Ambiguous input parquet for {symbol}: found {len(pats)} files (expected 1).')
    return pats[0].resolve()


def run_cmd(cmd: List[str], stdout_log: Path, stderr_log: Path, logger: logging.Logger):
    logger.info('CMD: %s', ' '.join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True)
    stdout_log.write_text(p.stdout or '', encoding='utf-8')
    stderr_log.write_text(p.stderr or '', encoding='utf-8')
    if p.returncode != 0:
        raise RuntimeError(f'Command failed ({p.returncode}): {" ".join(cmd)}')


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f'Missing JSON: {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def extract_hardened_nooverlap_metrics(phase14_summary: dict) -> Dict[str, float]:
    src = phase14_summary.get('hardened_test_metrics_nooverlap')
    if not isinstance(src, dict):
        raise RuntimeError('phase14 summary missing hardened_test_metrics_nooverlap')

    out: Dict[str, float] = {}
    for lvl in ['x1', 'x2', 'x3']:
        block = src.get(lvl)
        if not isinstance(block, dict):
            raise RuntimeError(f'phase14 summary missing hardened_test_metrics_nooverlap.{lvl}')
        for k in ['sharpe_ann', 'cum_net_return', 'max_drawdown', 'trade_rate', 'turnover_rate']:
            if k not in block:
                raise RuntimeError(f'phase14 summary missing hardened_test_metrics_nooverlap.{lvl}.{k}')
            out[f'hardened_test_nooverlap_{lvl}_{k}'] = float(block[k])
    return out


def configure_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger('phase15_multi_fx')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, mode='w')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def main():
    parser = argparse.ArgumentParser(description='Phase15 multi-FX benchmark 10->7 (phase13 -> phase14).')
    parser.add_argument('--input_root', required=True, type=str, help='Root folder with M1 parquet files (e.g., parquet_all).')
    parser.add_argument('--out_dir', required=True, type=str, help='Absolute output directory for this phase15 run.')
    parser.add_argument('--strict_mode', default=1, type=int, choices=[0, 1])
    parser.add_argument('--symbols', default=','.join(DEFAULT_SYMBOLS), type=str)
    parser.add_argument('--n_select', default=7, type=int)
    parser.add_argument('--priors_source', default='windows_only', choices=['windows_only', 'windows_plus_holdout'])
    parser.add_argument('--purged_gap_bars', default=24, type=int)
    parser.add_argument('--eval_stride', default=24, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--fail_fast', default=0, type=int, choices=[0, 1])
    parser.add_argument('--phase13_script', default=str(ROOT / 'phases/phase13_pipeline_cli/run_phase13_pipeline.py'), type=str)
    parser.add_argument('--phase14_script', default=str(ROOT / 'phases/phase14_production_hardening/run_phase14_1_hardening.py'), type=str)
    parser.add_argument('--phase13_sanity_script', default=str(ROOT / 'phases/phase13_pipeline_cli/run_phase13_sanity_checks.py'), type=str)
    args = parser.parse_args()

    np.random.seed(int(args.seed))

    symbols = parse_symbols(args.symbols)
    for s in symbols:
        if s not in DEFAULT_COST_PIPS:
            raise ValueError(f'Missing default conservative cost profile for symbol={s}')

    cfg = Phase15Config(
        input_root=str(Path(args.input_root).resolve()),
        out_dir=str(Path(args.out_dir).resolve()),
        strict_mode=bool(int(args.strict_mode)),
        symbols=symbols,
        n_select=int(args.n_select),
        priors_source=str(args.priors_source),
        purged_gap_bars=int(args.purged_gap_bars),
        eval_stride=int(args.eval_stride),
        seed=int(args.seed),
        fail_fast=bool(int(args.fail_fast)),
        phase13_script=str(Path(args.phase13_script).resolve()),
        phase14_script=str(Path(args.phase14_script).resolve()),
        phase13_sanity_script=str(Path(args.phase13_sanity_script).resolve()),
    )

    input_root = Path(cfg.input_root)
    if not input_root.exists():
        raise FileNotFoundError(f'input_root not found: {input_root}')

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    phase13_runs_root = out_dir / 'phase13_runs'
    phase14_runs_root = out_dir / 'phase14_runs'
    logs_root = out_dir / 'logs'
    dedup_inputs_root = out_dir / 'prepared_inputs'
    phase13_runs_root.mkdir(parents=True, exist_ok=True)
    phase14_runs_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    dedup_inputs_root.mkdir(parents=True, exist_ok=True)

    logger = configure_logger(out_dir / 'phase15.log')
    run_tag = now_tag()

    manifest = {
        'status': 'RUNNING',
        'created_at_utc': now_utc_iso(),
        'run_tag': run_tag,
        'config': asdict(cfg),
        'assumptions': {
            'paper_only': True,
            'live_orders': False,
            'ranking_primary_metric': 'hardened_test_metrics_nooverlap.x1.sharpe_ann',
        },
        'script_hashes': {
            'phase15_script_sha256': sha256_file(Path(__file__).resolve()),
            'phase13_script_sha256': sha256_file(Path(cfg.phase13_script)),
            'phase14_script_sha256': sha256_file(Path(cfg.phase14_script)),
            'phase13_sanity_script_sha256': sha256_file(Path(cfg.phase13_sanity_script)),
        },
    }
    safe_json_write(out_dir / 'phase15_manifest.json', manifest)

    strict_temporal_eval = 1 if cfg.strict_mode else 0
    strict_only = 1 if cfg.strict_mode else 0
    purged_gap = cfg.purged_gap_bars if cfg.strict_mode else 0
    eval_stride = cfg.eval_stride if cfg.strict_mode else 1

    results: List[dict] = []
    failures: List[dict] = []

    for symbol in cfg.symbols:
        logger.info('=== SYMBOL %s START ===', symbol)
        row = {
            'symbol': symbol,
            'status': 'FAILED',
            'error': None,
            'cost_pips': float(DEFAULT_COST_PIPS[symbol]),
            'input_m1': None,
            'phase13_run_dir': None,
            'phase14_run_dir': None,
            'phase13_sanity_json': None,
            'prepared_input_m1': None,
            'rows_in': None,
            'rows_out': None,
            'duplicates_removed': None,
            'leak_audit_ok': False,
            'phase14_metrics_ok': False,
            'strict_mode': bool(cfg.strict_mode),
        }

        try:
            input_file_raw = choose_input_file(input_root, symbol)
            row['input_m1'] = str(input_file_raw)

            prepared_input = (dedup_inputs_root / f'{symbol.lower()}_phase15_dedup.parquet').resolve()
            prep_meta = prepare_dedup_input(input_file_raw, prepared_input)
            row['prepared_input_m1'] = str(prepared_input)
            row['rows_in'] = int(prep_meta['rows_in'])
            row['rows_out'] = int(prep_meta['rows_out'])
            row['duplicates_removed'] = int(prep_meta['duplicates_removed'])

            phase13_run_id = f'phase15_{run_tag}_{symbol}_phase13'
            phase13_run_dir = (phase13_runs_root / phase13_run_id).resolve()
            row['phase13_run_dir'] = str(phase13_run_dir)

            cmd13 = [
                sys.executable,
                cfg.phase13_script,
                '--input_m1', str(prepared_input),
                '--symbol', symbol,
                '--mode', 'infer_frozen',
                '--cost_pips', str(DEFAULT_COST_PIPS[symbol]),
                '--priors_source', cfg.priors_source,
                '--strict_temporal_eval', str(strict_temporal_eval),
                '--out_dir', str(phase13_runs_root.resolve()),
                '--run_id', phase13_run_id,
            ]
            run_cmd(
                cmd13,
                logs_root / f'{symbol}_phase13.stdout.log',
                logs_root / f'{symbol}_phase13.stderr.log',
                logger,
            )

            p13_summary = load_json(phase13_run_dir / 'phase13_summary.json')
            if str(p13_summary.get('status')) != 'SUCCESS':
                raise RuntimeError('phase13 status != SUCCESS')

            leak_path = Path(
                p13_summary.get('outputs', {}).get('phase13_leak_audit_json', str(phase13_run_dir / 'phase13_leak_audit.json'))
            )
            leak = load_json(leak_path)
            if not bool(leak.get('all_eval_ts_gt_prior_max', False)):
                raise RuntimeError('Leak gate failed: all_eval_ts_gt_prior_max=false')
            if cfg.strict_mode and not bool(leak.get('strict_temporal_eval', False)):
                raise RuntimeError('Leak gate failed: strict_temporal_eval=false while strict_mode=1')
            row['leak_audit_ok'] = True

            cmd_sanity = [
                sys.executable,
                cfg.phase13_sanity_script,
                '--run_dir', str(phase13_run_dir),
                '--n_perm', '500',
                '--n_boot', '500',
                '--n_rand', '300',
                '--block_len', '24',
                '--shift_stride', str(eval_stride),
            ]
            run_cmd(
                cmd_sanity,
                logs_root / f'{symbol}_phase13_sanity.stdout.log',
                logs_root / f'{symbol}_phase13_sanity.stderr.log',
                logger,
            )

            sanity_json = phase13_run_dir / 'phase13_sanity_checks.json'
            if not sanity_json.exists():
                raise RuntimeError('phase13 sanity checks missing')
            row['phase13_sanity_json'] = str(sanity_json)

            phase14_run_dir = (phase14_runs_root / f'phase15_{run_tag}_{symbol}_phase14').resolve()
            row['phase14_run_dir'] = str(phase14_run_dir)
            cmd14 = [
                sys.executable,
                cfg.phase14_script,
                '--phase13_run_dir', str(phase13_run_dir),
                '--out_dir', str(phase14_run_dir),
                '--cost_pips', str(DEFAULT_COST_PIPS[symbol]),
                '--strict_only', str(strict_only),
                '--purged_gap_bars', str(purged_gap),
                '--eval_stride', str(eval_stride),
            ]
            run_cmd(
                cmd14,
                logs_root / f'{symbol}_phase14.stdout.log',
                logs_root / f'{symbol}_phase14.stderr.log',
                logger,
            )

            p14_summary = load_json(phase14_run_dir / 'phase14_summary.json')
            if str(p14_summary.get('status')) != 'SUCCESS':
                raise RuntimeError('phase14 status != SUCCESS')

            metrics = extract_hardened_nooverlap_metrics(p14_summary)
            row.update(metrics)
            row['phase14_metrics_ok'] = True

            x2_sh = float(row['hardened_test_nooverlap_x2_sharpe_ann'])
            x2_cum = float(row['hardened_test_nooverlap_x2_cum_net_return'])
            x1_sh = float(row['hardened_test_nooverlap_x1_sharpe_ann'])
            x1_cum = float(row['hardened_test_nooverlap_x1_cum_net_return'])
            row['quality_pass'] = bool((x1_sh > 0.0) and (x1_cum > 0.0))
            row['guardrail_x2_pass'] = bool((x2_sh > 0.0) and (x2_cum > 0.0))

            row['rank_primary_x1_sharpe'] = x1_sh
            row['rank_tiebreak_x2_sharpe'] = x2_sh
            row['rank_tiebreak_x1_max_drawdown'] = float(row['hardened_test_nooverlap_x1_max_drawdown'])

            row['status'] = 'SUCCESS'
            logger.info('=== SYMBOL %s SUCCESS ===', symbol)

        except Exception as e:
            row['error'] = f'{type(e).__name__}: {e}'
            failures.append({'symbol': symbol, 'error': row['error']})
            logger.error('=== SYMBOL %s FAILED === %s', symbol, row['error'])
            if cfg.fail_fast:
                results.append(row)
                break

        results.append(row)

    results_df = pd.DataFrame(results)
    results_csv = out_dir / 'phase15_symbol_results.csv'
    results_df.to_csv(results_csv, index=False)

    ok_df = results_df[results_df['status'] == 'SUCCESS'].copy()
    if len(ok_df) > 0:
        ok_df = ok_df.sort_values(
            by=['rank_primary_x1_sharpe', 'rank_tiebreak_x2_sharpe', 'rank_tiebreak_x1_max_drawdown'],
            ascending=[False, False, False],
        ).reset_index(drop=True)

    ranking_top10 = ok_df.head(10).copy()
    ranking_csv = out_dir / 'phase15_ranking_top10.csv'
    ranking_top10.to_csv(ranking_csv, index=False)

    selected = pd.DataFrame(columns=ranking_top10.columns.tolist() + ['borderline', 'selection_rank'])
    if len(ok_df) > 0:
        qualified = ok_df[ok_df.get('quality_pass', False) == True].copy()  # noqa: E712
        sel_main = qualified.head(cfg.n_select).copy()
        sel_main['borderline'] = False

        if len(sel_main) < cfg.n_select:
            need = cfg.n_select - len(sel_main)
            remain = ok_df[~ok_df['symbol'].isin(sel_main['symbol'])].copy()
            remain = remain.sort_values(by=['hardened_test_nooverlap_x1_sharpe_ann'], ascending=[False])
            fill = remain.head(need).copy()
            fill['borderline'] = True
            selected = pd.concat([sel_main, fill], ignore_index=True)
        else:
            selected = sel_main.copy()

        selected['selection_rank'] = np.arange(1, len(selected) + 1)

    selected_csv = out_dir / 'phase15_selected7.csv'
    selected.to_csv(selected_csv, index=False)

    failures_json = out_dir / 'phase15_failures.json'
    safe_json_write(failures_json, {'n_failures': int(len(failures)), 'failures': failures})

    summary = {
        'status': 'SUCCESS' if len(failures) == 0 else 'PARTIAL_SUCCESS',
        'created_at_utc': manifest['created_at_utc'],
        'finished_at_utc': now_utc_iso(),
        'strict_mode': bool(cfg.strict_mode),
        'symbols_requested': list(cfg.symbols),
        'symbols_succeeded': ok_df['symbol'].tolist() if len(ok_df) else [],
        'symbols_failed': [x['symbol'] for x in failures],
        'n_requested': int(len(cfg.symbols)),
        'n_succeeded': int(len(ok_df)),
        'n_failed': int(len(failures)),
        'n_selected': int(len(selected)),
        'n_select_target': int(cfg.n_select),
        'selection_rule': {
            'primary': 'hardened_test_metrics_nooverlap.x1.sharpe_ann (desc)',
            'tiebreak_1': 'hardened_test_metrics_nooverlap.x2.sharpe_ann (desc)',
            'tiebreak_2': 'hardened_test_metrics_nooverlap.x1.max_drawdown (desc)',
            'quality_filter': 'x1.sharpe_ann > 0 and x1.cum_net_return > 0',
            'guardrail': 'x2.sharpe_ann > 0 and x2.cum_net_return > 0 (informativo en columna guardrail_x2_pass)',
            'fallback_if_less_than_target': 'fill with best x1.sharpe_ann and borderline=true',
        },
        'cost_profile_pips': {s: float(DEFAULT_COST_PIPS[s]) for s in cfg.symbols},
        'outputs': {
            'symbol_results_csv': str(results_csv),
            'ranking_top10_csv': str(ranking_csv),
            'selected7_csv': str(selected_csv),
            'failures_json': str(failures_json),
            'phase15_manifest_json': str(out_dir / 'phase15_manifest.json'),
            'phase15_log': str(out_dir / 'phase15.log'),
        },
        'paper_only': True,
        'live_orders': False,
    }

    summary_path = out_dir / 'phase15_summary.json'
    safe_json_write(summary_path, summary)

    logger.info('Phase15 finished with status=%s | n_succeeded=%s | n_selected=%s', summary['status'], summary['n_succeeded'], summary['n_selected'])
    print(f'SUCCESS: {summary_path}')


if __name__ == '__main__':
    main()
