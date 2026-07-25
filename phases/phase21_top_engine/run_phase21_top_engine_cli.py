#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import math
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / 'phases/phase21_top_engine/config_phase21.json'


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def now_tag() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')


def parse_utc_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        s = str(ts)
        if s.endswith('Z'):
            s = s[:-1] + '+00:00'
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def age_hours(ts: Optional[str]) -> Optional[float]:
    dt = parse_utc_iso(ts)
    if dt is None:
        return None
    return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0)


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


def safe_json_write(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
    tmp.replace(path)


def append_jsonl(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=str) + '\n')


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None


def emit_alert(path: Path, level: str, message: str, payload: Dict[str, Any]):
    append_jsonl(
        path,
        {
            'ts_utc': now_utc_iso(),
            'level': str(level).upper(),
            'message': str(message),
            'payload': payload,
        },
    )


def load_config(path: Path) -> Dict[str, Any]:
    txt = path.read_text(encoding='utf-8')
    try:
        import yaml  # type: ignore

        out = yaml.safe_load(txt)
        if isinstance(out, dict):
            return out
    except Exception:
        pass
    out = json.loads(txt)
    if not isinstance(out, dict):
        raise ValueError('config must parse to dict/object')
    return out


def setup_logger(path: Path) -> logging.Logger:
    lg = logging.getLogger('phase21_top_engine')
    lg.setLevel(logging.INFO)
    lg.handlers.clear()

    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    fh = logging.FileHandler(path, mode='w', encoding='utf-8')
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    lg.addHandler(fh)
    lg.addHandler(sh)
    return lg


def resolve_path(x: str) -> Path:
    return Path(str(x)).expanduser().resolve()


def ensure_cols(df: pd.DataFrame, cols: Iterable[str], what: str):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise RuntimeError(f'{what} missing columns: {miss}')


def run_cmd_with_retry(
    cmd: List[str],
    stage_name: str,
    logs_dir: Path,
    alerts_file: Path,
    logger: logging.Logger,
    retries: int,
    backoff_sec: int,
):
    logs_dir.mkdir(parents=True, exist_ok=True)
    last = None
    for i in range(1, int(retries) + 1):
        logger.info('[%s] attempt %s/%s', stage_name, i, retries)
        p = subprocess.run(cmd, capture_output=True, text=True)
        (logs_dir / f'{stage_name}_attempt{i}.stdout.log').write_text(p.stdout or '', encoding='utf-8')
        (logs_dir / f'{stage_name}_attempt{i}.stderr.log').write_text(p.stderr or '', encoding='utf-8')
        if p.returncode == 0:
            return
        last = RuntimeError(f'{stage_name} failed rc={p.returncode}')
        emit_alert(
            alerts_file,
            'WARNING',
            f'{stage_name}_failed',
            {'attempt': i, 'retries': retries, 'returncode': p.returncode, 'cmd': cmd},
        )
        if i < int(retries):
            time.sleep(int(backoff_sec))
    raise last if last is not None else RuntimeError(f'{stage_name} failed')


def init_contract(run_dir: Path, command: str, cfg_path: Path, cfg: Dict[str, Any]) -> Dict[str, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    files = {
        'manifest': run_dir / 'phase21_manifest.json',
        'status': run_dir / 'phase21_status.json',
        'alerts': run_dir / 'phase21_alerts.jsonl',
        'guardrails': run_dir / 'phase21_guardrails.json',
        'snapshot': run_dir / 'phase21_snapshot.json',
        'log': run_dir / 'phase21.log',
        'links': run_dir / 'links_to_child_runs.json',
    }

    manifest = {
        'phase': 'phase21',
        'status': 'RUNNING',
        'created_at_utc': now_utc_iso(),
        'command': command,
        'paper_only': True,
        'live_order_execution': False,
        'config_path': str(cfg_path),
        'config': cfg,
        'stages': {},
        'child_runs': {},
        'script_hashes': {},
    }
    safe_json_write(files['manifest'], manifest)

    status = {
        'status': 'RUNNING',
        'command': command,
        'created_at_utc': manifest['created_at_utc'],
        'updated_at_utc': manifest['created_at_utc'],
        'paper_only': True,
    }
    safe_json_write(files['status'], status)
    return files


def update_manifest_stage(path: Path, stage: str, stage_status: str, details: Optional[Dict[str, Any]] = None):
    m = read_json(path) or {}
    s = m.setdefault('stages', {})
    r = s.setdefault(stage, {})
    r['status'] = stage_status
    r['updated_at_utc'] = now_utc_iso()
    if details:
        r.update(details)
    safe_json_write(path, m)


def update_status(path: Path, status: str, stage: str, extra: Optional[Dict[str, Any]] = None):
    d = read_json(path) or {}
    d.update({'status': status, 'stage': stage, 'updated_at_utc': now_utc_iso()})
    if extra:
        d.update(extra)
    safe_json_write(path, d)


def set_manifest_final(path: Path, status: str, extra: Optional[Dict[str, Any]] = None):
    m = read_json(path) or {}
    m['status'] = status
    m['ended_at_utc'] = now_utc_iso()
    if extra:
        m.update(extra)
    safe_json_write(path, m)


def load_top7_selected(path: Path) -> List[str]:
    df = pd.read_csv(path)
    ensure_cols(df, ['symbol'], f'{path.name}')
    syms = [str(x).upper() for x in df['symbol'].tolist() if str(x).strip()]
    # stable unique
    out: List[str] = []
    seen = set()
    for s in syms:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out


def choose_file_mode_input(input_root: Path, symbol: str) -> Path:
    pats = sorted(input_root.glob(f'{symbol.lower()}_*.parquet'))
    if len(pats) != 1:
        if len(pats) == 0:
            raise FileNotFoundError(f'No parquet for symbol={symbol} in {input_root}')
        raise RuntimeError(f'Ambiguous parquet for symbol={symbol}: {len(pats)} files')
    return pats[0].resolve()


def validate_m1(df: pd.DataFrame, symbol: str, min_rows: int, max_gap_rate: float) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    ensure_cols(df, ['timestamp', 'open', 'high', 'low', 'close'], f'M1[{symbol}]')
    d = df.copy()
    d['timestamp'] = pd.to_datetime(d['timestamp'], utc=True, errors='coerce')
    if d['timestamp'].isna().any():
        raise RuntimeError(f'M1[{symbol}] invalid timestamp rows={int(d["timestamp"].isna().sum())}')

    dup = int(d['timestamp'].duplicated().sum())
    if dup > 0:
        raise RuntimeError(f'M1[{symbol}] duplicated timestamps={dup}')

    d = d.sort_values('timestamp').reset_index(drop=True)
    if not d['timestamp'].is_monotonic_increasing:
        raise RuntimeError(f'M1[{symbol}] timestamp not monotonic increasing')

    o = pd.to_numeric(d['open'], errors='coerce')
    h = pd.to_numeric(d['high'], errors='coerce')
    l = pd.to_numeric(d['low'], errors='coerce')
    c = pd.to_numeric(d['close'], errors='coerce')
    bad_nan = int(pd.concat([o, h, l, c], axis=1).isna().any(axis=1).sum())
    if bad_nan > 0:
        raise RuntimeError(f'M1[{symbol}] NaN OHLC rows={bad_nan}')

    bad_hl = int((h < l).sum())
    bad_oh = int(((o > h) | (o < l) | (c > h) | (c < l)).sum())
    if bad_hl > 0 or bad_oh > 0:
        raise RuntimeError(f'M1[{symbol}] OHLC incoherent bad_hl={bad_hl} bad_open_close={bad_oh}')

    n = int(len(d))
    if n < int(min_rows):
        raise RuntimeError(f'M1[{symbol}] too few rows n={n} < min_rows={min_rows}')

    t0 = d['timestamp'].iloc[0]
    t1 = d['timestamp'].iloc[-1]
    span_min = int((t1 - t0).total_seconds() // 60) + 1
    expected = max(span_min, 1)
    gap_rate = float(max(0.0, expected - n) / expected)
    if gap_rate > float(max_gap_rate):
        raise RuntimeError(f'M1[{symbol}] gap_rate={gap_rate:.4f} > max_gap_rate={max_gap_rate:.4f}')

    stats = {
        'rows': n,
        'timestamp_min': str(t0),
        'timestamp_max': str(t1),
        'gap_rate': gap_rate,
        'expected_rows_1m': expected,
    }
    return d, stats


def _script_path(cfg: Dict[str, Any], key: str) -> Path:
    return resolve_path(cfg['paths'][key])


def _cost_map(cfg: Dict[str, Any]) -> Dict[str, float]:
    out = {str(k).upper(): float(v) for k, v in (cfg.get('cost_pips_by_symbol') or {}).items()}
    if not out:
        raise RuntimeError('config missing cost_pips_by_symbol')
    return out


def _write_run_snapshot(path: Path, payload: Dict[str, Any]):
    safe_json_write(path, payload)


def _fetch_m1_one(
    symbol: str,
    cfg: Dict[str, Any],
    mode: str,
    out_file: Path,
    lookback_hours: int,
    logs_dir: Path,
    alerts_file: Path,
    logger: logging.Logger,
):
    if mode == 'file':
        input_root = resolve_path(cfg['paths']['file_mode_input_root'])
        src = choose_file_mode_input(input_root, symbol)
        shutil.copy2(src, out_file)
        return

    cmd = [
        sys.executable,
        str(_script_path(cfg, 'phase19_fetch_script')),
        '--symbol', symbol,
        '--env_file', str(_script_path(cfg, 'ctrader_env_file')),
        '--output', str(out_file),
        '--lookback_hours', str(int(lookback_hours)),
        '--count', str(int(cfg['fetch'].get('count', 0))),
        '--account_id', str(int(cfg['fetch'].get('account_id', 0))),
        '--demo', str(int(cfg['fetch'].get('demo', 1))),
        '--require_demo_account', '1',
    ]
    run_cmd_with_retry(
        cmd=cmd,
        stage_name=f'fetch_{symbol}',
        logs_dir=logs_dir,
        alerts_file=alerts_file,
        logger=logger,
        retries=int(cfg['retry'].get('max', 3)),
        backoff_sec=int(cfg['retry'].get('backoff_sec', 20)),
    )


def _run_phase13_symbol(
    symbol: str,
    input_m1: Path,
    cost_pips: float,
    cfg: Dict[str, Any],
    out_root: Path,
    run_id: str,
    logs_dir: Path,
    alerts_file: Path,
    logger: logging.Logger,
) -> Path:
    run_dir = out_root / run_id
    cmd = [
        sys.executable,
        str(_script_path(cfg, 'phase13_script')),
        '--input_m1', str(input_m1),
        '--symbol', symbol,
        '--mode', 'infer_frozen',
        '--cost_pips', str(float(cost_pips)),
        '--priors_source', 'windows_only',
        '--strict_temporal_eval', '1',
        '--out_dir', str(out_root),
        '--run_id', run_id,
    ]
    run_cmd_with_retry(
        cmd=cmd,
        stage_name=f'phase13_{symbol}',
        logs_dir=logs_dir,
        alerts_file=alerts_file,
        logger=logger,
        retries=int(cfg['retry'].get('max', 3)),
        backoff_sec=int(cfg['retry'].get('backoff_sec', 20)),
    )

    s = read_json(run_dir / 'phase13_summary.json') or {}
    if str(s.get('status', '')).upper() != 'SUCCESS':
        raise RuntimeError(f'phase13 failed for {symbol}')

    leak = read_json(run_dir / 'phase13_leak_audit.json') or {}
    if bool(leak.get('all_eval_ts_gt_prior_max', False)) is not True:
        raise RuntimeError(f'phase13 leak gate failed for {symbol}')

    return run_dir


def _run_phase14_symbol(
    symbol: str,
    phase13_run_dir: Path,
    cost_pips: float,
    cfg: Dict[str, Any],
    out_root: Path,
    run_id: str,
    logs_dir: Path,
    alerts_file: Path,
    logger: logging.Logger,
) -> Path:
    run_dir = out_root / run_id
    strict_cfg = cfg['strict_eval']
    cmd = [
        sys.executable,
        str(_script_path(cfg, 'phase14_script')),
        '--phase13_run_dir', str(phase13_run_dir),
        '--out_dir', str(run_dir),
        '--cost_pips', str(float(cost_pips)),
        '--strict_only', '1',
        '--purged_gap_bars', str(int(strict_cfg.get('purged_gap_bars', 24))),
        '--eval_stride', str(int(strict_cfg.get('eval_stride', 8))),
    ]
    run_cmd_with_retry(
        cmd=cmd,
        stage_name=f'phase14_{symbol}',
        logs_dir=logs_dir,
        alerts_file=alerts_file,
        logger=logger,
        retries=int(cfg['retry'].get('max', 3)),
        backoff_sec=int(cfg['retry'].get('backoff_sec', 20)),
    )

    s = read_json(run_dir / 'phase14_summary.json') or {}
    if str(s.get('status', '')).upper() != 'SUCCESS':
        raise RuntimeError(f'phase14 failed for {symbol}')

    x2 = (((s.get('hardened_test_metrics_nooverlap') or {}).get('x2')) or {})
    if not (
        isinstance(x2.get('sharpe_ann', None), (int, float))
        and isinstance(x2.get('cum_net_return', None), (int, float))
    ):
        raise RuntimeError(f'phase14 x2 metrics missing for {symbol}')

    return run_dir


def _build_phase15_bridge(symbol_rows: List[Dict[str, Any]], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in symbol_rows:
        rows.append(
            {
                'symbol': r['symbol'],
                'status': 'SUCCESS',
                'phase14_run_dir': str(r['phase14_run_dir']),
                'phase13_run_dir': str(r['phase13_run_dir']),
                'cost_pips': float(r['cost_pips']),
            }
        )
    d = pd.DataFrame(rows).sort_values('symbol').reset_index(drop=True)
    d.to_csv(out_dir / 'phase15_selected7.csv', index=False)
    d.to_csv(out_dir / 'phase15_symbol_results.csv', index=False)
    safe_json_write(
        out_dir / 'phase15_summary.json',
        {
            'status': 'SUCCESS',
            'phase': 'phase21_bridge',
            'n_selected': int(len(d)),
            'symbols': d['symbol'].tolist(),
            'created_at_utc': now_utc_iso(),
        },
    )


def _run_phase16_chain(
    bridge_dir: Path,
    out_root: Path,
    cfg: Dict[str, Any],
    logs_dir: Path,
    alerts_file: Path,
    logger: logging.Logger,
) -> Dict[str, Path]:
    out_root.mkdir(parents=True, exist_ok=True)

    p16_1 = out_root / 'phase16_1'
    p16_2 = out_root / 'phase16_2'
    p16_3 = out_root / 'phase16_3'
    p16_4 = out_root / 'phase16_4'

    c16 = cfg.get('phase16', {})

    cmd1 = [
        sys.executable,
        str(_script_path(cfg, 'phase16_1_script')),
        '--phase15_run_dir', str(bridge_dir),
        '--out_dir', str(p16_1),
        '--universe', 'selected7',
        '--window_len', str(int(c16.get('window_len', 48))),
        '--window_stride', str(int(c16.get('window_stride', 1))),
        '--train_ratio', str(float(c16.get('train_ratio', 0.70))),
        '--val_ratio', str(float(c16.get('val_ratio', 0.15))),
        '--scaler_mode', str(c16.get('scaler_mode', 'per_symbol')),
        '--seed', str(int(c16.get('seed', 42))),
    ]
    run_cmd_with_retry(cmd1, 'phase16_1', logs_dir, alerts_file, logger, int(cfg['retry'].get('max', 3)), int(cfg['retry'].get('backoff_sec', 20)))

    cmd2 = [
        sys.executable,
        str(_script_path(cfg, 'phase16_2_script')),
        '--phase16_run_dir', str(p16_1),
        '--out_dir', str(p16_2),
        '--k', str(int(c16.get('k', 0))),
        '--k_min', str(int(c16.get('k_min', 6))),
        '--k_max', str(int(c16.get('k_max', 16))),
        '--seed', str(int(c16.get('seed', 42))),
    ]
    run_cmd_with_retry(cmd2, 'phase16_2', logs_dir, alerts_file, logger, int(cfg['retry'].get('max', 3)), int(cfg['retry'].get('backoff_sec', 20)))

    cmd3 = [
        sys.executable,
        str(_script_path(cfg, 'phase16_3_script')),
        '--phase16_2_run_dir', str(p16_2),
        '--out_dir', str(p16_3),
        '--seq_len', str(int(c16.get('seq_len', 32))),
        '--k_context', str(int(c16.get('k_context', 6))),
        '--seed', str(int(c16.get('seed', 42))),
    ]
    run_cmd_with_retry(cmd3, 'phase16_3', logs_dir, alerts_file, logger, int(cfg['retry'].get('max', 3)), int(cfg['retry'].get('backoff_sec', 20)))

    cmd4 = [
        sys.executable,
        str(_script_path(cfg, 'phase16_4_script')),
        '--phase16_3_run_dir', str(p16_3),
        '--phase16_1_run_dir', str(p16_1),
        '--out_dir', str(p16_4),
        '--window_len', str(int(c16.get('portfolio_window_len', 24))),
        '--window_stride', str(int(c16.get('window_stride', 1))),
        '--train_ratio', str(float(c16.get('train_ratio', 0.70))),
        '--val_ratio', str(float(c16.get('val_ratio', 0.15))),
        '--min_target_symbols', str(int(c16.get('min_target_symbols', 3))),
    ]
    run_cmd_with_retry(cmd4, 'phase16_4', logs_dir, alerts_file, logger, int(cfg['retry'].get('max', 3)), int(cfg['retry'].get('backoff_sec', 20)))

    for p in [p16_1 / 'phase16_summary.json', p16_2 / 'phase16_2_summary.json', p16_3 / 'phase16_3_summary.json', p16_4 / 'phase16_4_summary.json']:
        if not p.exists():
            raise RuntimeError(f'Missing phase16 artifact: {p}')

    return {
        'phase16_1': p16_1,
        'phase16_2': p16_2,
        'phase16_3': p16_3,
        'phase16_4': p16_4,
    }


def _run_phase18_chain(
    phase16_4_dir: Path,
    out_root: Path,
    cfg: Dict[str, Any],
    logs_dir: Path,
    alerts_file: Path,
    logger: logging.Logger,
) -> Dict[str, Path]:
    out_root.mkdir(parents=True, exist_ok=True)
    p18_2 = out_root / 'phase18_2'
    p18_5 = out_root / 'phase18_5'

    c18 = cfg.get('phase18', {})

    # Adapt fold sizes to available windows from recent pull.
    s16_4 = read_json(phase16_4_dir / 'phase16_4_summary.json') or {}
    n_windows = int(s16_4.get('window_count', 0) or 0)
    min_train_cfg = int(c18.get('min_train_windows', 1200))
    val_cfg = int(c18.get('val_windows', 128))
    test_cfg = int(c18.get('test_windows', 256))
    step_cfg = int(c18.get('step_windows', 256))

    min_train_eff = min_train_cfg
    val_eff = val_cfg
    test_eff = test_cfg
    step_eff = step_cfg

    if n_windows > 0:
        reserve = max(32, min(step_cfg, 128))
        if min_train_eff + val_eff + test_eff + reserve > n_windows:
            val_eff = min(val_cfg, max(48, int(round(0.15 * n_windows))))
            test_eff = min(test_cfg, max(72, int(round(0.20 * n_windows))))
            min_train_eff = min(min_train_cfg, max(160, n_windows - val_eff - test_eff - reserve))
            step_eff = min(step_cfg, max(64, int(round(0.25 * n_windows))))
        if min_train_eff + val_eff + test_eff > n_windows:
            raise RuntimeError(
                f'Insufficient windows for phase18_2 even after adaptation: '
                f'n_windows={n_windows}, min_train={min_train_eff}, val={val_eff}, test={test_eff}'
            )
    else:
        logger.warning('phase16_4 window_count unavailable; using configured phase18 window sizes.')

    logger.info(
        'phase18 window config effective: n_windows=%s configured(train/val/test/step)=%s/%s/%s/%s '
        'effective=%s/%s/%s/%s',
        n_windows,
        min_train_cfg, val_cfg, test_cfg, step_cfg,
        min_train_eff, val_eff, test_eff, step_eff,
    )

    cmd18_2 = [
        sys.executable,
        str(_script_path(cfg, 'phase18_2_script')),
        '--phase16_4_run_dir', str(phase16_4_dir),
        '--out_dir', str(p18_2),
        '--min_train_windows', str(int(min_train_eff)),
        '--val_windows', str(int(val_eff)),
        '--test_windows', str(int(test_eff)),
        '--step_windows', str(int(step_eff)),
        '--max_gross_leverage', str(float(c18.get('max_gross_leverage', 6.0))),
        '--target_gross_leverage', str(float(c18.get('target_gross_leverage', 3.0))),
        '--min_target_mult', str(float(c18.get('min_target_mult', 0.2))),
        '--max_target_mult', str(float(c18.get('max_target_mult', 2.2))),
        '--loss_w_turnover', str(float(c18.get('loss_w_turnover', 0.19))),
        '--cost_per_turnover', str(float(c18.get('cost_per_turnover', 9e-05))),
        '--turnover_smooth_alpha', str(float(c18.get('turnover_smooth_alpha', 0.3))),
        '--exec_rebalance_deadband', str(float(c18.get('exec_rebalance_deadband', 0.24))),
        '--exec_rebalance_cooldown_bars', str(int(c18.get('exec_rebalance_cooldown_bars', 0))),
        '--exec_min_turnover', str(float(c18.get('exec_min_turnover', 0.22))),
        '--gate_turnover_ratio_max', str(float(c18.get('gate_turnover_ratio_max', 0.85))),
        '--robust_score_turn_penalty', str(float(c18.get('robust_score_turn_penalty', 1.0))),
        '--robust_score_dd_penalty', str(float(c18.get('robust_score_dd_penalty', 0.28))),
        '--robust_score_trade_penalty', str(float(c18.get('robust_score_trade_penalty', 0.35))),
        '--robust_score_rebalance_penalty', str(float(c18.get('robust_score_rebalance_penalty', 0.2))),
        '--robust_alpha_candidates', str(c18.get('robust_alpha_candidates', '0.15,0.25,0.35,0.45')),
        '--robust_mode', '1',
        '--robust_cost_multipliers', str(c18.get('robust_cost_multipliers', '2,3')),
        '--export_inference_checkpoint', '1',
        '--checkpoint_fold', str(c18.get('checkpoint_fold', 'last')),
        '--checkpoint_seed_mode', str(c18.get('checkpoint_seed_mode', 'all')),
    ]
    run_cmd_with_retry(cmd18_2, 'phase18_2', logs_dir, alerts_file, logger, int(cfg['retry'].get('max', 3)), int(cfg['retry'].get('backoff_sec', 20)))

    steps_pq = p18_2 / 'phase18_2_test_steps.parquet'
    if not steps_pq.exists():
        raise RuntimeError(f'Missing phase18_2 test steps: {steps_pq}')

    exec_cfg = cfg.get('execution', {})
    cmd18_5 = [
        sys.executable,
        str(_script_path(cfg, 'phase18_5_script')),
        '--input_steps_parquet', str(steps_pq),
        '--run_phase18_2', '0',
        '--policy', str(exec_cfg.get('policy', 'model_ctde_robust')),
        '--cost_mult', str(float(exec_cfg.get('cost_mult', 1.0))),
        '--account_equity_usd', str(float(exec_cfg.get('account_equity_usd', 3000.0))),
        '--account_leverage_cap', str(float(exec_cfg.get('account_leverage_cap', 10000.0))),
        '--max_margin_utilization', str(float(exec_cfg.get('max_margin_utilization', 0.35))),
        '--leverage_tiers', str(exec_cfg.get('leverage_tiers', '1000000:500,5000000:200,inf:100')),
        '--extra_cost_per_turnover', str(float(exec_cfg.get('extra_cost_per_turnover', 2e-05))),
        '--slippage_per_turnover', str(float(exec_cfg.get('slippage_per_turnover', 2e-05))),
        '--shield_dd_limit', str(float(exec_cfg.get('shield_dd_limit', -0.08))),
        '--shield_risk_mult', str(float(exec_cfg.get('shield_risk_mult', 0.55))),
        '--min_exec_scale', str(float(exec_cfg.get('min_exec_scale', 0.05))),
        '--out_dir', str(p18_5),
    ]
    run_cmd_with_retry(cmd18_5, 'phase18_5', logs_dir, alerts_file, logger, int(cfg['retry'].get('max', 3)), int(cfg['retry'].get('backoff_sec', 20)))

    for p in [p18_2 / 'phase18_2_summary.json', p18_2 / 'phase18_2_inference_checkpoint.pt', p18_5 / 'phase18_5_summary.json']:
        if not p.exists():
            raise RuntimeError(f'Missing phase18 artifact: {p}')

    return {'phase18_2': p18_2, 'phase18_5': p18_5}


def _build_phase21_checkpoint(phase18_2_dir: Path, phase18_5_dir: Path, out_ckpt: Path, cfg: Dict[str, Any], source_run_dir: Path):
    ckpt18 = torch.load(phase18_2_dir / 'phase18_2_inference_checkpoint.pt', map_location='cpu', weights_only=False)
    s18_2 = read_json(phase18_2_dir / 'phase18_2_summary.json') or {}
    s18_5 = read_json(phase18_5_dir / 'phase18_5_summary.json') or {}

    payload = {
        'format': 'phase21_top_engine_checkpoint_v1',
        'created_at_utc': now_utc_iso(),
        'source_run_dir': str(source_run_dir),
        'paper_only': True,
        'live_order_execution': False,
        'model': ckpt18,
        'training_summary': {
            'phase18_2_comparisons': s18_2.get('comparisons', {}),
            'phase18_2_symbols': s18_2.get('symbols', []),
            'phase18_2_config': (s18_2.get('config') or {}),
        },
        'execution_layer': {
            'phase18_5_config': s18_5.get('config', {}),
            'selected_policy': s18_5.get('selected_policy', 'model_ctde_robust'),
            'selected_cost_mult': float(s18_5.get('selected_cost_mult', 1.0) or 1.0),
            'leverage_tiers_parsed': s18_5.get('leverage_tiers_parsed', []),
            'docs_used': s18_5.get('docs_used', []),
        },
        'runtime': {
            'phase18_2_script': str(_script_path(cfg, 'phase18_2_script')),
            'phase18_5_script': str(_script_path(cfg, 'phase18_5_script')),
        },
    }

    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_ckpt)


def _load_module(module_path: Path, name_prefix: str):
    spec = importlib.util.spec_from_file_location(f'{name_prefix}_{int(time.time() * 1000)}', str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot load module: {module_path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _infer_steps_from_checkpoint(
    ckpt_path: Path,
    phase16_4_dir: Path,
    cfg: Dict[str, Any],
    days: Optional[int],
    out_steps_parquet: Path,
) -> Dict[str, Any]:
    blob = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if str(blob.get('format')) != 'phase21_top_engine_checkpoint_v1':
        raise RuntimeError(f'Unsupported checkpoint format: {blob.get("format")}')

    model_blob = blob['model']
    m_arch = model_blob['model_arch']
    model_states = model_blob['ensemble_state_dicts']
    if not isinstance(model_states, list) or len(model_states) == 0:
        raise RuntimeError('Checkpoint has no ensemble_state_dicts')

    p18_mod = _load_module(resolve_path(blob['runtime']['phase18_2_script']), 'phase18_2_mod')

    npz = np.load(phase16_4_dir / 'phase16_4_windows.npz')
    x_raw = npz['x'].astype(np.float32)
    y_ret1 = npz['y_ret1'].astype(np.float32)
    y_mask = npz['y_mask'].astype(np.float32)

    wm = pd.read_parquet(phase16_4_dir / 'phase16_4_windows_meta.parquet').copy()
    wm['timestamp_end'] = pd.to_datetime(wm['timestamp_end'], utc=True)
    if len(wm) != len(x_raw):
        raise RuntimeError(f'windows/meta mismatch: x={len(x_raw)} meta={len(wm)}')

    feat_cols_runtime = [ln.strip() for ln in (phase16_4_dir / 'phase16_4_feature_columns.txt').read_text(encoding='utf-8').splitlines() if ln.strip()]
    feat_cols_ckpt = list((model_blob.get('feature_schema') or {}).get('feature_columns', []))
    if feat_cols_runtime != feat_cols_ckpt:
        raise RuntimeError('Feature schema mismatch between checkpoint and current phase16_4 run')

    x_mean = np.asarray(model_blob['normalization']['x_mean'], dtype=np.float32)
    x_std = np.asarray(model_blob['normalization']['x_std'], dtype=np.float32)
    x_std = np.where(np.abs(x_std) < 1e-6, 1.0, x_std)
    xz = (x_raw - x_mean[None, None, :]) / x_std[None, None, :]

    if days is not None:
        ts_max = wm['timestamp_end'].max()
        ts_min = ts_max - pd.Timedelta(days=int(days))
        keep = (wm['timestamp_end'] >= ts_min).to_numpy(dtype=bool)
    else:
        keep = np.ones((len(wm),), dtype=bool)

    if int(keep.sum()) < 10:
        raise RuntimeError(f'Not enough windows after days filter: kept={int(keep.sum())}')

    xz_k = xz[keep]
    xraw_k = x_raw[keep]
    yret_k = y_ret1[keep]
    ymask_k = y_mask[keep]
    wm_k = wm.loc[keep].reset_index(drop=True)

    symbols = list((model_blob.get('feature_schema') or {}).get('symbols', []))
    pparams = model_blob.get('policy_params', {})

    teacher = p18_mod.build_teacher_weights_from_features(
        x_last_raw=xraw_k[:, -1, :],
        feature_cols=feat_cols_runtime,
        symbols=symbols,
        y_mask=ymask_k,
        target_gross=float(pparams.get('target_gross_leverage', 1.0)),
        max_gross=float(pparams.get('max_gross_leverage', 2.0)),
    )

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bs = int(cfg.get('inference', {}).get('batch_size', 512))

    fake_cfg = SimpleNamespace(
        weight_temperature=float(pparams.get('weight_temperature', 1.0)),
        min_target_mult=float(pparams.get('min_target_mult', 0.2)),
        max_target_mult=float(pparams.get('max_target_mult', 2.2)),
        target_gross_leverage=float(pparams.get('target_gross_leverage', 1.0)),
        max_gross_leverage=float(pparams.get('max_gross_leverage', 2.0)),
    )

    model_weights_all = []
    for st in model_states:
        model = p18_mod.MultiAgentCTDETransformer(
            input_dim=int(m_arch['input_dim']),
            seq_len=int(m_arch['seq_len']),
            n_assets=int(m_arch['n_assets']),
            d_model=int(m_arch['d_model']),
            nhead=int(m_arch['nhead']),
            num_layers=int(m_arch['num_layers']),
            ff_dim=int(m_arch['ff_dim']),
            dropout=float(m_arch['dropout']),
        ).to(dev)
        model.load_state_dict(st)
        model.eval()

        outs = []
        with torch.no_grad():
            for i in range(0, len(xz_k), bs):
                xb = torch.from_numpy(xz_k[i:i + bs].astype(np.float32)).to(dev)
                mb = torch.from_numpy(ymask_k[i:i + bs].astype(np.float32)).to(dev)
                out = model(xb)
                w, _, _ = p18_mod.build_hier_weights(
                    gate_logits=out['gate_logits'],
                    dir_raw=out['dir_raw'],
                    mask=mb,
                    budget_raw=out['budget_raw'],
                    cfg=fake_cfg,
                )
                outs.append(w.detach().cpu().numpy())
        model_weights_all.append(np.concatenate(outs, axis=0))

    w_model = np.mean(np.stack(model_weights_all, axis=0), axis=0)

    w_blend = p18_mod.build_blend_weights(
        w_model=w_model,
        w_teacher=teacher,
        y_mask=ymask_k,
        mix_teacher=float(pparams.get('teacher_mix', 0.5)),
        risk_scale=float(pparams.get('risk_scale', 1.0)),
        target_gross=float(pparams.get('target_gross_leverage', 1.0)),
        max_gross=float(pparams.get('max_gross_leverage', 2.0)),
    )

    m_cfg = model_blob.get('mask_and_safety', {})
    t_smooth = float(pparams.get('turnover_smooth_alpha', 0.70))
    c_turn = float((blob.get('training_summary', {}).get('phase18_2_config', {}) or {}).get('cost_per_turnover', 5e-05))

    steps_df, base_metrics = p18_mod.backtest_from_weights(
        y_ret1=yret_k,
        y_mask=ymask_k,
        w_raw=w_blend,
        timestamps=wm_k['timestamp_end'].to_numpy(),
        cost_per_turnover=float(c_turn),
        cost_mult=1.0,
        smooth_alpha=float(t_smooth),
        max_gross=float(pparams.get('max_gross_leverage', 2.0)),
        shield_dd_limit=float(m_cfg.get('shield_dd_limit', -0.08)),
        shield_risk_mult=float(m_cfg.get('shield_risk_mult', 0.55)),
        rebalance_deadband=float(m_cfg.get('exec_rebalance_deadband', 0.0)),
        rebalance_cooldown_bars=int(m_cfg.get('exec_rebalance_cooldown_bars', 0)),
        min_turnover_exec=float(m_cfg.get('exec_min_turnover', 0.0)),
    )

    out_steps_parquet.parent.mkdir(parents=True, exist_ok=True)
    steps_df.to_parquet(out_steps_parquet, index=False)

    latest_w = w_blend[-1]
    latest_symbols = list(symbols)
    latest_weights = {latest_symbols[i]: float(latest_w[i]) for i in range(min(len(latest_symbols), len(latest_w)))}

    return {
        'n_windows': int(len(wm_k)),
        'timestamp_min': str(wm_k['timestamp_end'].min()),
        'timestamp_max': str(wm_k['timestamp_end'].max()),
        'base_metrics': {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in base_metrics.items()},
        'latest_weights': latest_weights,
        'latest_gross_lev': float(np.sum(np.abs(latest_w))),
        'steps_parquet': str(out_steps_parquet),
    }


def _run_phase18_5_on_steps(
    steps_parquet: Path,
    out_dir: Path,
    cfg: Dict[str, Any],
    logs_dir: Path,
    alerts_file: Path,
    logger: logging.Logger,
) -> Path:
    exec_cfg = cfg.get('execution', {})
    cmd = [
        sys.executable,
        str(_script_path(cfg, 'phase18_5_script')),
        '--input_steps_parquet', str(steps_parquet),
        '--run_phase18_2', '0',
        '--policy', str(exec_cfg.get('policy', 'model_ctde_robust')),
        '--cost_mult', str(float(exec_cfg.get('cost_mult', 1.0))),
        '--account_equity_usd', str(float(exec_cfg.get('account_equity_usd', 3000.0))),
        '--account_leverage_cap', str(float(exec_cfg.get('account_leverage_cap', 10000.0))),
        '--max_margin_utilization', str(float(exec_cfg.get('max_margin_utilization', 0.35))),
        '--leverage_tiers', str(exec_cfg.get('leverage_tiers', '1000000:500,5000000:200,inf:100')),
        '--extra_cost_per_turnover', str(float(exec_cfg.get('extra_cost_per_turnover', 2e-05))),
        '--slippage_per_turnover', str(float(exec_cfg.get('slippage_per_turnover', 2e-05))),
        '--shield_dd_limit', str(float(exec_cfg.get('shield_dd_limit', -0.08))),
        '--shield_risk_mult', str(float(exec_cfg.get('shield_risk_mult', 0.55))),
        '--min_exec_scale', str(float(exec_cfg.get('min_exec_scale', 0.05))),
        '--out_dir', str(out_dir),
    ]
    run_cmd_with_retry(
        cmd=cmd,
        stage_name=f'phase18_5_exec_{out_dir.name}',
        logs_dir=logs_dir,
        alerts_file=alerts_file,
        logger=logger,
        retries=int(cfg['retry'].get('max', 3)),
        backoff_sec=int(cfg['retry'].get('backoff_sec', 20)),
    )
    s = out_dir / 'phase18_5_summary.json'
    if not s.exists():
        raise RuntimeError(f'Missing phase18.5 summary: {s}')
    return s


def _emit_guardrails_from_backtests(run_dir: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    req_days = [90, 120]
    checks = []
    all_pass = True
    for d in req_days:
        s = read_json(run_dir / f'backtest_{d}d' / 'summary.json') or {}
        m = (((s.get('phase18_5_summary') or {}).get('ctrader_exec_metrics')) or {})
        sh = m.get('sharpe_ann', None)
        cr = m.get('cum_net_return', None)
        ok = isinstance(sh, (int, float)) and isinstance(cr, (int, float)) and float(sh) > 0.0 and float(cr) > 0.0
        checks.append({'days': int(d), 'sharpe_ann': sh, 'cum_net_return': cr, 'pass': bool(ok)})
        all_pass = all_pass and bool(ok)

    warm_cfg = cfg.get('paper_live', {})
    guard = {
        'phase': 'phase21',
        'status': 'PASS' if all_pass else 'FAILED',
        'generated_at_utc': now_utc_iso(),
        'stale_after_hours': float(warm_cfg.get('guardrail_stale_hours', 36.0)),
        'mode_for_live': 'NORMAL' if all_pass else 'DEGRADED',
        'reason': 'conservative_gates_pass' if all_pass else 'conservative_gates_failed',
        'gates': checks,
        'effective_limits': {
            'normal': {
                'max_abs_lev': float(warm_cfg.get('normal_max_abs_lev', 0.70)),
                'max_delta_lev_per_step': float(warm_cfg.get('normal_max_delta_lev_per_step', 0.20)),
            },
            'degraded': {
                'max_abs_lev': float(warm_cfg.get('degraded_max_abs_lev', 0.30)),
                'max_delta_lev_per_step': float(warm_cfg.get('degraded_max_delta_lev_per_step', 0.10)),
            },
            'warmup_24h': {
                'max_abs_lev': float(warm_cfg.get('warmup_max_abs_lev', 0.35)),
                'max_delta_lev_per_step': float(warm_cfg.get('warmup_max_delta_lev_per_step', 0.10)),
            },
        },
        'paper_only': True,
        'live_order_execution': False,
    }
    safe_json_write(run_dir / 'phase21_guardrails.json', guard)
    return guard


def _build_symbol_rows_from_stages(symbols: List[str], p13_dirs: Dict[str, Path], p14_dirs: Dict[str, Path], costs: Dict[str, float]) -> List[Dict[str, Any]]:
    rows = []
    for s in symbols:
        rows.append(
            {
                'symbol': s,
                'phase13_run_dir': p13_dirs[s],
                'phase14_run_dir': p14_dirs[s],
                'cost_pips': float(costs[s]),
            }
        )
    return rows


def _run_strict_symbol_pipeline(
    symbols: List[str],
    cfg: Dict[str, Any],
    run_dir: Path,
    mode: str,
    lookback_hours: int,
    logger: logging.Logger,
    files: Dict[str, Path],
) -> Dict[str, Any]:
    costs = _cost_map(cfg)

    data_root = run_dir / 'data' / 'm1'
    p13_root = run_dir / 'phase13_runs'
    p14_root = run_dir / 'phase14_runs'
    logs_root = run_dir / 'stage_logs'

    data_root.mkdir(parents=True, exist_ok=True)
    p13_root.mkdir(parents=True, exist_ok=True)
    p14_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    symbol_stats = {}
    p13_dirs: Dict[str, Path] = {}
    p14_dirs: Dict[str, Path] = {}

    for s in symbols:
        out_m1 = data_root / f'{s.lower()}_m1.parquet'
        _fetch_m1_one(
            symbol=s,
            cfg=cfg,
            mode=mode,
            out_file=out_m1,
            lookback_hours=int(lookback_hours),
            logs_dir=logs_root,
            alerts_file=files['alerts'],
            logger=logger,
        )

        d = pd.read_parquet(out_m1)
        clean, st = validate_m1(
            d,
            symbol=s,
            min_rows=int(cfg['validation'].get('min_rows_per_symbol', 5000)),
            max_gap_rate=float(cfg['validation'].get('max_gap_rate', 0.50)),
        )
        clean.to_parquet(out_m1, index=False)
        symbol_stats[s] = st

        p13_id = f'phase21_{run_dir.name}_{s}_p13'
        p13_dir = _run_phase13_symbol(
            symbol=s,
            input_m1=out_m1,
            cost_pips=float(costs[s]),
            cfg=cfg,
            out_root=p13_root,
            run_id=p13_id,
            logs_dir=logs_root,
            alerts_file=files['alerts'],
            logger=logger,
        )
        p13_dirs[s] = p13_dir

        p14_id = f'phase21_{run_dir.name}_{s}_p14'
        p14_dir = _run_phase14_symbol(
            symbol=s,
            phase13_run_dir=p13_dir,
            cost_pips=float(costs[s]),
            cfg=cfg,
            out_root=p14_root,
            run_id=p14_id,
            logs_dir=logs_root,
            alerts_file=files['alerts'],
            logger=logger,
        )
        p14_dirs[s] = p14_dir

    symbol_rows = _build_symbol_rows_from_stages(symbols, p13_dirs, p14_dirs, costs)
    return {
        'symbol_rows': symbol_rows,
        'symbol_stats': symbol_stats,
        'phase13_dirs': p13_dirs,
        'phase14_dirs': p14_dirs,
        'data_root': data_root,
        'logs_root': logs_root,
    }


def _collect_script_hashes(cfg: Dict[str, Any]) -> Dict[str, Optional[str]]:
    out = {}
    for k, v in (cfg.get('paths') or {}).items():
        p = Path(str(v))
        if p.exists() and p.is_file():
            out[f'{k}_sha256'] = sha256_file(p)
    out['phase21_cli_sha256'] = sha256_file(Path(__file__).resolve())
    return out


def _run_prepare_checkpoint(args: argparse.Namespace, cfg: Dict[str, Any], cfg_path: Path):
    out_root = resolve_path(args.out_dir or cfg['paths']['root_out_dir'])
    run_id = args.run_id or f'phase21_prepare_{now_tag()}'
    run_dir = out_root / run_id
    files = init_contract(run_dir, 'prepare-checkpoint', cfg_path, cfg)
    logger = setup_logger(files['log'])

    m = read_json(files['manifest']) or {}
    m['script_hashes'] = _collect_script_hashes(cfg)
    safe_json_write(files['manifest'], m)

    mode = str(args.data_source or cfg['prepare'].get('data_source', 'ctrader')).lower()
    if mode not in ('ctrader', 'file'):
        raise ValueError(f'Unsupported data_source={mode}')

    top7 = load_top7_selected(resolve_path(cfg['paths']['phase15_selected7_csv']))
    top7_fixed = [str(s).upper() for s in cfg.get('fixed_symbols', top7)]
    if set(top7) != set(top7_fixed):
        raise RuntimeError(f'Top7 mismatch: selected7={top7} fixed={top7_fixed}')

    update_status(files['status'], 'RUNNING', 'strict_symbol_pipeline')
    update_manifest_stage(files['manifest'], 'strict_symbol_pipeline', 'RUNNING', {'symbols': top7_fixed})

    p = _run_strict_symbol_pipeline(
        symbols=top7_fixed,
        cfg=cfg,
        run_dir=run_dir,
        mode=mode,
        lookback_hours=int(args.lookback_hours or cfg['prepare'].get('lookback_hours', 24 * 240)),
        logger=logger,
        files=files,
    )

    update_manifest_stage(files['manifest'], 'strict_symbol_pipeline', 'SUCCESS', {'symbol_stats': p['symbol_stats']})

    bridge_dir = run_dir / 'phase15_bridge'
    _build_phase15_bridge(p['symbol_rows'], bridge_dir)
    update_manifest_stage(files['manifest'], 'phase15_bridge', 'SUCCESS', {'phase15_bridge': str(bridge_dir)})

    update_status(files['status'], 'RUNNING', 'phase16_chain')
    update_manifest_stage(files['manifest'], 'phase16_chain', 'RUNNING')
    p16 = _run_phase16_chain(
        bridge_dir=bridge_dir,
        out_root=run_dir / 'phase16_runs',
        cfg=cfg,
        logs_dir=p['logs_root'],
        alerts_file=files['alerts'],
        logger=logger,
    )
    update_manifest_stage(files['manifest'], 'phase16_chain', 'SUCCESS', {k: str(v) for k, v in p16.items()})

    update_status(files['status'], 'RUNNING', 'phase18_chain')
    update_manifest_stage(files['manifest'], 'phase18_chain', 'RUNNING')
    p18 = _run_phase18_chain(
        phase16_4_dir=p16['phase16_4'],
        out_root=run_dir / 'phase18_runs',
        cfg=cfg,
        logs_dir=p['logs_root'],
        alerts_file=files['alerts'],
        logger=logger,
    )
    update_manifest_stage(files['manifest'], 'phase18_chain', 'SUCCESS', {k: str(v) for k, v in p18.items()})

    update_status(files['status'], 'RUNNING', 'checkpoint_export')
    update_manifest_stage(files['manifest'], 'checkpoint_export', 'RUNNING')
    ckpt_path = run_dir / 'checkpoint' / 'phase21_top_engine_checkpoint.pt'
    _build_phase21_checkpoint(
        phase18_2_dir=p18['phase18_2'],
        phase18_5_dir=p18['phase18_5'],
        out_ckpt=ckpt_path,
        cfg=cfg,
        source_run_dir=run_dir,
    )
    update_manifest_stage(files['manifest'], 'checkpoint_export', 'SUCCESS', {'checkpoint': str(ckpt_path)})

    links = {
        'phase13_dirs': {k: str(v) for k, v in p['phase13_dirs'].items()},
        'phase14_dirs': {k: str(v) for k, v in p['phase14_dirs'].items()},
        'phase16_dirs': {k: str(v) for k, v in p16.items()},
        'phase18_dirs': {k: str(v) for k, v in p18.items()},
        'checkpoint': str(ckpt_path),
    }
    safe_json_write(files['links'], links)

    snap = {
        'ts_utc': now_utc_iso(),
        'command': 'prepare-checkpoint',
        'run_dir': str(run_dir),
        'checkpoint': str(ckpt_path),
        'symbols': top7_fixed,
        'paper_only': True,
    }
    _write_run_snapshot(files['snapshot'], snap)

    update_status(files['status'], 'SUCCESS', 'done', {'checkpoint': str(ckpt_path)})
    set_manifest_final(files['manifest'], 'SUCCESS', {'checkpoint': str(ckpt_path)})

    print(json.dumps({'status': 'SUCCESS', 'run_dir': str(run_dir), 'checkpoint': str(ckpt_path)}, indent=2))


def _load_checkpoint_path(run_dir: Path) -> Path:
    p = run_dir / 'checkpoint' / 'phase21_top_engine_checkpoint.pt'
    if not p.exists():
        raise FileNotFoundError(f'Missing checkpoint: {p}')
    return p


def _run_backtest_recent(args: argparse.Namespace, cfg: Dict[str, Any], cfg_path: Path):
    run_dir = resolve_path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    files = {
        'manifest': run_dir / 'phase21_manifest.json',
        'status': run_dir / 'phase21_status.json',
        'alerts': run_dir / 'phase21_alerts.jsonl',
        'guardrails': run_dir / 'phase21_guardrails.json',
        'snapshot': run_dir / 'phase21_snapshot.json',
        'log': run_dir / 'phase21.log',
        'links': run_dir / 'links_to_child_runs.json',
    }
    logger = setup_logger(files['log'])

    ckpt = _load_checkpoint_path(run_dir)

    raw_days = str(args.days or '90').strip()
    days_list = [int(x.strip()) for x in raw_days.split(',') if x.strip()]
    if not days_list:
        raise ValueError('No days provided')

    mode = str(args.data_source or cfg['backtest'].get('data_source', 'ctrader')).lower()
    max_days = max(days_list)
    buffer_days = int(cfg['backtest'].get('buffer_days', 45))
    lookback_h = int((max_days + buffer_days) * 24)

    top7 = [str(s).upper() for s in cfg.get('fixed_symbols', [])]
    if not top7:
        top7 = load_top7_selected(resolve_path(cfg['paths']['phase15_selected7_csv']))

    bt_root = run_dir / 'backtest_source'
    update_status(files['status'], 'RUNNING', 'backtest_symbol_pipeline', {'days': days_list})
    p = _run_strict_symbol_pipeline(
        symbols=top7,
        cfg=cfg,
        run_dir=bt_root,
        mode=mode,
        lookback_hours=lookback_h,
        logger=logger,
        files=files,
    )

    bridge_dir = bt_root / 'phase15_bridge'
    _build_phase15_bridge(p['symbol_rows'], bridge_dir)

    p16 = _run_phase16_chain(
        bridge_dir=bridge_dir,
        out_root=bt_root / 'phase16_runs',
        cfg=cfg,
        logs_dir=p['logs_root'],
        alerts_file=files['alerts'],
        logger=logger,
    )

    summaries = {}
    for d in days_list:
        out_bt = run_dir / f'backtest_{int(d)}d'
        out_bt.mkdir(parents=True, exist_ok=True)
        steps_pq = out_bt / 'phase21_inferred_steps.parquet'

        infer = _infer_steps_from_checkpoint(
            ckpt_path=ckpt,
            phase16_4_dir=p16['phase16_4'],
            cfg=cfg,
            days=int(d),
            out_steps_parquet=steps_pq,
        )

        p18_5_out = out_bt / 'phase18_5_exec'
        s18_5 = _run_phase18_5_on_steps(
            steps_parquet=steps_pq,
            out_dir=p18_5_out,
            cfg=cfg,
            logs_dir=p['logs_root'],
            alerts_file=files['alerts'],
            logger=logger,
        )
        s18 = read_json(s18_5) or {}

        summary = {
            'status': 'SUCCESS',
            'days': int(d),
            'created_at_utc': now_utc_iso(),
            'run_dir': str(out_bt),
            'checkpoint': str(ckpt),
            'infer_summary': infer,
            'phase18_5_summary': s18,
            'outputs': {
                'steps_parquet': str(steps_pq),
                'phase18_5_summary_json': str(s18_5),
            },
            'paper_only': True,
        }
        safe_json_write(out_bt / 'summary.json', summary)
        summaries[f'{int(d)}d'] = summary

    guard = _emit_guardrails_from_backtests(run_dir, cfg)

    snap = {
        'ts_utc': now_utc_iso(),
        'command': 'backtest-recent',
        'run_dir': str(run_dir),
        'days': days_list,
        'guardrail_status': guard.get('status'),
        'paper_only': True,
    }
    _write_run_snapshot(files['snapshot'], snap)
    update_status(files['status'], 'SUCCESS', 'done', {'days': days_list, 'guardrail_status': guard.get('status')})

    print(json.dumps({'status': 'SUCCESS', 'run_dir': str(run_dir), 'days': days_list, 'guardrail': guard.get('status')}, indent=2))


def _load_guardrail(guardrail_file: Path, stale_hours: float) -> Dict[str, Any]:
    if not guardrail_file.exists():
        return {
            'status': 'MISSING',
            'mode': 'DEGRADED',
            'reason': 'guardrail_missing',
            'is_stale': True,
        }
    g = read_json(guardrail_file) or {}
    st = str(g.get('status', 'UNKNOWN')).upper()
    gen = g.get('generated_at_utc')
    ah = age_hours(gen)
    stale = (ah is None) or (ah > float(stale_hours))
    if st == 'PASS' and not stale:
        mode = 'NORMAL'
        reason = 'guardrail_pass'
    elif stale:
        mode = 'DEGRADED'
        reason = f'guardrail_stale_gt_{float(stale_hours):.1f}h'
    else:
        mode = 'DEGRADED'
        reason = f'guardrail_status_{st}'
    return {
        'status': st,
        'mode': mode,
        'reason': reason,
        'generated_at_utc': gen,
        'age_hours': ah,
        'is_stale': bool(stale),
        'payload': g,
    }


def _seconds_to_next_hour(delay_sec: int) -> int:
    now = datetime.now(timezone.utc)
    nxt = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    return max(1, int((nxt - now).total_seconds()) + int(delay_sec))


def _run_paper_live(args: argparse.Namespace, cfg: Dict[str, Any], cfg_path: Path):
    run_dir = resolve_path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    files = {
        'manifest': run_dir / 'phase21_manifest.json',
        'status': run_dir / 'phase21_status.json',
        'alerts': run_dir / 'phase21_alerts.jsonl',
        'guardrails': run_dir / 'phase21_guardrails.json',
        'snapshot': run_dir / 'phase21_snapshot.json',
        'log': run_dir / 'phase21.log',
        'links': run_dir / 'links_to_child_runs.json',
    }
    logger = setup_logger(files['log'])

    ckpt = _load_checkpoint_path(run_dir)
    top7 = [str(s).upper() for s in cfg.get('fixed_symbols', [])]
    if not top7:
        top7 = load_top7_selected(resolve_path(cfg['paths']['phase15_selected7_csv']))

    p_cfg = cfg.get('paper_live', {})
    mode = str(args.data_source or p_cfg.get('data_source', 'ctrader')).lower()

    service_dir = run_dir / 'paper_live'
    service_dir.mkdir(parents=True, exist_ok=True)
    state_file = service_dir / 'state.json'
    decisions_pq = service_dir / 'decisions.parquet'

    state = read_json(state_file) or {
        'status': 'INIT',
        'created_at_utc': now_utc_iso(),
        'updated_at_utc': now_utc_iso(),
        'n_cycles': 0,
        'prev_weights': {},
        'service_start_utc': now_utc_iso(),
        'mode': 'NORMAL',
    }

    once = int(args.once if args.once is not None else p_cfg.get('once', 0))
    loop = int(args.loop if args.loop is not None else p_cfg.get('loop', 1))
    align_hour = int(args.align_hour if args.align_hour is not None else p_cfg.get('align_hour', 1))
    hour_delay_sec = int(args.hour_delay_sec if args.hour_delay_sec is not None else p_cfg.get('hour_delay_sec', 20))

    while True:
        t0 = now_utc_iso()
        try:
            guard = _load_guardrail(run_dir / 'phase21_guardrails.json', stale_hours=float(p_cfg.get('guardrail_stale_hours', 36.0)))
            warm_h = age_hours(state.get('service_start_utc')) or 0.0
            in_warmup = warm_h < float(p_cfg.get('warmup_hours', 24))

            if guard['mode'] == 'DEGRADED':
                cap_abs = float(p_cfg.get('degraded_max_abs_lev', 0.30))
                cap_delta = float(p_cfg.get('degraded_max_delta_lev_per_step', 0.10))
                mode_live = 'DEGRADED'
                reason = str(guard['reason'])
            elif in_warmup:
                cap_abs = float(p_cfg.get('warmup_max_abs_lev', 0.35))
                cap_delta = float(p_cfg.get('warmup_max_delta_lev_per_step', 0.10))
                mode_live = 'WARMUP'
                reason = 'warmup_24h'
            else:
                cap_abs = float(p_cfg.get('normal_max_abs_lev', 0.70))
                cap_delta = float(p_cfg.get('normal_max_delta_lev_per_step', 0.20))
                mode_live = 'NORMAL'
                reason = 'guardrail_pass'

            cyc_root = service_dir / f'cycle_{now_tag()}'
            p = _run_strict_symbol_pipeline(
                symbols=top7,
                cfg=cfg,
                run_dir=cyc_root,
                mode=mode,
                lookback_hours=int(p_cfg.get('lookback_hours', 24 * 200)),
                logger=logger,
                files=files,
            )
            bridge_dir = cyc_root / 'phase15_bridge'
            _build_phase15_bridge(p['symbol_rows'], bridge_dir)
            p16 = _run_phase16_chain(
                bridge_dir=bridge_dir,
                out_root=cyc_root / 'phase16_runs',
                cfg=cfg,
                logs_dir=p['logs_root'],
                alerts_file=files['alerts'],
                logger=logger,
            )

            infer_steps = cyc_root / 'live_infer_steps.parquet'
            infer = _infer_steps_from_checkpoint(
                ckpt_path=ckpt,
                phase16_4_dir=p16['phase16_4'],
                cfg=cfg,
                days=None,
                out_steps_parquet=infer_steps,
            )

            # latest target weights
            target_w = infer['latest_weights']
            symbols = sorted(target_w.keys())
            vec = np.array([float(target_w[s]) for s in symbols], dtype=float)
            gross = float(np.sum(np.abs(vec)))
            if gross > cap_abs and gross > 1e-12:
                vec = vec * (float(cap_abs) / gross)

            prev = state.get('prev_weights', {}) or {}
            prev_vec = np.array([float(prev.get(s, 0.0)) for s in symbols], dtype=float)
            l1 = float(np.sum(np.abs(vec - prev_vec)))
            if l1 > cap_delta and l1 > 1e-12:
                alpha = float(cap_delta / l1)
                vec = prev_vec + alpha * (vec - prev_vec)

            gross_final = float(np.sum(np.abs(vec)))
            final_w = {s: float(vec[i]) for i, s in enumerate(symbols)}

            row = {
                'ts_utc': now_utc_iso(),
                'mode': mode_live,
                'reason': reason,
                'guardrail_status': str(guard.get('status')),
                'guardrail_is_stale': bool(guard.get('is_stale', True)),
                'warmup_active': bool(in_warmup),
                'cap_abs_lev': float(cap_abs),
                'cap_delta_lev': float(cap_delta),
                'gross_lev_target': float(infer.get('latest_gross_lev', gross_final)),
                'gross_lev_final': float(gross_final),
                'weights_json': json.dumps(final_w),
                'pipeline_ts_max': str(infer.get('timestamp_max')),
                'paper_only': True,
            }

            if decisions_pq.exists():
                d = pd.read_parquet(decisions_pq)
                d = pd.concat([d, pd.DataFrame([row])], ignore_index=True)
            else:
                d = pd.DataFrame([row])
            d.to_parquet(decisions_pq, index=False)

            state.update(
                {
                    'status': 'RUNNING',
                    'updated_at_utc': now_utc_iso(),
                    'n_cycles': int(state.get('n_cycles', 0)) + 1,
                    'prev_weights': final_w,
                    'mode': mode_live,
                    'last_reason': reason,
                    'last_ts': row['ts_utc'],
                    'last_pipeline_ts_max': row['pipeline_ts_max'],
                }
            )
            safe_json_write(state_file, state)

            snap = {
                'ts_utc': now_utc_iso(),
                'command': 'paper-live',
                'run_dir': str(run_dir),
                'mode': mode_live,
                'reason': reason,
                'n_cycles': int(state['n_cycles']),
                'paper_only': True,
            }
            _write_run_snapshot(files['snapshot'], snap)
            update_status(files['status'], 'RUNNING', 'paper_live_cycle', {'mode': mode_live, 'n_cycles': int(state['n_cycles'])})

            emit_alert(files['alerts'], 'INFO', 'paper_live_cycle_ok', {'mode': mode_live, 'reason': reason, 'gross_lev_final': gross_final})
        except Exception as e:
            state['status'] = 'ERROR'
            state['updated_at_utc'] = now_utc_iso()
            state['last_error'] = f'{type(e).__name__}: {e}'
            safe_json_write(state_file, state)
            emit_alert(files['alerts'], 'ERROR', 'paper_live_cycle_failed', {'error': f'{type(e).__name__}: {e}', 'trace': traceback.format_exc()[-2000:]})
            update_status(files['status'], 'FAILED', 'paper_live_cycle_failed', {'error': f'{type(e).__name__}: {e}'})
            if int(p_cfg.get('continue_on_error', 1)) != 1:
                raise

        if int(loop) != 1 or int(once) == 1:
            break

        sleep_s = _seconds_to_next_hour(hour_delay_sec) if int(align_hour) == 1 else int(p_cfg.get('poll_interval_sec', 3600))
        logger.info('paper-live cycle done (%s -> %s). sleeping %ss', t0, now_utc_iso(), sleep_s)
        time.sleep(sleep_s)

    print(json.dumps({'status': 'SUCCESS', 'run_dir': str(run_dir), 'decisions': str(decisions_pq)}, indent=2))


def _run_status(args: argparse.Namespace, cfg: Dict[str, Any], cfg_path: Path):
    run_dir = resolve_path(args.run_dir)
    guard = read_json(run_dir / 'phase21_guardrails.json') or {}
    status = read_json(run_dir / 'phase21_status.json') or {}
    snap = read_json(run_dir / 'phase21_snapshot.json') or {}
    state = read_json(run_dir / 'paper_live' / 'state.json') or {}

    out = {
        'ts_utc': now_utc_iso(),
        'run_dir': str(run_dir),
        'phase21_status': status.get('status'),
        'phase21_stage': status.get('stage'),
        'guardrail_status': guard.get('status'),
        'guardrail_generated_at_utc': guard.get('generated_at_utc'),
        'guardrail_age_hours': age_hours(guard.get('generated_at_utc')),
        'snapshot': snap,
        'paper_live': {
            'status': state.get('status'),
            'mode': state.get('mode'),
            'n_cycles': state.get('n_cycles'),
            'updated_at_utc': state.get('updated_at_utc'),
            'freshness_hours': age_hours(state.get('updated_at_utc')),
        },
        'paper_only': True,
    }
    safe_json_write(run_dir / 'phase21_snapshot.json', out)
    print(json.dumps(out, indent=2))


def _run_full(args: argparse.Namespace, cfg: Dict[str, Any], cfg_path: Path):
    out_root = resolve_path(args.out_dir or cfg['paths']['root_out_dir'])
    run_id = args.run_id or f'phase21_full_{now_tag()}'
    run_dir = out_root / run_id

    # prepare-checkpoint
    prepare_args = argparse.Namespace(
        out_dir=str(out_root),
        run_id=run_id,
        data_source=args.data_source,
        lookback_hours=args.lookback_hours,
    )
    _run_prepare_checkpoint(prepare_args, cfg, cfg_path)

    # backtest-recent on same run
    days = args.days or '90,120'
    bt_args = argparse.Namespace(run_dir=str(run_dir), days=days, data_source=args.data_source)
    _run_backtest_recent(bt_args, cfg, cfg_path)

    s = {
        'status': 'SUCCESS',
        'run_dir': str(run_dir),
        'checkpoint': str(run_dir / 'checkpoint' / 'phase21_top_engine_checkpoint.pt'),
        'guardrail': read_json(run_dir / 'phase21_guardrails.json'),
        'finished_at_utc': now_utc_iso(),
    }
    safe_json_write(run_dir / 'phase21_full_run_summary.json', s)
    print(json.dumps(s, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description='Phase21 - E2E top-engine CLI (phase16->18.2->18.5) for 90/120 backtests and paper live.')
    ap.add_argument('--config', default=str(DEFAULT_CONFIG), type=str)

    sub = ap.add_subparsers(dest='command', required=True)

    p1 = sub.add_parser('prepare-checkpoint', help='Pull recent data, run strict pipeline and export frozen phase21 checkpoint.')
    p1.add_argument('--out_dir', default='', type=str)
    p1.add_argument('--run_id', default='', type=str)
    p1.add_argument('--data_source', default=None, choices=['ctrader', 'file'])
    p1.add_argument('--lookback_hours', default=None, type=int)

    p2 = sub.add_parser('backtest-recent', help='Run frozen-only backtest for recent days using existing phase21 checkpoint.')
    p2.add_argument('--run_dir', required=True, type=str)
    p2.add_argument('--days', default='90', type=str, help='Single value or CSV list, e.g. 90 or 90,120')
    p2.add_argument('--data_source', default=None, choices=['ctrader', 'file'])

    p3 = sub.add_parser('paper-live', help='Run H1 paper-live loop using frozen phase21 checkpoint.')
    p3.add_argument('--run_dir', required=True, type=str)
    p3.add_argument('--once', default=None, type=int, choices=[0, 1])
    p3.add_argument('--loop', default=None, type=int, choices=[0, 1])
    p3.add_argument('--align_hour', default=None, type=int, choices=[0, 1])
    p3.add_argument('--hour_delay_sec', default=None, type=int)
    p3.add_argument('--data_source', default=None, choices=['ctrader', 'file'])

    p4 = sub.add_parser('status', help='Consolidated status for a phase21 run.')
    p4.add_argument('--run_dir', required=True, type=str)

    p5 = sub.add_parser('full-run', help='prepare-checkpoint + backtest-recent (90,120) in one command.')
    p5.add_argument('--out_dir', default='', type=str)
    p5.add_argument('--run_id', default='', type=str)
    p5.add_argument('--days', default='90,120', type=str)
    p5.add_argument('--data_source', default=None, choices=['ctrader', 'file'])
    p5.add_argument('--lookback_hours', default=None, type=int)

    return ap


def main():
    ap = build_arg_parser()
    args = ap.parse_args()

    cfg_path = resolve_path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    cfg = load_config(cfg_path)

    if args.command == 'prepare-checkpoint':
        _run_prepare_checkpoint(args, cfg, cfg_path)
        return
    if args.command == 'backtest-recent':
        _run_backtest_recent(args, cfg, cfg_path)
        return
    if args.command == 'paper-live':
        _run_paper_live(args, cfg, cfg_path)
        return
    if args.command == 'status':
        _run_status(args, cfg, cfg_path)
        return
    if args.command == 'full-run':
        _run_full(args, cfg, cfg_path)
        return

    raise RuntimeError(f'Unknown command: {args.command}')


if __name__ == '__main__':
    main()
