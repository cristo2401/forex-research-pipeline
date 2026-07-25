#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FETCH_SCRIPT = ROOT / 'phases/phase19_ctrader_paper_live/run_phase19_fetch_m1_once.py'
DEFAULT_PHASE13_SCRIPT = ROOT / 'phases/phase13_pipeline_cli/run_phase13_pipeline.py'
DEFAULT_PHASE19_SCRIPT = ROOT / 'phases/phase19_ctrader_paper_live/run_phase19_paper_live.py'
DEFAULT_ENV_FILE = ROOT / 'phases/phase19_ctrader_paper_live/.env'


@dataclass
class Config:
    symbol: str
    env_file: str
    out_root: str
    data_source: str
    input_m1_file: str
    fetch_script: str
    phase13_script: str
    phase19_script: str

    demo: int
    account_id: int
    fetch_count: int
    max_probe_hours: int
    probe_hours_csv: str

    phase13_cost_pips: float
    phase13_priors_source: str
    phase13_strict_temporal_eval: int

    min_segments: int
    freshness_max_hours: float
    safety_buffer_hours: int
    allow_degraded_start: int

    conf_min: float
    max_abs_lev: float
    max_delta_lev_per_step: float
    daily_loss_limit: float
    start_equity: float

    once: int
    loop: int
    align_hour: int
    hour_delay_sec: int

    retry_max: int
    retry_backoff_sec: int


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def now_utc_tag() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger('phase13_5')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}


def write_json(path: Path, payload: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding='utf-8')


def run_cmd_with_retry(cmd: List[str], stage: str, logger: logging.Logger, retries: int, backoff_sec: int) -> str:
    last_err: Optional[RuntimeError] = None
    for k in range(1, retries + 1):
        logger.info('[%s] attempt %s/%s', stage, k, retries)
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode == 0:
            if p.stdout.strip():
                logger.info('[%s] stdout: %s', stage, p.stdout.strip()[:1200])
            return p.stdout
        msg = p.stderr.strip() or p.stdout.strip() or f'rc={p.returncode}'
        logger.warning('[%s] failed: %s', stage, msg[-1000:])
        last_err = RuntimeError(f'{stage} failed rc={p.returncode}: {msg[-1200:]}')
        if k < retries:
            time.sleep(backoff_sec)
    raise last_err if last_err is not None else RuntimeError(f'{stage} failed')


def load_m1(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == '.parquet':
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if 'timestamp' not in df.columns:
        raise RuntimeError(f'M1 file missing timestamp column: {path}')
    req = {'open', 'high', 'low', 'close'}
    if not req.issubset(set(df.columns)):
        raise RuntimeError(f'M1 file missing OHLC columns: {path}')
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    df = df.dropna(subset=['timestamp']).sort_values('timestamp').drop_duplicates(subset=['timestamp'])
    return df.reset_index(drop=True)


def fetch_reference_m1(cfg: Config, run_dir: Path, logger: logging.Logger) -> Path:
    data_dir = run_dir / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    out_m1 = data_dir / f'{cfg.symbol.lower()}_probe_m1.parquet'

    if cfg.data_source == 'file':
        src = Path(cfg.input_m1_file)
        if not src.exists():
            raise FileNotFoundError(f'input_m1_file not found: {src}')
        if src.resolve() != out_m1.resolve():
            shutil.copy2(src, out_m1)
        return out_m1

    cmd = [
        sys.executable,
        cfg.fetch_script,
        '--symbol', cfg.symbol,
        '--env_file', cfg.env_file,
        '--output', str(out_m1),
        '--lookback_hours', str(int(cfg.max_probe_hours)),
        '--count', str(int(cfg.fetch_count)),
        '--account_id', str(int(cfg.account_id)),
        '--demo', str(int(cfg.demo)),
        '--require_demo_account', '1',
    ]
    run_cmd_with_retry(cmd, 'fetch_reference_m1', logger, cfg.retry_max, cfg.retry_backoff_sec)
    return out_m1


def parse_probe_hours(csv_text: str) -> List[int]:
    vals: List[int] = []
    for part in str(csv_text).split(','):
        p = part.strip()
        if not p:
            continue
        vals.append(int(p))
    vals = sorted(set(v for v in vals if v > 0))
    if not vals:
        raise RuntimeError('probe_hours_csv produced empty list')
    return vals


def run_phase13_probe(
    cfg: Config,
    input_path: Path,
    probe_out_root: Path,
    probe_id: str,
    logger: logging.Logger,
) -> Path:
    run_id = f'phase13_5_{cfg.symbol}_{probe_id}'
    cmd = [
        sys.executable,
        cfg.phase13_script,
        '--input_m1', str(input_path),
        '--symbol', cfg.symbol,
        '--mode', 'infer_frozen',
        '--cost_pips', str(float(cfg.phase13_cost_pips)),
        '--priors_source', cfg.phase13_priors_source,
        '--strict_temporal_eval', str(int(cfg.phase13_strict_temporal_eval)),
        '--out_dir', str(probe_out_root),
        '--run_id', run_id,
    ]
    # Probe failures like "too few segments" are deterministic for a given window.
    # Keep this at one attempt to avoid wasting minutes in backoff loops.
    run_cmd_with_retry(cmd, f'phase13_probe_{probe_id}', logger, 1, cfg.retry_backoff_sec)
    p13_run = probe_out_root / run_id
    if not (p13_run / 'phase13_summary.json').exists():
        raise RuntimeError(f'phase13 summary missing at {p13_run}')
    return p13_run


def evaluate_probe(
    p13_run: Path,
    cfg: Config,
    ref_max_ts: pd.Timestamp,
    lookback_h: int,
) -> Dict[str, Any]:
    summary = read_json(p13_run / 'phase13_summary.json')
    status = str(summary.get('status', 'FAILED')).upper()
    stage02 = (summary.get('quality') or {}).get('stage02') or {}
    n_segments = int(stage02.get('n_segments', 0) or 0)

    st5 = p13_run / 'stage05_phase12_predictions.parquet'
    rows_st5 = 0
    latest_ts = None
    age_h = None
    if st5.exists():
        df = pd.read_parquet(st5)
        rows_st5 = int(len(df))
        if rows_st5 > 0 and 'timestamp' in df.columns:
            ts = pd.to_datetime(df['timestamp'], utc=True, errors='coerce').dropna()
            if len(ts) > 0:
                latest_ts = ts.max()
                age_h = float((ref_max_ts - latest_ts).total_seconds() / 3600.0)

    pass_status = status == 'SUCCESS'
    pass_segments = n_segments >= int(cfg.min_segments)
    pass_rows = rows_st5 > 0
    pass_fresh = (age_h is not None) and (age_h <= float(cfg.freshness_max_hours))
    pass_all = bool(pass_status and pass_segments and pass_rows and pass_fresh)

    return {
        'lookback_h': int(lookback_h),
        'phase13_run_dir': str(p13_run),
        'status': status,
        'n_segments': n_segments,
        'rows_stage05': rows_st5,
        'latest_signal_ts_utc': latest_ts.strftime('%Y-%m-%dT%H:%M:%SZ') if latest_ts is not None else None,
        'age_h_vs_latest_m1': age_h,
        'pass_status': bool(pass_status),
        'pass_segments': bool(pass_segments),
        'pass_rows': bool(pass_rows),
        'pass_fresh': bool(pass_fresh),
        'pass_all': bool(pass_all),
    }


def select_lookback(probe_rows: List[Dict[str, Any]], cfg: Config) -> Dict[str, Any]:
    good = [r for r in probe_rows if bool(r.get('pass_all'))]
    if good:
        best = sorted(good, key=lambda x: int(x['lookback_h']))[0]
        selected = min(int(cfg.max_probe_hours), int(best['lookback_h']) + int(cfg.safety_buffer_hours))
        return {
            'live_ready': True,
            'reason': 'first_pass_all_plus_safety_buffer',
            'selected_probe': best,
            'selected_lookback_h': int(selected),
        }

    succ = [r for r in probe_rows if str(r.get('status', '')).upper() == 'SUCCESS']
    if succ:
        # Best degraded candidate: freshest signal, then more segments, then shorter lookback.
        def key_fn(x: Dict[str, Any]):
            age = x.get('age_h_vs_latest_m1')
            age_val = float(age) if age is not None else 1e9
            return (age_val, -int(x.get('n_segments', 0) or 0), int(x['lookback_h']))

        best = sorted(succ, key=key_fn)[0]
        selected = min(int(cfg.max_probe_hours), int(best['lookback_h']) + int(cfg.safety_buffer_hours))
        return {
            'live_ready': False,
            'reason': 'no_pass_all_using_best_success_degraded',
            'selected_probe': best,
            'selected_lookback_h': int(selected),
        }

    # No successful probe.
    fallback = max(int(cfg.max_probe_hours), 24)
    return {
        'live_ready': False,
        'reason': 'no_successful_probe',
        'selected_probe': None,
        'selected_lookback_h': int(fallback),
    }


def launch_phase19(cfg: Config, run_dir: Path, lookback_h: int, degraded_start: bool, logger: logging.Logger) -> Dict[str, Any]:
    live_dir = run_dir / 'live_service'
    p13_root = live_dir / 'phase13_runs'
    live_dir.mkdir(parents=True, exist_ok=True)

    conf_min = float(cfg.conf_min)
    max_abs_lev = float(cfg.max_abs_lev)
    max_delta = float(cfg.max_delta_lev_per_step)
    daily_loss = float(cfg.daily_loss_limit)

    if degraded_start:
        conf_min = max(conf_min, 0.75)
        max_abs_lev = min(max_abs_lev, 0.30)
        max_delta = min(max_delta, 0.10)
        daily_loss = min(daily_loss, 0.005)

    cmd = [
        sys.executable,
        cfg.phase19_script,
        '--symbol', cfg.symbol,
        '--env_file', cfg.env_file,
        '--out_dir', str(live_dir),
        '--data_source', cfg.data_source,
        '--input_m1_file', cfg.input_m1_file,
        '--fetch_script', cfg.fetch_script,
        '--phase13_script', cfg.phase13_script,
        '--phase13_out_root', str(p13_root),
        '--phase13_cost_pips', str(float(cfg.phase13_cost_pips)),
        '--phase13_priors_source', cfg.phase13_priors_source,
        '--phase13_strict_temporal_eval', str(int(cfg.phase13_strict_temporal_eval)),
        '--lookback_hours', str(int(lookback_h)),
        '--fetch_count', str(int(cfg.fetch_count)),
        '--account_id', str(int(cfg.account_id)),
        '--demo', str(int(cfg.demo)),
        '--start_equity', str(float(cfg.start_equity)),
        '--conf_min', str(float(conf_min)),
        '--max_abs_lev', str(float(max_abs_lev)),
        '--max_delta_lev_per_step', str(float(max_delta)),
        '--daily_loss_limit', str(float(daily_loss)),
        '--once', str(int(cfg.once)),
        '--loop', str(int(cfg.loop)),
        '--align_hour', str(int(cfg.align_hour)),
        '--hour_delay_sec', str(int(cfg.hour_delay_sec)),
        '--retry_max', str(int(cfg.retry_max)),
        '--retry_backoff_sec', str(int(cfg.retry_backoff_sec)),
        '--phase20_guardrail_enable', '0',
    ]

    # If running in loop, keep process attached and return quickly is not possible.
    # For robust automation, always run command directly and let caller choose once/loop.
    run_cmd_with_retry(cmd, 'phase19_live', logger, 1, int(cfg.retry_backoff_sec))

    status_path = live_dir / 'status.json'
    latest_signal = live_dir / 'latest_signal.json'
    out = {
        'live_dir': str(live_dir),
        'status_json': str(status_path),
        'latest_signal_json': str(latest_signal),
        'status': read_json(status_path),
        'latest_signal': read_json(latest_signal),
        'effective_risk': {
            'conf_min': conf_min,
            'max_abs_lev': max_abs_lev,
            'max_delta_lev_per_step': max_delta,
            'daily_loss_limit': daily_loss,
            'degraded_start': bool(degraded_start),
        },
    }
    return out


def build_config() -> Config:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', default='', type=str)
    pre_ns, _ = pre.parse_known_args()

    cfg_json: Dict[str, Any] = {}
    if str(pre_ns.config).strip():
        cfg_path = Path(str(pre_ns.config)).expanduser().resolve()
        cfg_json = read_json(cfg_path)

    def cget(key: str, default: Any) -> Any:
        return cfg_json.get(key, default)

    ap = argparse.ArgumentParser(description='Phase13.5 robust individual paper-live launcher (auto lookback probing).')
    ap.add_argument('--config', default=str(pre_ns.config), type=str)

    ap.add_argument('--symbol', default=cget('symbol', 'EURUSD'), type=str)
    ap.add_argument('--env_file', default=cget('env_file', str(DEFAULT_ENV_FILE)), type=str)
    ap.add_argument('--out_root', default=cget('out_root', str(ROOT / 'phases/phase13_5_individual_paper_live/output_runs')), type=str)
    ap.add_argument('--data_source', default=cget('data_source', 'ctrader'), choices=['ctrader', 'file'])
    ap.add_argument('--input_m1_file', default=cget('input_m1_file', str(ROOT / 'phases/phase21_top_engine/output_runs/phase21_full_ctrader_20260325_1/data/m1/eurusd_m1.parquet')), type=str)
    ap.add_argument('--fetch_script', default=cget('fetch_script', str(DEFAULT_FETCH_SCRIPT)), type=str)
    ap.add_argument('--phase13_script', default=cget('phase13_script', str(DEFAULT_PHASE13_SCRIPT)), type=str)
    ap.add_argument('--phase19_script', default=cget('phase19_script', str(DEFAULT_PHASE19_SCRIPT)), type=str)

    ap.add_argument('--demo', default=int(cget('demo', 1)), type=int, choices=[0, 1])
    ap.add_argument('--account_id', default=int(cget('account_id', 0)), type=int)
    ap.add_argument('--fetch_count', default=int(cget('fetch_count', 0)), type=int)
    ap.add_argument('--max_probe_hours', default=int(cget('max_probe_hours', 24 * 120)), type=int)
    ap.add_argument('--probe_hours_csv', default=str(cget('probe_hours_csv', '24,48,72,96,120,168,240,336,504,720,1080,1440,2160,2880')), type=str)

    ap.add_argument('--phase13_cost_pips', default=float(cget('phase13_cost_pips', 0.9)), type=float)
    ap.add_argument('--phase13_priors_source', default=str(cget('phase13_priors_source', 'windows_only')), choices=['windows_only', 'windows_plus_holdout'])
    ap.add_argument('--phase13_strict_temporal_eval', default=int(cget('phase13_strict_temporal_eval', 1)), type=int, choices=[0, 1])

    ap.add_argument('--min_segments', default=int(cget('min_segments', 5)), type=int)
    ap.add_argument('--freshness_max_hours', default=float(cget('freshness_max_hours', 6.0)), type=float)
    ap.add_argument('--safety_buffer_hours', default=int(cget('safety_buffer_hours', 24)), type=int)
    ap.add_argument('--allow_degraded_start', default=int(cget('allow_degraded_start', 1)), type=int, choices=[0, 1])

    ap.add_argument('--conf_min', default=float(cget('conf_min', 0.60)), type=float)
    ap.add_argument('--max_abs_lev', default=float(cget('max_abs_lev', 0.70)), type=float)
    ap.add_argument('--max_delta_lev_per_step', default=float(cget('max_delta_lev_per_step', 0.20)), type=float)
    ap.add_argument('--daily_loss_limit', default=float(cget('daily_loss_limit', 0.0075)), type=float)
    ap.add_argument('--start_equity', default=float(cget('start_equity', 3000.0)), type=float)

    ap.add_argument('--once', default=int(cget('once', 1)), type=int, choices=[0, 1])
    ap.add_argument('--loop', default=int(cget('loop', 0)), type=int, choices=[0, 1])
    ap.add_argument('--align_hour', default=int(cget('align_hour', 1)), type=int, choices=[0, 1])
    ap.add_argument('--hour_delay_sec', default=int(cget('hour_delay_sec', 20)), type=int)

    ap.add_argument('--retry_max', default=int(cget('retry_max', 3)), type=int)
    ap.add_argument('--retry_backoff_sec', default=int(cget('retry_backoff_sec', 15)), type=int)

    ns = ap.parse_args()

    return Config(
        symbol=str(ns.symbol).upper(),
        env_file=str(ns.env_file),
        out_root=str(ns.out_root),
        data_source=str(ns.data_source),
        input_m1_file=str(ns.input_m1_file),
        fetch_script=str(ns.fetch_script),
        phase13_script=str(ns.phase13_script),
        phase19_script=str(ns.phase19_script),
        demo=int(ns.demo),
        account_id=int(ns.account_id),
        fetch_count=int(ns.fetch_count),
        max_probe_hours=int(ns.max_probe_hours),
        probe_hours_csv=str(ns.probe_hours_csv),
        phase13_cost_pips=float(ns.phase13_cost_pips),
        phase13_priors_source=str(ns.phase13_priors_source),
        phase13_strict_temporal_eval=int(ns.phase13_strict_temporal_eval),
        min_segments=int(ns.min_segments),
        freshness_max_hours=float(ns.freshness_max_hours),
        safety_buffer_hours=int(ns.safety_buffer_hours),
        allow_degraded_start=int(ns.allow_degraded_start),
        conf_min=float(ns.conf_min),
        max_abs_lev=float(ns.max_abs_lev),
        max_delta_lev_per_step=float(ns.max_delta_lev_per_step),
        daily_loss_limit=float(ns.daily_loss_limit),
        start_equity=float(ns.start_equity),
        once=int(ns.once),
        loop=int(ns.loop),
        align_hour=int(ns.align_hour),
        hour_delay_sec=int(ns.hour_delay_sec),
        retry_max=int(ns.retry_max),
        retry_backoff_sec=int(ns.retry_backoff_sec),
    )


def main():
    cfg = build_config()
    run_dir = Path(cfg.out_root) / f'phase13_5_{cfg.symbol}_{now_utc_tag()}'
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(run_dir / 'phase13_5.log')
    write_json(run_dir / 'config.json', {'created_at_utc': now_utc_iso(), 'config': asdict(cfg), 'paper_only': True})

    logger.info('Phase13.5 start | symbol=%s | data_source=%s | run_dir=%s', cfg.symbol, cfg.data_source, run_dir)

    ref_m1 = fetch_reference_m1(cfg, run_dir, logger)
    ref_df = load_m1(ref_m1)
    if len(ref_df) < 1000:
        raise RuntimeError(f'Fetched data too short: {len(ref_df)} rows')

    ref_min = pd.Timestamp(ref_df['timestamp'].min())
    ref_max = pd.Timestamp(ref_df['timestamp'].max())
    logger.info('Fetched reference M1 rows=%s ts_min=%s ts_max=%s', len(ref_df), ref_min, ref_max)

    probe_hours = parse_probe_hours(cfg.probe_hours_csv)
    probe_hours = [h for h in probe_hours if h <= int(cfg.max_probe_hours)]
    if int(cfg.max_probe_hours) not in probe_hours:
        probe_hours.append(int(cfg.max_probe_hours))
    probe_hours = sorted(set(probe_hours))

    probe_data_dir = run_dir / 'probe_inputs'
    probe_data_dir.mkdir(parents=True, exist_ok=True)
    probe_out_root = run_dir / 'phase13_probe_runs'
    probe_out_root.mkdir(parents=True, exist_ok=True)

    probe_rows: List[Dict[str, Any]] = []
    for h in probe_hours:
        t0 = time.time()
        ts_start = ref_max - pd.Timedelta(hours=int(h))
        sub = ref_df.loc[ref_df['timestamp'] >= ts_start].copy()
        inp = probe_data_dir / f'{cfg.symbol.lower()}_m1_{int(h)}h.parquet'
        sub.to_parquet(inp, index=False)

        rec: Dict[str, Any] = {
            'lookback_h': int(h),
            'input_rows': int(len(sub)),
            'input_ts_min': pd.Timestamp(sub['timestamp'].min()).strftime('%Y-%m-%dT%H:%M:%SZ') if len(sub) else None,
            'input_ts_max': pd.Timestamp(sub['timestamp'].max()).strftime('%Y-%m-%dT%H:%M:%SZ') if len(sub) else None,
            'probe_runtime_sec': None,
            'probe_error': None,
        }

        if len(sub) < 500:
            rec.update({'status': 'FAILED', 'probe_error': 'too_few_rows(<500)'})
            probe_rows.append(rec)
            continue

        try:
            p13_run = run_phase13_probe(cfg, inp, probe_out_root, f'{int(h)}h', logger)
            ev = evaluate_probe(p13_run, cfg, ref_max, int(h))
            rec.update(ev)
        except Exception as e:
            rec.update({'status': 'FAILED', 'probe_error': f'{type(e).__name__}: {e}'})

        rec['probe_runtime_sec'] = round(float(time.time() - t0), 3)
        probe_rows.append(rec)

    probe_df = pd.DataFrame(probe_rows)
    probe_csv = run_dir / 'probe_results.csv'
    probe_json = run_dir / 'probe_results.json'
    probe_df.to_csv(probe_csv, index=False)
    write_json(probe_json, {'rows': probe_rows})

    sel = select_lookback(probe_rows, cfg)
    live_ready = bool(sel['live_ready'])
    selected_lookback_h = int(sel['selected_lookback_h'])
    degraded_start = not live_ready

    if degraded_start and int(cfg.allow_degraded_start) != 1:
        summary = {
            'status': 'FAILED_BLOCKED',
            'created_at_utc': now_utc_iso(),
            'symbol': cfg.symbol,
            'live_ready': False,
            'selection': sel,
            'reference_data': {
                'rows': int(len(ref_df)),
                'ts_min': ref_min.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'ts_max': ref_max.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'ref_m1_file': str(ref_m1),
            },
            'probe_outputs': {
                'probe_results_csv': str(probe_csv),
                'probe_results_json': str(probe_json),
            },
            'reason': 'live_not_ready_and_allow_degraded_start=0',
        }
        write_json(run_dir / 'phase13_5_summary.json', summary)
        logger.error('Phase13.5 blocked: no fresh probe and degraded start disabled')
        print(json.dumps(summary, indent=2, ensure_ascii=True))
        return

    live_info = launch_phase19(cfg, run_dir, selected_lookback_h, degraded_start, logger)

    summary = {
        'status': 'SUCCESS',
        'created_at_utc': now_utc_iso(),
        'symbol': cfg.symbol,
        'live_ready': bool(live_ready),
        'selection': sel,
        'reference_data': {
            'rows': int(len(ref_df)),
            'ts_min': ref_min.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'ts_max': ref_max.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'ref_m1_file': str(ref_m1),
        },
        'probe_outputs': {
            'probe_results_csv': str(probe_csv),
            'probe_results_json': str(probe_json),
            'phase13_probe_runs_root': str(probe_out_root),
        },
        'paper_live': live_info,
        'paper_only': True,
        'live_order_execution': False,
    }

    write_json(run_dir / 'phase13_5_summary.json', summary)
    logger.info('Phase13.5 SUCCESS | selected_lookback_h=%s | live_ready=%s', selected_lookback_h, live_ready)
    print(json.dumps({'status': 'SUCCESS', 'run_dir': str(run_dir), 'summary': str(run_dir / 'phase13_5_summary.json')}, indent=2, ensure_ascii=True))


if __name__ == '__main__':
    main()
