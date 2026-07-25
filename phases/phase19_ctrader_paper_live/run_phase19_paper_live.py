#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from phase19_utils import append_jsonl, now_utc_iso, now_utc_tag, safe_json_write

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHASE13_SCRIPT = ROOT / 'phases/phase13_pipeline_cli/run_phase13_pipeline.py'
DEFAULT_FETCH_SCRIPT = ROOT / 'phases/phase19_ctrader_paper_live/run_phase19_fetch_m1_once.py'


@dataclass
class LiveConfig:
    symbol: str
    env_file: str
    out_dir: str
    data_source: str  # ctrader|file
    input_m1_file: str
    fetch_script: str
    phase13_script: str

    phase13_out_root: str
    phase13_cost_pips: float
    phase13_priors_source: str
    phase13_strict_temporal_eval: int

    lookback_hours: int
    fetch_count: int
    account_id: int
    demo: int

    # Risk engine (paper only)
    start_equity: float
    conf_min: float
    max_abs_lev: float
    max_delta_lev_per_step: float
    daily_loss_limit: float

    # Loop controls
    loop: int
    once: int
    align_hour: int
    hour_delay_sec: int
    retry_max: int
    retry_backoff_sec: int
    # Optional guardrail controls from phase20.
    phase20_guardrail_enable: int
    phase20_guardrail_file: str
    phase20_guardrail_stale_hours: float
    phase20_degraded_max_abs_lev: float
    phase20_degraded_max_delta_lev_per_step: float


def pip_size_for_symbol(symbol: str) -> float:
    return 0.01 if symbol.upper().endswith('JPY') else 0.0001


def setup_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger('phase19_paper_live')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    fh = logging.FileHandler(path, mode='w')
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def run_with_retry(cmd: list[str], retries: int, backoff_sec: int, logger: logging.Logger, stage: str):
    last_err = None
    for k in range(1, retries + 1):
        logger.info('[%s] attempt %s/%s', stage, k, retries)
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode == 0:
            if p.stdout.strip():
                logger.info('[%s] stdout: %s', stage, p.stdout.strip()[:1200])
            return
        last_err = RuntimeError(f'{stage} failed rc={p.returncode} stderr={p.stderr[-1000:]}')
        logger.warning('[%s] failed rc=%s', stage, p.returncode)
        if k < retries:
            time.sleep(backoff_sec)
    raise last_err if last_err else RuntimeError(f'{stage} failed')


def load_or_init_state(path: Path, start_equity: float) -> Dict:
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    return {
        'status': 'INIT',
        'created_at_utc': now_utc_iso(),
        'updated_at_utc': now_utc_iso(),
        'equity': float(start_equity),
        'cum_pnl': 0.0,
        'position_lev': 0.0,
        'last_close': None,
        'last_ts': None,
        'day': None,
        'day_pnl': 0.0,
        'kill_locked_day': None,
        'n_cycles': 0,
        'n_errors': 0,
        'last_guardrail_mode': None,
    }


def parse_utc_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    s = str(ts).strip()
    if not s:
        return None
    try:
        # Accept "...Z" and offsets.
        if s.endswith('Z'):
            s = s[:-1] + '+00:00'
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def load_guardrail_state(cfg: LiveConfig) -> Dict[str, object]:
    # Backward compatible: when disabled, keep NORMAL behavior.
    if int(cfg.phase20_guardrail_enable) != 1:
        return {
            'mode': 'NORMAL',
            'status': 'DISABLED',
            'reason': 'guardrail_disabled',
            'guardrail_file': str(cfg.phase20_guardrail_file),
            'generated_at_utc': None,
            'is_stale': False,
        }

    path = Path(cfg.phase20_guardrail_file)
    if not path.exists():
        return {
            'mode': 'DEGRADED',
            'status': 'MISSING',
            'reason': 'guardrail_file_missing',
            'guardrail_file': str(path),
            'generated_at_utc': None,
            'is_stale': True,
        }

    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except Exception as e:
        return {
            'mode': 'DEGRADED',
            'status': 'INVALID',
            'reason': f'guardrail_json_error:{type(e).__name__}',
            'guardrail_file': str(path),
            'generated_at_utc': None,
            'is_stale': True,
        }

    status = str(payload.get('status', '')).upper()
    gts = payload.get('generated_at_utc') or payload.get('ts_utc')
    dt = parse_utc_iso(gts)
    is_stale = True
    if dt is not None:
        age_h = (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0
        is_stale = bool(age_h > float(cfg.phase20_guardrail_stale_hours))
    mode = 'NORMAL'
    reason = 'guardrail_pass'
    if status != 'PASS':
        mode = 'DEGRADED'
        reason = f'guardrail_status_{status or "UNKNOWN"}'
    if is_stale:
        mode = 'DEGRADED'
        reason = f'guardrail_stale_gt_{float(cfg.phase20_guardrail_stale_hours):.1f}h'

    return {
        'mode': mode,
        'status': status or 'UNKNOWN',
        'reason': reason,
        'guardrail_file': str(path),
        'generated_at_utc': gts,
        'is_stale': bool(is_stale),
    }


def seconds_to_next_hour(delay_sec: int) -> int:
    now = datetime.now(timezone.utc)
    nxt = (now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))
    sec = int((nxt - now).total_seconds()) + int(delay_sec)
    return max(sec, 1)


def emit_alert(alerts_file: Path, level: str, message: str, payload: dict):
    append_jsonl(
        alerts_file,
        {
            'ts_utc': now_utc_iso(),
            'level': level,
            'message': message,
            'payload': payload,
        },
    )


def _fetch_input(cfg: LiveConfig, run_dir: Path, logger: logging.Logger):
    data_dir = run_dir / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    out_m1 = data_dir / f'{cfg.symbol.lower()}_latest_m1.parquet'

    if cfg.data_source == 'file':
        src = Path(cfg.input_m1_file)
        if not src.exists():
            raise FileNotFoundError(f'input_m1_file not found: {src}')
        if src.resolve() != out_m1.resolve():
            shutil.copy2(src, out_m1)
        return out_m1

    fetch_cmd = [
        sys.executable,
        cfg.fetch_script,
        '--symbol', cfg.symbol,
        '--env_file', cfg.env_file,
        '--output', str(out_m1),
        '--lookback_hours', str(cfg.lookback_hours),
        '--count', str(cfg.fetch_count),
        '--account_id', str(cfg.account_id),
        '--demo', str(int(cfg.demo)),
        '--require_demo_account', '1',
    ]
    run_with_retry(fetch_cmd, retries=cfg.retry_max, backoff_sec=cfg.retry_backoff_sec, logger=logger, stage='fetch_m1')
    return out_m1


def _run_phase13(cfg: LiveConfig, input_m1: Path, run_dir: Path, logger: logging.Logger) -> Path:
    p13_root = Path(cfg.phase13_out_root)
    p13_root.mkdir(parents=True, exist_ok=True)
    run_id = f'phase19_p13_{cfg.symbol}_{now_utc_tag()}'

    cmd = [
        sys.executable,
        cfg.phase13_script,
        '--input_m1', str(input_m1),
        '--symbol', cfg.symbol,
        '--mode', 'infer_frozen',
        '--cost_pips', str(cfg.phase13_cost_pips),
        '--priors_source', cfg.phase13_priors_source,
        '--strict_temporal_eval', str(int(cfg.phase13_strict_temporal_eval)),
        '--out_dir', str(p13_root),
        '--run_id', run_id,
    ]
    run_with_retry(cmd, retries=cfg.retry_max, backoff_sec=cfg.retry_backoff_sec, logger=logger, stage='phase13')

    p13_run = p13_root / run_id
    if not (p13_run / 'phase13_summary.json').exists():
        raise RuntimeError(f'phase13 summary missing: {p13_run / "phase13_summary.json"}')
    return p13_run


def _extract_latest_signal(p13_run: Path) -> Dict:
    summary = json.loads((p13_run / 'phase13_summary.json').read_text(encoding='utf-8'))
    if summary.get('status') != 'SUCCESS':
        raise RuntimeError('phase13 summary status != SUCCESS')

    st5_path = p13_run / 'stage05_phase12_predictions.parquet'
    if not st5_path.exists():
        raise RuntimeError(f'Missing stage05 output: {st5_path}')

    df = pd.read_parquet(st5_path)
    if len(df) == 0:
        raise RuntimeError('Empty stage05 dataframe')

    if 'timestamp' not in df.columns or 'close' not in df.columns:
        raise RuntimeError('stage05 missing timestamp/close')

    d = df.copy()
    d['timestamp'] = pd.to_datetime(d['timestamp'], errors='coerce', utc=True)
    d = d.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    if len(d) == 0:
        raise RuntimeError('No valid timestamp rows in stage05')

    last = d.iloc[-1]
    lev = float(last['lev_phase12_2_frozen']) if 'lev_phase12_2_frozen' in d.columns else float(last.get('pred_action', 0.0))
    conf = float(last['pred_conf_max']) if 'pred_conf_max' in d.columns else 1.0

    return {
        'timestamp': pd.Timestamp(last['timestamp']).to_pydatetime(),
        'close': float(last['close']),
        'lev_raw': float(lev),
        'conf': float(conf),
        'run_dir': str(p13_run),
    }


def _risk_and_paper_step(signal: Dict, state: Dict, cfg: LiveConfig, guardrail: Dict[str, object]) -> Dict:
    ts: datetime = signal['timestamp']
    close = float(signal['close'])
    lev_raw = float(signal['lev_raw'])
    conf = float(signal['conf'])

    prev_lev = float(state.get('position_lev', 0.0) or 0.0)
    prev_close = state.get('last_close', None)

    # Realize previous interval PnL (if possible)
    ret_h = 0.0
    pnl = 0.0
    if prev_close is not None and float(prev_close) > 0:
        ret_h = (close / float(prev_close)) - 1.0
        cost_ret = float(cfg.phase13_cost_pips) * pip_size_for_symbol(cfg.symbol)
        pnl = prev_lev * ret_h - abs(prev_lev) * cost_ret
        state['equity'] = float(state.get('equity', 0.0)) + pnl
        state['cum_pnl'] = float(state.get('cum_pnl', 0.0)) + pnl

    # Day bookkeeping
    day = ts.date().isoformat()
    if state.get('day') != day:
        state['day'] = day
        state['day_pnl'] = 0.0
        # release lock on new day
        if state.get('kill_locked_day') != day:
            state['kill_locked_day'] = None

    state['day_pnl'] = float(state.get('day_pnl', 0.0)) + pnl

    # Risk-adjusted target leverage for next interval
    max_abs_lev_eff = float(cfg.max_abs_lev)
    max_delta_lev_eff = float(cfg.max_delta_lev_per_step)
    if str(guardrail.get('mode', 'NORMAL')).upper() == 'DEGRADED':
        max_abs_lev_eff = min(max_abs_lev_eff, float(cfg.phase20_degraded_max_abs_lev))
        max_delta_lev_eff = min(max_delta_lev_eff, float(cfg.phase20_degraded_max_delta_lev_per_step))

    target = lev_raw if conf >= float(cfg.conf_min) else 0.0
    target = float(np.clip(target, -max_abs_lev_eff, max_abs_lev_eff))

    # Smooth turnover by limiting step delta
    max_d = max_delta_lev_eff
    target = float(np.clip(target, prev_lev - max_d, prev_lev + max_d))

    # Daily kill-switch
    if float(state.get('day_pnl', 0.0)) <= -abs(float(cfg.daily_loss_limit)):
        state['kill_locked_day'] = day
    if state.get('kill_locked_day') == day:
        target = 0.0

    trade_event = 1 if abs(target - prev_lev) > 1e-9 else 0

    # Update state for next bar
    state['position_lev'] = float(target)
    state['last_close'] = float(close)
    state['last_ts'] = ts.strftime('%Y-%m-%dT%H:%M:%SZ')
    state['updated_at_utc'] = now_utc_iso()
    state['status'] = 'RUNNING'
    state['n_cycles'] = int(state.get('n_cycles', 0)) + 1

    return {
        'timestamp_utc': ts.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'close': close,
        'conf': conf,
        'lev_raw': lev_raw,
        'lev_prev': prev_lev,
        'lev_target_next': float(target),
        'ret_h_realized': float(ret_h),
        'pnl_realized': float(pnl),
        'equity': float(state['equity']),
        'day': day,
        'day_pnl': float(state['day_pnl']),
        'kill_locked_day': state.get('kill_locked_day'),
        'trade_event': int(trade_event),
        'op_mode': str(guardrail.get('mode', 'NORMAL')),
        'guardrail_status': str(guardrail.get('status', 'UNKNOWN')),
        'guardrail_reason': str(guardrail.get('reason', 'n/a')),
        'guardrail_file': str(guardrail.get('guardrail_file', '')),
        'max_abs_lev_effective': float(max_abs_lev_eff),
        'max_delta_lev_per_step_effective': float(max_d),
        'phase13_run_dir': signal['run_dir'],
    }


def _append_decision(decisions_csv: Path, row: Dict):
    df = pd.DataFrame([row])
    header = not decisions_csv.exists()
    df.to_csv(decisions_csv, mode='a', header=header, index=False)


def run_cycle(cfg: LiveConfig, run_dir: Path, state_path: Path, status_path: Path, alerts_file: Path, decisions_csv: Path, logger: logging.Logger):
    state = load_or_init_state(state_path, start_equity=float(cfg.start_equity))

    safe_json_write(
        status_path,
        {
            'status': 'RUNNING',
            'updated_at_utc': now_utc_iso(),
            'stage': 'fetch_input',
            'symbol': cfg.symbol,
            'run_dir': str(run_dir),
        },
    )

    input_m1 = _fetch_input(cfg=cfg, run_dir=run_dir, logger=logger)

    safe_json_write(
        status_path,
        {
            'status': 'RUNNING',
            'updated_at_utc': now_utc_iso(),
            'stage': 'phase13_inference',
            'input_m1': str(input_m1),
        },
    )

    p13_run = _run_phase13(cfg=cfg, input_m1=input_m1, run_dir=run_dir, logger=logger)
    signal = _extract_latest_signal(p13_run)

    last_ts_prev = state.get('last_ts')
    signal_ts_iso = signal['timestamp'].strftime('%Y-%m-%dT%H:%M:%SZ')
    if last_ts_prev and signal_ts_iso <= str(last_ts_prev):
        emit_alert(
            alerts_file,
            'WARNING',
            'no_new_bar',
            {'last_ts_prev': last_ts_prev, 'signal_ts': signal_ts_iso, 'phase13_run_dir': str(p13_run)},
        )
        safe_json_write(
            status_path,
            {
                'status': 'RUNNING',
                'updated_at_utc': now_utc_iso(),
                'stage': 'idle_no_new_bar',
                'last_ts_prev': last_ts_prev,
                'signal_ts': signal_ts_iso,
            },
        )
        safe_json_write(state_path, state)
        return

    guardrail = load_guardrail_state(cfg=cfg)
    prev_mode = str(state.get('last_guardrail_mode') or '')
    cur_mode = str(guardrail.get('mode', 'NORMAL'))
    if prev_mode != cur_mode:
        emit_alert(
            alerts_file,
            'WARNING' if cur_mode == 'DEGRADED' else 'INFO',
            'guardrail_mode_changed',
            {
                'prev_mode': prev_mode or None,
                'new_mode': cur_mode,
                'reason': guardrail.get('reason'),
                'status': guardrail.get('status'),
                'guardrail_file': guardrail.get('guardrail_file'),
            },
        )
    state['last_guardrail_mode'] = cur_mode

    decision = _risk_and_paper_step(signal=signal, state=state, cfg=cfg, guardrail=guardrail)
    _append_decision(decisions_csv, decision)
    safe_json_write(state_path, state)

    latest_signal = run_dir / 'latest_signal.json'
    safe_json_write(latest_signal, decision)

    safe_json_write(
        status_path,
        {
            'status': 'RUNNING',
            'updated_at_utc': now_utc_iso(),
            'stage': 'ok',
            'latest_ts': decision['timestamp_utc'],
            'equity': decision['equity'],
            'lev_target_next': decision['lev_target_next'],
            'op_mode': decision['op_mode'],
            'guardrail_reason': decision['guardrail_reason'],
            'phase13_run_dir': decision['phase13_run_dir'],
            'latest_signal_json': str(latest_signal),
        },
    )


def main():
    ap = argparse.ArgumentParser(description='Phase19 paper-live daemon (cTrader M1 -> phase13 -> paper decision).')
    ap.add_argument('--symbol', default='GBPUSD', type=str)
    ap.add_argument('--env_file', default='./phases/phase19_ctrader_paper_live/.env', type=str)
    ap.add_argument('--out_dir', default='./phases/phase19_ctrader_paper_live/output_phase19_live_v1', type=str)
    ap.add_argument('--data_source', default='ctrader', choices=['ctrader', 'file'])
    ap.add_argument('--input_m1_file', default='./phases/phase13_pipeline_cli/gbpusd_20000530_20251205_dedup_from_phase1.parquet', type=str)
    ap.add_argument('--fetch_script', default=str(DEFAULT_FETCH_SCRIPT), type=str)
    ap.add_argument('--phase13_script', default=str(DEFAULT_PHASE13_SCRIPT), type=str)
    ap.add_argument('--phase13_out_root', default='', type=str)
    ap.add_argument('--phase13_cost_pips', default=1.1, type=float)
    ap.add_argument('--phase13_priors_source', default='windows_only', choices=['windows_only', 'windows_plus_holdout'])
    ap.add_argument('--phase13_strict_temporal_eval', default=1, type=int, choices=[0, 1])

    ap.add_argument('--lookback_hours', default=24 * 120, type=int)
    ap.add_argument('--fetch_count', default=0, type=int)
    ap.add_argument('--account_id', default=0, type=int)
    ap.add_argument('--demo', default=1, type=int, choices=[0, 1])

    ap.add_argument('--start_equity', default=0.0, type=float)
    ap.add_argument('--conf_min', default=0.55, type=float)
    ap.add_argument('--max_abs_lev', default=1.0, type=float)
    ap.add_argument('--max_delta_lev_per_step', default=0.50, type=float)
    ap.add_argument('--daily_loss_limit', default=0.01, type=float)

    ap.add_argument('--loop', default=1, type=int, choices=[0, 1])
    ap.add_argument('--once', default=0, type=int, choices=[0, 1])
    ap.add_argument('--align_hour', default=1, type=int, choices=[0, 1])
    ap.add_argument('--hour_delay_sec', default=20, type=int)
    ap.add_argument('--retry_max', default=3, type=int)
    ap.add_argument('--retry_backoff_sec', default=15, type=int)
    ap.add_argument('--phase20_guardrail_enable', default=0, type=int, choices=[0, 1])
    ap.add_argument('--phase20_guardrail_file', default='', type=str)
    ap.add_argument('--phase20_guardrail_stale_hours', default=36.0, type=float)
    ap.add_argument('--phase20_degraded_max_abs_lev', default=0.30, type=float)
    ap.add_argument('--phase20_degraded_max_delta_lev_per_step', default=0.10, type=float)
    args = ap.parse_args()

    phase13_out_root = str(args.phase13_out_root).strip() or str(Path(args.out_dir) / 'phase13_runs')

    cfg = LiveConfig(
        symbol=str(args.symbol).upper(),
        env_file=str(args.env_file),
        out_dir=str(args.out_dir),
        data_source=str(args.data_source),
        input_m1_file=str(args.input_m1_file),
        fetch_script=str(args.fetch_script),
        phase13_script=str(args.phase13_script),
        phase13_out_root=phase13_out_root,
        phase13_cost_pips=float(args.phase13_cost_pips),
        phase13_priors_source=str(args.phase13_priors_source),
        phase13_strict_temporal_eval=int(args.phase13_strict_temporal_eval),
        lookback_hours=int(args.lookback_hours),
        fetch_count=int(args.fetch_count),
        account_id=int(args.account_id),
        demo=int(args.demo),
        start_equity=float(args.start_equity),
        conf_min=float(args.conf_min),
        max_abs_lev=float(args.max_abs_lev),
        max_delta_lev_per_step=float(args.max_delta_lev_per_step),
        daily_loss_limit=float(args.daily_loss_limit),
        loop=int(args.loop),
        once=int(args.once),
        align_hour=int(args.align_hour),
        hour_delay_sec=int(args.hour_delay_sec),
        retry_max=int(args.retry_max),
        retry_backoff_sec=int(args.retry_backoff_sec),
        phase20_guardrail_enable=int(args.phase20_guardrail_enable),
        phase20_guardrail_file=str(args.phase20_guardrail_file),
        phase20_guardrail_stale_hours=float(args.phase20_guardrail_stale_hours),
        phase20_degraded_max_abs_lev=float(args.phase20_degraded_max_abs_lev),
        phase20_degraded_max_delta_lev_per_step=float(args.phase20_degraded_max_delta_lev_per_step),
    )

    run_dir = Path(cfg.out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / 'phase19.log'
    status_path = run_dir / 'status.json'
    state_path = run_dir / 'state.json'
    alerts_file = run_dir / 'alerts.jsonl'
    decisions_csv = run_dir / 'paper_decisions.csv'

    logger = setup_logger(log_path)

    safe_json_write(
        run_dir / 'config.json',
        {
            'created_at_utc': now_utc_iso(),
            'config': asdict(cfg),
            'paper_only': True,
            'live_order_execution': False,
        },
    )

    logger.info('Phase19 start | symbol=%s | source=%s | out_dir=%s', cfg.symbol, cfg.data_source, run_dir)
    emit_alert(alerts_file, 'INFO', 'phase19_started', {'symbol': cfg.symbol, 'out_dir': str(run_dir)})

    try:
        if int(cfg.once) == 1:
            run_cycle(cfg, run_dir, state_path, status_path, alerts_file, decisions_csv, logger)
            logger.info('Phase19 one-shot cycle SUCCESS')
            return

        while True:
            cycle_started = now_utc_iso()
            try:
                run_cycle(cfg, run_dir, state_path, status_path, alerts_file, decisions_csv, logger)
                logger.info('cycle success | started=%s', cycle_started)
            except Exception as e:
                logger.exception('cycle failed: %s', e)
                st = load_or_init_state(state_path, cfg.start_equity)
                st['n_errors'] = int(st.get('n_errors', 0)) + 1
                st['updated_at_utc'] = now_utc_iso()
                safe_json_write(state_path, st)
                safe_json_write(
                    status_path,
                    {
                        'status': 'RUNNING_WITH_ERRORS',
                        'updated_at_utc': now_utc_iso(),
                        'error': f'{type(e).__name__}: {e}',
                        'n_errors': st['n_errors'],
                    },
                )
                emit_alert(alerts_file, 'ERROR', 'cycle_failed', {'error': f'{type(e).__name__}: {e}'})

            if int(cfg.loop) != 1:
                break

            if int(cfg.align_hour) == 1:
                sec = seconds_to_next_hour(cfg.hour_delay_sec)
            else:
                sec = 3600
            logger.info('sleeping %s sec to next cycle', sec)
            time.sleep(sec)

    finally:
        emit_alert(alerts_file, 'INFO', 'phase19_stopped', {'ts_utc': now_utc_iso()})


if __name__ == '__main__':
    main()
