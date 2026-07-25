#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / 'phases/phase20_e2e_orchestration/config_phase20.json'


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def now_utc_tag() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')


def parse_utc_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    s = str(ts).strip()
    if not s:
        return None
    try:
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


def safe_json_write(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    tmp.replace(path)


def append_jsonl(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(payload, ensure_ascii=False) + '\n')


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


def setup_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger('phase20_orchestrator')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    fh = logging.FileHandler(path, mode='w', encoding='utf-8')
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def load_config(path: Path) -> Dict[str, Any]:
    txt = path.read_text(encoding='utf-8')
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(txt)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # Fallback: allow JSON-formatted YAML.
    data = json.loads(txt)
    if not isinstance(data, dict):
        raise ValueError('config file must parse to a dict/object')
    return data


def get_required(cfg: Dict[str, Any], *keys: str) -> Any:
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            raise KeyError(f'missing config key: {".".join(keys)}')
        cur = cur[k]
    return cur


def resolve_path(v: str) -> Path:
    return Path(str(v)).expanduser().resolve()


def assert_gbpusd(symbol: str):
    if str(symbol).upper() != 'GBPUSD':
        raise ValueError(f'phase20 v1 only supports GBPUSD, got {symbol!r}')


def run_cmd_with_retry(
    cmd: list[str],
    stage_name: str,
    stage_logs_dir: Path,
    alerts_file: Path,
    logger: logging.Logger,
    retries: int,
    backoff_sec: int,
):
    stage_logs_dir.mkdir(parents=True, exist_ok=True)
    last_err = None
    for i in range(1, int(retries) + 1):
        logger.info('[%s] attempt %s/%s', stage_name, i, retries)
        p = subprocess.run(cmd, capture_output=True, text=True)
        (stage_logs_dir / f'{stage_name}_attempt{i}.stdout.log').write_text(p.stdout or '', encoding='utf-8')
        (stage_logs_dir / f'{stage_name}_attempt{i}.stderr.log').write_text(p.stderr or '', encoding='utf-8')

        if p.returncode == 0:
            return

        last_err = RuntimeError(f'{stage_name} failed rc={p.returncode}')
        emit_alert(
            alerts_file,
            'WARNING',
            f'{stage_name}_failed',
            {
                'attempt': i,
                'retries': retries,
                'returncode': p.returncode,
                'cmd': cmd,
            },
        )
        if i < int(retries):
            time.sleep(int(backoff_sec))

    if last_err is None:
        last_err = RuntimeError(f'{stage_name} failed')
    raise last_err


def init_run_contract(run_dir: Path, command: str, symbol: str, config_path: Path, cfg: Dict[str, Any]) -> Dict[str, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    files = {
        'manifest': run_dir / 'phase20_manifest.json',
        'status': run_dir / 'phase20_status.json',
        'alerts': run_dir / 'phase20_alerts.jsonl',
        'guardrails': run_dir / 'phase20_live_guardrails.json',
        'snapshot': run_dir / 'phase20_status_snapshot.json',
        'log': run_dir / 'phase20.log',
    }
    manifest = {
        'phase': 'phase20',
        'status': 'RUNNING',
        'created_at_utc': now_utc_iso(),
        'command': command,
        'symbol': symbol,
        'paper_only': True,
        'live_order_execution': False,
        'config_path': str(config_path),
        'config': cfg,
        'stages': {},
        'child_runs': {},
    }
    safe_json_write(files['manifest'], manifest)

    status = {
        'status': 'RUNNING',
        'created_at_utc': manifest['created_at_utc'],
        'updated_at_utc': manifest['created_at_utc'],
        'command': command,
        'symbol': symbol,
        'paper_only': True,
    }
    safe_json_write(files['status'], status)
    return files


def update_status(path: Path, status: str, stage: str, extra: Optional[Dict[str, Any]] = None):
    payload = read_json(path) or {}
    payload.update(
        {
            'status': status,
            'stage': stage,
            'updated_at_utc': now_utc_iso(),
        }
    )
    if extra:
        payload.update(extra)
    safe_json_write(path, payload)


def update_manifest_stage(manifest_path: Path, stage: str, stage_status: str, details: Optional[Dict[str, Any]] = None):
    m = read_json(manifest_path) or {}
    stages = m.setdefault('stages', {})
    rec = stages.setdefault(stage, {})
    rec['status'] = stage_status
    rec['updated_at_utc'] = now_utc_iso()
    if details:
        rec.update(details)
    safe_json_write(manifest_path, m)


def compute_mode_from_guardrail(guardrail: Optional[Dict[str, Any]], stale_hours: float) -> Dict[str, Any]:
    if not guardrail:
        return {'mode': 'DEGRADED', 'reason': 'missing_guardrail', 'is_stale': True, 'age_hours': None}

    status = str(guardrail.get('status', 'UNKNOWN')).upper()
    gts = guardrail.get('generated_at_utc') or guardrail.get('ts_utc')
    ah = age_hours(gts)
    stale = (ah is None) or (ah > float(stale_hours))

    if status == 'PASS' and not stale:
        return {'mode': 'NORMAL', 'reason': 'guardrail_pass', 'is_stale': False, 'age_hours': ah}

    if stale:
        return {
            'mode': 'DEGRADED',
            'reason': f'guardrail_stale_gt_{float(stale_hours):.1f}h',
            'is_stale': True,
            'age_hours': ah,
        }
    return {'mode': 'DEGRADED', 'reason': f'guardrail_status_{status}', 'is_stale': False, 'age_hours': ah}


def run_live_h1(args: argparse.Namespace, cfg: Dict[str, Any], config_path: Path):
    symbol = str(get_required(cfg, 'symbol')).upper()
    assert_gbpusd(symbol)

    paths = get_required(cfg, 'paths')
    live_cfg = get_required(cfg, 'live_h1')
    stale_hours = float(live_cfg.get('phase20_guardrail_stale_hours', 36.0))

    out_root = resolve_path(args.out_dir or paths['root_out_dir'])
    run_dir = out_root / 'live_service'
    shared_guardrail = out_root / 'shared' / 'phase20_live_guardrails.json'
    files = init_run_contract(run_dir=run_dir, command='live-h1', symbol=symbol, config_path=config_path, cfg=cfg)
    logger = setup_logger(files['log'])

    emit_alert(files['alerts'], 'INFO', 'live_h1_starting', {'run_dir': str(run_dir)})

    phase19_script = resolve_path(paths['phase19_live_script'])
    phase13_script = resolve_path(paths['phase13_script'])
    fetch_script = resolve_path(paths['phase19_fetch_script'])
    env_file = resolve_path(paths['phase19_env_file'])

    once = int(args.once if args.once is not None else live_cfg.get('once', 0))
    loop = int(args.loop if args.loop is not None else live_cfg.get('loop', 1))
    align_hour = int(args.align_hour if args.align_hour is not None else live_cfg.get('align_hour', 1))
    hour_delay_sec = int(args.hour_delay_sec if args.hour_delay_sec is not None else live_cfg.get('hour_delay_sec', 20))
    data_source = str(args.data_source or live_cfg.get('data_source', 'ctrader'))
    input_m1_file = str(args.input_m1_file or live_cfg.get('input_m1_file', ''))
    account_id = int(args.account_id if args.account_id is not None else live_cfg.get('account_id', 0))
    demo = int(args.demo if args.demo is not None else live_cfg.get('demo', 1))

    cmd = [
        sys.executable,
        str(phase19_script),
        '--symbol', symbol,
        '--env_file', str(env_file),
        '--out_dir', str(run_dir),
        '--data_source', data_source,
        '--fetch_script', str(fetch_script),
        '--phase13_script', str(phase13_script),
        '--phase13_out_root', str(run_dir / 'phase13_runs'),
        '--phase13_cost_pips', str(float(live_cfg.get('phase13_cost_pips', 1.1))),
        '--phase13_priors_source', str(live_cfg.get('phase13_priors_source', 'windows_only')),
        '--phase13_strict_temporal_eval', str(int(live_cfg.get('phase13_strict_temporal_eval', 1))),
        '--start_equity', str(float(live_cfg.get('start_equity', 0.0))),
        '--conf_min', str(float(live_cfg.get('conf_min', 0.60))),
        '--max_abs_lev', str(float(live_cfg.get('max_abs_lev', 0.70))),
        '--max_delta_lev_per_step', str(float(live_cfg.get('max_delta_lev_per_step', 0.20))),
        '--daily_loss_limit', str(float(live_cfg.get('daily_loss_limit', 0.0075))),
        '--lookback_hours', str(int(live_cfg.get('lookback_hours', 24 * 120))),
        '--fetch_count', str(int(live_cfg.get('fetch_count', 0))),
        '--account_id', str(account_id),
        '--demo', str(demo),
        '--once', str(once),
        '--loop', str(loop),
        '--align_hour', str(align_hour),
        '--hour_delay_sec', str(hour_delay_sec),
        '--retry_max', str(int(live_cfg.get('retry_max', 3))),
        '--retry_backoff_sec', str(int(live_cfg.get('retry_backoff_sec', 15))),
        '--phase20_guardrail_enable', '1',
        '--phase20_guardrail_file', str(shared_guardrail),
        '--phase20_guardrail_stale_hours', str(stale_hours),
        '--phase20_degraded_max_abs_lev', str(float(live_cfg.get('phase20_degraded_max_abs_lev', 0.30))),
        '--phase20_degraded_max_delta_lev_per_step', str(float(live_cfg.get('phase20_degraded_max_delta_lev_per_step', 0.10))),
    ]
    if data_source == 'file' and input_m1_file:
        cmd.extend(['--input_m1_file', input_m1_file])

    update_status(files['status'], 'RUNNING', 'launch_phase19', {'cmd': cmd})
    update_manifest_stage(files['manifest'], 'phase19_live_h1', 'RUNNING', {'cmd': cmd})

    logger.info('Launching phase19 live engine...')
    p = subprocess.run(cmd)

    guardrail_payload = read_json(shared_guardrail)
    mode_info = compute_mode_from_guardrail(guardrail_payload, stale_hours=stale_hours)
    if not guardrail_payload:
        guardrail_payload = {
            'phase': 'phase20',
            'status': 'FAILED',
            'symbol': symbol,
            'generated_at_utc': now_utc_iso(),
            'mode_for_live': mode_info['mode'],
            'reason': mode_info['reason'],
            'paper_only': True,
            'live_order_execution': False,
            'refs': {'shared_guardrail_file': str(shared_guardrail)},
        }
    safe_json_write(files['guardrails'], guardrail_payload)

    snapshot = {
        'ts_utc': now_utc_iso(),
        'command': 'live-h1',
        'symbol': symbol,
        'live_run_dir': str(run_dir),
        'phase19_status_file': str(run_dir / 'status.json'),
        'phase19_state_file': str(run_dir / 'state.json'),
        'guardrail_file': str(shared_guardrail),
        'mode': mode_info['mode'],
        'mode_reason': mode_info['reason'],
        'guardrail_age_hours': mode_info['age_hours'],
        'guardrail_is_stale': mode_info['is_stale'],
        'paper_only': True,
        'returncode': int(p.returncode),
    }
    safe_json_write(files['snapshot'], snapshot)
    safe_json_write(out_root / 'phase20_status_snapshot.json', snapshot)

    if p.returncode == 0:
        update_manifest_stage(files['manifest'], 'phase19_live_h1', 'SUCCESS', {'ended_at_utc': now_utc_iso()})
        update_status(files['status'], 'SUCCESS', 'done', {'returncode': 0})
        emit_alert(files['alerts'], 'INFO', 'live_h1_finished', {'returncode': 0})
        m = read_json(files['manifest']) or {}
        m['status'] = 'SUCCESS'
        m['ended_at_utc'] = now_utc_iso()
        safe_json_write(files['manifest'], m)
        return

    update_manifest_stage(files['manifest'], 'phase19_live_h1', 'FAILED', {'ended_at_utc': now_utc_iso(), 'returncode': int(p.returncode)})
    update_status(files['status'], 'FAILED', 'phase19_live_h1_failed', {'returncode': int(p.returncode)})
    emit_alert(files['alerts'], 'ERROR', 'live_h1_failed', {'returncode': int(p.returncode)})
    m = read_json(files['manifest']) or {}
    m['status'] = 'FAILED'
    m['ended_at_utc'] = now_utc_iso()
    safe_json_write(files['manifest'], m)
    raise SystemExit(int(p.returncode))


def run_nightly_hardening(args: argparse.Namespace, cfg: Dict[str, Any], config_path: Path):
    symbol = str(get_required(cfg, 'symbol')).upper()
    assert_gbpusd(symbol)

    paths = get_required(cfg, 'paths')
    nightly_cfg = get_required(cfg, 'nightly_hardening')
    live_cfg = get_required(cfg, 'live_h1')

    out_root = resolve_path(args.out_dir or paths['root_out_dir'])
    run_id = str(args.run_id).strip() if args.run_id else f'phase20_nightly_{symbol}_{now_utc_tag()}'
    run_dir = out_root / 'nightly_runs' / run_id
    stage_logs = run_dir / 'stage_logs'
    shared_guardrail = out_root / 'shared' / 'phase20_live_guardrails.json'
    live_guardrail_copy = out_root / 'live_service' / 'phase20_live_guardrails.json'

    files = init_run_contract(run_dir=run_dir, command='nightly-hardening', symbol=symbol, config_path=config_path, cfg=cfg)
    logger = setup_logger(files['log'])

    retries = int(nightly_cfg.get('retry_max', 3))
    backoff = int(nightly_cfg.get('retry_backoff_sec', 20))
    cost_pips = float(args.cost_pips if args.cost_pips is not None else live_cfg.get('phase13_cost_pips', 1.1))

    manifest = read_json(files['manifest']) or {}
    manifest['run_id'] = run_id
    safe_json_write(files['manifest'], manifest)

    emit_alert(files['alerts'], 'INFO', 'nightly_started', {'run_id': run_id, 'run_dir': str(run_dir)})

    fetch_script = resolve_path(paths['phase19_fetch_script'])
    phase13_script = resolve_path(paths['phase13_script'])
    phase13_sanity_script = resolve_path(paths['phase13_sanity_script'])
    phase14_script = resolve_path(paths['phase14_script'])
    env_file = resolve_path(paths['phase19_env_file'])

    data_source = str(args.data_source or nightly_cfg.get('data_source', 'file'))
    input_file = str(args.input_m1_file or nightly_cfg.get('input_m1_file', ''))
    account_id = int(args.account_id if args.account_id is not None else nightly_cfg.get('account_id', 0))
    demo = int(args.demo if args.demo is not None else nightly_cfg.get('demo', 1))

    phase13_out_root = run_dir / 'phase13_runs'
    phase14_out_root = run_dir / 'phase14_runs'
    data_dir = run_dir / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    input_m1 = data_dir / f'{symbol.lower()}_nightly_m1.parquet'

    try:
        # Stage 1: get M1 window.
        update_status(files['status'], 'RUNNING', 'fetch_m1', {'run_id': run_id})
        update_manifest_stage(files['manifest'], 'fetch_m1', 'RUNNING')
        if data_source == 'file':
            if not input_file:
                raise ValueError('nightly file mode requires --input_m1_file or config nightly_hardening.input_m1_file')
            src = resolve_path(input_file)
            if not src.exists():
                raise FileNotFoundError(src)
            shutil.copy2(src, input_m1)
        else:
            fetch_cmd = [
                sys.executable,
                str(fetch_script),
                '--symbol', symbol,
                '--env_file', str(env_file),
                '--output', str(input_m1),
                '--lookback_hours', str(int(nightly_cfg.get('lookback_hours', 24 * 180))),
                '--count', str(int(nightly_cfg.get('fetch_count', 0))),
                '--account_id', str(account_id),
                '--demo', str(demo),
                '--require_demo_account', '1',
            ]
            run_cmd_with_retry(
                cmd=fetch_cmd,
                stage_name='fetch_m1',
                stage_logs_dir=stage_logs,
                alerts_file=files['alerts'],
                logger=logger,
                retries=retries,
                backoff_sec=backoff,
            )
        update_manifest_stage(files['manifest'], 'fetch_m1', 'SUCCESS', {'input_m1': str(input_m1)})

        # Stage 2: phase13 strict frozen.
        p13_run_id = f'phase20_p13_{symbol}_{now_utc_tag()}'
        p13_cmd = [
            sys.executable,
            str(phase13_script),
            '--input_m1', str(input_m1),
            '--symbol', symbol,
            '--mode', 'infer_frozen',
            '--cost_pips', str(cost_pips),
            '--priors_source', 'windows_only',
            '--strict_temporal_eval', '1',
            '--out_dir', str(phase13_out_root),
            '--run_id', p13_run_id,
        ]
        update_status(files['status'], 'RUNNING', 'phase13_pipeline', {'phase13_run_id': p13_run_id})
        update_manifest_stage(files['manifest'], 'phase13_pipeline', 'RUNNING', {'run_id': p13_run_id})
        run_cmd_with_retry(
            cmd=p13_cmd,
            stage_name='phase13_pipeline',
            stage_logs_dir=stage_logs,
            alerts_file=files['alerts'],
            logger=logger,
            retries=retries,
            backoff_sec=backoff,
        )
        p13_run = phase13_out_root / p13_run_id
        if not (p13_run / 'phase13_summary.json').exists():
            raise RuntimeError('phase13_summary.json missing after phase13 run')
        update_manifest_stage(files['manifest'], 'phase13_pipeline', 'SUCCESS', {'phase13_run_dir': str(p13_run)})

        # Stage 3: sanity checks.
        update_status(files['status'], 'RUNNING', 'phase13_sanity_checks')
        update_manifest_stage(files['manifest'], 'phase13_sanity_checks', 'RUNNING')
        sanity_cmd = [
            sys.executable,
            str(phase13_sanity_script),
            '--run_dir', str(p13_run),
        ]
        run_cmd_with_retry(
            cmd=sanity_cmd,
            stage_name='phase13_sanity_checks',
            stage_logs_dir=stage_logs,
            alerts_file=files['alerts'],
            logger=logger,
            retries=retries,
            backoff_sec=backoff,
        )
        sanity_json = p13_run / 'phase13_sanity_checks.json'
        if not sanity_json.exists():
            raise RuntimeError('phase13_sanity_checks.json missing after sanity run')
        update_manifest_stage(files['manifest'], 'phase13_sanity_checks', 'SUCCESS', {'sanity_json': str(sanity_json)})

        # Stage 4: phase14 hardening strict.
        p14_run_id = f'phase20_p14_{symbol}_{now_utc_tag()}'
        p14_out = phase14_out_root / p14_run_id
        update_status(files['status'], 'RUNNING', 'phase14_hardening', {'phase14_run_id': p14_run_id})
        update_manifest_stage(files['manifest'], 'phase14_hardening', 'RUNNING', {'run_id': p14_run_id})
        p14_cmd = [
            sys.executable,
            str(phase14_script),
            '--phase13_run_dir', str(p13_run),
            '--out_dir', str(p14_out),
            '--cost_pips', str(cost_pips),
            '--strict_only', '1',
            '--purged_gap_bars', str(int(nightly_cfg.get('purged_gap_bars', 24))),
            '--eval_stride', str(int(nightly_cfg.get('eval_stride', 8))),
        ]
        run_cmd_with_retry(
            cmd=p14_cmd,
            stage_name='phase14_hardening',
            stage_logs_dir=stage_logs,
            alerts_file=files['alerts'],
            logger=logger,
            retries=retries,
            backoff_sec=backoff,
        )
        p14_summary_path = p14_out / 'phase14_summary.json'
        if not p14_summary_path.exists():
            raise RuntimeError('phase14_summary.json missing after phase14 run')
        update_manifest_stage(files['manifest'], 'phase14_hardening', 'SUCCESS', {'phase14_run_dir': str(p14_out)})

        # Stage 5: gating and guardrail emission.
        update_status(files['status'], 'RUNNING', 'gating')
        update_manifest_stage(files['manifest'], 'gating', 'RUNNING')

        p13_summary = read_json(p13_run / 'phase13_summary.json') or {}
        p13_leak = read_json(p13_run / 'phase13_leak_audit.json') or {}
        p13_sanity = read_json(sanity_json) or {}
        p14_summary = read_json(p14_summary_path) or {}

        c1 = str(p13_summary.get('status', '')).upper() == 'SUCCESS'
        c2 = bool(p13_leak.get('all_eval_ts_gt_prior_max', False) is True)

        x2 = (((p14_summary.get('hardened_test_metrics_nooverlap') or {}).get('x2')) or {})
        sharpe_x2 = x2.get('sharpe_ann', None)
        cum_x2 = x2.get('cum_net_return', None)
        c3 = isinstance(sharpe_x2, (int, float)) and isinstance(cum_x2, (int, float)) and float(sharpe_x2) > 0.0 and float(cum_x2) > 0.0

        perm = (((p13_sanity.get('tests') or {}).get('phase12_2_frozen') or {}).get('permutation_iid') or {})
        pval_sh = perm.get('pvalue_sharpe_ge_obs', None)
        c4 = isinstance(pval_sh, (int, float)) and float(pval_sh) <= 0.10

        if int(args.force_guardrail_fail) == 1:
            c4 = False

        passed = bool(c1 and c2 and c3 and c4)
        guardrail_status = 'PASS' if passed else 'FAILED'
        live_mode = 'NORMAL' if passed else 'DEGRADED'

        normal_max_abs = float(live_cfg.get('max_abs_lev', 0.70))
        normal_max_delta = float(live_cfg.get('max_delta_lev_per_step', 0.20))
        degraded_max_abs = float(live_cfg.get('phase20_degraded_max_abs_lev', 0.30))
        degraded_max_delta = float(live_cfg.get('phase20_degraded_max_delta_lev_per_step', 0.10))

        guardrail_payload = {
            'phase': 'phase20',
            'status': guardrail_status,
            'symbol': symbol,
            'generated_at_utc': now_utc_iso(),
            'stale_after_hours': float(live_cfg.get('phase20_guardrail_stale_hours', 36.0)),
            'mode_for_live': live_mode,
            'reason': 'all_gates_passed' if passed else 'one_or_more_gates_failed',
            'effective_limits': {
                'max_abs_lev': normal_max_abs if passed else min(normal_max_abs, degraded_max_abs),
                'max_delta_lev_per_step': normal_max_delta if passed else min(normal_max_delta, degraded_max_delta),
            },
            'gating': {
                'phase13_status_success': c1,
                'phase13_all_eval_ts_gt_prior_max': c2,
                'phase14_test_nooverlap_x2_positive': c3,
                'phase13_sanity_pvalue_sharpe_le_0_10': c4,
                'phase14_x2_sharpe_ann': sharpe_x2,
                'phase14_x2_cum_net_return': cum_x2,
                'phase13_sanity_pvalue_sharpe_ge_obs': pval_sh,
            },
            'refs': {
                'phase13_run_dir': str(p13_run),
                'phase14_run_dir': str(p14_out),
                'phase13_sanity_checks_json': str(sanity_json),
            },
            'paper_only': True,
            'live_order_execution': False,
        }
        safe_json_write(files['guardrails'], guardrail_payload)
        safe_json_write(shared_guardrail, guardrail_payload)
        safe_json_write(live_guardrail_copy, guardrail_payload)

        # Keep child pointers in manifest.
        m = read_json(files['manifest']) or {}
        m['child_runs'] = {
            'phase13_run_dir': str(p13_run),
            'phase14_run_dir': str(p14_out),
            'phase13_sanity_checks_json': str(sanity_json),
        }
        m['status'] = 'SUCCESS' if passed else 'FAILED'
        m['ended_at_utc'] = now_utc_iso()
        safe_json_write(files['manifest'], m)

        update_manifest_stage(files['manifest'], 'gating', 'SUCCESS', {'guardrail_status': guardrail_status})

        snap = {
            'ts_utc': now_utc_iso(),
            'command': 'nightly-hardening',
            'run_id': run_id,
            'run_dir': str(run_dir),
            'symbol': symbol,
            'guardrail_status': guardrail_status,
            'live_mode_next': live_mode,
            'phase13_run_dir': str(p13_run),
            'phase14_run_dir': str(p14_out),
            'phase13_sanity_checks_json': str(sanity_json),
            'paper_only': True,
        }
        safe_json_write(files['snapshot'], snap)
        safe_json_write(out_root / 'phase20_status_snapshot.json', snap)

        status_text = 'SUCCESS' if passed else 'FAILED'
        update_status(files['status'], status_text, 'done', {'guardrail_status': guardrail_status, 'live_mode_next': live_mode})
        emit_alert(
            files['alerts'],
            'INFO' if passed else 'WARNING',
            'nightly_finished',
            {
                'guardrail_status': guardrail_status,
                'live_mode_next': live_mode,
                'phase13_run_dir': str(p13_run),
                'phase14_run_dir': str(p14_out),
            },
        )
        if not passed:
            raise RuntimeError('nightly gating failed; guardrail set to FAILED/DEGRADED')

    except Exception as e:
        logger.exception('nightly-hardening failed: %s', e)
        emit_alert(files['alerts'], 'ERROR', 'nightly_failed', {'error': f'{type(e).__name__}: {e}'})
        m = read_json(files['manifest']) or {}
        m['status'] = 'FAILED'
        m['ended_at_utc'] = now_utc_iso()
        m['error'] = f'{type(e).__name__}: {e}'
        safe_json_write(files['manifest'], m)
        update_status(files['status'], 'FAILED', 'failed', {'error': f'{type(e).__name__}: {e}'})
        raise


def run_status(args: argparse.Namespace, cfg: Dict[str, Any], config_path: Path):
    symbol = str(get_required(cfg, 'symbol')).upper()
    assert_gbpusd(symbol)

    paths = get_required(cfg, 'paths')
    live_cfg = get_required(cfg, 'live_h1')
    out_root = resolve_path(args.out_dir or paths['root_out_dir'])

    live_dir = out_root / 'live_service'
    guardrail_path = out_root / 'shared' / 'phase20_live_guardrails.json'
    nightly_root = out_root / 'nightly_runs'

    nightly_dirs = [p for p in nightly_root.glob('*') if p.is_dir()]
    nightly_dirs = sorted(nightly_dirs, key=lambda p: p.stat().st_mtime, reverse=True)
    latest_nightly = nightly_dirs[0] if nightly_dirs else None

    live_phase19_status = read_json(live_dir / 'status.json') or {}
    live_state = read_json(live_dir / 'state.json') or {}
    live_phase20_status = read_json(live_dir / 'phase20_status.json') or {}

    guardrail = read_json(guardrail_path)
    mode_info = compute_mode_from_guardrail(guardrail, stale_hours=float(live_cfg.get('phase20_guardrail_stale_hours', 36.0)))

    nightly_phase20_status = read_json(latest_nightly / 'phase20_status.json') if latest_nightly else None
    nightly_manifest = read_json(latest_nightly / 'phase20_manifest.json') if latest_nightly else None

    live_ts = live_phase19_status.get('updated_at_utc') or live_state.get('updated_at_utc')
    nightly_ts = None
    if nightly_phase20_status:
        nightly_ts = nightly_phase20_status.get('updated_at_utc')
    if (not nightly_ts) and nightly_manifest:
        nightly_ts = nightly_manifest.get('ended_at_utc')

    snapshot = {
        'ts_utc': now_utc_iso(),
        'command': 'status',
        'symbol': symbol,
        'paper_only': True,
        'mode': mode_info['mode'],
        'mode_reason': mode_info['reason'],
        'guardrail': {
            'path': str(guardrail_path),
            'exists': bool(guardrail is not None),
            'status': (guardrail or {}).get('status'),
            'generated_at_utc': (guardrail or {}).get('generated_at_utc'),
            'age_hours': mode_info['age_hours'],
            'is_stale': mode_info['is_stale'],
        },
        'live': {
            'run_dir': str(live_dir),
            'phase19_status': live_phase19_status.get('status'),
            'phase19_stage': live_phase19_status.get('stage'),
            'equity': live_phase19_status.get('equity', live_state.get('equity')),
            'latest_ts': live_phase19_status.get('latest_ts', live_state.get('last_ts')),
            'updated_at_utc': live_ts,
            'freshness_hours': age_hours(live_ts),
            'phase20_status': live_phase20_status.get('status'),
        },
        'nightly': {
            'latest_run_dir': str(latest_nightly) if latest_nightly else None,
            'status': (nightly_phase20_status or {}).get('status'),
            'guardrail_status': (nightly_phase20_status or {}).get('guardrail_status'),
            'updated_at_utc': nightly_ts,
            'freshness_hours': age_hours(nightly_ts),
        },
    }

    safe_json_write(out_root / 'phase20_status_snapshot.json', snapshot)
    if live_dir.exists():
        safe_json_write(live_dir / 'phase20_status_snapshot.json', snapshot)

    print(json.dumps(snapshot, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description='Phase20 orchestrator: live-h1 + nightly-hardening + status.')
    ap.add_argument('--config', default=str(DEFAULT_CONFIG), type=str)

    sub = ap.add_subparsers(dest='command', required=True)

    p_live = sub.add_parser('live-h1', help='Run/launch H1 paper-live process.')
    p_live.add_argument('--out_dir', default='', type=str)
    p_live.add_argument('--once', default=None, type=int, choices=[0, 1])
    p_live.add_argument('--loop', default=None, type=int, choices=[0, 1])
    p_live.add_argument('--align_hour', default=None, type=int, choices=[0, 1])
    p_live.add_argument('--hour_delay_sec', default=None, type=int)
    p_live.add_argument('--data_source', default=None, choices=['ctrader', 'file'])
    p_live.add_argument('--input_m1_file', default='', type=str)
    p_live.add_argument('--account_id', default=None, type=int)
    p_live.add_argument('--demo', default=None, type=int, choices=[0, 1])

    p_n = sub.add_parser('nightly-hardening', help='Run nightly strict validation and emit live guardrails.')
    p_n.add_argument('--out_dir', default='', type=str)
    p_n.add_argument('--run_id', default='', type=str)
    p_n.add_argument('--data_source', default=None, choices=['ctrader', 'file'])
    p_n.add_argument('--input_m1_file', default='', type=str)
    p_n.add_argument('--account_id', default=None, type=int)
    p_n.add_argument('--demo', default=None, type=int, choices=[0, 1])
    p_n.add_argument('--cost_pips', default=None, type=float)
    p_n.add_argument('--force_guardrail_fail', default=0, type=int, choices=[0, 1])

    p_s = sub.add_parser('status', help='Consolidated health status for live + nightly + guardrails.')
    p_s.add_argument('--out_dir', default='', type=str)

    return ap


def main():
    ap = build_arg_parser()
    args = ap.parse_args()

    config_path = resolve_path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    cfg = load_config(config_path)

    if args.command == 'live-h1':
        run_live_h1(args=args, cfg=cfg, config_path=config_path)
        return
    if args.command == 'nightly-hardening':
        run_nightly_hardening(args=args, cfg=cfg, config_path=config_path)
        return
    if args.command == 'status':
        run_status(args=args, cfg=cfg, config_path=config_path)
        return

    raise RuntimeError(f'Unknown command: {args.command}')


if __name__ == '__main__':
    main()
