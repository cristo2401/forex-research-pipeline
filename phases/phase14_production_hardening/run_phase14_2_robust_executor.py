#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List


@dataclass
class ExecConfig:
    input_m1: str
    symbol: str
    phase13_script: str
    phase14_script: str
    phase13_out_root: str
    phase14_out_root: str
    phase13_run_id: str
    phase14_run_id: str
    priors_source: str
    strict_temporal_eval: int
    cost_pips: float
    max_retries: int
    retry_backoff_sec: int
    alert_file: str


def now_utc_tag() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')


def emit_alert(alert_path: Path, level: str, message: str, payload: dict):
    rec = {
        'ts_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'level': level,
        'message': message,
        'payload': payload,
    }
    alert_path.parent.mkdir(parents=True, exist_ok=True)
    with alert_path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')


def run_cmd_with_retry(
    name: str,
    cmd: List[str],
    max_retries: int,
    retry_backoff_sec: int,
    stage_dir: Path,
    alert_file: Path,
    logger: logging.Logger,
):
    stage_dir.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_retries + 1):
        logger.info('[%s] attempt %s/%s', name, attempt, max_retries)
        p = subprocess.run(cmd, capture_output=True, text=True)

        (stage_dir / f'{name}_attempt{attempt}.stdout.log').write_text(p.stdout or '', encoding='utf-8')
        (stage_dir / f'{name}_attempt{attempt}.stderr.log').write_text(p.stderr or '', encoding='utf-8')

        if p.returncode == 0:
            logger.info('[%s] success at attempt %s', name, attempt)
            return

        emit_alert(
            alert_file,
            level='WARNING',
            message=f'{name} failed',
            payload={
                'attempt': attempt,
                'returncode': p.returncode,
                'cmd': cmd,
            },
        )
        logger.warning('[%s] failed at attempt %s | returncode=%s', name, attempt, p.returncode)

        if attempt < max_retries:
            time.sleep(int(retry_backoff_sec))

    raise RuntimeError(f'{name} failed after {max_retries} attempts')


def main():
    parser = argparse.ArgumentParser(description='Phase14 robust E2E runner: phase13 pipeline + phase14 hardening with retries/alerts.')
    parser.add_argument('--input_m1', required=True, type=str)
    parser.add_argument('--symbol', default='GBPUSD', type=str)
    parser.add_argument('--phase13_script', default='./phases/phase13_pipeline_cli/run_phase13_pipeline.py', type=str)
    parser.add_argument('--phase14_script', default='./phases/phase14_production_hardening/run_phase14_1_hardening.py', type=str)
    parser.add_argument('--phase13_out_root', default='./phases/phase13_pipeline_cli/output_runs', type=str)
    parser.add_argument('--phase14_out_root', default='./phases/phase14_production_hardening/output_runs', type=str)
    parser.add_argument('--phase13_run_id', default='', type=str)
    parser.add_argument('--phase14_run_id', default='', type=str)
    parser.add_argument('--priors_source', default='windows_only', choices=['windows_only', 'windows_plus_holdout'])
    parser.add_argument('--strict_temporal_eval', default=1, type=int, choices=[0, 1])
    parser.add_argument('--cost_pips', default=1.2, type=float)
    parser.add_argument('--max_retries', default=3, type=int)
    parser.add_argument('--retry_backoff_sec', default=20, type=int)
    parser.add_argument('--alert_file', default='', type=str)
    args = parser.parse_args()

    phase13_run_id = args.phase13_run_id.strip() or f'phase13_exec_{args.symbol}_{now_utc_tag()}'
    phase14_run_id = args.phase14_run_id.strip() or f'phase14_exec_{args.symbol}_{now_utc_tag()}'

    out_root = Path(args.phase14_out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir = out_root / phase14_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    alert_file = Path(args.alert_file) if args.alert_file else (run_dir / 'phase14_alerts.jsonl')
    status_path = run_dir / 'status.json'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(run_dir / 'phase14_executor.log', mode='w'),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger('phase14_executor')

    cfg = ExecConfig(
        input_m1=str(args.input_m1),
        symbol=str(args.symbol),
        phase13_script=str(args.phase13_script),
        phase14_script=str(args.phase14_script),
        phase13_out_root=str(args.phase13_out_root),
        phase14_out_root=str(args.phase14_out_root),
        phase13_run_id=phase13_run_id,
        phase14_run_id=phase14_run_id,
        priors_source=str(args.priors_source),
        strict_temporal_eval=int(args.strict_temporal_eval),
        cost_pips=float(args.cost_pips),
        max_retries=int(args.max_retries),
        retry_backoff_sec=int(args.retry_backoff_sec),
        alert_file=str(alert_file),
    )

    status = {
        'status': 'RUNNING',
        'created_at_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'config': cfg.__dict__,
        'stage_phase13': 'PENDING',
        'stage_phase14': 'PENDING',
    }
    status_path.write_text(json.dumps(status, indent=2))

    try:
        # Stage 1: phase13 single-command pipeline.
        phase13_cmd = [
            sys.executable,
            cfg.phase13_script,
            '--input_m1', cfg.input_m1,
            '--symbol', cfg.symbol,
            '--mode', 'infer_frozen',
            '--priors_source', cfg.priors_source,
            '--strict_temporal_eval', str(cfg.strict_temporal_eval),
            '--out_dir', cfg.phase13_out_root,
            '--run_id', cfg.phase13_run_id,
        ]
        run_cmd_with_retry(
            name='phase13_pipeline',
            cmd=phase13_cmd,
            max_retries=cfg.max_retries,
            retry_backoff_sec=cfg.retry_backoff_sec,
            stage_dir=run_dir / 'stage_logs',
            alert_file=alert_file,
            logger=logger,
        )

        phase13_run_dir = Path(cfg.phase13_out_root) / cfg.phase13_run_id
        if not (phase13_run_dir / 'phase13_summary.json').exists():
            raise RuntimeError(f'phase13 summary missing: {phase13_run_dir / "phase13_summary.json"}')

        status['stage_phase13'] = 'SUCCESS'
        status_path.write_text(json.dumps(status, indent=2))

        # Stage 2: phase14 hardening + robust validation.
        phase14_out_dir = Path(cfg.phase14_out_root) / cfg.phase14_run_id
        phase14_cmd = [
            sys.executable,
            cfg.phase14_script,
            '--phase13_run_dir', str(phase13_run_dir),
            '--out_dir', str(phase14_out_dir),
            '--cost_pips', str(cfg.cost_pips),
            '--strict_only', '1',
        ]
        run_cmd_with_retry(
            name='phase14_hardening',
            cmd=phase14_cmd,
            max_retries=cfg.max_retries,
            retry_backoff_sec=cfg.retry_backoff_sec,
            stage_dir=run_dir / 'stage_logs',
            alert_file=alert_file,
            logger=logger,
        )

        if not (phase14_out_dir / 'phase14_summary.json').exists():
            raise RuntimeError(f'phase14 summary missing: {phase14_out_dir / "phase14_summary.json"}')

        status['stage_phase14'] = 'SUCCESS'
        status['status'] = 'SUCCESS'
        status['finished_at_utc'] = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        status['phase13_run_dir'] = str(phase13_run_dir)
        status['phase14_run_dir'] = str(phase14_out_dir)
        status_path.write_text(json.dumps(status, indent=2))

        emit_alert(
            alert_file,
            level='INFO',
            message='phase14 executor success',
            payload={'phase13_run_dir': str(phase13_run_dir), 'phase14_run_dir': str(phase14_out_dir)},
        )

        print('SUCCESS:', status_path)

    except Exception as e:
        status['status'] = 'FAILED'
        status['error'] = f'{type(e).__name__}: {e}'
        status['finished_at_utc'] = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        status_path.write_text(json.dumps(status, indent=2))

        emit_alert(
            alert_file,
            level='ERROR',
            message='phase14 executor failed',
            payload={'error': status['error']},
        )

        raise


if __name__ == '__main__':
    main()
