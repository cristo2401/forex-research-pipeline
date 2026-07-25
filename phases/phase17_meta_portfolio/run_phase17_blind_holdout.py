#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class BlindConfig:
    phase16_4_run_dir: str
    out_dir: str
    runner_script: str
    val_windows: int = 128
    holdout_ratio: float = 0.15
    holdout_windows: int = 0
    ensemble_seeds: str = '42,43'
    seed: int = 42
    device: str = 'auto'

    # Frozen from phase17.6 v2 baseline
    d_model: int = 192
    nhead: int = 6
    num_layers: int = 3
    ff_dim: int = 384
    dropout: float = 0.1
    batch_size: int = 128
    imitation_epochs: int = 8
    rl_epochs: int = 18
    patience: int = 6
    lr: float = 3e-4
    weight_decay: float = 1e-5

    max_gross_leverage: float = 2.0
    target_gross_leverage: float = 0.9
    min_target_mult: float = 0.35
    max_target_mult: float = 1.45
    weight_temperature: float = 1.0

    loss_w_ret: float = 1.0
    loss_w_risk: float = 0.30
    loss_w_excess: float = 1.0
    loss_w_teacher_excess: float = 0.25
    loss_w_exposure: float = 0.12
    loss_w_l1: float = 0.002
    loss_w_turnover: float = 0.06
    loss_w_bc_anchor: float = 0.18
    loss_w_value: float = 0.15
    loss_w_gate: float = 0.10

    temporal_weight_min: float = 0.6
    temporal_weight_max: float = 1.6

    cost_per_turnover: float = 5e-5
    turnover_smooth_alpha: float = 0.7

    shield_dd_limit: float = -0.08
    shield_risk_mult: float = 0.55

    gate_min_excess_sharpe: float = 0.05
    gate_min_excess_cum: float = 0.0
    gate_turnover_ratio_max: float = 1.20
    teacher_mix_eval: float = 0.80


REQUIRED_OUTPUTS = [
    'phase17_6_summary.json',
    'phase17_6_pooled_metrics.csv',
    'phase17_6_fold_metrics.csv',
    'phase17_6_gate_decisions.csv',
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def build_runner_cmd(cfg: BlindConfig, train_windows: int, holdout_windows: int, run_dir: Path) -> List[str]:
    cmd = [
        sys.executable,
        cfg.runner_script,
        '--phase16_4_run_dir', cfg.phase16_4_run_dir,
        '--out_dir', str(run_dir),
        '--min_train_windows', str(int(train_windows)),
        '--val_windows', str(int(cfg.val_windows)),
        '--test_windows', str(int(holdout_windows)),
        '--step_windows', str(int(holdout_windows)),
        '--d_model', str(int(cfg.d_model)),
        '--nhead', str(int(cfg.nhead)),
        '--num_layers', str(int(cfg.num_layers)),
        '--ff_dim', str(int(cfg.ff_dim)),
        '--dropout', str(float(cfg.dropout)),
        '--batch_size', str(int(cfg.batch_size)),
        '--imitation_epochs', str(int(cfg.imitation_epochs)),
        '--rl_epochs', str(int(cfg.rl_epochs)),
        '--patience', str(int(cfg.patience)),
        '--lr', str(float(cfg.lr)),
        '--weight_decay', str(float(cfg.weight_decay)),
        '--max_gross_leverage', str(float(cfg.max_gross_leverage)),
        '--target_gross_leverage', str(float(cfg.target_gross_leverage)),
        '--min_target_mult', str(float(cfg.min_target_mult)),
        '--max_target_mult', str(float(cfg.max_target_mult)),
        '--weight_temperature', str(float(cfg.weight_temperature)),
        '--loss_w_ret', str(float(cfg.loss_w_ret)),
        '--loss_w_risk', str(float(cfg.loss_w_risk)),
        '--loss_w_excess', str(float(cfg.loss_w_excess)),
        '--loss_w_teacher_excess', str(float(cfg.loss_w_teacher_excess)),
        '--loss_w_exposure', str(float(cfg.loss_w_exposure)),
        '--loss_w_l1', str(float(cfg.loss_w_l1)),
        '--loss_w_turnover', str(float(cfg.loss_w_turnover)),
        '--loss_w_bc_anchor', str(float(cfg.loss_w_bc_anchor)),
        '--loss_w_value', str(float(cfg.loss_w_value)),
        '--loss_w_gate', str(float(cfg.loss_w_gate)),
        '--temporal_weight_min', str(float(cfg.temporal_weight_min)),
        '--temporal_weight_max', str(float(cfg.temporal_weight_max)),
        '--cost_per_turnover', str(float(cfg.cost_per_turnover)),
        '--turnover_smooth_alpha', str(float(cfg.turnover_smooth_alpha)),
        '--shield_dd_limit', str(float(cfg.shield_dd_limit)),
        '--shield_risk_mult', str(float(cfg.shield_risk_mult)),
        '--ensemble_seeds', str(cfg.ensemble_seeds),
        '--gate_min_excess_sharpe', str(float(cfg.gate_min_excess_sharpe)),
        '--gate_min_excess_cum', str(float(cfg.gate_min_excess_cum)),
        '--gate_turnover_ratio_max', str(float(cfg.gate_turnover_ratio_max)),
        '--teacher_mix_eval', str(float(cfg.teacher_mix_eval)),
        '--seed', str(int(cfg.seed)),
        '--device', str(cfg.device),
    ]
    return cmd


def pick_holdout_windows(n: int, holdout_windows: int, holdout_ratio: float, val_windows: int) -> int:
    if holdout_windows > 0:
        h = int(holdout_windows)
    else:
        h = max(128, int(round(n * float(holdout_ratio))))
    # keep enough train history
    max_h = n - int(val_windows) - 1000
    if max_h <= 0:
        raise RuntimeError(f'Not enough windows for blind split: n={n}, val={val_windows}')
    h = min(h, max_h)
    if h < 128:
        raise RuntimeError(f'Holdout too small after constraints: {h}')
    return int(h)


def main() -> int:
    ap = argparse.ArgumentParser(description='Phase17 blind holdout final (single-fold, frozen config, no post-holdout tuning).')
    ap.add_argument('--phase16_4_run_dir', required=True, type=str)
    ap.add_argument('--out_dir', default='', type=str)
    ap.add_argument('--runner_script', default='', type=str)
    ap.add_argument('--val_windows', type=int, default=128)
    ap.add_argument('--holdout_ratio', type=float, default=0.15)
    ap.add_argument('--holdout_windows', type=int, default=0)
    ap.add_argument('--ensemble_seeds', type=str, default='42,43')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    ap.add_argument('--d_model', type=int, default=192)
    ap.add_argument('--nhead', type=int, default=6)
    ap.add_argument('--num_layers', type=int, default=3)
    ap.add_argument('--ff_dim', type=int, default=384)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--imitation_epochs', type=int, default=8)
    ap.add_argument('--rl_epochs', type=int, default=18)
    ap.add_argument('--patience', type=int, default=6)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--weight_decay', type=float, default=1e-5)
    ap.add_argument('--loss_w_turnover', type=float, default=0.06)
    ap.add_argument('--loss_w_teacher_excess', type=float, default=0.25)
    ap.add_argument('--teacher_mix_eval', type=float, default=0.80)
    ap.add_argument('--gate_turnover_ratio_max', type=float, default=1.20)
    args = ap.parse_args()

    p16 = Path(args.phase16_4_run_dir).resolve()
    if not p16.exists():
        raise FileNotFoundError(p16)

    runner = Path(args.runner_script).resolve() if args.runner_script else (Path(__file__).resolve().parent / 'run_phase17_6_distill_turnover.py')
    if not runner.exists():
        raise FileNotFoundError(runner)

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (Path(__file__).resolve().parent / 'output_phase17_blind_holdout_v1')
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = out_dir / 'blind_run'
    run_dir.mkdir(parents=True, exist_ok=True)

    npz = p16 / 'phase16_4_windows.npz'
    meta_pq = p16 / 'phase16_4_windows_meta.parquet'
    if not npz.exists() or not meta_pq.exists():
        raise FileNotFoundError('Missing phase16.4 windows artifacts')

    data = np.load(npz)
    n = int(data['x'].shape[0])
    meta = pd.read_parquet(meta_pq)
    ts = pd.to_datetime(meta['timestamp_end'], utc=True)

    holdout_w = pick_holdout_windows(
        n=n,
        holdout_windows=int(args.holdout_windows),
        holdout_ratio=float(args.holdout_ratio),
        val_windows=int(args.val_windows),
    )
    train_w = int(n - int(args.val_windows) - holdout_w)
    if train_w < 1000:
        raise RuntimeError(f'train_windows too small: {train_w}')

    cfg = BlindConfig(
        phase16_4_run_dir=str(p16),
        out_dir=str(out_dir),
        runner_script=str(runner),
        val_windows=int(args.val_windows),
        holdout_ratio=float(args.holdout_ratio),
        holdout_windows=int(args.holdout_windows),
        ensemble_seeds=str(args.ensemble_seeds),
        seed=int(args.seed),
        device=str(args.device),
        d_model=int(args.d_model),
        nhead=int(args.nhead),
        num_layers=int(args.num_layers),
        ff_dim=int(args.ff_dim),
        dropout=float(args.dropout),
        batch_size=int(args.batch_size),
        imitation_epochs=int(args.imitation_epochs),
        rl_epochs=int(args.rl_epochs),
        patience=int(args.patience),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        loss_w_turnover=float(args.loss_w_turnover),
        loss_w_teacher_excess=float(args.loss_w_teacher_excess),
        teacher_mix_eval=float(args.teacher_mix_eval),
        gate_turnover_ratio_max=float(args.gate_turnover_ratio_max),
    )

    cmd = build_runner_cmd(cfg=cfg, train_windows=train_w, holdout_windows=holdout_w, run_dir=run_dir)
    cmd_str = ' '.join(shlex.quote(x) for x in cmd)

    (out_dir / 'phase17_blind_command.sh').write_text(cmd_str + '\n', encoding='utf-8')

    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        manifest = {
            'status': 'FAILED',
            'return_code': int(proc.returncode),
            'command': cmd,
        }
        (out_dir / 'phase17_blind_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        return int(proc.returncode)

    missing = [x for x in REQUIRED_OUTPUTS if not (run_dir / x).exists()]
    if missing:
        raise RuntimeError(f'Missing runner outputs: {missing}')

    s = json.loads((run_dir / 'phase17_6_summary.json').read_text(encoding='utf-8'))
    if int(s.get('n_folds', -1)) != 1:
        raise RuntimeError(f'Blind run must have exactly 1 fold, got {s.get("n_folds")}')

    pooled = pd.read_csv(run_dir / 'phase17_6_pooled_metrics.csv')

    def row(policy: str, mult: float) -> Dict[str, float]:
        r = pooled[(pooled['policy'] == policy) & (np.isclose(pooled['cost_mult'], float(mult)))]
        if len(r) == 0:
            return {}
        d = r.iloc[0].to_dict()
        return {k: float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v for k, v in d.items()}

    holdout_start_idx = train_w + int(args.val_windows)
    holdout_end_idx = n - 1

    summary = {
        'status': 'SUCCESS',
        'protocol': 'blind_final_holdout_single_fold_frozen_config',
        'n_total_windows': int(n),
        'train_windows': int(train_w),
        'val_windows': int(args.val_windows),
        'holdout_windows': int(holdout_w),
        'holdout_start_ts': str(ts.iloc[holdout_start_idx]),
        'holdout_end_ts': str(ts.iloc[holdout_end_idx]),
        'policies_x1': {
            'model_distill_gated': row('model_distill_gated', 1.0),
            'teacher_lag1': row('teacher_lag1', 1.0),
            'equal_weight_target': row('equal_weight_target', 1.0),
            'equal_weight_max': row('equal_weight_max', 1.0),
        },
        'policies_x2': {
            'model_distill_gated': row('model_distill_gated', 2.0),
            'teacher_lag1': row('teacher_lag1', 2.0),
            'equal_weight_target': row('equal_weight_target', 2.0),
            'equal_weight_max': row('equal_weight_max', 2.0),
        },
        'policies_x3': {
            'model_distill_gated': row('model_distill_gated', 3.0),
            'teacher_lag1': row('teacher_lag1', 3.0),
            'equal_weight_target': row('equal_weight_target', 3.0),
            'equal_weight_max': row('equal_weight_max', 3.0),
        },
        'artifacts': {
            'runner_out_dir': str(run_dir),
            'runner_summary': str(run_dir / 'phase17_6_summary.json'),
            'runner_pooled_metrics': str(run_dir / 'phase17_6_pooled_metrics.csv'),
            'runner_gate_decisions': str(run_dir / 'phase17_6_gate_decisions.csv'),
            'command_file': str(out_dir / 'phase17_blind_command.sh'),
        },
    }

    manifest = {
        'status': 'SUCCESS',
        'config': asdict(cfg),
        'derived_windows': {
            'train_windows': int(train_w),
            'val_windows': int(args.val_windows),
            'holdout_windows': int(holdout_w),
        },
        'script_hashes': {
            'blind_runner_sha256': sha256_file(Path(__file__).resolve()),
            'phase17_6_runner_sha256': sha256_file(runner),
            'phase16_4_npz_sha256': sha256_file(npz),
        },
        'command': cmd,
        'summary_file': str(out_dir / 'phase17_blind_holdout_summary.json'),
    }

    (out_dir / 'phase17_blind_holdout_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (out_dir / 'phase17_blind_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    print(json.dumps({'status': 'SUCCESS', 'out_dir': str(out_dir), 'summary': str(out_dir / 'phase17_blind_holdout_summary.json')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
