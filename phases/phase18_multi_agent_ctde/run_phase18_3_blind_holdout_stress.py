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
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class BlindConfig:
    phase16_4_run_dir: str
    out_dir: str
    runner_script: str

    # Split controls
    val_windows: int = 128
    holdout_ratio: float = 0.15
    holdout_windows: int = 0
    min_train_floor: int = 1200

    # Runtime
    seed: int = 42
    device: str = 'auto'

    # Frozen model/training config
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
    loss_w_exposure: float = 0.12
    loss_w_l1: float = 0.002
    loss_w_turnover: float = 0.06
    loss_w_bc_anchor: float = 0.18
    loss_w_value: float = 0.15
    loss_w_value_local: float = 0.08
    loss_w_gate: float = 0.10
    loss_w_teacher_excess: float = 0.25
    loss_w_agent_excess: float = 0.20

    temporal_weight_min: float = 0.6
    temporal_weight_max: float = 1.6

    cost_per_turnover: float = 5e-5
    turnover_smooth_alpha: float = 0.7

    shield_dd_limit: float = -0.08
    shield_risk_mult: float = 0.55

    # Ensemble / robust selector
    ensemble_seeds: str = '42,43'
    teacher_mix_eval: float = 0.80
    robust_mode: int = 1
    robust_cost_multipliers: str = '2,3'
    robust_teacher_mix_candidates: str = '0.00,0.25,0.50,0.75,1.00'
    robust_risk_scales: str = '0.70,0.85,1.00'
    robust_alpha_candidates: str = '0.55,0.70,0.85'
    robust_score_turn_penalty: float = 0.30
    robust_score_dd_penalty: float = 0.20

    # Stress post-eval
    stress_latency_per_trade: float = 1e-5
    stress_extra_cost_per_turnover: float = 5e-5


REQUIRED_OUTPUTS = [
    'phase18_2_summary.json',
    'phase18_2_pooled_metrics.csv',
    'phase18_2_fold_metrics.csv',
    'phase18_2_gate_decisions.csv',
    'phase18_2_test_steps.parquet',
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


def to_num(v):
    if isinstance(v, (np.integer, np.floating)):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    return v


def pick_holdout_windows(n: int, holdout_windows: int, holdout_ratio: float, val_windows: int, min_train_floor: int) -> int:
    if holdout_windows > 0:
        h = int(holdout_windows)
    else:
        h = max(128, int(round(n * float(holdout_ratio))))

    max_h = n - int(val_windows) - int(min_train_floor)
    if max_h <= 0:
        raise RuntimeError(f'Not enough windows for blind split: n={n}, val={val_windows}, min_train_floor={min_train_floor}')

    h = min(h, max_h)
    if h < 128:
        raise RuntimeError(f'Holdout too small after constraints: {h}')
    return int(h)


def build_runner_cmd(cfg: BlindConfig, train_windows: int, holdout_windows: int, run_dir: Path) -> List[str]:
    return [
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
        '--loss_w_exposure', str(float(cfg.loss_w_exposure)),
        '--loss_w_l1', str(float(cfg.loss_w_l1)),
        '--loss_w_turnover', str(float(cfg.loss_w_turnover)),
        '--loss_w_bc_anchor', str(float(cfg.loss_w_bc_anchor)),
        '--loss_w_value', str(float(cfg.loss_w_value)),
        '--loss_w_value_local', str(float(cfg.loss_w_value_local)),
        '--loss_w_gate', str(float(cfg.loss_w_gate)),
        '--loss_w_teacher_excess', str(float(cfg.loss_w_teacher_excess)),
        '--loss_w_agent_excess', str(float(cfg.loss_w_agent_excess)),
        '--temporal_weight_min', str(float(cfg.temporal_weight_min)),
        '--temporal_weight_max', str(float(cfg.temporal_weight_max)),
        '--cost_per_turnover', str(float(cfg.cost_per_turnover)),
        '--turnover_smooth_alpha', str(float(cfg.turnover_smooth_alpha)),
        '--shield_dd_limit', str(float(cfg.shield_dd_limit)),
        '--shield_risk_mult', str(float(cfg.shield_risk_mult)),
        '--ensemble_seeds', str(cfg.ensemble_seeds),
        '--teacher_mix_eval', str(float(cfg.teacher_mix_eval)),
        '--robust_mode', str(int(cfg.robust_mode)),
        '--robust_cost_multipliers', str(cfg.robust_cost_multipliers),
        '--robust_teacher_mix_candidates', str(cfg.robust_teacher_mix_candidates),
        '--robust_risk_scales', str(cfg.robust_risk_scales),
        '--robust_alpha_candidates', str(cfg.robust_alpha_candidates),
        '--robust_score_turn_penalty', str(float(cfg.robust_score_turn_penalty)),
        '--robust_score_dd_penalty', str(float(cfg.robust_score_dd_penalty)),
        '--seed', str(int(cfg.seed)),
        '--device', str(cfg.device),
    ]


def compute_drawdown(equity_curve: np.ndarray) -> float:
    if len(equity_curve) == 0:
        return float('nan')
    peak = np.maximum.accumulate(equity_curve)
    dd = equity_curve / np.maximum(peak, 1e-12) - 1.0
    return float(np.min(dd))


def aggregate_returns(df: pd.DataFrame, ret_col: str) -> Dict[str, float]:
    if len(df) == 0:
        return {
            'n_steps': 0,
            'mean_net_ret': float('nan'),
            'std_net_ret': float('nan'),
            'sharpe_ann': float('nan'),
            'cum_net_return': float('nan'),
            'max_drawdown': float('nan'),
            'trade_rate': float('nan'),
            'avg_turnover': float('nan'),
            'avg_gross_lev': float('nan'),
            'avg_active_assets': float('nan'),
            'shield_rate': float('nan'),
        }

    d = df.sort_values('timestamp_end').reset_index(drop=True).copy()
    d['equity_tmp'] = (1.0 + d[ret_col].astype(float)).cumprod()

    r = d[ret_col].to_numpy(dtype=float)
    mu = float(np.mean(r))
    sd = float(np.std(r))
    sharpe = float(mu / sd * np.sqrt(24 * 252)) if sd > 1e-12 else float('nan')

    return {
        'n_steps': int(len(d)),
        'mean_net_ret': mu,
        'std_net_ret': sd,
        'sharpe_ann': sharpe,
        'cum_net_return': float(d['equity_tmp'].iloc[-1] - 1.0),
        'max_drawdown': compute_drawdown(d['equity_tmp'].to_numpy(dtype=float)),
        'trade_rate': float((d['turnover'] > 1e-6).mean()),
        'avg_turnover': float(d['turnover'].mean()),
        'avg_gross_lev': float(d['gross_lev'].mean()),
        'avg_active_assets': float(d['active_assets'].mean()),
        'shield_rate': float(d['shield_on'].mean()) if 'shield_on' in d.columns else 0.0,
    }


def stress_eval(steps: pd.DataFrame, policy: str, cfg: BlindConfig) -> pd.DataFrame:
    p_all = steps[steps['policy'] == policy].copy()
    # Use a single canonical trajectory to avoid triple-counting rows from cost_mult x1/x2/x3.
    p = p_all[np.isclose(p_all['cost_mult'].astype(float), 1.0)].copy()
    if len(p) == 0:
        p = p_all.copy()
    if len(p) == 0:
        return pd.DataFrame()

    scenarios = [
        ('x2_base', 2.0, 0.0, 0.0),
        ('x3_base', 3.0, 0.0, 0.0),
        ('x4_cost', 4.0, 0.0, 0.0),
        ('x5_cost', 5.0, 0.0, 0.0),
        ('x3_slip_latency', 3.0, float(cfg.stress_extra_cost_per_turnover), float(cfg.stress_latency_per_trade)),
    ]

    rows = []
    for name, cost_mult, extra_cost_turn, latency_trade in scenarios:
        d = p.copy()
        turnover = d['turnover'].astype(float)
        gross_ret = d['gross_ret'].astype(float)
        trade_flag = (turnover > 1e-6).astype(float)

        cost = (float(cfg.cost_per_turnover) * float(cost_mult) + float(extra_cost_turn)) * turnover
        latency_cost = float(latency_trade) * trade_flag
        d['net_stress'] = gross_ret - cost - latency_cost

        m = aggregate_returns(d, 'net_stress')
        rows.append({
            'policy': policy,
            'scenario': name,
            'cost_mult': float(cost_mult),
            'extra_cost_per_turnover': float(extra_cost_turn),
            'latency_per_trade': float(latency_trade),
            **m,
        })

    return pd.DataFrame(rows)


def pooled_row(pooled: pd.DataFrame, policy: str, cost_mult: float) -> Dict[str, float]:
    r = pooled[(pooled['policy'] == policy) & (np.isclose(pooled['cost_mult'], float(cost_mult)))]
    if len(r) == 0:
        return {}
    d = r.iloc[0].to_dict()
    return {k: to_num(v) for k, v in d.items()}


def evaluate_go_no_go(summary_18_2: Dict, pooled: pd.DataFrame, stress: pd.DataFrame) -> Tuple[str, Dict[str, bool], Dict[str, float]]:
    n_folds = int(summary_18_2.get('n_folds', -1))

    x2 = pooled_row(pooled, 'model_ctde_robust', 2.0)
    x3 = pooled_row(pooled, 'model_ctde_robust', 3.0)

    s_x4 = stress[(stress['policy'] == 'model_ctde_robust') & (stress['scenario'] == 'x4_cost')]
    s_x5 = stress[(stress['policy'] == 'model_ctde_robust') & (stress['scenario'] == 'x5_cost')]
    s_sl = stress[(stress['policy'] == 'model_ctde_robust') & (stress['scenario'] == 'x3_slip_latency')]

    m_x4 = s_x4.iloc[0].to_dict() if len(s_x4) else {}
    m_x5 = s_x5.iloc[0].to_dict() if len(s_x5) else {}
    m_sl = s_sl.iloc[0].to_dict() if len(s_sl) else {}

    checks = {
        'single_fold_blind': (n_folds == 1),
        'x2_sharpe_ge_1_0': float(x2.get('sharpe_ann', -1e9)) >= 1.0,
        'x2_cumret_gt_0': float(x2.get('cum_net_return', -1e9)) > 0.0,
        'x2_mdd_ge_-0_08': float(x2.get('max_drawdown', -1.0)) >= -0.08,
        'x2_turnover_le_0_45': float(x2.get('avg_turnover', 1e9)) <= 0.45,
        'x3_sharpe_ge_0_8': float(x3.get('sharpe_ann', -1e9)) >= 0.8,
        'x4_sharpe_ge_0_4': float(m_x4.get('sharpe_ann', -1e9)) >= 0.4,
        'x5_cumret_gt_-0_02': float(m_x5.get('cum_net_return', -1e9)) > -0.02,
        'x3_slip_latency_sharpe_ge_0_4': float(m_sl.get('sharpe_ann', -1e9)) >= 0.4,
    }

    score = {
        'x2_sharpe': float(x2.get('sharpe_ann', float('nan'))),
        'x2_cumret': float(x2.get('cum_net_return', float('nan'))),
        'x2_mdd': float(x2.get('max_drawdown', float('nan'))),
        'x2_turnover': float(x2.get('avg_turnover', float('nan'))),
        'x3_sharpe': float(x3.get('sharpe_ann', float('nan'))),
        'x4_sharpe': float(m_x4.get('sharpe_ann', float('nan'))),
        'x5_cumret': float(m_x5.get('cum_net_return', float('nan'))),
        'x3_slip_latency_sharpe': float(m_sl.get('sharpe_ann', float('nan'))),
    }

    decision = 'GO_PAPER' if all(checks.values()) else 'NO_GO'
    return decision, checks, score


def main() -> int:
    ap = argparse.ArgumentParser(description='Phase18.3 - Blind holdout final + stress + go/no-go (frozen phase18.2).')
    ap.add_argument('--phase16_4_run_dir', required=True, type=str)
    ap.add_argument('--out_dir', default='', type=str)
    ap.add_argument('--runner_script', default='', type=str)

    ap.add_argument('--val_windows', type=int, default=128)
    ap.add_argument('--holdout_ratio', type=float, default=0.15)
    ap.add_argument('--holdout_windows', type=int, default=0)
    ap.add_argument('--min_train_floor', type=int, default=1200)

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

    ap.add_argument('--max_gross_leverage', type=float, default=2.0)
    ap.add_argument('--target_gross_leverage', type=float, default=0.9)
    ap.add_argument('--min_target_mult', type=float, default=0.35)
    ap.add_argument('--max_target_mult', type=float, default=1.45)
    ap.add_argument('--weight_temperature', type=float, default=1.0)

    ap.add_argument('--loss_w_ret', type=float, default=1.0)
    ap.add_argument('--loss_w_risk', type=float, default=0.30)
    ap.add_argument('--loss_w_excess', type=float, default=1.0)
    ap.add_argument('--loss_w_exposure', type=float, default=0.12)
    ap.add_argument('--loss_w_l1', type=float, default=0.002)
    ap.add_argument('--loss_w_turnover', type=float, default=0.06)
    ap.add_argument('--loss_w_bc_anchor', type=float, default=0.18)
    ap.add_argument('--loss_w_value', type=float, default=0.15)
    ap.add_argument('--loss_w_value_local', type=float, default=0.08)
    ap.add_argument('--loss_w_gate', type=float, default=0.10)
    ap.add_argument('--loss_w_teacher_excess', type=float, default=0.25)
    ap.add_argument('--loss_w_agent_excess', type=float, default=0.20)

    ap.add_argument('--temporal_weight_min', type=float, default=0.6)
    ap.add_argument('--temporal_weight_max', type=float, default=1.6)

    ap.add_argument('--cost_per_turnover', type=float, default=5e-5)
    ap.add_argument('--turnover_smooth_alpha', type=float, default=0.7)

    ap.add_argument('--shield_dd_limit', type=float, default=-0.08)
    ap.add_argument('--shield_risk_mult', type=float, default=0.55)

    ap.add_argument('--ensemble_seeds', type=str, default='42,43')
    ap.add_argument('--teacher_mix_eval', type=float, default=0.80)
    ap.add_argument('--robust_mode', type=int, default=1)
    ap.add_argument('--robust_cost_multipliers', type=str, default='2,3')
    ap.add_argument('--robust_teacher_mix_candidates', type=str, default='0.00,0.25,0.50,0.75,1.00')
    ap.add_argument('--robust_risk_scales', type=str, default='0.70,0.85,1.00')
    ap.add_argument('--robust_alpha_candidates', type=str, default='0.55,0.70,0.85')
    ap.add_argument('--robust_score_turn_penalty', type=float, default=0.30)
    ap.add_argument('--robust_score_dd_penalty', type=float, default=0.20)

    ap.add_argument('--stress_latency_per_trade', type=float, default=1e-5)
    ap.add_argument('--stress_extra_cost_per_turnover', type=float, default=5e-5)

    args = ap.parse_args()

    p16 = Path(args.phase16_4_run_dir).resolve()
    if not p16.exists():
        raise FileNotFoundError(p16)

    script_dir = Path(__file__).resolve().parent
    runner = Path(args.runner_script).resolve() if args.runner_script else (script_dir / 'run_phase18_2_ctde_robust.py')
    if not runner.exists():
        raise FileNotFoundError(runner)

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (script_dir / 'output_phase18_3_blind_holdout_v1')
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
        min_train_floor=int(args.min_train_floor),
    )
    train_w = int(n - int(args.val_windows) - holdout_w)
    if train_w < int(args.min_train_floor):
        raise RuntimeError(f'train_windows too small: {train_w} < min_train_floor={args.min_train_floor}')

    cfg = BlindConfig(
        phase16_4_run_dir=str(p16),
        out_dir=str(out_dir),
        runner_script=str(runner),
        val_windows=int(args.val_windows),
        holdout_ratio=float(args.holdout_ratio),
        holdout_windows=int(args.holdout_windows),
        min_train_floor=int(args.min_train_floor),
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
        max_gross_leverage=float(args.max_gross_leverage),
        target_gross_leverage=float(args.target_gross_leverage),
        min_target_mult=float(args.min_target_mult),
        max_target_mult=float(args.max_target_mult),
        weight_temperature=float(args.weight_temperature),
        loss_w_ret=float(args.loss_w_ret),
        loss_w_risk=float(args.loss_w_risk),
        loss_w_excess=float(args.loss_w_excess),
        loss_w_exposure=float(args.loss_w_exposure),
        loss_w_l1=float(args.loss_w_l1),
        loss_w_turnover=float(args.loss_w_turnover),
        loss_w_bc_anchor=float(args.loss_w_bc_anchor),
        loss_w_value=float(args.loss_w_value),
        loss_w_value_local=float(args.loss_w_value_local),
        loss_w_gate=float(args.loss_w_gate),
        loss_w_teacher_excess=float(args.loss_w_teacher_excess),
        loss_w_agent_excess=float(args.loss_w_agent_excess),
        temporal_weight_min=float(args.temporal_weight_min),
        temporal_weight_max=float(args.temporal_weight_max),
        cost_per_turnover=float(args.cost_per_turnover),
        turnover_smooth_alpha=float(args.turnover_smooth_alpha),
        shield_dd_limit=float(args.shield_dd_limit),
        shield_risk_mult=float(args.shield_risk_mult),
        ensemble_seeds=str(args.ensemble_seeds),
        teacher_mix_eval=float(args.teacher_mix_eval),
        robust_mode=int(args.robust_mode),
        robust_cost_multipliers=str(args.robust_cost_multipliers),
        robust_teacher_mix_candidates=str(args.robust_teacher_mix_candidates),
        robust_risk_scales=str(args.robust_risk_scales),
        robust_alpha_candidates=str(args.robust_alpha_candidates),
        robust_score_turn_penalty=float(args.robust_score_turn_penalty),
        robust_score_dd_penalty=float(args.robust_score_dd_penalty),
        stress_latency_per_trade=float(args.stress_latency_per_trade),
        stress_extra_cost_per_turnover=float(args.stress_extra_cost_per_turnover),
    )

    cmd = build_runner_cmd(cfg=cfg, train_windows=train_w, holdout_windows=holdout_w, run_dir=run_dir)
    cmd_str = ' '.join(shlex.quote(x) for x in cmd)
    (out_dir / 'phase18_3_blind_command.sh').write_text(cmd_str + '\n', encoding='utf-8')

    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        manifest = {
            'status': 'FAILED',
            'return_code': int(proc.returncode),
            'command': cmd,
        }
        (out_dir / 'phase18_3_blind_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        return int(proc.returncode)

    missing = [x for x in REQUIRED_OUTPUTS if not (run_dir / x).exists()]
    if missing:
        raise RuntimeError(f'Missing runner outputs: {missing}')

    s18 = json.loads((run_dir / 'phase18_2_summary.json').read_text(encoding='utf-8'))
    if int(s18.get('n_folds', -1)) != 1:
        raise RuntimeError(f'Blind run must have exactly 1 fold, got {s18.get("n_folds")}')

    pooled = pd.read_csv(run_dir / 'phase18_2_pooled_metrics.csv')
    steps = pd.read_parquet(run_dir / 'phase18_2_test_steps.parquet')

    stress_all = []
    for pol in ['model_ctde_robust', 'model_ctde', 'teacher_lag1']:
        d = stress_eval(steps=steps, policy=pol, cfg=cfg)
        if len(d):
            stress_all.append(d)
    stress_df = pd.concat(stress_all, axis=0, ignore_index=True) if stress_all else pd.DataFrame()
    stress_path = out_dir / 'phase18_3_stress_metrics.csv'
    stress_df.to_csv(stress_path, index=False)

    decision, checks, score = evaluate_go_no_go(summary_18_2=s18, pooled=pooled, stress=stress_df)

    holdout_start_idx = train_w + int(args.val_windows)
    holdout_end_idx = n - 1

    summary = {
        'status': 'SUCCESS',
        'protocol': 'phase18.3_blind_holdout_single_fold_frozen_phase18.2_plus_stress',
        'decision': decision,
        'checks': checks,
        'score_snapshot': score,
        'n_total_windows': int(n),
        'train_windows': int(train_w),
        'val_windows': int(args.val_windows),
        'holdout_windows': int(holdout_w),
        'holdout_start_ts': str(ts.iloc[holdout_start_idx]),
        'holdout_end_ts': str(ts.iloc[holdout_end_idx]),
        'policy_metrics_x1': {
            'model_ctde_robust': pooled_row(pooled, 'model_ctde_robust', 1.0),
            'model_ctde': pooled_row(pooled, 'model_ctde', 1.0),
            'teacher_lag1': pooled_row(pooled, 'teacher_lag1', 1.0),
        },
        'policy_metrics_x2': {
            'model_ctde_robust': pooled_row(pooled, 'model_ctde_robust', 2.0),
            'model_ctde': pooled_row(pooled, 'model_ctde', 2.0),
            'teacher_lag1': pooled_row(pooled, 'teacher_lag1', 2.0),
        },
        'policy_metrics_x3': {
            'model_ctde_robust': pooled_row(pooled, 'model_ctde_robust', 3.0),
            'model_ctde': pooled_row(pooled, 'model_ctde', 3.0),
            'teacher_lag1': pooled_row(pooled, 'teacher_lag1', 3.0),
        },
        'artifacts': {
            'runner_out_dir': str(run_dir),
            'runner_summary': str(run_dir / 'phase18_2_summary.json'),
            'runner_pooled_metrics': str(run_dir / 'phase18_2_pooled_metrics.csv'),
            'runner_fold_metrics': str(run_dir / 'phase18_2_fold_metrics.csv'),
            'runner_gate_decisions': str(run_dir / 'phase18_2_gate_decisions.csv'),
            'runner_test_steps': str(run_dir / 'phase18_2_test_steps.parquet'),
            'stress_metrics_csv': str(stress_path),
            'command_file': str(out_dir / 'phase18_3_blind_command.sh'),
        },
        'proposed_live_limits': {
            'max_daily_drawdown': 0.03,
            'max_gross_leverage': float(cfg.max_gross_leverage),
            'max_turnover_hour': 0.70,
            'kill_switch_on_3_consecutive_days_loss': True,
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
            'phase18_3_runner_sha256': sha256_file(Path(__file__).resolve()),
            'phase18_2_runner_sha256': sha256_file(runner),
            'phase16_4_npz_sha256': sha256_file(npz),
        },
        'command': cmd,
        'summary_file': str(out_dir / 'phase18_3_blind_summary.json'),
    }

    (out_dir / 'phase18_3_blind_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (out_dir / 'phase18_3_blind_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    print(json.dumps({'status': 'SUCCESS', 'out_dir': str(out_dir), 'summary': str(out_dir / 'phase18_3_blind_summary.json')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
