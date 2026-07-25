#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class Config:
    phase16_4_run_dir: str = './phases/phase16_state_embeddings/output_phase16_4_portfolio_state_full12_tuned_v3'
    phase18_2_runner: str = './phases/phase18_multi_agent_ctde/run_phase18_2_ctde_robust.py'
    out_dir: str = './phases/phase18_5_dynamic_leverage/output_phase18_5_dynamic_ctrader_v1'
    input_steps_parquet: str = ''

    # Phase18.2 retrain/infer setup
    run_phase18_2: int = 1
    skip_if_exists: int = 1
    min_train_windows: int = 1200
    val_windows: int = 128
    test_windows: int = 256
    step_windows: int = 256
    max_gross_leverage: float = 3.5
    target_gross_leverage: float = 1.8
    min_target_mult: float = 0.2
    max_target_mult: float = 2.2
    loss_w_turnover: float = 0.08
    cost_per_turnover: float = 5e-05
    turnover_smooth_alpha: float = 0.70
    exec_rebalance_deadband: float = 0.0
    exec_rebalance_cooldown_bars: int = 0
    exec_min_turnover: float = 0.0
    gate_turnover_ratio_max: float = 1.20
    robust_score_turn_penalty: float = 0.30
    robust_score_dd_penalty: float = 0.20
    robust_score_trade_penalty: float = 0.0
    robust_score_rebalance_penalty: float = 0.0
    robust_alpha_candidates: str = '0.55,0.70,0.85'

    # cTrader-like execution layer
    policy: str = 'model_ctde_robust'
    cost_mult: float = 1.0
    account_equity_usd: float = 3000.0
    account_leverage_cap: float = 10000.0
    max_margin_utilization: float = 0.35
    leverage_tiers: str = '1000000:500,5000000:200,inf:100'
    extra_cost_per_turnover: float = 2e-05
    slippage_per_turnover: float = 2e-05
    shield_dd_limit: float = -0.08
    shield_risk_mult: float = 0.55
    min_exec_scale: float = 0.05


DOCS_USED = [
    'https://help.ctrader.com/open-api/account-authentication/',
    'https://help.ctrader.com/open-api/id/model-messages/',
    'https://help.ctrader.com/trading-with-ctrader/dynamic-leverage/',
    'https://help.ctrader.com/open-api/symbol-data/',
]


def setup_logger(path: Path) -> logging.Logger:
    lg = logging.getLogger('phase18_5_dynamic_ctrader')
    lg.setLevel(logging.INFO)
    lg.handlers.clear()

    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    fh = logging.FileHandler(path, encoding='utf-8')
    fh.setFormatter(fmt)
    lg.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    lg.addHandler(sh)
    return lg


def parse_tiers(s: str) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for raw in str(s).split(','):
        raw = raw.strip()
        if not raw:
            continue
        if ':' not in raw:
            raise ValueError(f'Invalid tier format: {raw!r}. Use cap:leverage, e.g. 1000000:500')
        cap_s, lev_s = raw.split(':', 1)
        cap_s = cap_s.strip().lower()
        lev = float(lev_s.strip())
        if lev <= 0:
            raise ValueError(f'Leverage must be > 0 in tier: {raw!r}')

        if cap_s in ('inf', 'infinity', 'max'):
            cap = math.inf
        else:
            cap = float(cap_s)
            if cap <= 0:
                raise ValueError(f'Tier cap must be > 0 in tier: {raw!r}')
        out.append((cap, lev))

    if not out:
        raise ValueError('No valid leverage tiers provided')

    out.sort(key=lambda x: (math.inf if math.isinf(x[0]) else x[0]))
    return out


def required_margin_usd(exposure_usd: float, tiers: List[Tuple[float, float]], account_lev_cap: float) -> float:
    exp = float(max(0.0, exposure_usd))
    if exp <= 0:
        return 0.0

    margin = 0.0
    prev_cap = 0.0
    for cap, lev in tiers:
        lev_eff = min(float(lev), float(account_lev_cap))
        if lev_eff <= 0:
            continue

        if math.isinf(cap):
            width = exp
        else:
            width = min(exp, max(0.0, cap - prev_cap))

        if width > 0:
            margin += width / lev_eff
            exp -= width

        if exp <= 1e-12:
            break

        if not math.isinf(cap):
            prev_cap = cap

    if exp > 1e-12:
        lev_eff = min(float(tiers[-1][1]), float(account_lev_cap))
        margin += exp / max(lev_eff, 1e-12)

    return float(margin)


def compute_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return float('nan')
    peak = np.maximum.accumulate(equity)
    dd = equity / np.maximum(peak, 1e-12) - 1.0
    return float(np.min(dd))


def metrics_from_returns(net_ret: np.ndarray, turnover: np.ndarray, gross_lev: np.ndarray, margin_util: np.ndarray, scale: np.ndarray) -> Dict[str, float]:
    if len(net_ret) == 0:
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
            'avg_margin_util': float('nan'),
            'avg_exec_scale': float('nan'),
        }

    eq = np.cumprod(1.0 + net_ret)
    mu = float(np.mean(net_ret))
    sd = float(np.std(net_ret))
    sharpe = float(mu / sd * np.sqrt(24 * 252)) if sd > 1e-12 else float('nan')

    return {
        'n_steps': int(len(net_ret)),
        'mean_net_ret': mu,
        'std_net_ret': sd,
        'sharpe_ann': sharpe,
        'cum_net_return': float(eq[-1] - 1.0),
        'max_drawdown': compute_drawdown(eq),
        'trade_rate': float(np.mean(turnover > 1e-6)),
        'avg_turnover': float(np.mean(turnover)),
        'avg_gross_lev': float(np.mean(gross_lev)),
        'avg_margin_util': float(np.mean(margin_util)),
        'avg_exec_scale': float(np.mean(scale)),
    }


def run_phase18_2_if_needed(cfg: Config, out_dir: Path, logger: logging.Logger) -> Path:
    run_dir = out_dir / 'phase18_2_base_run'
    summary_path = run_dir / 'phase18_2_summary.json'

    if int(cfg.skip_if_exists) == 1 and summary_path.exists():
        logger.info('Reusing existing phase18.2 run: %s', run_dir)
        return run_dir

    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(Path(cfg.phase18_2_runner).resolve()),
        '--phase16_4_run_dir', str(Path(cfg.phase16_4_run_dir).resolve()),
        '--out_dir', str(run_dir.resolve()),
        '--min_train_windows', str(int(cfg.min_train_windows)),
        '--val_windows', str(int(cfg.val_windows)),
        '--test_windows', str(int(cfg.test_windows)),
        '--step_windows', str(int(cfg.step_windows)),
        '--max_gross_leverage', str(float(cfg.max_gross_leverage)),
        '--target_gross_leverage', str(float(cfg.target_gross_leverage)),
        '--min_target_mult', str(float(cfg.min_target_mult)),
        '--max_target_mult', str(float(cfg.max_target_mult)),
        '--loss_w_turnover', str(float(cfg.loss_w_turnover)),
        '--cost_per_turnover', str(float(cfg.cost_per_turnover)),
        '--turnover_smooth_alpha', str(float(cfg.turnover_smooth_alpha)),
        '--exec_rebalance_deadband', str(float(cfg.exec_rebalance_deadband)),
        '--exec_rebalance_cooldown_bars', str(int(cfg.exec_rebalance_cooldown_bars)),
        '--exec_min_turnover', str(float(cfg.exec_min_turnover)),
        '--gate_turnover_ratio_max', str(float(cfg.gate_turnover_ratio_max)),
        '--robust_score_turn_penalty', str(float(cfg.robust_score_turn_penalty)),
        '--robust_score_dd_penalty', str(float(cfg.robust_score_dd_penalty)),
        '--robust_score_trade_penalty', str(float(cfg.robust_score_trade_penalty)),
        '--robust_score_rebalance_penalty', str(float(cfg.robust_score_rebalance_penalty)),
        '--robust_alpha_candidates', str(cfg.robust_alpha_candidates),
        '--robust_mode', '1',
        '--robust_cost_multipliers', '2,3',
    ]

    (run_dir / 'phase18_5_runner_command.sh').write_text(' '.join(cmd), encoding='utf-8')
    logger.info('Running phase18.2 base model for phase18.5 ...')

    proc = subprocess.run(cmd, text=True, capture_output=True)
    (run_dir / 'phase18_5_runner.stdout.log').write_text(proc.stdout or '', encoding='utf-8')
    (run_dir / 'phase18_5_runner.stderr.log').write_text(proc.stderr or '', encoding='utf-8')

    if proc.returncode != 0:
        raise RuntimeError(f'phase18.2 runner failed (rc={proc.returncode}). See logs in {run_dir}')
    if not summary_path.exists():
        raise RuntimeError(f'phase18.2 completed but summary not found: {summary_path}')

    logger.info('phase18.2 finished: %s', summary_path)
    return run_dir


def apply_ctrader_layer(
    d: pd.DataFrame,
    cfg: Config,
    tiers: List[Tuple[float, float]],
) -> pd.DataFrame:
    rows = []

    equity = 1.0
    peak = 1.0

    for r in d.itertuples(index=False):
        desired_gross = float(getattr(r, 'gross_lev'))
        desired_turn = float(getattr(r, 'turnover'))
        desired_cost = float(getattr(r, 'cost'))
        desired_gross_ret = float(getattr(r, 'gross_ret'))

        dd_prev = equity / max(peak, 1e-12) - 1.0

        eq_usd = float(cfg.account_equity_usd) * equity
        desired_exposure_usd = abs(desired_gross) * eq_usd
        req_margin = required_margin_usd(desired_exposure_usd, tiers=tiers, account_lev_cap=float(cfg.account_leverage_cap))
        allowed_margin = float(cfg.max_margin_utilization) * eq_usd

        scale_margin = 1.0
        if req_margin > allowed_margin and req_margin > 1e-12:
            scale_margin = float(max(0.0, min(1.0, allowed_margin / req_margin)))

        scale_shield = float(cfg.shield_risk_mult) if dd_prev <= float(cfg.shield_dd_limit) else 1.0
        scale = float(max(0.0, min(1.0, scale_margin * scale_shield)))
        if 0.0 < scale < float(cfg.min_exec_scale):
            scale = 0.0

        eff_gross = desired_gross * scale
        eff_turn = desired_turn * scale
        gross_ret_adj = desired_gross_ret * scale

        base_cost_adj = desired_cost * scale
        extra_cost_adj = (float(cfg.extra_cost_per_turnover) + float(cfg.slippage_per_turnover)) * eff_turn
        net_ret_adj = gross_ret_adj - base_cost_adj - extra_cost_adj

        net_ret_adj = float(max(net_ret_adj, -0.95))
        equity *= (1.0 + net_ret_adj)
        peak = max(peak, equity)

        eff_exposure_usd = abs(eff_gross) * eq_usd
        eff_margin = required_margin_usd(eff_exposure_usd, tiers=tiers, account_lev_cap=float(cfg.account_leverage_cap))
        margin_util = eff_margin / max(eq_usd, 1e-12)

        rows.append(
            {
                'timestamp_end': pd.Timestamp(getattr(r, 'timestamp_end')),
                'desired_gross_lev': desired_gross,
                'effective_gross_lev': eff_gross,
                'desired_turnover': desired_turn,
                'effective_turnover': eff_turn,
                'scale_margin': scale_margin,
                'scale_shield': scale_shield,
                'exec_scale': scale,
                'gross_ret_base': desired_gross_ret,
                'gross_ret_exec': gross_ret_adj,
                'base_cost_exec': base_cost_adj,
                'extra_cost_exec': extra_cost_adj,
                'net_ret_exec': net_ret_adj,
                'equity_exec': equity,
                'dd_prev': dd_prev,
                'desired_exposure_usd': desired_exposure_usd,
                'effective_exposure_usd': eff_exposure_usd,
                'required_margin_usd': eff_margin,
                'margin_utilization': margin_util,
            }
        )

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description='Phase18.5 - cTrader-aware dynamic leverage execution layer over phase18.2 CTDE model.')

    ap.add_argument('--phase16_4_run_dir', type=str, default=Config.phase16_4_run_dir)
    ap.add_argument('--phase18_2_runner', type=str, default=Config.phase18_2_runner)
    ap.add_argument('--out_dir', type=str, default=Config.out_dir)
    ap.add_argument('--input_steps_parquet', type=str, default=Config.input_steps_parquet, help='Optional. If provided, use this steps parquet directly (frozen inference mode).')

    ap.add_argument('--run_phase18_2', type=int, default=Config.run_phase18_2, choices=[0, 1])
    ap.add_argument('--skip_if_exists', type=int, default=Config.skip_if_exists, choices=[0, 1])

    ap.add_argument('--min_train_windows', type=int, default=Config.min_train_windows)
    ap.add_argument('--val_windows', type=int, default=Config.val_windows)
    ap.add_argument('--test_windows', type=int, default=Config.test_windows)
    ap.add_argument('--step_windows', type=int, default=Config.step_windows)

    ap.add_argument('--max_gross_leverage', type=float, default=Config.max_gross_leverage)
    ap.add_argument('--target_gross_leverage', type=float, default=Config.target_gross_leverage)
    ap.add_argument('--min_target_mult', type=float, default=Config.min_target_mult)
    ap.add_argument('--max_target_mult', type=float, default=Config.max_target_mult)
    ap.add_argument('--loss_w_turnover', type=float, default=Config.loss_w_turnover)
    ap.add_argument('--cost_per_turnover', type=float, default=Config.cost_per_turnover)
    ap.add_argument('--turnover_smooth_alpha', type=float, default=Config.turnover_smooth_alpha)
    ap.add_argument('--exec_rebalance_deadband', type=float, default=Config.exec_rebalance_deadband)
    ap.add_argument('--exec_rebalance_cooldown_bars', type=int, default=Config.exec_rebalance_cooldown_bars)
    ap.add_argument('--exec_min_turnover', type=float, default=Config.exec_min_turnover)
    ap.add_argument('--gate_turnover_ratio_max', type=float, default=Config.gate_turnover_ratio_max)
    ap.add_argument('--robust_score_turn_penalty', type=float, default=Config.robust_score_turn_penalty)
    ap.add_argument('--robust_score_dd_penalty', type=float, default=Config.robust_score_dd_penalty)
    ap.add_argument('--robust_score_trade_penalty', type=float, default=Config.robust_score_trade_penalty)
    ap.add_argument('--robust_score_rebalance_penalty', type=float, default=Config.robust_score_rebalance_penalty)
    ap.add_argument('--robust_alpha_candidates', type=str, default=Config.robust_alpha_candidates)

    ap.add_argument('--policy', type=str, default=Config.policy)
    ap.add_argument('--cost_mult', type=float, default=Config.cost_mult)

    ap.add_argument('--account_equity_usd', type=float, default=Config.account_equity_usd)
    ap.add_argument('--account_leverage_cap', type=float, default=Config.account_leverage_cap)
    ap.add_argument('--max_margin_utilization', type=float, default=Config.max_margin_utilization)
    ap.add_argument('--leverage_tiers', type=str, default=Config.leverage_tiers)
    ap.add_argument('--extra_cost_per_turnover', type=float, default=Config.extra_cost_per_turnover)
    ap.add_argument('--slippage_per_turnover', type=float, default=Config.slippage_per_turnover)
    ap.add_argument('--shield_dd_limit', type=float, default=Config.shield_dd_limit)
    ap.add_argument('--shield_risk_mult', type=float, default=Config.shield_risk_mult)
    ap.add_argument('--min_exec_scale', type=float, default=Config.min_exec_scale)

    args = ap.parse_args()
    cfg = Config(**vars(args))

    out_dir = Path(cfg.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_dir / 'phase18_5.log')
    logger.info('Config: %s', cfg)

    tiers = parse_tiers(cfg.leverage_tiers)
    logger.info('Parsed leverage tiers: %s', tiers)

    base_dir = None
    if str(cfg.input_steps_parquet).strip():
        p_steps = Path(str(cfg.input_steps_parquet)).resolve()
        if not p_steps.exists():
            raise FileNotFoundError(p_steps)
        d_all = pd.read_parquet(p_steps)
        if 'timestamp_end' not in d_all.columns:
            raise RuntimeError(f'input_steps_parquet missing timestamp_end: {p_steps}')
        d_all['timestamp_end'] = pd.to_datetime(d_all['timestamp_end'], utc=True)
        d_sel = d_all.copy()
        if 'policy' in d_sel.columns:
            d_tmp = d_sel[d_sel['policy'] == str(cfg.policy)].copy()
            if len(d_tmp) > 0:
                d_sel = d_tmp
        if 'cost_mult' in d_sel.columns:
            d_tmp = d_sel[np.isclose(d_sel['cost_mult'].astype(float), float(cfg.cost_mult))].copy()
            if len(d_tmp) > 0:
                d_sel = d_tmp
        d_sel = d_sel.sort_values('timestamp_end').reset_index(drop=True)
    else:
        if int(cfg.run_phase18_2) == 1:
            base_dir = run_phase18_2_if_needed(cfg=cfg, out_dir=out_dir, logger=logger)
        else:
            base_dir = (out_dir / 'phase18_2_base_run').resolve()

        p_steps = base_dir / 'phase18_2_test_steps.parquet'
        p_summary = base_dir / 'phase18_2_summary.json'
        if not p_steps.exists() or not p_summary.exists():
            raise FileNotFoundError(f'Missing phase18.2 artifacts in {base_dir}')

        d_all = pd.read_parquet(p_steps)
        d_all['timestamp_end'] = pd.to_datetime(d_all['timestamp_end'], utc=True)

        d_sel = d_all[(d_all['policy'] == str(cfg.policy)) & (d_all['cost_mult'] == float(cfg.cost_mult))].copy()
        if d_sel.empty:
            raise RuntimeError(f'No rows for policy={cfg.policy}, cost_mult={cfg.cost_mult} in {p_steps}')
        d_sel = d_sel.sort_values('timestamp_end').reset_index(drop=True)

    need_cols = ['gross_ret', 'turnover', 'cost', 'gross_lev']
    missing = [c for c in need_cols if c not in d_sel.columns]
    if missing:
        raise RuntimeError(f'Steps missing required cols {missing}. Got cols={list(d_sel.columns)}')
    if 'net_ret' not in d_sel.columns:
        d_sel['net_ret'] = d_sel['gross_ret'].astype(float) - d_sel['cost'].astype(float)

    # Base metrics from original backtest rows
    m_base = metrics_from_returns(
        net_ret=d_sel['net_ret'].to_numpy(dtype=float),
        turnover=d_sel['turnover'].to_numpy(dtype=float),
        gross_lev=d_sel['gross_lev'].to_numpy(dtype=float),
        margin_util=np.zeros((len(d_sel),), dtype=float),
        scale=np.ones((len(d_sel),), dtype=float),
    )

    d_exec = apply_ctrader_layer(d=d_sel, cfg=cfg, tiers=tiers)
    m_exec = metrics_from_returns(
        net_ret=d_exec['net_ret_exec'].to_numpy(dtype=float),
        turnover=d_exec['effective_turnover'].to_numpy(dtype=float),
        gross_lev=d_exec['effective_gross_lev'].to_numpy(dtype=float),
        margin_util=d_exec['margin_utilization'].to_numpy(dtype=float),
        scale=d_exec['exec_scale'].to_numpy(dtype=float),
    )

    d_exec.to_parquet(out_dir / 'phase18_5_ctrader_execution_steps.parquet', index=False)
    d_exec.to_csv(out_dir / 'phase18_5_ctrader_execution_steps.csv', index=False)

    cmp_df = pd.DataFrame([
        {'layer': 'base_phase18_2', **m_base},
        {'layer': 'phase18_5_ctrader_exec', **m_exec},
    ])
    cmp_df.to_csv(out_dir / 'phase18_5_metrics_comparison.csv', index=False)

    # Equity chart
    e_base = np.cumprod(1.0 + d_sel['net_ret'].to_numpy(dtype=float))
    e_exec = d_exec['equity_exec'].to_numpy(dtype=float)
    plt.figure(figsize=(11, 4))
    plt.plot(d_sel['timestamp_end'], e_base, label='base_phase18_2')
    plt.plot(d_exec['timestamp_end'], e_exec, label='phase18_5_ctrader_exec')
    plt.title('Phase18.5 Equity Comparison (selected policy)')
    plt.xlabel('timestamp')
    plt.ylabel('equity (start=1.0)')
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'phase18_5_equity_compare.png', dpi=140)
    plt.close()

    # Margin utilization chart
    plt.figure(figsize=(11, 3.5))
    plt.plot(d_exec['timestamp_end'], d_exec['margin_utilization'])
    plt.axhline(float(cfg.max_margin_utilization), color='r', linestyle='--', linewidth=1.0, label='max_margin_utilization')
    plt.title('Phase18.5 Estimated Margin Utilization')
    plt.xlabel('timestamp')
    plt.ylabel('utilization')
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'phase18_5_margin_utilization.png', dpi=140)
    plt.close()

    summary = {
        'status': 'SUCCESS',
        'model_base': 'phase18.2 CTDE robust',
        'selected_policy': str(cfg.policy),
        'selected_cost_mult': float(cfg.cost_mult),
        'config': asdict(cfg),
        'leverage_tiers_parsed': [{'cap_usd': ('inf' if math.isinf(c) else float(c)), 'leverage': float(l)} for c, l in tiers],
        'docs_used': DOCS_USED,
        'base_metrics': m_base,
        'ctrader_exec_metrics': m_exec,
        'delta': {
            'delta_sharpe': float(m_exec['sharpe_ann'] - m_base['sharpe_ann']) if np.isfinite(m_exec['sharpe_ann']) and np.isfinite(m_base['sharpe_ann']) else float('nan'),
            'delta_cum_net_return': float(m_exec['cum_net_return'] - m_base['cum_net_return']) if np.isfinite(m_exec['cum_net_return']) and np.isfinite(m_base['cum_net_return']) else float('nan'),
            'delta_max_drawdown': float(m_exec['max_drawdown'] - m_base['max_drawdown']) if np.isfinite(m_exec['max_drawdown']) and np.isfinite(m_base['max_drawdown']) else float('nan'),
            'delta_avg_turnover': float(m_exec['avg_turnover'] - m_base['avg_turnover']) if np.isfinite(m_exec['avg_turnover']) and np.isfinite(m_base['avg_turnover']) else float('nan'),
        },
        'outputs': {
            'base_run_dir': str(base_dir) if base_dir is not None else None,
            'input_steps_parquet': str(Path(cfg.input_steps_parquet).resolve()) if str(cfg.input_steps_parquet).strip() else None,
            'steps_parquet': str(out_dir / 'phase18_5_ctrader_execution_steps.parquet'),
            'steps_csv': str(out_dir / 'phase18_5_ctrader_execution_steps.csv'),
            'comparison_csv': str(out_dir / 'phase18_5_metrics_comparison.csv'),
            'equity_png': str(out_dir / 'phase18_5_equity_compare.png'),
            'margin_png': str(out_dir / 'phase18_5_margin_utilization.png'),
            'log': str(out_dir / 'phase18_5.log'),
        },
    }

    (out_dir / 'phase18_5_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    logger.info('SUCCESS: %s', out_dir / 'phase18_5_summary.json')


if __name__ == '__main__':
    main()
