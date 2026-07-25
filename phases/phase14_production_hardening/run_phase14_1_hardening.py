#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)


@dataclass(frozen=True)
class PolicyParams:
    enter_nonflat: float
    enter_margin: float
    exit_nonflat: float
    exit_margin: float
    min_hold_bars: int
    max_leverage: float
    max_trades_per_hour: int
    daily_dd_limit: float


@dataclass
class Config:
    phase13_run_dir: str
    out_dir: str
    cost_pips: float = 1.2
    strict_only: bool = True
    search_train_ratio: float = 0.65
    wf_folds: int = 6
    recent_days: str = '90,180,365'
    target_trade_rate: float = 0.35
    purged_gap_bars: int = 24
    eval_stride: int = 24


def pip_size_for_symbol(symbol: str) -> float:
    return 0.01 if str(symbol).upper().endswith('JPY') else 0.0001


def macro_f1_fixed_labels(y_true: np.ndarray, y_pred: np.ndarray, labels=(-1, 0, 1)) -> float:
    f1s = []
    for lb in labels:
        tp = float(np.sum((y_true == lb) & (y_pred == lb)))
        fp = float(np.sum((y_true != lb) & (y_pred == lb)))
        fn = float(np.sum((y_true == lb) & (y_pred != lb)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def evaluate_levels(
    levels: np.ndarray,
    y_true_action: np.ndarray,
    future_ret: np.ndarray,
    cost_ret: float,
    bars_per_year: float = 24.0 * 252.0,
) -> Dict[str, float]:
    action = np.sign(levels).astype(np.int64)
    net = levels * future_ret - np.abs(levels) * cost_ret
    trade = (levels != 0).astype(np.float32)
    turnover = np.zeros_like(trade)
    if len(levels) > 1:
        turnover[1:] = (np.abs(np.diff(levels)) > 1e-12).astype(np.float32)

    out = {
        'acc_action_3': float((action == y_true_action).mean()),
        'f1_macro_action_3': macro_f1_fixed_labels(y_true_action, action, labels=(-1, 0, 1)),
        'avg_net_return': float(np.mean(net)),
        'cum_net_return': float(np.sum(net)),
        'std_net_return': float(np.std(net)),
        'trade_rate': float(np.mean(trade)),
        'turnover_rate': float(np.mean(turnover)),
    }
    eps = 1e-12
    out['sharpe_ann'] = float((out['avg_net_return'] / (out['std_net_return'] + eps)) * np.sqrt(float(bars_per_year)))
    traded = trade > 0
    out['win_rate_trades'] = float((net[traded] > 0).mean()) if traded.any() else 0.0
    out['avg_abs_lev'] = float(np.mean(np.abs(levels)))
    out['avg_abs_lev_on_trade'] = float(np.mean(np.abs(levels[traded]))) if traded.any() else 0.0

    eq = np.cumsum(net)
    run_max = np.maximum.accumulate(eq) if len(eq) else np.array([0.0])
    dd = eq - run_max if len(eq) else np.array([0.0])
    out['max_drawdown'] = float(dd.min())
    return out


def ensure_signal_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if 'p_long' in out.columns and 'p_short' in out.columns:
        out['score_ls'] = out['p_long'].astype(float) - out['p_short'].astype(float)
    else:
        out['score_ls'] = out['pred_action'].astype(float)

    if 'p_nonflat' in out.columns:
        out['nonflat_score'] = out['p_nonflat'].astype(float)
    elif 'p_flat' in out.columns:
        out['nonflat_score'] = 1.0 - out['p_flat'].astype(float)
    else:
        out['nonflat_score'] = (np.abs(out['score_ls']) > 0).astype(float)

    out['timestamp'] = pd.to_datetime(out['timestamp'], errors='coerce', utc=True)
    out = out.dropna(subset=['timestamp']).copy()
    out = out.sort_values('timestamp').reset_index(drop=True)

    # Causal realized return proxy (t-1 -> t), used by kill-switch logic.
    if 'close' in out.columns:
        close = pd.to_numeric(out['close'], errors='coerce').astype(float)
        rr = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out['realized_return_1bar'] = rr.astype(float)
    else:
        out['realized_return_1bar'] = 0.0
    return out


def apply_hardened_policy(df: pd.DataFrame, params: PolicyParams, cost_ret_for_kill: float) -> pd.DataFrame:
    x = ensure_signal_columns(df)

    score = x['score_ls'].to_numpy(dtype=float)
    nonflat = x['nonflat_score'].to_numpy(dtype=float)
    ret_realized = x['realized_return_1bar'].fillna(0.0).to_numpy(dtype=float)
    ts = pd.to_datetime(x['timestamp'], utc=True)

    n = len(x)
    lev = np.zeros(n, dtype=np.float32)
    trade_event = np.zeros(n, dtype=np.int8)
    turnover_event = np.zeros(n, dtype=np.int8)
    kill_locked = np.zeros(n, dtype=np.int8)
    day_dd = np.zeros(n, dtype=np.float32)
    trades_in_hour = np.zeros(n, dtype=np.int16)

    pos = 0.0
    hold = 0

    curr_day = None
    day_pnl = 0.0
    day_peak = 0.0
    day_lock = False
    hourly_trade_count: Dict[pd.Timestamp, int] = {}

    for i in range(n):
        t = ts.iloc[i]
        d = t.date()
        h = t.floor('h')

        if curr_day is None or d != curr_day:
            curr_day = d
            day_pnl = 0.0
            day_peak = 0.0
            day_lock = False
            hourly_trade_count = {}

        # Update daily PnL with causal realized move from the already-held position.
        day_pnl += float(pos) * float(ret_realized[i])
        day_peak = max(day_peak, day_pnl)
        ddraw = day_pnl - day_peak
        if ddraw <= -float(params.daily_dd_limit):
            day_lock = True

        hour_count = hourly_trade_count.get(h, 0)
        can_trade_hour = hour_count < int(params.max_trades_per_hour)

        if score[i] >= params.enter_margin and nonflat[i] >= params.enter_nonflat:
            desired_sign = 1
        elif score[i] <= -params.enter_margin and nonflat[i] >= params.enter_nonflat:
            desired_sign = -1
        else:
            desired_sign = 0

        if hold > 0:
            hold -= 1

        new_pos = pos

        if day_lock and abs(pos) > 1e-12:
            if can_trade_hour:
                new_pos = 0.0
                trade_event[i] = 1
        elif abs(pos) <= 1e-12:
            if (not day_lock) and can_trade_hour and desired_sign != 0:
                new_pos = float(desired_sign) * float(params.max_leverage)
                hold = int(params.min_hold_bars)
                trade_event[i] = 1
        else:
            curr_sign = 1 if pos > 0 else -1
            if hold <= 0:
                if (abs(score[i]) <= params.exit_margin) or (nonflat[i] < params.exit_nonflat):
                    if can_trade_hour:
                        new_pos = 0.0
                        trade_event[i] = 1
                elif desired_sign != 0 and desired_sign != curr_sign and (not day_lock):
                    if can_trade_hour:
                        new_pos = float(desired_sign) * float(params.max_leverage)
                        hold = int(params.min_hold_bars)
                        trade_event[i] = 1

        # Transaction-cost impact for risk accounting only when position changes.
        if abs(float(new_pos) - float(pos)) > 1e-12:
            day_pnl -= abs(float(new_pos) - float(pos)) * float(cost_ret_for_kill)
            day_peak = max(day_peak, day_pnl)
            ddraw = day_pnl - day_peak
            if ddraw <= -float(params.daily_dd_limit):
                day_lock = True

        pos = float(new_pos)
        if i > 0 and abs(pos - lev[i - 1]) > 1e-12:
            turnover_event[i] = 1

        if trade_event[i] == 1:
            hourly_trade_count[h] = hour_count + 1
        trades_in_hour[i] = hourly_trade_count.get(h, 0)

        kill_locked[i] = 1 if day_lock else 0
        day_dd[i] = float(ddraw)
        lev[i] = pos

    out = x.copy()
    out['lev_phase14_hardened'] = lev
    out['action_phase14_hardened'] = np.sign(lev).astype(np.int8)
    out['phase14_trade_event'] = trade_event
    out['phase14_turnover_event'] = turnover_event
    out['phase14_kill_locked'] = kill_locked
    out['phase14_day_drawdown'] = day_dd
    out['phase14_trades_in_hour'] = trades_in_hour
    return out


def evaluate_cost_grid(
    df: pd.DataFrame,
    lev_col: str,
    cost_ret_base: float,
    multipliers: List[float],
    bars_per_year: float = 24.0 * 252.0,
) -> List[Dict[str, float]]:
    y = df['y_action'].to_numpy(dtype=np.int64)
    ret = df['future_return_h'].to_numpy(dtype=np.float32)
    lev = df[lev_col].to_numpy(dtype=np.float32)
    rows = []
    for m in multipliers:
        met = evaluate_levels(lev, y, ret, cost_ret=float(cost_ret_base * m), bars_per_year=bars_per_year)
        met['cost_mult'] = float(m)
        rows.append(met)
    return rows


def robust_score(metrics_by_cost: Dict[float, Dict[str, float]], target_trade_rate: float) -> float:
    m1 = metrics_by_cost.get(1.0)
    m2 = metrics_by_cost.get(2.0)
    m3 = metrics_by_cost.get(3.0)
    if not (m1 and m2 and m3):
        return -1e12

    edge_floor = min(float(m2['cum_net_return']), float(m3['cum_net_return']))
    score = (
        0.20 * float(m1['sharpe_ann'])
        + 0.35 * float(m2['sharpe_ann'])
        + 0.45 * float(m3['sharpe_ann'])
        + 8.0 * edge_floor
        - 2.0 * float(m1['turnover_rate'])
        - 2.0 * abs(float(m1['trade_rate']) - float(target_trade_rate))
    )
    return float(score)


def parameter_grid() -> List[PolicyParams]:
    grid = []
    for enter_nonflat in [0.55, 0.60]:
        for enter_margin in [0.03, 0.05, 0.08]:
            for exit_nonflat in [0.40, 0.45]:
                for exit_margin in [0.005, 0.015]:
                    for min_hold_bars in [1, 3, 6]:
                        for max_leverage in [1.0, 1.5]:
                            for max_trades_per_hour in [2, 4]:
                                for daily_dd_limit in [0.003, 0.006]:
                                    grid.append(PolicyParams(
                                        enter_nonflat=enter_nonflat,
                                        enter_margin=enter_margin,
                                        exit_nonflat=exit_nonflat,
                                        exit_margin=exit_margin,
                                        min_hold_bars=min_hold_bars,
                                        max_leverage=max_leverage,
                                        max_trades_per_hour=max_trades_per_hour,
                                        daily_dd_limit=daily_dd_limit,
                                    ))
    return grid


def search_best_policy(df_train: pd.DataFrame, cost_ret_base: float, target_trade_rate: float, logger: logging.Logger):
    candidates = parameter_grid()
    rows = []
    best_score = -1e18
    best_params = None
    best_df = None

    for idx, p in enumerate(candidates, start=1):
        d = apply_hardened_policy(df_train, p, cost_ret_for_kill=cost_ret_base)
        mets = evaluate_cost_grid(d, 'lev_phase14_hardened', cost_ret_base=cost_ret_base, multipliers=[1.0, 2.0, 3.0])
        by_cost = {float(m['cost_mult']): m for m in mets}
        score = robust_score(by_cost, target_trade_rate=target_trade_rate)

        row = {'candidate_idx': idx, 'robust_score': score, **asdict(p)}
        for m in mets:
            c = int(m['cost_mult'])
            row[f'cum_x{c}'] = m['cum_net_return']
            row[f'sharpe_x{c}'] = m['sharpe_ann']
            row[f'trade_rate_x{c}'] = m['trade_rate']
            row[f'turnover_x{c}'] = m['turnover_rate']
        rows.append(row)

        if score > best_score:
            best_score = score
            best_params = p
            best_df = d

    logger.info('Policy search completed: %s candidates | best_score=%.6f', len(candidates), best_score)
    return best_params, best_df, pd.DataFrame(rows).sort_values('robust_score', ascending=False)


def split_train_test(df: pd.DataFrame, train_ratio: float):
    n = len(df)
    cut = int(max(1, min(n - 1, round(n * float(train_ratio)))))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def maybe_nonoverlap(df: pd.DataFrame, stride: int) -> pd.DataFrame:
    s = int(max(1, stride))
    if s <= 1:
        return df.copy()
    return df.iloc[::s].copy().reset_index(drop=True)


def walkforward_extended(
    df: pd.DataFrame,
    wf_folds: int,
    cost_ret_base: float,
    target_trade_rate: float,
    purged_gap_bars: int,
    eval_stride: int,
    logger: logging.Logger,
):
    d = df.sort_values('timestamp').reset_index(drop=True)
    n = len(d)
    start_test = int(max(50, round(n * 0.50)))
    remaining = n - start_test
    if remaining <= wf_folds:
        return pd.DataFrame(), pd.DataFrame()

    fold_size = max(1, remaining // wf_folds)
    fold_rows = []
    regime_rows = []

    for f in range(wf_folds):
        te0 = start_test + f * fold_size
        te1 = n if f == wf_folds - 1 else min(n, te0 + fold_size)
        if te1 - te0 < 24:
            continue
        tr_end = max(0, te0 - int(max(0, purged_gap_bars)))
        tr = d.iloc[:tr_end].copy()
        te = d.iloc[te0:te1].copy()
        if len(tr) < 240:
            continue

        best_params, _, _ = search_best_policy(tr, cost_ret_base, target_trade_rate, logger)
        te_h = apply_hardened_policy(te, best_params, cost_ret_for_kill=cost_ret_base)
        te_h_eval = maybe_nonoverlap(te_h, stride=eval_stride)
        bars_per_year_eval = (24.0 * 252.0) / float(max(1, int(eval_stride)))
        if len(te_h_eval) < 24:
            continue

        for model_name, lev_col in [('baseline_phase12', 'lev_phase12_2_frozen'), ('phase14_hardened', 'lev_phase14_hardened')]:
            for m in [1.0, 2.0, 3.0]:
                met = evaluate_levels(
                    te_h_eval[lev_col].to_numpy(dtype=np.float32),
                    te_h_eval['y_action'].to_numpy(dtype=np.int64),
                    te_h_eval['future_return_h'].to_numpy(dtype=np.float32),
                    cost_ret=float(cost_ret_base * m),
                    bars_per_year=bars_per_year_eval,
                )
                fold_rows.append({
                    'fold_id': f,
                    'test_start': str(te['timestamp'].iloc[0]),
                    'test_end': str(te['timestamp'].iloc[-1]),
                    'n_test_rows': int(len(te)),
                    'n_test_eval_rows': int(len(te_h_eval)),
                    'purged_gap_bars': int(max(0, purged_gap_bars)),
                    'eval_stride': int(max(1, eval_stride)),
                    'bars_per_year_eval': float(bars_per_year_eval),
                    'model': model_name,
                    'cost_mult': float(m),
                    **met,
                })

        # Regime-level stats on hardened model at cost x2.
        for cluster_id, g in te_h_eval.groupby('cluster_id', dropna=False):
            met_r = evaluate_levels(
                g['lev_phase14_hardened'].to_numpy(dtype=np.float32),
                g['y_action'].to_numpy(dtype=np.int64),
                g['future_return_h'].to_numpy(dtype=np.float32),
                cost_ret=float(cost_ret_base * 2.0),
                bars_per_year=bars_per_year_eval,
            )
            regime_rows.append({
                'fold_id': f,
                'cluster_id': int(cluster_id) if pd.notna(cluster_id) else -1,
                'n_rows': int(len(g)),
                'purged_gap_bars': int(max(0, purged_gap_bars)),
                'eval_stride': int(max(1, eval_stride)),
                'bars_per_year_eval': float(bars_per_year_eval),
                **met_r,
            })

    return pd.DataFrame(fold_rows), pd.DataFrame(regime_rows)


def compute_recent_metrics(df: pd.DataFrame, cost_ret_base: float, recent_days: List[int]) -> pd.DataFrame:
    out = []
    d = df.sort_values('timestamp').copy()
    tmax = pd.to_datetime(d['timestamp']).max()

    for days in recent_days:
        t0 = tmax - pd.Timedelta(days=int(days))
        g = d[pd.to_datetime(d['timestamp']) >= t0].copy()
        if len(g) < 24:
            continue

        for model_name, lev_col in [('baseline_phase12', 'lev_phase12_2_frozen'), ('phase14_hardened', 'lev_phase14_hardened')]:
            for cm in [1.0, 2.0, 3.0]:
                met = evaluate_levels(
                    g[lev_col].to_numpy(dtype=np.float32),
                    g['y_action'].to_numpy(dtype=np.int64),
                    g['future_return_h'].to_numpy(dtype=np.float32),
                    cost_ret=float(cost_ret_base * cm),
                )
                out.append({
                    'recent_days': int(days),
                    'n_rows': int(len(g)),
                    'model': model_name,
                    'cost_mult': float(cm),
                    **met,
                })
    return pd.DataFrame(out)


def plot_equity_compare(df: pd.DataFrame, cost_ret_base: float, out_png: Path, title: str):
    d = df.sort_values('timestamp').copy()
    ts = pd.to_datetime(d['timestamp'])
    ret = d['future_return_h'].to_numpy(dtype=np.float32)

    lev_b = d['lev_phase12_2_frozen'].to_numpy(dtype=np.float32)
    lev_h = d['lev_phase14_hardened'].to_numpy(dtype=np.float32)

    eq_b = np.cumsum(lev_b * ret - np.abs(lev_b) * cost_ret_base)
    eq_h = np.cumsum(lev_h * ret - np.abs(lev_h) * cost_ret_base)

    plt.figure(figsize=(12, 5))
    plt.plot(ts, eq_b, label='baseline_phase12', linewidth=1.2)
    plt.plot(ts, eq_h, label='phase14_hardened', linewidth=1.2)
    plt.title(title)
    plt.xlabel('timestamp')
    plt.ylabel('cumulative net return')
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=130)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Phase14 hardening: turnover control, robust costs, walk-forward, risk limits.')
    parser.add_argument('--phase13_run_dir', required=True, type=str)
    parser.add_argument('--out_dir', type=str, default='')
    parser.add_argument('--cost_pips', type=float, default=1.2)
    parser.add_argument('--strict_only', type=int, choices=[0, 1], default=1)
    parser.add_argument('--search_train_ratio', type=float, default=0.65)
    parser.add_argument('--wf_folds', type=int, default=6)
    parser.add_argument('--recent_days', type=str, default='90,180,365')
    parser.add_argument('--target_trade_rate', type=float, default=0.35)
    parser.add_argument('--purged_gap_bars', type=int, default=24)
    parser.add_argument('--eval_stride', type=int, default=24)
    args = parser.parse_args()

    cfg = Config(
        phase13_run_dir=str(args.phase13_run_dir),
        out_dir=str(args.out_dir or ''),
        cost_pips=float(args.cost_pips),
        strict_only=bool(int(args.strict_only)),
        search_train_ratio=float(args.search_train_ratio),
        wf_folds=int(args.wf_folds),
        recent_days=str(args.recent_days),
        target_trade_rate=float(args.target_trade_rate),
        purged_gap_bars=int(args.purged_gap_bars),
        eval_stride=int(args.eval_stride),
    )

    run_dir = Path(cfg.phase13_run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    out_dir = Path(cfg.out_dir) if cfg.out_dir else (run_dir.parent / f'phase14_hardening_{run_dir.name}')
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(out_dir / 'phase14.log', mode='w'),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger('phase14')

    summary = json.loads((run_dir / 'phase13_summary.json').read_text())
    leak_audit = json.loads((run_dir / 'phase13_leak_audit.json').read_text())
    st5 = pd.read_parquet(run_dir / 'stage05_phase12_predictions.parquet')

    symbol = summary.get('symbol', 'GBPUSD')
    pip = pip_size_for_symbol(symbol)
    cost_ret_base = float(cfg.cost_pips) * pip

    st5 = ensure_signal_columns(st5)
    use = np.isfinite(st5['future_return_h'].to_numpy(dtype=float))
    strict_start = pd.to_datetime(leak_audit.get('strict_eval_start_exclusive'), errors='coerce', utc=True)
    if cfg.strict_only and pd.notna(strict_start):
        use &= (pd.to_datetime(st5['timestamp'], utc=True) > strict_start).to_numpy()

    d = st5.loc[use].copy().reset_index(drop=True)
    # Keep a minimum sample-size guard, but allow short smoke runs.
    if len(d) < 240:
        raise RuntimeError(f'too few rows after strict filter: {len(d)}')

    logger.info('Rows loaded strict=%s | symbol=%s | cost_ret_base=%.8f', len(d), symbol, cost_ret_base)

    d_train, d_test = split_train_test(d, train_ratio=cfg.search_train_ratio)
    logger.info('Train rows=%s | Test rows=%s', len(d_train), len(d_test))

    # Baseline metrics.
    baseline_full = evaluate_cost_grid(d, 'lev_phase12_2_frozen', cost_ret_base, multipliers=[1.0, 2.0, 3.0])

    # Search best anti-overtrading policy on train.
    best_params, _, search_df = search_best_policy(d_train, cost_ret_base, target_trade_rate=cfg.target_trade_rate, logger=logger)
    d_hardened = apply_hardened_policy(d, best_params, cost_ret_for_kill=cost_ret_base)
    d_test_hardened = apply_hardened_policy(d_test, best_params, cost_ret_for_kill=cost_ret_base)
    d_nonoverlap = maybe_nonoverlap(d, stride=cfg.eval_stride)
    d_hardened_nonoverlap = maybe_nonoverlap(d_hardened, stride=cfg.eval_stride)
    d_test_nonoverlap = maybe_nonoverlap(d_test, stride=cfg.eval_stride)
    d_test_hardened_nonoverlap = maybe_nonoverlap(d_test_hardened, stride=cfg.eval_stride)

    hardened_full = evaluate_cost_grid(d_hardened, 'lev_phase14_hardened', cost_ret_base, multipliers=[1.0, 2.0, 3.0])
    baseline_test = evaluate_cost_grid(d_test, 'lev_phase12_2_frozen', cost_ret_base, multipliers=[1.0, 2.0, 3.0])
    hardened_test = evaluate_cost_grid(d_test_hardened, 'lev_phase14_hardened', cost_ret_base, multipliers=[1.0, 2.0, 3.0])
    bars_per_year_nonoverlap = (24.0 * 252.0) / float(max(1, int(cfg.eval_stride)))
    baseline_full_no = evaluate_cost_grid(
        d_nonoverlap,
        'lev_phase12_2_frozen',
        cost_ret_base,
        multipliers=[1.0, 2.0, 3.0],
        bars_per_year=bars_per_year_nonoverlap,
    )
    hardened_full_no = evaluate_cost_grid(
        d_hardened_nonoverlap,
        'lev_phase14_hardened',
        cost_ret_base,
        multipliers=[1.0, 2.0, 3.0],
        bars_per_year=bars_per_year_nonoverlap,
    )
    baseline_test_no = evaluate_cost_grid(
        d_test_nonoverlap,
        'lev_phase12_2_frozen',
        cost_ret_base,
        multipliers=[1.0, 2.0, 3.0],
        bars_per_year=bars_per_year_nonoverlap,
    )
    hardened_test_no = evaluate_cost_grid(
        d_test_hardened_nonoverlap,
        'lev_phase14_hardened',
        cost_ret_base,
        multipliers=[1.0, 2.0, 3.0],
        bars_per_year=bars_per_year_nonoverlap,
    )

    # Extended walk-forward by regime.
    wf_df, wf_regime_df = walkforward_extended(
        d,
        wf_folds=cfg.wf_folds,
        cost_ret_base=cost_ret_base,
        target_trade_rate=cfg.target_trade_rate,
        purged_gap_bars=cfg.purged_gap_bars,
        eval_stride=cfg.eval_stride,
        logger=logger,
    )

    recent_days = [int(x.strip()) for x in cfg.recent_days.split(',') if x.strip()]
    recent_df = compute_recent_metrics(d_hardened, cost_ret_base=cost_ret_base, recent_days=recent_days)

    # Save artifacts.
    search_df.to_csv(out_dir / 'phase14_param_search.csv', index=False)

    rows_full = []
    for m in baseline_full:
        rows_full.append({'model': 'baseline_phase12', **m})
    for m in hardened_full:
        rows_full.append({'model': 'phase14_hardened', **m})
    full_df = pd.DataFrame(rows_full)
    full_df.to_csv(out_dir / 'phase14_cost_validation_full.csv', index=False)
    rows_full_no = []
    for m in baseline_full_no:
        rows_full_no.append({'model': 'baseline_phase12', **m})
    for m in hardened_full_no:
        rows_full_no.append({'model': 'phase14_hardened', **m})
    pd.DataFrame(rows_full_no).to_csv(out_dir / 'phase14_cost_validation_full_nooverlap.csv', index=False)

    rows_test = []
    for m in baseline_test:
        rows_test.append({'model': 'baseline_phase12', **m})
    for m in hardened_test:
        rows_test.append({'model': 'phase14_hardened', **m})
    test_df = pd.DataFrame(rows_test)
    test_df.to_csv(out_dir / 'phase14_cost_validation_test.csv', index=False)
    rows_test_no = []
    for m in baseline_test_no:
        rows_test_no.append({'model': 'baseline_phase12', **m})
    for m in hardened_test_no:
        rows_test_no.append({'model': 'phase14_hardened', **m})
    pd.DataFrame(rows_test_no).to_csv(out_dir / 'phase14_cost_validation_test_nooverlap.csv', index=False)

    if len(wf_df):
        wf_df.to_csv(out_dir / 'phase14_walkforward_folds.csv', index=False)
    if len(wf_regime_df):
        wf_regime_df.to_csv(out_dir / 'phase14_walkforward_regime_metrics.csv', index=False)
    if len(recent_df):
        recent_df.to_csv(out_dir / 'phase14_recent_period_metrics.csv', index=False)

    d_hardened.to_parquet(out_dir / 'phase14_policy_strict_predictions.parquet', index=False)
    d_hardened.to_csv(out_dir / 'phase14_policy_strict_predictions.csv', index=False)

    plot_equity_compare(
        d_hardened,
        cost_ret_base=cost_ret_base,
        out_png=out_dir / 'phase14_equity_compare_x1.png',
        title=f'{symbol} Phase14 Hardened vs Baseline (cost x1)',
    )

    # Summary JSON.
    b_map = {float(x['cost_mult']): x for x in baseline_full}
    h_map = {float(x['cost_mult']): x for x in hardened_full}
    bt_map = {float(x['cost_mult']): x for x in baseline_test}
    ht_map = {float(x['cost_mult']): x for x in hardened_test}
    bno_map = {float(x['cost_mult']): x for x in baseline_full_no}
    hno_map = {float(x['cost_mult']): x for x in hardened_full_no}
    btno_map = {float(x['cost_mult']): x for x in baseline_test_no}
    htno_map = {float(x['cost_mult']): x for x in hardened_test_no}

    summary_out = {
        'status': 'SUCCESS',
        'phase13_run_dir': str(run_dir),
        'symbol': symbol,
        'strict_rows': int(len(d)),
        'strict_rows_nooverlap': int(len(d_nonoverlap)),
        'bars_per_year_nonoverlap': float(bars_per_year_nonoverlap),
        'cost_pips_base': float(cfg.cost_pips),
        'cost_ret_base': float(cost_ret_base),
        'purged_gap_bars': int(cfg.purged_gap_bars),
        'eval_stride': int(cfg.eval_stride),
        'best_policy_params': asdict(best_params),
        'baseline_metrics': {f'x{int(k)}': v for k, v in b_map.items()},
        'hardened_metrics': {f'x{int(k)}': v for k, v in h_map.items()},
        'baseline_metrics_nooverlap': {f'x{int(k)}': v for k, v in bno_map.items()},
        'hardened_metrics_nooverlap': {f'x{int(k)}': v for k, v in hno_map.items()},
        'baseline_test_metrics': {f'x{int(k)}': v for k, v in bt_map.items()},
        'hardened_test_metrics': {f'x{int(k)}': v for k, v in ht_map.items()},
        'baseline_test_metrics_nooverlap': {f'x{int(k)}': v for k, v in btno_map.items()},
        'hardened_test_metrics_nooverlap': {f'x{int(k)}': v for k, v in htno_map.items()},
        'deltas_hardened_minus_baseline': {
            f'x{int(k)}': {
                'cum_net_return': float(h_map[k]['cum_net_return'] - b_map[k]['cum_net_return']),
                'sharpe_ann': float(h_map[k]['sharpe_ann'] - b_map[k]['sharpe_ann']),
                'trade_rate': float(h_map[k]['trade_rate'] - b_map[k]['trade_rate']),
                'turnover_rate': float(h_map[k]['turnover_rate'] - b_map[k]['turnover_rate']),
            }
            for k in b_map.keys()
        },
        'deltas_test_hardened_minus_baseline': {
            f'x{int(k)}': {
                'cum_net_return': float(ht_map[k]['cum_net_return'] - bt_map[k]['cum_net_return']),
                'sharpe_ann': float(ht_map[k]['sharpe_ann'] - bt_map[k]['sharpe_ann']),
                'trade_rate': float(ht_map[k]['trade_rate'] - bt_map[k]['trade_rate']),
                'turnover_rate': float(ht_map[k]['turnover_rate'] - bt_map[k]['turnover_rate']),
            }
            for k in bt_map.keys()
        },
        'deltas_nooverlap_hardened_minus_baseline': {
            f'x{int(k)}': {
                'cum_net_return': float(hno_map[k]['cum_net_return'] - bno_map[k]['cum_net_return']),
                'sharpe_ann': float(hno_map[k]['sharpe_ann'] - bno_map[k]['sharpe_ann']),
                'trade_rate': float(hno_map[k]['trade_rate'] - bno_map[k]['trade_rate']),
                'turnover_rate': float(hno_map[k]['turnover_rate'] - bno_map[k]['turnover_rate']),
            }
            for k in bno_map.keys()
        },
        'deltas_test_nooverlap_hardened_minus_baseline': {
            f'x{int(k)}': {
                'cum_net_return': float(htno_map[k]['cum_net_return'] - btno_map[k]['cum_net_return']),
                'sharpe_ann': float(htno_map[k]['sharpe_ann'] - btno_map[k]['sharpe_ann']),
                'trade_rate': float(htno_map[k]['trade_rate'] - btno_map[k]['trade_rate']),
                'turnover_rate': float(htno_map[k]['turnover_rate'] - btno_map[k]['turnover_rate']),
            }
            for k in btno_map.keys()
        },
        'n_walkforward_rows': int(len(wf_df)),
        'n_walkforward_regime_rows': int(len(wf_regime_df)),
        'n_recent_rows': int(len(recent_df)),
        'outputs': {
            'param_search_csv': str(out_dir / 'phase14_param_search.csv'),
            'cost_validation_full_csv': str(out_dir / 'phase14_cost_validation_full.csv'),
            'cost_validation_full_nooverlap_csv': str(out_dir / 'phase14_cost_validation_full_nooverlap.csv'),
            'cost_validation_test_csv': str(out_dir / 'phase14_cost_validation_test.csv'),
            'cost_validation_test_nooverlap_csv': str(out_dir / 'phase14_cost_validation_test_nooverlap.csv'),
            'walkforward_folds_csv': str(out_dir / 'phase14_walkforward_folds.csv'),
            'walkforward_regime_csv': str(out_dir / 'phase14_walkforward_regime_metrics.csv'),
            'recent_period_csv': str(out_dir / 'phase14_recent_period_metrics.csv'),
            'policy_predictions_parquet': str(out_dir / 'phase14_policy_strict_predictions.parquet'),
            'equity_png': str(out_dir / 'phase14_equity_compare_x1.png'),
            'log_file': str(out_dir / 'phase14.log'),
        },
    }
    (out_dir / 'phase14_summary.json').write_text(json.dumps(summary_out, indent=2))

    logger.info('Phase14 SUCCESS. Summary: %s', out_dir / 'phase14_summary.json')
    print('SUCCESS:', out_dir / 'phase14_summary.json')


if __name__ == '__main__':
    main()
