#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)


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


def evaluate_levels(levels: np.ndarray, y_true_action: np.ndarray, future_ret: np.ndarray, cost_ret: float) -> Dict[str, float]:
    action = np.sign(levels).astype(np.int64)
    net = levels * future_ret - np.abs(levels) * cost_ret
    trade = (levels != 0).astype(np.float32)

    out = {
        'acc_action_3': float((action == y_true_action).mean()),
        'f1_macro_action_3': macro_f1_fixed_labels(y_true_action, action, labels=(-1, 0, 1)),
        'avg_net_return': float(np.mean(net)),
        'cum_net_return': float(np.sum(net)),
        'std_net_return': float(np.std(net)),
        'trade_rate': float(np.mean(trade)),
    }
    eps = 1e-12
    out['sharpe_ann'] = float((out['avg_net_return'] / (out['std_net_return'] + eps)) * np.sqrt(24.0 * 252.0))
    traded = trade > 0
    out['win_rate_trades'] = float((net[traded] > 0).mean()) if traded.any() else 0.0
    out['avg_abs_lev'] = float(np.mean(np.abs(levels)))
    out['avg_abs_lev_on_trade'] = float(np.mean(np.abs(levels[traded]))) if traded.any() else 0.0
    return out


def permutation_test(levels: np.ndarray, ret: np.ndarray, cost_ret: float, obs_cum: float, obs_sharpe: float, n_perm: int):
    n = len(ret)
    null_cum = np.zeros(n_perm, dtype=np.float64)
    null_sh = np.zeros(n_perm, dtype=np.float64)

    abs_lev = np.abs(levels)
    for i in range(n_perm):
        rp = ret[np.random.permutation(n)]
        net = levels * rp - abs_lev * cost_ret
        m = net.mean()
        s = net.std()
        sh = (m / (s + 1e-12)) * np.sqrt(24.0 * 252.0)
        null_cum[i] = net.sum()
        null_sh[i] = sh

    p_cum = float((1 + np.sum(null_cum >= obs_cum)) / (n_perm + 1))
    p_sh = float((1 + np.sum(null_sh >= obs_sharpe)) / (n_perm + 1))

    return {
        'n_perm': int(n_perm),
        'obs_cum': float(obs_cum),
        'obs_sharpe': float(obs_sharpe),
        'null_cum_mean': float(null_cum.mean()),
        'null_cum_std': float(null_cum.std()),
        'null_sharpe_mean': float(null_sh.mean()),
        'null_sharpe_std': float(null_sh.std()),
        'pvalue_cum_ge_obs': p_cum,
        'pvalue_sharpe_ge_obs': p_sh,
    }


def _block_bootstrap_indices(n: int, block_len: int) -> np.ndarray:
    idx = []
    while len(idx) < n:
        s = np.random.randint(0, n)
        e = min(n, s + block_len)
        idx.extend(range(s, e))
    return np.asarray(idx[:n], dtype=np.int64)


def bootstrap_ci(net: np.ndarray, n_boot: int = 2000, block_len: int = 24):
    n = len(net)
    b_cum = np.zeros(n_boot, dtype=np.float64)
    b_sh = np.zeros(n_boot, dtype=np.float64)

    for i in range(n_boot):
        idx = _block_bootstrap_indices(n, block_len)
        x = net[idx]
        b_cum[i] = x.sum()
        m = x.mean()
        s = x.std()
        b_sh[i] = (m / (s + 1e-12)) * np.sqrt(24.0 * 252.0)

    return {
        'n_boot': int(n_boot),
        'block_len': int(block_len),
        'cum_p2_5': float(np.quantile(b_cum, 0.025)),
        'cum_p50': float(np.quantile(b_cum, 0.50)),
        'cum_p97_5': float(np.quantile(b_cum, 0.975)),
        'sharpe_p2_5': float(np.quantile(b_sh, 0.025)),
        'sharpe_p50': float(np.quantile(b_sh, 0.50)),
        'sharpe_p97_5': float(np.quantile(b_sh, 0.975)),
    }


def random_baseline_test(ret: np.ndarray, y_true: np.ndarray, levels_ref: np.ndarray, cost_ret: float, n_rand: int = 1000):
    # match empirical action distribution of reference strategy
    actions_ref = np.sign(levels_ref).astype(np.int64)
    vals, cnts = np.unique(actions_ref, return_counts=True)
    probs = cnts / cnts.sum()

    cum = np.zeros(n_rand, dtype=np.float64)
    sh = np.zeros(n_rand, dtype=np.float64)

    for i in range(n_rand):
        a = np.random.choice(vals, size=len(ret), p=probs)
        lev = a.astype(np.float32)
        net = lev * ret - np.abs(lev) * cost_ret
        cum[i] = net.sum()
        sh[i] = (net.mean() / (net.std() + 1e-12)) * np.sqrt(24.0 * 252.0)

    return {
        'n_rand': int(n_rand),
        'null_cum_mean': float(cum.mean()),
        'null_cum_p95': float(np.quantile(cum, 0.95)),
        'null_sharpe_mean': float(sh.mean()),
        'null_sharpe_p95': float(np.quantile(sh, 0.95)),
    }


def shift_alignment_nonoverlap(levels: np.ndarray, ret: np.ndarray, cost_ret: float, stride: int = 24, shifts=(-3, -1, 0, 1, 3)):
    if stride <= 1:
        a = levels
        r = ret
    else:
        a = levels[::stride]
        r = ret[::stride]

    out = {
        'stride': int(stride),
        'n_rows': int(len(a)),
        'results': [],
    }
    for k in shifts:
        if k > 0:
            lev = np.r_[np.zeros(k, dtype=np.float32), a[:-k]]
        elif k < 0:
            kk = -k
            lev = np.r_[a[kk:], np.zeros(kk, dtype=np.float32)]
        else:
            lev = a.copy()
        net = lev * r - np.abs(lev) * cost_ret
        m = net.mean()
        s = net.std()
        sh = (m / (s + 1e-12)) * np.sqrt(24.0 * 252.0)
        out['results'].append({
            'shift': int(k),
            'cum_net_return': float(net.sum()),
            'sharpe_ann': float(sh),
        })
    return out


def main():
    parser = argparse.ArgumentParser(description='Sanity checks for phase13 runs (leak-safe diagnostics).')
    parser.add_argument('--run_dir', required=True, type=str)
    parser.add_argument('--n_perm', default=2000, type=int)
    parser.add_argument('--n_boot', default=2000, type=int)
    parser.add_argument('--n_rand', default=1000, type=int)
    parser.add_argument('--block_len', default=24, type=int)
    parser.add_argument('--shift_stride', default=24, type=int)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    summary = json.loads((run_dir / 'phase13_summary.json').read_text())
    leak_audit = json.loads((run_dir / 'phase13_leak_audit.json').read_text())
    df = pd.read_parquet(run_dir / 'stage05_phase12_predictions.parquet')

    ts = pd.to_datetime(df['timestamp'], errors='coerce')
    strict_start = pd.to_datetime(leak_audit.get('strict_eval_start_exclusive'), errors='coerce')

    use = np.isfinite(df['future_return_h'].to_numpy(dtype=float))
    if pd.notna(strict_start):
        use &= (ts > strict_start).to_numpy()

    d = df.loc[use].copy()
    if len(d) < 200:
        raise RuntimeError(f'too few strict rows: {len(d)}')

    symbol = summary.get('symbol', 'GBPUSD')
    cost_pips = float(summary.get('quality', {}).get('stage04', {}).get('cost_pips', 1.2))
    cost_ret = cost_pips * pip_size_for_symbol(symbol)

    y = d['y_action'].to_numpy(dtype=np.int64)
    ret = d['future_return_h'].to_numpy(dtype=np.float32)

    levels = {
        'behavior_1x': d['pred_action'].to_numpy(dtype=np.float32),
        'behavior_2x': (2.0 * d['pred_action'].to_numpy(dtype=np.float32)),
        'phase12_2_frozen': d['lev_phase12_2_frozen'].to_numpy(dtype=np.float32),
    }

    out = {
        'run_dir': str(run_dir),
        'strict_rows_used': int(len(d)),
        'strict_start_exclusive': str(strict_start),
        'tests': {},
    }

    metrics = {}
    for name, lev in levels.items():
        m = evaluate_levels(lev, y, ret, cost_ret)
        metrics[name] = m

        net = lev * ret - np.abs(lev) * cost_ret
        out['tests'][name] = {
            'observed_metrics': m,
            'permutation_iid': permutation_test(lev, ret, cost_ret, m['cum_net_return'], m['sharpe_ann'], args.n_perm),
            'bootstrap_block': bootstrap_ci(net, n_boot=args.n_boot, block_len=args.block_len),
            'random_baseline': random_baseline_test(ret, y, lev, cost_ret, n_rand=args.n_rand),
            'shift_alignment_nonoverlap': shift_alignment_nonoverlap(lev, ret, cost_ret, stride=args.shift_stride),
        }

    # delta check vs behavior_1x
    net_b1 = levels['behavior_1x'] * ret - np.abs(levels['behavior_1x']) * cost_ret
    net_q = levels['phase12_2_frozen'] * ret - np.abs(levels['phase12_2_frozen']) * cost_ret
    delta = net_q - net_b1
    out['tests']['delta_phase12_minus_behavior1x'] = {
        'obs_cum_delta': float(delta.sum()),
        'obs_avg_delta': float(delta.mean()),
        'bootstrap_block': bootstrap_ci(delta, n_boot=args.n_boot, block_len=args.block_len),
    }

    out_json = run_dir / 'phase13_sanity_checks.json'
    out_json.write_text(json.dumps(out, indent=2))
    print('saved', out_json)

    # concise stdout
    print('strict_rows', len(d))
    for k in ['behavior_1x', 'phase12_2_frozen']:
        m = out['tests'][k]['observed_metrics']
        p = out['tests'][k]['permutation_iid']
        print(k, 'cum=', round(m['cum_net_return'], 6), 'sh=', round(m['sharpe_ann'], 4), 'p_cum=', p['pvalue_cum_ge_obs'], 'p_sh=', p['pvalue_sharpe_ge_obs'])


if __name__ == '__main__':
    main()
