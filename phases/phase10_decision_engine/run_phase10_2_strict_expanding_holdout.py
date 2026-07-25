#!/usr/bin/env python3
import importlib.util
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score

SEED = 42
np.random.seed(SEED)


@dataclass
class StrictConfig:
    symbol: str = 'GBPUSD'

    holdout_ratio: float = 0.15
    pre_min_train_ratio: float = 0.45
    pre_block_ratio: float = 0.07
    max_windows: int = 6

    min_train_rows: int = 4000
    min_test_rows: int = 700


ROOT = Path(__file__).resolve().parents[2]
PHASE10_DIR = ROOT / 'phases/phase10_decision_engine'
PHASE10_1_SCRIPT = PHASE10_DIR / 'run_phase10_1_decision_engine_leak_safe.py'
OUT_DIR = PHASE10_DIR / 'output_phase10_2_strict_expanding_GBPUSD'
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_WINDOWS_CSV = OUT_DIR / 'GBPUSD_phase10_2_windows_metrics.csv'
OUT_HOLDOUT_CSV = OUT_DIR / 'GBPUSD_phase10_2_holdout_metrics.csv'
OUT_PHASE9_CSV = OUT_DIR / 'GBPUSD_phase10_2_phase9_metrics.csv'
OUT_PRED_WINDOWS_CSV = OUT_DIR / 'GBPUSD_phase10_2_windows_predictions.csv'
OUT_PRED_HOLDOUT_CSV = OUT_DIR / 'GBPUSD_phase10_2_holdout_predictions.csv'
OUT_SUMMARY_JSON = OUT_DIR / 'GBPUSD_phase10_2_summary.json'
OUT_WINDOWS_PNG = OUT_DIR / 'GBPUSD_phase10_2_windows_equity.png'
OUT_HOLDOUT_PNG = OUT_DIR / 'GBPUSD_phase10_2_holdout_equity.png'
OUT_BAR_PNG = OUT_DIR / 'GBPUSD_phase10_2_windows_bar_metrics.png'


def load_phase10_1_module(path: Path):
    spec = importlib.util.spec_from_file_location('phase10_1_mod', str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot load module from {path}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_expanding_windows(n_pre: int, cfg10, cfgs: StrictConfig) -> List[Dict[str, int]]:
    train_end = max(int(n_pre * cfgs.pre_min_train_ratio), cfgs.min_train_rows)
    block_size = max(int(n_pre * cfgs.pre_block_ratio), cfgs.min_test_rows)

    windows: List[Dict[str, int]] = []
    for wid in range(1, cfgs.max_windows + 1):
        test_start = train_end + cfg10.embargo_bars
        test_end = test_start + block_size
        if test_end > n_pre:
            break

        windows.append(
            {
                'window_id': wid,
                'train_start': 0,
                'train_end': int(train_end),
                'test_start': int(test_start),
                'test_end': int(test_end),
            }
        )
        # expanding window: next train includes previous test block
        train_end = test_end

    return windows


def evaluate_split(
    split_name: str,
    tr: pd.DataFrame,
    te: pd.DataFrame,
    module,
    cfg10,
    x_all: np.ndarray,
    lengths_all: np.ndarray,
    y_class_all: np.ndarray,
    y_inf_all: np.ndarray,
    y_sub_all: np.ndarray,
    y_prog_all: np.ndarray,
    reg_mask_all: np.ndarray,
) -> Tuple[Dict[str, float], pd.DataFrame, Dict[str, float]]:
    feature_cols = [
        'pred_inf_prob',
        'pred_progress',
        'pred_sub_conf',
        'pred_sub_entropy',
        'pred_class',
        'ret_1',
        'ret_2',
        'ret_4',
        'vol_4',
        'range_1',
        'body_1',
        'segment_len',
        'step_frac',
    ]

    train_master_idx = tr['master_idx'].to_numpy(dtype=np.int64)
    model9, thr9, m9 = module.train_phase9_fold(
        cfg=cfg10,
        x=x_all,
        lengths=lengths_all,
        y_class=y_class_all,
        y_inf=y_inf_all,
        y_sub=y_sub_all,
        y_prog=y_prog_all,
        reg_mask=reg_mask_all,
        train_idx=train_master_idx,
    )

    tr_idx = tr['master_idx'].to_numpy(dtype=np.int64)
    te_idx = te['master_idx'].to_numpy(dtype=np.int64)

    tr_sig = module.infer_phase9_fold(model9, cfg10, x_all[tr_idx], lengths_all[tr_idx], thr9)
    te_sig = module.infer_phase9_fold(model9, cfg10, x_all[te_idx], lengths_all[te_idx], thr9)

    tr = tr.copy()
    te = te.copy()
    for k in tr_sig:
        tr[k] = tr_sig[k]
        te[k] = te_sig[k]

    calib_n = max(300, int(len(tr) * cfg10.calib_ratio))
    if calib_n >= len(tr):
        calib_n = max(100, len(tr) // 5)

    fit = tr.iloc[:-calib_n].copy()
    calib = tr.iloc[-calib_n:].copy()

    x_fit = fit[feature_cols].to_numpy(dtype=np.float32)
    y_fit = fit['y_action_idx'].to_numpy(dtype=np.int64)

    cw = module.class_weights(y_fit)
    sw = np.array([cw[int(v)] for v in y_fit], dtype=np.float32)

    clf = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=6,
        max_iter=280,
        min_samples_leaf=80,
        l2_regularization=1e-3,
        random_state=SEED,
    )
    clf.fit(x_fit, y_fit, sample_weight=sw)

    p_cal = clf.predict_proba(calib[feature_cols].to_numpy(dtype=np.float32))
    conf_thr, inf_min, cal_best = module.tune_policy_thresholds(calib, p_cal, cfg10)

    x_tr = tr[feature_cols].to_numpy(dtype=np.float32)
    y_tr = tr['y_action_idx'].to_numpy(dtype=np.int64)
    cw2 = module.class_weights(y_tr)
    sw2 = np.array([cw2[int(v)] for v in y_tr], dtype=np.float32)
    clf.fit(x_tr, y_tr, sample_weight=sw2)

    x_te = te[feature_cols].to_numpy(dtype=np.float32)
    p_te = clf.predict_proba(x_te)
    a_te = module.apply_policy(p_te, conf_thr, te['pred_inf_prob'].to_numpy(), inf_min)

    y_true = te['y_action'].to_numpy(dtype=np.int64)
    acc = float((y_true == a_te).mean())
    f1m = float(f1_score(y_true, a_te, labels=[-1, 0, 1], average='macro', zero_division=0))

    cost_ret = cfg10.cost_pips * module.pip_size_for_symbol(cfg10.symbol)
    tm = module.evaluate_trading(a_te, te['future_return_h'].to_numpy(), cost_ret)

    m = {
        'split': split_name,
        'train_rows': int(len(tr)),
        'test_rows': int(len(te)),
        'timestamp_test_start': str(te['timestamp'].iloc[0]),
        'timestamp_test_end': str(te['timestamp'].iloc[-1]),
        'phase9_thr': float(thr9),
        'policy_conf_thr': float(conf_thr),
        'policy_inf_min': float(inf_min),
        'policy_calib_score': float(cal_best['score']),
        'acc_action_3': acc,
        'f1_macro_action_3': f1m,
        **tm,
    }

    preds = te[
        [
            'global_segment_id',
            'timestamp',
            'step',
            'y_action',
            'future_return_h',
            'future_pips_h',
            'pred_inf_prob',
            'pred_progress',
            'pred_sub_conf',
            'pred_sub_entropy',
            'pred_class',
        ]
    ].copy()
    preds['split'] = split_name
    preds['pred_action'] = a_te.astype(int)
    preds['pred_conf_max'] = np.max(p_te, axis=1)
    preds['p_short'] = p_te[:, 0]
    preds['p_flat'] = p_te[:, 1]
    preds['p_long'] = p_te[:, 2]
    preds['net_return'] = preds['pred_action'] * preds['future_return_h'] - (preds['pred_action'] != 0).astype(float) * cost_ret

    phase9_fold_metrics = {'split': split_name, **m9}
    return m, preds, phase9_fold_metrics


def plot_curves(preds_windows: pd.DataFrame, preds_holdout: pd.DataFrame, windows_metrics: pd.DataFrame):
    pw = preds_windows.sort_values('timestamp').copy()
    pw['cum_net'] = pw['net_return'].cumsum()

    plt.figure(figsize=(11, 4))
    plt.plot(pd.to_datetime(pw['timestamp']), pw['cum_net'], linewidth=1.6)
    plt.title('Phase10.2 - Expanding windows cumulative net return')
    plt.xlabel('timestamp')
    plt.ylabel('cumulative return')
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_WINDOWS_PNG, dpi=140)
    plt.close()

    ph = preds_holdout.sort_values('timestamp').copy()
    ph['cum_net'] = ph['net_return'].cumsum()

    plt.figure(figsize=(11, 4))
    plt.plot(pd.to_datetime(ph['timestamp']), ph['cum_net'], linewidth=1.6, color='tab:orange')
    plt.title('Phase10.2 - Final holdout cumulative net return')
    plt.xlabel('timestamp')
    plt.ylabel('cumulative return')
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_HOLDOUT_PNG, dpi=140)
    plt.close()

    x = np.arange(len(windows_metrics))
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.bar(x - 0.15, windows_metrics['cum_net_return'], width=0.3)
    ax1.set_ylabel('cum_net_return')
    ax1.set_xlabel('window')
    ax1.set_xticks(x)
    ax1.set_xticklabels(windows_metrics['split'].tolist(), rotation=35, ha='right')

    ax2 = ax1.twinx()
    ax2.plot(x + 0.15, windows_metrics['sharpe_ann'], marker='o', color='tab:red')
    ax2.set_ylabel('sharpe_ann')

    ax1.set_title('Phase10.2 - Window metrics (net return & Sharpe)')
    ax1.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT_BAR_PNG, dpi=140)
    plt.close(fig)


def main():
    cfgs = StrictConfig()
    module = load_phase10_1_module(PHASE10_1_SCRIPT)

    cfg10 = module.Config()
    cfg10.epochs_phase9 = 8
    cfg10.patience_phase9 = 3
    cfg10.embargo_bars = max(int(cfg10.embargo_bars), 16)

    master, x_all, lengths_all = module.load_master(cfg10)

    samples_full = pd.read_csv(module.IN_SAMPLES_CSV)
    y_class_all = samples_full['y_class'].to_numpy(dtype=np.int64)
    y_inf_all = samples_full['y_inf'].to_numpy(dtype=np.int64)
    y_sub_all = samples_full['y_sub'].to_numpy(dtype=np.int64)
    y_prog_all = samples_full['y_progress'].to_numpy(dtype=np.float32)
    reg_mask_all = samples_full['reg_mask'].to_numpy(dtype=np.int64)

    n_total = len(master)
    holdout_start = int(n_total * (1.0 - cfgs.holdout_ratio))
    holdout_start = max(holdout_start, cfgs.min_train_rows + cfg10.embargo_bars + cfgs.min_test_rows)

    pre = master.iloc[:holdout_start].copy().reset_index(drop=True)
    holdout = master.iloc[holdout_start:].copy().reset_index(drop=True)

    windows = build_expanding_windows(len(pre), cfg10, cfgs)
    if not windows:
        raise RuntimeError('No expanding windows generated; adjust ratios/min rows.')

    win_rows = []
    phase9_rows = []
    win_preds = []

    for w in windows:
        tr = pre.iloc[w['train_start'] : w['train_end']].copy()
        te = pre.iloc[w['test_start'] : w['test_end']].copy()

        m, p, pm = evaluate_split(
            split_name=f"w{w['window_id']}",
            tr=tr,
            te=te,
            module=module,
            cfg10=cfg10,
            x_all=x_all,
            lengths_all=lengths_all,
            y_class_all=y_class_all,
            y_inf_all=y_inf_all,
            y_sub_all=y_sub_all,
            y_prog_all=y_prog_all,
            reg_mask_all=reg_mask_all,
        )
        win_rows.append(m)
        phase9_rows.append(pm)
        win_preds.append(p)

    windows_metrics = pd.DataFrame(win_rows)
    phase9_metrics = pd.DataFrame(phase9_rows)
    preds_windows = pd.concat(win_preds, axis=0, ignore_index=True)

    # final holdout: strictly train until pre-holdout with embargo gap
    hold_train_end = max(cfgs.min_train_rows, holdout_start - cfg10.embargo_bars)
    tr_final = master.iloc[:hold_train_end].copy()
    te_holdout = holdout.copy()

    hold_metrics, preds_holdout, phase9_hold = evaluate_split(
        split_name='holdout_final',
        tr=tr_final,
        te=te_holdout,
        module=module,
        cfg10=cfg10,
        x_all=x_all,
        lengths_all=lengths_all,
        y_class_all=y_class_all,
        y_inf_all=y_inf_all,
        y_sub_all=y_sub_all,
        y_prog_all=y_prog_all,
        reg_mask_all=reg_mask_all,
    )

    holdout_metrics_df = pd.DataFrame([hold_metrics])
    phase9_metrics = pd.concat([phase9_metrics, pd.DataFrame([phase9_hold])], ignore_index=True)

    windows_metrics.to_csv(OUT_WINDOWS_CSV, index=False)
    holdout_metrics_df.to_csv(OUT_HOLDOUT_CSV, index=False)
    phase9_metrics.to_csv(OUT_PHASE9_CSV, index=False)
    preds_windows.to_csv(OUT_PRED_WINDOWS_CSV, index=False)
    preds_holdout.to_csv(OUT_PRED_HOLDOUT_CSV, index=False)

    plot_curves(preds_windows, preds_holdout, windows_metrics)

    summary = {
        'symbol': cfgs.symbol,
        'model_tag': 'phase10_2_strict_expanding_holdout',
        'description': 'Strict expanding-window plus final untouched holdout (leak-safe)',
        'strict_config': asdict(cfgs),
        'phase10_config': {
            'epochs_phase9': int(cfg10.epochs_phase9),
            'embargo_bars': int(cfg10.embargo_bars),
            'horizon_bars': int(cfg10.horizon_bars),
            'cost_pips': float(cfg10.cost_pips),
        },
        'dataset': {
            'n_total_rows': int(n_total),
            'n_pre_rows': int(len(pre)),
            'n_holdout_rows': int(len(holdout)),
            'pre_date_min': str(pre['timestamp'].min()),
            'pre_date_max': str(pre['timestamp'].max()),
            'holdout_date_min': str(holdout['timestamp'].min()),
            'holdout_date_max': str(holdout['timestamp'].max()),
        },
        'windows': {
            'n_windows': int(len(windows_metrics)),
            'mean_metrics': windows_metrics.mean(numeric_only=True).to_dict(),
            'std_metrics': windows_metrics.std(numeric_only=True).to_dict(),
        },
        'holdout_final': hold_metrics,
        'outputs': {
            'windows_metrics_csv': str(OUT_WINDOWS_CSV),
            'holdout_metrics_csv': str(OUT_HOLDOUT_CSV),
            'phase9_metrics_csv': str(OUT_PHASE9_CSV),
            'pred_windows_csv': str(OUT_PRED_WINDOWS_CSV),
            'pred_holdout_csv': str(OUT_PRED_HOLDOUT_CSV),
            'windows_equity_png': str(OUT_WINDOWS_PNG),
            'holdout_equity_png': str(OUT_HOLDOUT_PNG),
            'windows_bar_png': str(OUT_BAR_PNG),
        },
    }

    OUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

    print('Saved summary:', OUT_SUMMARY_JSON)
    print('Windows mean metrics:')
    print(windows_metrics.mean(numeric_only=True).to_dict())
    print('Holdout metrics:')
    print(hold_metrics)


if __name__ == '__main__':
    main()
