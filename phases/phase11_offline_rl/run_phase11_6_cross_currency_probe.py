#!/usr/bin/env python3
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

SEED = 42
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[2]
PHASE11_DIR = ROOT / 'phases/phase11_offline_rl'

IN_GBP = ROOT / 'phases/phase05_context_sequencing/output_superclustering_contextual_GBPUSD_v5_2_chronos_pretrained/GBPUSD_context_window_embeddings_v5_2.csv'
IN_EUR = ROOT / 'phases/phase05_context_sequencing/output_superclustering_contextual_EURUSD_v5_3_chronos_pretrained/EURUSD_context_window_embeddings_v5_3.csv'

OUT_DIR = PHASE11_DIR / 'output_phase11_6_cross_currency_probe'
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_EXPERIMENTS_CSV = OUT_DIR / 'phase11_6_experiments.csv'
OUT_PRED_CSV = OUT_DIR / 'phase11_6_predictions.csv'
OUT_SUMMARY_JSON = OUT_DIR / 'phase11_6_summary.json'
OUT_PNG_CUM = OUT_DIR / 'phase11_6_cum_net_pips_by_experiment.png'
OUT_PNG_TRADE = OUT_DIR / 'phase11_6_trade_rate_by_experiment.png'

RET_PIPS_COL = 'ret_fwd_24h_pips'
COST_PIPS = 1.2
NOISE_PIPS = 20.0

C_GRID = [0.1, 0.3, 1.0, 3.0, 10.0]
CONF_GRID = [0.34, 0.40, 0.46, 0.52, 0.58, 0.64, 0.70]


@dataclass
class RunSpec:
    name: str
    train_symbols: Tuple[str, ...]
    val_symbols: Tuple[str, ...]
    test_symbol: str


RUNS = [
    RunSpec('intra_GBP_to_GBP', ('GBPUSD',), ('GBPUSD',), 'GBPUSD'),
    RunSpec('intra_EUR_to_EUR', ('EURUSD',), ('EURUSD',), 'EURUSD'),
    RunSpec('cross_GBP_to_EUR', ('GBPUSD',), ('GBPUSD',), 'EURUSD'),
    RunSpec('cross_EUR_to_GBP', ('EURUSD',), ('EURUSD',), 'GBPUSD'),
    RunSpec('pooled_to_GBP', ('GBPUSD', 'EURUSD'), ('GBPUSD', 'EURUSD'), 'GBPUSD'),
    RunSpec('pooled_to_EUR', ('GBPUSD', 'EURUSD'), ('GBPUSD', 'EURUSD'), 'EURUSD'),
]


def label_from_ret_pips(ret: np.ndarray, noise: float) -> np.ndarray:
    y = np.zeros_like(ret, dtype=np.int64)
    y[ret > noise] = 1
    y[ret < -noise] = -1
    return y


def metrics_from_actions(actions: np.ndarray, y_true: np.ndarray, ret_pips: np.ndarray, cost_pips: float) -> Dict[str, float]:
    net = actions * ret_pips - (actions != 0).astype(np.float32) * float(cost_pips)
    trade = (actions != 0).astype(np.float32)

    out = {
        'acc_action_3': float((actions == y_true).mean()),
        'f1_macro_action_3': float(f1_score(y_true, actions, labels=[-1, 0, 1], average='macro', zero_division=0)),
        'avg_net_pips': float(np.mean(net)),
        'cum_net_pips': float(np.sum(net)),
        'std_net_pips': float(np.std(net)),
        'trade_rate': float(np.mean(trade)),
    }

    eps = 1e-12
    out['sharpe_like'] = float((out['avg_net_pips'] / (out['std_net_pips'] + eps)) * np.sqrt(252.0))
    traded = trade > 0
    out['win_rate_trades'] = float((net[traded] > 0).mean()) if traded.any() else 0.0
    return out


def predict_actions_prob(clf: LogisticRegression, x: np.ndarray, conf_thr: float) -> Tuple[np.ndarray, np.ndarray]:
    p = clf.predict_proba(x)
    classes = clf.classes_.astype(int)

    idx = np.argmax(p, axis=1)
    conf = p[np.arange(len(p)), idx]
    pred = classes[idx]
    pred = np.where(conf >= float(conf_thr), pred, 0)
    return pred.astype(np.int64), conf.astype(np.float32)


def score_for_model(m: Dict[str, float]) -> float:
    return float(m['avg_net_pips']) + 0.05 * float(m['sharpe_like'])


def fit_and_calibrate(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    ret_val: np.ndarray,
) -> Dict:
    best = None

    for c in C_GRID:
        clf = LogisticRegression(
            C=float(c),
            max_iter=1200,
            class_weight='balanced',
            random_state=SEED,
            solver='lbfgs',
        )
        clf.fit(x_train, y_train)

        for conf_thr in CONF_GRID:
            a_val, _ = predict_actions_prob(clf, x_val, conf_thr=conf_thr)
            m = metrics_from_actions(a_val, y_val, ret_val, cost_pips=COST_PIPS)
            score = score_for_model(m)

            row = {
                'C': float(c),
                'conf_thr': float(conf_thr),
                'score': float(score),
                'clf': clf,
                'metrics_val': m,
            }
            if best is None or row['score'] > best['score']:
                best = row

    return best


def main():
    gbp = pd.read_csv(IN_GBP)
    eur = pd.read_csv(IN_EUR)

    gbp['symbol'] = 'GBPUSD'
    eur['symbol'] = 'EURUSD'

    df = pd.concat([gbp, eur], axis=0, ignore_index=True).copy()

    emb_cols = [c for c in df.columns if c.startswith('ctx_emb_')]
    feature_cols = emb_cols + ['regime_id', 'regime_dist']

    df['ret_pips'] = df[RET_PIPS_COL].astype(float)
    df['y_action'] = label_from_ret_pips(df['ret_pips'].to_numpy(dtype=np.float32), noise=NOISE_PIPS)

    experiments: List[Dict] = []
    pred_rows: List[Dict] = []

    for run in RUNS:
        tr = df[(df['symbol'].isin(run.train_symbols)) & (df['split'] == 'train')].copy()
        va = df[(df['symbol'].isin(run.val_symbols)) & (df['split'] == 'val')].copy()
        te = df[(df['symbol'] == run.test_symbol) & (df['split'] == 'test')].copy()

        x_tr_raw = tr[feature_cols].to_numpy(dtype=np.float32)
        x_va_raw = va[feature_cols].to_numpy(dtype=np.float32)
        x_te_raw = te[feature_cols].to_numpy(dtype=np.float32)

        scaler = StandardScaler()
        x_tr = scaler.fit_transform(x_tr_raw)
        x_va = scaler.transform(x_va_raw)
        x_te = scaler.transform(x_te_raw)

        y_tr = tr['y_action'].to_numpy(dtype=np.int64)
        y_va = va['y_action'].to_numpy(dtype=np.int64)
        y_te = te['y_action'].to_numpy(dtype=np.int64)

        ret_va = va['ret_pips'].to_numpy(dtype=np.float32)
        ret_te = te['ret_pips'].to_numpy(dtype=np.float32)

        best = fit_and_calibrate(x_tr, y_tr, x_va, y_va, ret_va)
        clf = best['clf']

        a_test, conf_test = predict_actions_prob(clf, x_te, conf_thr=best['conf_thr'])
        m_test = metrics_from_actions(a_test, y_te, ret_te, cost_pips=COST_PIPS)

        experiments.append({
            'run': run.name,
            'train_symbols': ','.join(run.train_symbols),
            'val_symbols': ','.join(run.val_symbols),
            'test_symbol': run.test_symbol,
            'n_train': int(len(tr)),
            'n_val': int(len(va)),
            'n_test': int(len(te)),
            'best_C': float(best['C']),
            'best_conf_thr': float(best['conf_thr']),
            'val_score': float(best['score']),
            'val_avg_net_pips': float(best['metrics_val']['avg_net_pips']),
            'val_cum_net_pips': float(best['metrics_val']['cum_net_pips']),
            'test_acc_action_3': float(m_test['acc_action_3']),
            'test_f1_macro_action_3': float(m_test['f1_macro_action_3']),
            'test_avg_net_pips': float(m_test['avg_net_pips']),
            'test_cum_net_pips': float(m_test['cum_net_pips']),
            'test_sharpe_like': float(m_test['sharpe_like']),
            'test_trade_rate': float(m_test['trade_rate']),
            'test_win_rate_trades': float(m_test['win_rate_trades']),
        })

        tmp = te[['window_id', 'symbol', 'start_ts', 'end_ts', 'split', 'ret_pips', 'y_action']].copy()
        tmp['run'] = run.name
        tmp['pred_action'] = a_test
        tmp['pred_conf'] = conf_test
        tmp['net_pips'] = tmp['pred_action'] * tmp['ret_pips'] - (tmp['pred_action'] != 0).astype(float) * COST_PIPS
        tmp['cum_net_pips'] = tmp['net_pips'].cumsum()
        pred_rows.append(tmp)

    exp_df = pd.DataFrame(experiments).sort_values('test_cum_net_pips', ascending=False)
    pred_df = pd.concat(pred_rows, axis=0, ignore_index=True)

    exp_df.to_csv(OUT_EXPERIMENTS_CSV, index=False)
    pred_df.to_csv(OUT_PRED_CSV, index=False)

    # Plots
    plt.figure(figsize=(10, 4))
    x = np.arange(len(exp_df))
    plt.bar(x, exp_df['test_cum_net_pips'])
    plt.xticks(x, exp_df['run'], rotation=30, ha='right')
    plt.ylabel('test cum net pips')
    plt.title('Phase11.6 cross-currency probe: cumulative net pips')
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_PNG_CUM, dpi=140)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.bar(x, exp_df['test_trade_rate'])
    plt.xticks(x, exp_df['run'], rotation=30, ha='right')
    plt.ylabel('trade rate')
    plt.title('Phase11.6 cross-currency probe: trade rate')
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_PNG_TRADE, dpi=140)
    plt.close()

    summary = {
        'model_tag': 'phase11_6_cross_currency_probe',
        'description': 'Cross-currency probe using contextual embeddings from GBPUSD/EURUSD',
        'seed': SEED,
        'cost_pips': COST_PIPS,
        'noise_pips_for_label': NOISE_PIPS,
        'n_rows_total': int(len(df)),
        'symbols': sorted(df['symbol'].unique().tolist()),
        'split_counts': df.groupby(['symbol', 'split']).size().unstack(fill_value=0).to_dict(),
        'features': {
            'n_features': int(len(feature_cols)),
            'first_features': feature_cols[:10],
        },
        'experiments_ranked': exp_df.to_dict(orient='records'),
        'outputs': {
            'experiments_csv': str(OUT_EXPERIMENTS_CSV),
            'predictions_csv': str(OUT_PRED_CSV),
            'cum_plot_png': str(OUT_PNG_CUM),
            'trade_plot_png': str(OUT_PNG_TRADE),
        },
    }

    OUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

    print('Saved summary:', OUT_SUMMARY_JSON)
    print('Top run:', exp_df.iloc[0]['run'])
    print('Top test_cum_net_pips:', float(exp_df.iloc[0]['test_cum_net_pips']))


if __name__ == '__main__':
    main()
