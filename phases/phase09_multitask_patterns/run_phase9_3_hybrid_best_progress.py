#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent

IN_SUM_V9 = ROOT / 'output_phase9_partial_pattern_GBPUSD/GBPUSD_phase9_summary.json'
IN_SUM_V91 = ROOT / 'output_phase9_1_partial_pattern_GBPUSD/GBPUSD_phase9_1_summary.json'
IN_SUM_V92 = ROOT / 'output_phase9_2_hybrid_ensemble_GBPUSD/GBPUSD_phase9_2_summary.json'
IN_SUM_V93 = ROOT / 'output_phase9_3_search_best_GBPUSD/GBPUSD_phase9_3_summary.json'

IN_PRED_V9 = ROOT / 'output_phase9_partial_pattern_GBPUSD/GBPUSD_phase9_test_predictions.csv'
IN_PRED_V93 = ROOT / 'output_phase9_3_search_best_GBPUSD/GBPUSD_phase9_3_best_test_predictions.csv'

OUT_DIR = ROOT / 'output_phase9_3_hybrid_best_progress_GBPUSD'
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PRED_CSV = OUT_DIR / 'GBPUSD_phase9_3_hybrid_test_predictions.csv'
OUT_COMPARE_CSV = OUT_DIR / 'GBPUSD_phase9_3_hybrid_model_comparison.csv'
OUT_SUMMARY_JSON = OUT_DIR / 'GBPUSD_phase9_3_hybrid_summary.json'


def compute_metrics(y_true_class, y_pred_class, y_true_prog, y_pred_prog, reg_mask):
    n_classes = int(max(np.max(y_true_class), np.max(y_pred_class))) + 1
    labels_all = list(range(n_classes))

    out = {}
    out['acc'] = float((y_true_class == y_pred_class).mean())
    out['f1_macro_all'] = float(f1_score(y_true_class, y_pred_class, labels=labels_all, average='macro', zero_division=0))

    inf = y_true_class > 0
    labels_inf = sorted(list(set(y_true_class[inf].tolist()))) if inf.any() else [1, 2]
    out['f1_macro_informative'] = float(f1_score(y_true_class[inf], y_pred_class[inf], labels=labels_inf, average='macro', zero_division=0)) if inf.any() else np.nan

    out['recall_class0'] = float((y_pred_class[y_true_class == 0] == 0).mean()) if (y_true_class == 0).any() else np.nan

    m = reg_mask.astype(bool)
    if m.any():
        err = np.abs(y_true_prog[m] - y_pred_prog[m])
        out['mae_progress'] = float(err.mean())
        out['rmse_progress'] = float(np.sqrt(np.mean((y_true_prog[m] - y_pred_prog[m]) ** 2)))
        early = m & (y_true_prog < 0.5)
        out['mae_progress_early'] = float(np.abs(y_true_prog[early] - y_pred_prog[early]).mean()) if early.any() else np.nan
    else:
        out['mae_progress'] = np.nan
        out['rmse_progress'] = np.nan
        out['mae_progress_early'] = np.nan

    out['n_informative'] = int((y_true_class > 0).sum())
    return out


def main():
    s9 = json.loads(IN_SUM_V9.read_text())
    s91 = json.loads(IN_SUM_V91.read_text())
    s92 = json.loads(IN_SUM_V92.read_text())
    s93 = json.loads(IN_SUM_V93.read_text())

    p9 = pd.read_csv(IN_PRED_V9)
    p93 = pd.read_csv(IN_PRED_V93)

    key = ['global_segment_id', 'step', 'timestamp']
    p9['timestamp'] = p9['timestamp'].astype(str)
    p93['timestamp'] = p93['timestamp'].astype(str)

    m = p93.merge(p9[key + ['pred_progress']], on=key, how='inner', suffixes=('_v93', '_v9'))

    # class from phase9.3, progress from phase9
    pred_class = m['pred_class'].to_numpy()
    pred_prog = m['pred_progress_v9'].to_numpy()

    y_true = m['y_class'].to_numpy()
    y_prog_true = m['y_progress'].to_numpy()
    reg_mask = m['reg_mask'].to_numpy().astype(int)

    metrics = compute_metrics(y_true, pred_class, y_prog_true, pred_prog, reg_mask)

    out = m.copy()
    out['pred_class_hybrid'] = pred_class
    out['pred_progress_hybrid'] = pred_prog
    out.to_csv(OUT_PRED_CSV, index=False)

    rows = [
        {'model': 'phase9', **s9['test_metrics']},
        {'model': 'phase9_1', **s91['test_metrics']},
        {'model': 'phase9_2_hybrid', **s92['test_metrics']},
        {'model': 'phase9_3', **s93['test_metrics']},
        {'model': 'phase9_3_hybrid_best_progress', **metrics},
    ]
    cmp = pd.DataFrame(rows)
    cmp.to_csv(OUT_COMPARE_CSV, index=False)

    summary = {
        'symbol': s9['symbol'],
        'model_tag': 'phase9_3_hybrid_best_progress',
        'description': 'class from phase9.3 + progress from phase9',
        'n_rows': int(len(out)),
        'test_metrics': metrics,
        'compare': rows,
        'outputs': {
            'predictions_csv': str(OUT_PRED_CSV),
            'comparison_csv': str(OUT_COMPARE_CSV),
        },
    }
    OUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

    print('Saved summary:', OUT_SUMMARY_JSON)
    print('Hybrid metrics:', metrics)
    return summary, cmp


if __name__ == '__main__':
    main()
