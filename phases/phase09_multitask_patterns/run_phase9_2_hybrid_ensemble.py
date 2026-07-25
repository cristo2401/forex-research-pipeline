#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent

IN_SUM_V9 = ROOT / 'output_phase9_partial_pattern_GBPUSD/GBPUSD_phase9_summary.json'
IN_SUM_V91 = ROOT / 'output_phase9_1_partial_pattern_GBPUSD/GBPUSD_phase9_1_summary.json'
IN_PRED_V9 = ROOT / 'output_phase9_partial_pattern_GBPUSD/GBPUSD_phase9_test_predictions.csv'
IN_PRED_V91 = ROOT / 'output_phase9_1_partial_pattern_GBPUSD/GBPUSD_phase9_1_test_predictions.csv'

OUT_DIR = ROOT / 'output_phase9_2_hybrid_ensemble_GBPUSD'
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PRED_CSV = OUT_DIR / 'GBPUSD_phase9_2_hybrid_test_predictions.csv'
OUT_COMPARE_CSV = OUT_DIR / 'GBPUSD_phase9_2_model_comparison.csv'
OUT_SUMMARY_JSON = OUT_DIR / 'GBPUSD_phase9_2_summary.json'

MODELS_REGISTRY_DIR = ROOT / 'models_registry'
MODELS_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
OUT_REGISTRY_JSON = MODELS_REGISTRY_DIR / 'phase9_models_registry.json'


def compute_metrics(y_true_class, y_pred_class, y_true_prog, y_pred_prog, reg_mask, n_classes=3):
    out = {}
    out['acc'] = float((y_true_class == y_pred_class).mean())

    labels_all = list(range(n_classes))
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

    p9 = pd.read_csv(IN_PRED_V9)
    p91 = pd.read_csv(IN_PRED_V91)

    key = ['global_segment_id', 'step', 'timestamp']
    p9['timestamp'] = p9['timestamp'].astype(str)
    p91['timestamp'] = p91['timestamp'].astype(str)

    m = p9.merge(
        p91[key + ['pred_inf_prob', 'pred_class', 'pred_progress']],
        on=key,
        how='inner',
        suffixes=('_v9', '_v91'),
    )

    gate_thr = float(s91['setup']['best_threshold'])

    # Hybrid rule:
    # - gate informative/noise with v9.1
    # - if informative, subclass from v9 (better informative discrimination)
    pred_class_h = np.where(m['pred_inf_prob'].to_numpy() >= gate_thr, m['pred_class_v9'].to_numpy(), 0)

    # progress from v9 (better regression quality in previous runs)
    pred_prog_h = m['pred_progress_v9'].to_numpy()

    y_true = m['y_class'].to_numpy()
    y_prog_true = m['y_progress'].to_numpy()
    reg_mask = m['reg_mask'].to_numpy().astype(int)

    n_classes = int(max(np.max(y_true), np.max(pred_class_h))) + 1
    metrics_h = compute_metrics(y_true, pred_class_h, y_prog_true, pred_prog_h, reg_mask, n_classes=n_classes)

    out = m.copy()
    out['hybrid_threshold'] = gate_thr
    out['pred_class_hybrid'] = pred_class_h
    out['pred_progress_hybrid'] = pred_prog_h
    out.to_csv(OUT_PRED_CSV, index=False)

    cmp_rows = [
        {
            'model': 'phase9',
            **s9['test_metrics'],
        },
        {
            'model': 'phase9_1',
            **s91['test_metrics'],
        },
        {
            'model': 'phase9_2_hybrid',
            **metrics_h,
        },
    ]
    cmp_df = pd.DataFrame(cmp_rows)
    cmp_df.to_csv(OUT_COMPARE_CSV, index=False)

    registry = {
        'base_models': {
            'phase9': {
                'model_pt': s9['outputs']['model_pt'],
                'summary_json': str(IN_SUM_V9),
                'predictions_csv': str(IN_PRED_V9),
            },
            'phase9_1': {
                'model_pt': s91['outputs']['model_pt'],
                'summary_json': str(IN_SUM_V91),
                'predictions_csv': str(IN_PRED_V91),
            },
        },
        'hybrid_model': {
            'name': 'phase9_2_hybrid',
            'gating_source': 'phase9_1.pred_inf_prob',
            'gating_threshold': gate_thr,
            'informative_subclass_source': 'phase9.pred_class',
            'progress_source': 'phase9.pred_progress',
            'outputs': {
                'summary_json': str(OUT_SUMMARY_JSON),
                'predictions_csv': str(OUT_PRED_CSV),
                'comparison_csv': str(OUT_COMPARE_CSV),
            },
        },
    }
    OUT_REGISTRY_JSON.write_text(json.dumps(registry, indent=2))

    summary = {
        'symbol': s9['symbol'],
        'model_tag': 'phase9_2_hybrid_ensemble_gated',
        'description': 'phase9.1 gate (noise vs informative) + phase9 informative subclass/progress',
        'hybrid_threshold': gate_thr,
        'n_rows': int(len(out)),
        'test_metrics': metrics_h,
        'compare': cmp_rows,
        'outputs': {
            'hybrid_predictions_csv': str(OUT_PRED_CSV),
            'comparison_csv': str(OUT_COMPARE_CSV),
            'models_registry_json': str(OUT_REGISTRY_JSON),
        },
    }

    OUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

    print('Saved summary:', OUT_SUMMARY_JSON)
    print('Hybrid metrics:', metrics_h)
    return summary, cmp_df, out


if __name__ == '__main__':
    main()
