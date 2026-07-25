#!/usr/bin/env python3
import importlib.util
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

ROOT = Path(__file__).resolve().parents[2]
PHASE11_DIR = ROOT / 'phases/phase11_offline_rl'
PHASE10_OUT_DIR = ROOT / 'phases/phase10_decision_engine/output_phase10_2_strict_expanding_GBPUSD'
PHASE11_BASE_SCRIPT = PHASE11_DIR / 'run_phase11_offline_rl_cql.py'
PHASE11_2_SCRIPT = PHASE11_DIR / 'run_phase11_2_offline_rl_reward_input.py'
PHASE11_2_SUMMARY = PHASE11_DIR / 'output_phase11_2_offline_rl_reward_input_GBPUSD/GBPUSD_phase11_2_summary.json'
PHASE11_2_MODEL_CQL = PHASE11_DIR / 'output_phase11_2_offline_rl_reward_input_GBPUSD/phase11_2_best_cql_model.pt'

IN_WINDOWS = PHASE10_OUT_DIR / 'GBPUSD_phase10_2_windows_predictions.csv'
IN_HOLDOUT = PHASE10_OUT_DIR / 'GBPUSD_phase10_2_holdout_predictions.csv'

OUT_DIR = PHASE11_DIR / 'output_phase11_3_meta_policy_GBPUSD'
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_COMPARE_CSV = OUT_DIR / 'GBPUSD_phase11_3_policy_comparison.csv'
OUT_HOLDOUT_PRED_CSV = OUT_DIR / 'GBPUSD_phase11_3_holdout_predictions.csv'
OUT_META_GRID_CSV = OUT_DIR / 'GBPUSD_phase11_3_meta_threshold_grid.csv'
OUT_META_COEF_CSV = OUT_DIR / 'GBPUSD_phase11_3_meta_coefficients.csv'
OUT_META_MODEL_PKL = OUT_DIR / 'phase11_3_meta_gate.pkl'
OUT_EQUITY_PNG = OUT_DIR / 'GBPUSD_phase11_3_holdout_equity_compare.png'
OUT_PROBA_PNG = OUT_DIR / 'GBPUSD_phase11_3_gate_probability_hist.png'
OUT_SUMMARY_JSON = OUT_DIR / 'GBPUSD_phase11_3_summary.json'


ACTION_TO_IDX = {-1: 0, 0: 1, 1: 2}
IDX_TO_ACTION = {0: -1, 1: 0, 2: 1}


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot load module {path}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def action_to_idx(actions: np.ndarray) -> np.ndarray:
    return np.array([ACTION_TO_IDX[int(a)] for a in actions], dtype=np.int64)


def idx_to_action(idxs: np.ndarray) -> np.ndarray:
    return np.array([IDX_TO_ACTION[int(i)] for i in idxs], dtype=np.int64)


def q_logits_np(model, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        bs = 2048
        for i in range(0, len(x), bs):
            xb = torch.tensor(x[i:i + bs], dtype=torch.float32, device=device)
            q = model(xb)
            out.append(q.cpu().numpy())
    return np.concatenate(out, axis=0)


def cql_actions_from_logits(q_logits: np.ndarray, conf_thr: float) -> np.ndarray:
    m_idx = np.argmax(q_logits, axis=1)
    exp = np.exp(q_logits - q_logits.max(axis=1, keepdims=True))
    p = exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-8)
    conf = p.max(axis=1)
    act = idx_to_action(m_idx)
    act = np.where(conf >= float(conf_thr), act, 0)
    return act.astype(np.int64)


def compute_net(actions: np.ndarray, future_ret: np.ndarray, cost_ret: float) -> np.ndarray:
    return actions * future_ret - (actions != 0).astype(np.float32) * float(cost_ret)


def build_meta_features(df: pd.DataFrame, q_logits: np.ndarray, behavior_actions: np.ndarray, hybrid_actions: np.ndarray) -> tuple[np.ndarray, list[str], np.ndarray]:
    b_idx = action_to_idx(behavior_actions)
    m_idx = np.argmax(q_logits, axis=1)

    q_best = q_logits[np.arange(len(q_logits)), m_idx]
    q_beh = q_logits[np.arange(len(q_logits)), b_idx]
    q_adv = q_best - q_beh

    exp = np.exp(q_logits - q_logits.max(axis=1, keepdims=True))
    p = exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-8)
    q_conf = p.max(axis=1)

    p_sorted = np.sort(p, axis=1)
    q_margin = p_sorted[:, -1] - p_sorted[:, -2]

    changed = (hybrid_actions != behavior_actions).astype(np.float32)
    behavior_nonflat = (behavior_actions != 0).astype(np.float32)
    hybrid_nonflat = (hybrid_actions != 0).astype(np.float32)

    X = pd.DataFrame({
        'pred_inf_prob': df['pred_inf_prob'].to_numpy(dtype=np.float32),
        'pred_progress': df['pred_progress'].to_numpy(dtype=np.float32),
        'pred_sub_conf': df['pred_sub_conf'].to_numpy(dtype=np.float32),
        'pred_sub_entropy': df['pred_sub_entropy'].to_numpy(dtype=np.float32),
        'pred_conf_max': df['pred_conf_max'].to_numpy(dtype=np.float32),
        'p_short': df['p_short'].to_numpy(dtype=np.float32),
        'p_flat': df['p_flat'].to_numpy(dtype=np.float32),
        'p_long': df['p_long'].to_numpy(dtype=np.float32),
        'step': df['step'].to_numpy(dtype=np.float32),
        'q_adv': q_adv.astype(np.float32),
        'q_conf': q_conf.astype(np.float32),
        'q_margin': q_margin.astype(np.float32),
        'changed': changed,
        'behavior_nonflat': behavior_nonflat,
        'hybrid_nonflat': hybrid_nonflat,
        'behavior_action': behavior_actions.astype(np.float32),
        'hybrid_action': hybrid_actions.astype(np.float32),
        'hybrid_minus_behavior': (hybrid_actions - behavior_actions).astype(np.float32),
    })

    feature_cols = X.columns.tolist()
    x = X.to_numpy(dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x, feature_cols, changed


def meta_gate_actions(
    behavior_actions: np.ndarray,
    hybrid_actions: np.ndarray,
    gate_prob: np.ndarray,
    pred_inf_prob: np.ndarray,
    q_conf: np.ndarray,
    gate_thr: float,
    inf_min: float,
    q_conf_min: float,
) -> np.ndarray:
    out = behavior_actions.copy()
    mask = (
        (hybrid_actions != behavior_actions)
        & (gate_prob >= float(gate_thr))
        & (pred_inf_prob >= float(inf_min))
        & (q_conf >= float(q_conf_min))
    )
    out[mask] = hybrid_actions[mask]
    return out.astype(np.int64)


def main():
    base = load_module(PHASE11_BASE_SCRIPT, module_name='phase11_base_v3')
    p112 = load_module(PHASE11_2_SCRIPT, module_name='phase11_2_v3')

    if not PHASE11_2_MODEL_CQL.exists():
        raise FileNotFoundError(f'Missing model file: {PHASE11_2_MODEL_CQL}')

    w = pd.read_csv(IN_WINDOWS)
    h = pd.read_csv(IN_HOLDOUT)
    w['timestamp'] = pd.to_datetime(w['timestamp'], errors='coerce')
    h['timestamp'] = pd.to_datetime(h['timestamp'], errors='coerce')

    w_feat, _ = p112.build_features(w)
    h_feat, _ = p112.build_features(h)

    win = w_feat.sort_values('timestamp').reset_index(drop=True)
    hold = h_feat.sort_values('timestamp').reset_index(drop=True)

    ckpt = torch.load(PHASE11_2_MODEL_CQL, map_location='cpu', weights_only=False)
    feature_cols = list(ckpt['feature_cols'])
    mu = np.asarray(ckpt['mu'], dtype=np.float32)
    sd = np.asarray(ckpt['sd'], dtype=np.float32)

    cfg = base.Config()
    cfg_dict = dict(ckpt.get('config', {}))
    for k, v in cfg_dict.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    in_dim = len(feature_cols)
    hidden_dim = int(getattr(cfg, 'hidden_dim', 128))
    cql_model = base.MLPPolicy(in_dim=in_dim, hidden_dim=hidden_dim).to('cpu')
    cql_model.load_state_dict(ckpt['state_dict'])

    x_win = base.normalize_apply(win[feature_cols].to_numpy(dtype=np.float32), mu, sd)
    x_hold = base.normalize_apply(hold[feature_cols].to_numpy(dtype=np.float32), mu, sd)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cql_model = cql_model.to(device)

    q_win = q_logits_np(cql_model, x_win, device)
    q_hold = q_logits_np(cql_model, x_hold, device)

    hyb_cfg = ckpt.get('hybrid', {})
    hyb_conf_thr = float(hyb_cfg.get('conf_thr', 0.55))
    hyb_adv_thr = float(hyb_cfg.get('adv_thr', 0.0))
    hyb_inf_min = float(hyb_cfg.get('inf_min', 0.4))

    b_win = win['pred_action'].to_numpy(dtype=np.int64)
    b_hold = hold['pred_action'].to_numpy(dtype=np.int64)

    a_cql_win = cql_actions_from_logits(q_win, conf_thr=float(ckpt.get('calibration', {}).get('conf_thr', 0.5)))
    a_cql_hold = cql_actions_from_logits(q_hold, conf_thr=float(ckpt.get('calibration', {}).get('conf_thr', 0.5)))

    a_hyb_win = p112.hybrid_by_advantage(
        behavior_actions=b_win,
        q_logits=q_win,
        pred_inf_prob=win['pred_inf_prob'].to_numpy(dtype=np.float32),
        conf_thr=hyb_conf_thr,
        adv_thr=hyb_adv_thr,
        inf_min=hyb_inf_min,
    )
    a_hyb_hold = p112.hybrid_by_advantage(
        behavior_actions=b_hold,
        q_logits=q_hold,
        pred_inf_prob=hold['pred_inf_prob'].to_numpy(dtype=np.float32),
        conf_thr=hyb_conf_thr,
        adv_thr=hyb_adv_thr,
        inf_min=hyb_inf_min,
    )

    # Use the same validation span as phase11.x for meta-gate training/calibration.
    cut = int(len(win) * float(getattr(cfg, 'train_ratio', 0.8)))
    val = win.iloc[cut:].copy().reset_index(drop=True)
    q_val = q_win[cut:]
    b_val = b_win[cut:]
    a_hyb_val = a_hyb_win[cut:]
    a_cql_val = a_cql_win[cut:]

    x_meta_val, meta_feature_cols, changed_val = build_meta_features(val, q_val, b_val, a_hyb_val)

    cost_ret = float(getattr(cfg, 'cost_pips', base.CFG.cost_pips)) * float(base.pip_size_for_symbol(getattr(cfg, 'symbol', base.CFG.symbol)))

    ret_val = val['future_return_h'].to_numpy(dtype=np.float32)
    net_b_val = compute_net(b_val, ret_val, cost_ret)
    net_hyb_val = compute_net(a_hyb_val, ret_val, cost_ret)

    # Gate target: choose hybrid when expected to outperform behavior by at least small margin.
    margin = 0.0
    y_gate = (net_hyb_val > (net_b_val + margin)).astype(np.int64)

    # train/cal split inside validation (temporal order to avoid leakage)
    n_val = len(val)
    split_meta = int(n_val * 0.70)
    x_mtr = x_meta_val[:split_meta]
    y_mtr = y_gate[:split_meta]
    x_mcal = x_meta_val[split_meta:]

    val_cal = val.iloc[split_meta:].copy().reset_index(drop=True)
    b_cal = b_val[split_meta:]
    a_hyb_cal = a_hyb_val[split_meta:]
    q_val_cal = q_val[split_meta:]

    # Keep only points where hybrid differs from behavior for model fit; otherwise decision is trivial.
    diff_train_mask = changed_val[:split_meta] > 0.5
    if diff_train_mask.sum() < 50:
        diff_train_mask = np.ones_like(diff_train_mask, dtype=bool)

    gate_model = LogisticRegression(
        C=0.7,
        max_iter=300,
        class_weight='balanced',
        random_state=SEED,
    )
    gate_model.fit(x_mtr[diff_train_mask], y_mtr[diff_train_mask])

    p_cal = gate_model.predict_proba(x_mcal)[:, 1]
    q_exp_cal = np.exp(q_val_cal - q_val_cal.max(axis=1, keepdims=True))
    q_prob_cal = q_exp_cal / np.maximum(q_exp_cal.sum(axis=1, keepdims=True), 1e-8)
    q_conf_cal = q_prob_cal.max(axis=1)

    y_cal = val_cal['y_action'].to_numpy(dtype=np.int64)
    r_cal = val_cal['future_return_h'].to_numpy(dtype=np.float32)

    grid_rows = []
    best = None
    for gate_thr in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        for inf_min in [0.30, 0.40, 0.50, 0.60]:
            for q_conf_min in [0.30, 0.40, 0.50, 0.60, 0.70]:
                a_meta_cal = meta_gate_actions(
                    behavior_actions=b_cal,
                    hybrid_actions=a_hyb_cal,
                    gate_prob=p_cal,
                    pred_inf_prob=val_cal['pred_inf_prob'].to_numpy(dtype=np.float32),
                    q_conf=q_conf_cal,
                    gate_thr=float(gate_thr),
                    inf_min=float(inf_min),
                    q_conf_min=float(q_conf_min),
                )
                m = base.evaluate_actions(a_meta_cal, y_cal, r_cal, cost_ret)
                score = base.score_from_metrics(m)
                row = {
                    'gate_thr': float(gate_thr),
                    'inf_min': float(inf_min),
                    'q_conf_min': float(q_conf_min),
                    'score': float(score),
                    **m,
                }
                grid_rows.append(row)
                if best is None or row['score'] > best['score']:
                    best = row

    grid_df = pd.DataFrame(grid_rows).sort_values('score', ascending=False)
    grid_df.to_csv(OUT_META_GRID_CSV, index=False)

    # Holdout evaluation
    x_meta_hold, _, changed_hold = build_meta_features(hold, q_hold, b_hold, a_hyb_hold)
    p_hold = gate_model.predict_proba(x_meta_hold)[:, 1]

    q_exp_hold = np.exp(q_hold - q_hold.max(axis=1, keepdims=True))
    q_prob_hold = q_exp_hold / np.maximum(q_exp_hold.sum(axis=1, keepdims=True), 1e-8)
    q_conf_hold = q_prob_hold.max(axis=1)

    a_meta_hold = meta_gate_actions(
        behavior_actions=b_hold,
        hybrid_actions=a_hyb_hold,
        gate_prob=p_hold,
        pred_inf_prob=hold['pred_inf_prob'].to_numpy(dtype=np.float32),
        q_conf=q_conf_hold,
        gate_thr=float(best['gate_thr']),
        inf_min=float(best['inf_min']),
        q_conf_min=float(best['q_conf_min']),
    )

    y_hold = hold['y_action'].to_numpy(dtype=np.int64)
    r_hold = hold['future_return_h'].to_numpy(dtype=np.float32)

    m_behavior_hold = base.evaluate_actions(b_hold, y_hold, r_hold, cost_ret)
    m_cql_hold = base.evaluate_actions(a_cql_hold, y_hold, r_hold, cost_ret)
    m_hybrid_hold = base.evaluate_actions(a_hyb_hold, y_hold, r_hold, cost_ret)
    m_meta_hold = base.evaluate_actions(a_meta_hold, y_hold, r_hold, cost_ret)

    # Also keep validation-side reference metrics
    y_val = val['y_action'].to_numpy(dtype=np.int64)
    r_val = val['future_return_h'].to_numpy(dtype=np.float32)
    m_behavior_val = base.evaluate_actions(b_val, y_val, r_val, cost_ret)
    m_cql_val = base.evaluate_actions(a_cql_val, y_val, r_val, cost_ret)
    m_hybrid_val = base.evaluate_actions(a_hyb_val, y_val, r_val, cost_ret)
    a_meta_val = meta_gate_actions(
        behavior_actions=b_val,
        hybrid_actions=a_hyb_val,
        gate_prob=gate_model.predict_proba(x_meta_val)[:, 1],
        pred_inf_prob=val['pred_inf_prob'].to_numpy(dtype=np.float32),
        q_conf=(np.exp(q_val - q_val.max(axis=1, keepdims=True)) / np.maximum(np.exp(q_val - q_val.max(axis=1, keepdims=True)).sum(axis=1, keepdims=True), 1e-8)).max(axis=1),
        gate_thr=float(best['gate_thr']),
        inf_min=float(best['inf_min']),
        q_conf_min=float(best['q_conf_min']),
    )
    m_meta_val = base.evaluate_actions(a_meta_val, y_val, r_val, cost_ret)

    cmp_rows = [
        {'split': 'validation', 'policy': 'behavior', **m_behavior_val},
        {'split': 'validation', 'policy': 'cql', **m_cql_val},
        {'split': 'validation', 'policy': 'hybrid_advantage', **m_hybrid_val},
        {'split': 'validation', 'policy': 'meta_gate', **m_meta_val},
        {'split': 'holdout', 'policy': 'behavior', **m_behavior_hold},
        {'split': 'holdout', 'policy': 'cql', **m_cql_hold},
        {'split': 'holdout', 'policy': 'hybrid_advantage', **m_hybrid_hold},
        {'split': 'holdout', 'policy': 'meta_gate', **m_meta_hold},
    ]
    cmp_df = pd.DataFrame(cmp_rows)
    cmp_df.to_csv(OUT_COMPARE_CSV, index=False)

    pred = hold[['global_segment_id', 'timestamp', 'step', 'y_action', 'future_return_h']].copy()
    pred['action_behavior'] = b_hold
    pred['action_cql'] = a_cql_hold
    pred['action_hybrid'] = a_hyb_hold
    pred['action_meta'] = a_meta_hold
    pred['meta_gate_prob'] = p_hold
    pred['meta_changed_candidate'] = changed_hold.astype(int)

    for nm, a in [('behavior', b_hold), ('cql', a_cql_hold), ('hybrid', a_hyb_hold), ('meta', a_meta_hold)]:
        pred[f'net_{nm}'] = compute_net(a, r_hold, cost_ret)
        pred[f'cum_{nm}'] = pred[f'net_{nm}'].cumsum()

    pred.to_csv(OUT_HOLDOUT_PRED_CSV, index=False)

    plt.figure(figsize=(11, 4))
    plt.plot(pred['timestamp'], pred['cum_behavior'], label='behavior')
    plt.plot(pred['timestamp'], pred['cum_cql'], label='cql')
    plt.plot(pred['timestamp'], pred['cum_hybrid'], label='hybrid_advantage')
    plt.plot(pred['timestamp'], pred['cum_meta'], label='meta_gate', linewidth=2.0)
    plt.title('Phase11.3 holdout cumulative net return')
    plt.xlabel('timestamp')
    plt.ylabel('cumulative return')
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_EQUITY_PNG, dpi=140)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.hist(p_hold, bins=40, alpha=0.8)
    plt.title('Meta gate probability (holdout)')
    plt.xlabel('P(use_hybrid)')
    plt.ylabel('count')
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_PROBA_PNG, dpi=140)
    plt.close()

    coef_df = pd.DataFrame({'feature': meta_feature_cols, 'coef': gate_model.coef_[0]})
    coef_df['abs_coef'] = coef_df['coef'].abs()
    coef_df = coef_df.sort_values('abs_coef', ascending=False)
    coef_df.to_csv(OUT_META_COEF_CSV, index=False)

    model_payload = {
        'model': gate_model,
        'feature_cols': meta_feature_cols,
        'thresholds': {
            'gate_thr': float(best['gate_thr']),
            'inf_min': float(best['inf_min']),
            'q_conf_min': float(best['q_conf_min']),
        },
        'source_phase11_2_model': str(PHASE11_2_MODEL_CQL),
        'seed': SEED,
    }
    joblib.dump(model_payload, OUT_META_MODEL_PKL)

    phase11_2_summary = {}
    if PHASE11_2_SUMMARY.exists():
        phase11_2_summary = json.loads(PHASE11_2_SUMMARY.read_text())

    summary = {
        'symbol': getattr(cfg, 'symbol', 'GBPUSD'),
        'model_tag': 'phase11_3_meta_policy',
        'description': 'Meta-policy gate over phase11.2 hybrid vs behavior using leak-safe temporal calibration',
        'seed': SEED,
        'source': {
            'phase11_2_model': str(PHASE11_2_MODEL_CQL),
            'phase11_2_summary': str(PHASE11_2_SUMMARY),
            'phase11_2_best_trial': phase11_2_summary.get('best_trial', {}),
        },
        'dataset': {
            'n_windows_total': int(len(win)),
            'n_windows_meta_val': int(len(val)),
            'n_windows_meta_train': int(split_meta),
            'n_windows_meta_cal': int(len(val) - split_meta),
            'n_holdout': int(len(hold)),
            'windows_date_min': str(win['timestamp'].min()),
            'windows_date_max': str(win['timestamp'].max()),
            'holdout_date_min': str(hold['timestamp'].min()),
            'holdout_date_max': str(hold['timestamp'].max()),
        },
        'phase11_2_hybrid_config': {
            'conf_thr': hyb_conf_thr,
            'adv_thr': hyb_adv_thr,
            'inf_min': hyb_inf_min,
        },
        'meta_calibration_best': {
            'gate_thr': float(best['gate_thr']),
            'inf_min': float(best['inf_min']),
            'q_conf_min': float(best['q_conf_min']),
            'score': float(best['score']),
        },
        'validation_metrics': {
            'behavior': m_behavior_val,
            'cql': m_cql_val,
            'hybrid_advantage': m_hybrid_val,
            'meta_gate': m_meta_val,
        },
        'holdout_metrics': {
            'behavior': m_behavior_hold,
            'cql': m_cql_hold,
            'hybrid_advantage': m_hybrid_hold,
            'meta_gate': m_meta_hold,
        },
        'outputs': {
            'compare_csv': str(OUT_COMPARE_CSV),
            'holdout_predictions_csv': str(OUT_HOLDOUT_PRED_CSV),
            'meta_threshold_grid_csv': str(OUT_META_GRID_CSV),
            'meta_coefficients_csv': str(OUT_META_COEF_CSV),
            'meta_model_pkl': str(OUT_META_MODEL_PKL),
            'equity_png': str(OUT_EQUITY_PNG),
            'gate_probability_png': str(OUT_PROBA_PNG),
        },
    }

    OUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

    print('Saved summary:', OUT_SUMMARY_JSON)
    print('Holdout metrics:', summary['holdout_metrics'])


if __name__ == '__main__':
    main()
