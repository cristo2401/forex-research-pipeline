#!/usr/bin/env python3
import importlib.util
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

SEEDS = [7, 11, 19, 42, 77]
SEED_GLOBAL = 42
np.random.seed(SEED_GLOBAL)
torch.manual_seed(SEED_GLOBAL)
torch.cuda.manual_seed_all(SEED_GLOBAL)

ROOT = Path(__file__).resolve().parents[2]
PHASE11_DIR = ROOT / 'phases/phase11_offline_rl'
PHASE10_OUT_DIR = ROOT / 'phases/phase10_decision_engine/output_phase10_2_strict_expanding_GBPUSD'
PHASE11_BASE_SCRIPT = PHASE11_DIR / 'run_phase11_offline_rl_cql.py'
PHASE11_2_SCRIPT = PHASE11_DIR / 'run_phase11_2_offline_rl_reward_input.py'
PHASE11_2_SUMMARY = PHASE11_DIR / 'output_phase11_2_offline_rl_reward_input_GBPUSD/GBPUSD_phase11_2_summary.json'
PHASE11_2_MODEL_CQL = PHASE11_DIR / 'output_phase11_2_offline_rl_reward_input_GBPUSD/phase11_2_best_cql_model.pt'

IN_WINDOWS = PHASE10_OUT_DIR / 'GBPUSD_phase10_2_windows_predictions.csv'
IN_HOLDOUT = PHASE10_OUT_DIR / 'GBPUSD_phase10_2_holdout_predictions.csv'

OUT_DIR = PHASE11_DIR / 'output_phase11_5_meta_ensemble_walkforward_GBPUSD'
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_WF_ROWS_CSV = OUT_DIR / 'GBPUSD_phase11_5_walkforward_rows.csv'
OUT_WF_SEED_SUMMARY_CSV = OUT_DIR / 'GBPUSD_phase11_5_walkforward_seed_summary.csv'
OUT_HOLDOUT_BY_SEED_CSV = OUT_DIR / 'GBPUSD_phase11_5_holdout_by_seed.csv'
OUT_COMPARE_CSV = OUT_DIR / 'GBPUSD_phase11_5_policy_comparison.csv'
OUT_HOLDOUT_PRED_CSV = OUT_DIR / 'GBPUSD_phase11_5_holdout_predictions_selected_seed.csv'
OUT_FEATURES_CSV = OUT_DIR / 'GBPUSD_phase11_5_meta_features.csv'
OUT_MODEL_PKL = OUT_DIR / 'phase11_5_selected_meta_ensemble.pkl'
OUT_PNG_HOLDOUT_EQUITY = OUT_DIR / 'GBPUSD_phase11_5_holdout_equity_compare.png'
OUT_PNG_WF_BOX = OUT_DIR / 'GBPUSD_phase11_5_walkforward_cum_boxplot.png'
OUT_PNG_WF_SEED = OUT_DIR / 'GBPUSD_phase11_5_walkforward_seed_mean_cum.png'
OUT_SUMMARY_JSON = OUT_DIR / 'GBPUSD_phase11_5_summary.json'


ACTION_TO_IDX = {-1: 0, 0: 1, 1: 2}
IDX_TO_ACTION = {0: -1, 1: 0, 2: 1}

WEIGHT_GRID = [
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.34, 0.33, 0.33),
    (0.50, 0.30, 0.20),
    (0.50, 0.20, 0.30),
    (0.30, 0.50, 0.20),
    (0.20, 0.50, 0.30),
    (0.40, 0.40, 0.20),
    (0.20, 0.30, 0.50),
    (0.60, 0.20, 0.20),
]


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


def q_confidence(q_logits: np.ndarray) -> np.ndarray:
    exp = np.exp(q_logits - q_logits.max(axis=1, keepdims=True))
    p = exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-8)
    return p.max(axis=1).astype(np.float32)


def cql_actions_from_logits(q_logits: np.ndarray, conf_thr: float) -> np.ndarray:
    m_idx = np.argmax(q_logits, axis=1)
    conf = q_confidence(q_logits)
    act = idx_to_action(m_idx)
    act = np.where(conf >= float(conf_thr), act, 0)
    return act.astype(np.int64)


def compute_net(actions: np.ndarray, future_ret: np.ndarray, cost_ret: float) -> np.ndarray:
    return actions * future_ret - (actions != 0).astype(np.float32) * float(cost_ret)


def build_meta_features(
    df: pd.DataFrame,
    q_logits: np.ndarray,
    behavior_actions: np.ndarray,
    hybrid_actions: np.ndarray,
) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray]:
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

    xdf = pd.DataFrame({
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

    feature_cols = xdf.columns.tolist()
    x = xdf.to_numpy(dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x, feature_cols, changed, q_conf.astype(np.float32)


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


def choose_walkforward_folds(n: int, min_train: int = 3200, test_size: int = 800, max_folds: int = 6):
    folds = []
    train_end = min_train
    f = 0
    while train_end + test_size <= n and f < max_folds:
        cal_len = max(400, int(train_end * 0.20))
        if train_end - cal_len < 800:
            break
        tr0, tr1 = 0, train_end - cal_len
        ca0, ca1 = train_end - cal_len, train_end
        te0, te1 = train_end, train_end + test_size
        folds.append((tr0, tr1, ca0, ca1, te0, te1))
        train_end += test_size
        f += 1
    return folds


def fit_binary_model(kind: str, x: np.ndarray, y: np.ndarray, seed: int):
    uniq = np.unique(y)
    if len(uniq) < 2:
        return {'kind': 'constant', 'prob': float(uniq[0]) if len(uniq) else 0.0}

    n = len(y)
    pos = max(int((y == 1).sum()), 1)
    neg = max(int((y == 0).sum()), 1)
    w_pos = n / (2.0 * pos)
    w_neg = n / (2.0 * neg)
    sw = np.where(y == 1, w_pos, w_neg).astype(np.float32)

    if kind == 'hgb':
        model = HistGradientBoostingClassifier(
            loss='log_loss',
            learning_rate=0.05,
            max_iter=240,
            max_leaf_nodes=31,
            min_samples_leaf=70,
            l2_regularization=0.25,
            random_state=int(seed),
        )
        model.fit(x, y, sample_weight=sw)
    elif kind == 'rf':
        model = RandomForestClassifier(
            n_estimators=360,
            max_depth=8,
            min_samples_leaf=60,
            class_weight='balanced_subsample',
            random_state=int(seed),
            n_jobs=-1,
        )
        model.fit(x, y, sample_weight=sw)
    elif kind == 'lr':
        model = LogisticRegression(
            C=0.65,
            max_iter=500,
            class_weight='balanced',
            random_state=int(seed),
            solver='lbfgs',
        )
        model.fit(x, y, sample_weight=sw)
    else:
        raise ValueError(kind)

    return {'kind': kind, 'model': model}


def predict_binary_proba(model_obj: Dict, x: np.ndarray) -> np.ndarray:
    if model_obj['kind'] == 'constant':
        return np.full((len(x),), float(model_obj['prob']), dtype=np.float32)
    p = model_obj['model'].predict_proba(x)
    return p[:, 1].astype(np.float32)


def train_gate_ensemble(x: np.ndarray, y: np.ndarray, changed: np.ndarray, seed: int):
    mask = changed > 0.5
    xg = x[mask]
    yg = y[mask]

    if len(xg) < 250:
        xg = x
        yg = y

    return {
        'hgb': fit_binary_model('hgb', xg, yg, seed=seed),
        'rf': fit_binary_model('rf', xg, yg, seed=seed),
        'lr': fit_binary_model('lr', xg, yg, seed=seed),
    }


def predict_ensemble_proba(models: Dict, x: np.ndarray):
    p_hgb = predict_binary_proba(models['hgb'], x)
    p_rf = predict_binary_proba(models['rf'], x)
    p_lr = predict_binary_proba(models['lr'], x)
    return p_hgb, p_rf, p_lr


def tune_ensemble_and_thresholds(
    base_mod,
    df_cal: pd.DataFrame,
    behavior_actions: np.ndarray,
    hybrid_actions: np.ndarray,
    q_conf: np.ndarray,
    p_hgb: np.ndarray,
    p_rf: np.ndarray,
    p_lr: np.ndarray,
    cost_ret: float,
):
    y = df_cal['y_action'].to_numpy(dtype=np.int64)
    ret = df_cal['future_return_h'].to_numpy(dtype=np.float32)
    inf = df_cal['pred_inf_prob'].to_numpy(dtype=np.float32)

    best = None
    rows = []

    for w_hgb, w_rf, w_lr in WEIGHT_GRID:
        p = (w_hgb * p_hgb + w_rf * p_rf + w_lr * p_lr).astype(np.float32)

        for gate_thr in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
            for inf_min in [0.25, 0.35, 0.45, 0.55, 0.65]:
                for q_conf_min in [0.25, 0.35, 0.45, 0.55, 0.65]:
                    a = meta_gate_actions(
                        behavior_actions=behavior_actions,
                        hybrid_actions=hybrid_actions,
                        gate_prob=p,
                        pred_inf_prob=inf,
                        q_conf=q_conf,
                        gate_thr=float(gate_thr),
                        inf_min=float(inf_min),
                        q_conf_min=float(q_conf_min),
                    )
                    m = base_mod.evaluate_actions(a, y, ret, cost_ret)
                    score = base_mod.score_from_metrics(m)
                    override_rate = float((a != behavior_actions).mean())

                    row = {
                        'w_hgb': float(w_hgb),
                        'w_rf': float(w_rf),
                        'w_lr': float(w_lr),
                        'gate_thr': float(gate_thr),
                        'inf_min': float(inf_min),
                        'q_conf_min': float(q_conf_min),
                        'override_rate': override_rate,
                        'score': float(score),
                        **m,
                    }
                    rows.append(row)
                    if best is None or row['score'] > best['score']:
                        best = row

    return best, rows


def evaluate_subset(base_mod, actions: np.ndarray, df: pd.DataFrame, cost_ret: float) -> Dict[str, float]:
    y = df['y_action'].to_numpy(dtype=np.int64)
    ret = df['future_return_h'].to_numpy(dtype=np.float32)
    return base_mod.evaluate_actions(actions, y, ret, cost_ret)


def main():
    base = load_module(PHASE11_BASE_SCRIPT, module_name='phase11_base_v5')
    p112 = load_module(PHASE11_2_SCRIPT, module_name='phase11_2_v5')

    if not PHASE11_2_MODEL_CQL.exists():
        raise FileNotFoundError(f'Missing model file: {PHASE11_2_MODEL_CQL}')

    windows = pd.read_csv(IN_WINDOWS)
    holdout = pd.read_csv(IN_HOLDOUT)
    windows['timestamp'] = pd.to_datetime(windows['timestamp'], errors='coerce')
    holdout['timestamp'] = pd.to_datetime(holdout['timestamp'], errors='coerce')

    w_feat, _ = p112.build_features(windows)
    h_feat, _ = p112.build_features(holdout)

    win = w_feat.sort_values('timestamp').reset_index(drop=True)
    ho = h_feat.sort_values('timestamp').reset_index(drop=True)

    ckpt = torch.load(PHASE11_2_MODEL_CQL, map_location='cpu', weights_only=False)
    feature_cols = list(ckpt['feature_cols'])
    mu = np.asarray(ckpt['mu'], dtype=np.float32)
    sd = np.asarray(ckpt['sd'], dtype=np.float32)

    cfg = base.Config()
    cfg_dict = dict(ckpt.get('config', {}))
    for k, v in cfg_dict.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    cql_model = base.MLPPolicy(in_dim=len(feature_cols), hidden_dim=int(getattr(cfg, 'hidden_dim', 128))).to('cpu')
    cql_model.load_state_dict(ckpt['state_dict'])

    xw = base.normalize_apply(win[feature_cols].to_numpy(dtype=np.float32), mu, sd)
    xh = base.normalize_apply(ho[feature_cols].to_numpy(dtype=np.float32), mu, sd)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cql_model = cql_model.to(device)

    qw = q_logits_np(cql_model, xw, device)
    qh = q_logits_np(cql_model, xh, device)

    b_w = win['pred_action'].to_numpy(dtype=np.int64)
    b_h = ho['pred_action'].to_numpy(dtype=np.int64)

    cql_conf_thr = float(ckpt.get('calibration', {}).get('conf_thr', 0.5))
    a_cql_w = cql_actions_from_logits(qw, conf_thr=cql_conf_thr)
    a_cql_h = cql_actions_from_logits(qh, conf_thr=cql_conf_thr)

    hyb_cfg = ckpt.get('hybrid', {})
    hyb_conf_thr = float(hyb_cfg.get('conf_thr', 0.55))
    hyb_adv_thr = float(hyb_cfg.get('adv_thr', 0.0))
    hyb_inf_min = float(hyb_cfg.get('inf_min', 0.4))

    a_hyb_w = p112.hybrid_by_advantage(
        behavior_actions=b_w,
        q_logits=qw,
        pred_inf_prob=win['pred_inf_prob'].to_numpy(dtype=np.float32),
        conf_thr=hyb_conf_thr,
        adv_thr=hyb_adv_thr,
        inf_min=hyb_inf_min,
    )
    a_hyb_h = p112.hybrid_by_advantage(
        behavior_actions=b_h,
        q_logits=qh,
        pred_inf_prob=ho['pred_inf_prob'].to_numpy(dtype=np.float32),
        conf_thr=hyb_conf_thr,
        adv_thr=hyb_adv_thr,
        inf_min=hyb_inf_min,
    )

    x_meta_w, meta_feature_cols, changed_w, q_conf_w = build_meta_features(win, qw, b_w, a_hyb_w)
    x_meta_h, _f2, changed_h, q_conf_h = build_meta_features(ho, qh, b_h, a_hyb_h)

    pd.DataFrame({'feature': meta_feature_cols}).to_csv(OUT_FEATURES_CSV, index=False)

    cost_ret = float(getattr(cfg, 'cost_pips', base.CFG.cost_pips)) * float(base.pip_size_for_symbol(getattr(cfg, 'symbol', base.CFG.symbol)))

    ret_w = win['future_return_h'].to_numpy(dtype=np.float32)
    net_b_w = compute_net(b_w, ret_w, cost_ret)
    net_hyb_w = compute_net(a_hyb_w, ret_w, cost_ret)
    y_gate_w = (net_hyb_w > net_b_w).astype(np.int64)

    folds = choose_walkforward_folds(len(win), min_train=3200, test_size=800, max_folds=6)
    if len(folds) < 3:
        raise RuntimeError(f'Not enough folds built: {len(folds)}')

    wf_rows = []

    for seed in SEEDS:
        for fold_id, (tr0, tr1, ca0, ca1, te0, te1) in enumerate(folds, start=1):
            x_tr = x_meta_w[tr0:tr1]
            y_tr = y_gate_w[tr0:tr1]
            ch_tr = changed_w[tr0:tr1]

            x_cal = x_meta_w[ca0:ca1]
            df_cal = win.iloc[ca0:ca1].copy()
            b_cal = b_w[ca0:ca1]
            hyb_cal = a_hyb_w[ca0:ca1]
            qc_cal = q_conf_w[ca0:ca1]

            x_te = x_meta_w[te0:te1]
            df_te = win.iloc[te0:te1].copy()
            b_te = b_w[te0:te1]
            cql_te = a_cql_w[te0:te1]
            hyb_te = a_hyb_w[te0:te1]
            qc_te = q_conf_w[te0:te1]

            models = train_gate_ensemble(x_tr, y_tr, ch_tr, seed=seed)
            p_hgb_cal, p_rf_cal, p_lr_cal = predict_ensemble_proba(models, x_cal)

            th_best, _ = tune_ensemble_and_thresholds(
                base_mod=base,
                df_cal=df_cal,
                behavior_actions=b_cal,
                hybrid_actions=hyb_cal,
                q_conf=qc_cal,
                p_hgb=p_hgb_cal,
                p_rf=p_rf_cal,
                p_lr=p_lr_cal,
                cost_ret=cost_ret,
            )

            p_hgb_te, p_rf_te, p_lr_te = predict_ensemble_proba(models, x_te)
            p_te = (th_best['w_hgb'] * p_hgb_te + th_best['w_rf'] * p_rf_te + th_best['w_lr'] * p_lr_te).astype(np.float32)

            a_meta_te = meta_gate_actions(
                behavior_actions=b_te,
                hybrid_actions=hyb_te,
                gate_prob=p_te,
                pred_inf_prob=df_te['pred_inf_prob'].to_numpy(dtype=np.float32),
                q_conf=qc_te,
                gate_thr=th_best['gate_thr'],
                inf_min=th_best['inf_min'],
                q_conf_min=th_best['q_conf_min'],
            )

            m_beh = evaluate_subset(base, b_te, df_te, cost_ret)
            m_cql = evaluate_subset(base, cql_te, df_te, cost_ret)
            m_hyb = evaluate_subset(base, hyb_te, df_te, cost_ret)
            m_meta = evaluate_subset(base, a_meta_te, df_te, cost_ret)

            wf_rows.append({
                'seed': int(seed),
                'fold_id': int(fold_id),
                'train_n': int(tr1 - tr0),
                'cal_n': int(ca1 - ca0),
                'test_n': int(te1 - te0),
                'train_end_ts': str(win.iloc[tr1 - 1]['timestamp']),
                'cal_end_ts': str(win.iloc[ca1 - 1]['timestamp']),
                'test_start_ts': str(win.iloc[te0]['timestamp']),
                'test_end_ts': str(win.iloc[te1 - 1]['timestamp']),
                'w_hgb': float(th_best['w_hgb']),
                'w_rf': float(th_best['w_rf']),
                'w_lr': float(th_best['w_lr']),
                'gate_thr': float(th_best['gate_thr']),
                'inf_min': float(th_best['inf_min']),
                'q_conf_min': float(th_best['q_conf_min']),
                'override_rate': float(th_best['override_rate']),
                'meta_score_cal': float(th_best['score']),
                'cum_behavior': float(m_beh['cum_net_return']),
                'cum_cql': float(m_cql['cum_net_return']),
                'cum_hybrid': float(m_hyb['cum_net_return']),
                'cum_meta': float(m_meta['cum_net_return']),
                'avg_behavior': float(m_beh['avg_net_return']),
                'avg_cql': float(m_cql['avg_net_return']),
                'avg_hybrid': float(m_hyb['avg_net_return']),
                'avg_meta': float(m_meta['avg_net_return']),
                'sharpe_behavior': float(m_beh['sharpe_ann']),
                'sharpe_cql': float(m_cql['sharpe_ann']),
                'sharpe_hybrid': float(m_hyb['sharpe_ann']),
                'sharpe_meta': float(m_meta['sharpe_ann']),
                'trade_behavior': float(m_beh['trade_rate']),
                'trade_hybrid': float(m_hyb['trade_rate']),
                'trade_meta': float(m_meta['trade_rate']),
            })

    wf_df = pd.DataFrame(wf_rows)
    wf_df.to_csv(OUT_WF_ROWS_CSV, index=False)

    seed_rows = []
    for seed, g in wf_df.groupby('seed'):
        row = {
            'seed': int(seed),
            'n_folds': int(len(g)),
            'cum_behavior_mean': float(g['cum_behavior'].mean()),
            'cum_behavior_std': float(g['cum_behavior'].std(ddof=0)),
            'cum_hybrid_mean': float(g['cum_hybrid'].mean()),
            'cum_hybrid_std': float(g['cum_hybrid'].std(ddof=0)),
            'cum_meta_mean': float(g['cum_meta'].mean()),
            'cum_meta_std': float(g['cum_meta'].std(ddof=0)),
            'avg_meta_mean': float(g['avg_meta'].mean()),
            'sharpe_meta_mean': float(g['sharpe_meta'].mean()),
        }
        row['uplift_meta_vs_behavior_mean'] = row['cum_meta_mean'] - row['cum_behavior_mean']
        row['uplift_meta_vs_hybrid_mean'] = row['cum_meta_mean'] - row['cum_hybrid_mean']
        row['score_meta_mean'] = row['avg_meta_mean'] * 1e4 + 0.10 * row['sharpe_meta_mean']
        seed_rows.append(row)

    seed_df = pd.DataFrame(seed_rows).sort_values('score_meta_mean', ascending=False)
    seed_df.to_csv(OUT_WF_SEED_SUMMARY_CSV, index=False)

    selected_seed_wf = int(seed_df.iloc[0]['seed'])

    # Final train/cal split for holdout deployment
    n_w = len(win)
    cal_len = max(1000, int(n_w * 0.20))
    tr0, tr1 = 0, n_w - cal_len
    ca0, ca1 = n_w - cal_len, n_w

    x_train_final = x_meta_w[tr0:tr1]
    y_train_final = y_gate_w[tr0:tr1]
    ch_train_final = changed_w[tr0:tr1]

    x_cal_final = x_meta_w[ca0:ca1]
    df_cal_final = win.iloc[ca0:ca1].copy()
    b_cal_final = b_w[ca0:ca1]
    hyb_cal_final = a_hyb_w[ca0:ca1]
    qc_cal_final = q_conf_w[ca0:ca1]

    hold_rows = []
    seed_objects = {}

    for seed in SEEDS:
        models = train_gate_ensemble(x_train_final, y_train_final, ch_train_final, seed=seed)
        p_hgb_cal, p_rf_cal, p_lr_cal = predict_ensemble_proba(models, x_cal_final)
        th_best, _ = tune_ensemble_and_thresholds(
            base_mod=base,
            df_cal=df_cal_final,
            behavior_actions=b_cal_final,
            hybrid_actions=hyb_cal_final,
            q_conf=qc_cal_final,
            p_hgb=p_hgb_cal,
            p_rf=p_rf_cal,
            p_lr=p_lr_cal,
            cost_ret=cost_ret,
        )

        p_hgb_h, p_rf_h, p_lr_h = predict_ensemble_proba(models, x_meta_h)
        p_hold = (th_best['w_hgb'] * p_hgb_h + th_best['w_rf'] * p_rf_h + th_best['w_lr'] * p_lr_h).astype(np.float32)

        a_meta_h = meta_gate_actions(
            behavior_actions=b_h,
            hybrid_actions=a_hyb_h,
            gate_prob=p_hold,
            pred_inf_prob=ho['pred_inf_prob'].to_numpy(dtype=np.float32),
            q_conf=q_conf_h,
            gate_thr=th_best['gate_thr'],
            inf_min=th_best['inf_min'],
            q_conf_min=th_best['q_conf_min'],
        )

        m_meta_h = evaluate_subset(base, a_meta_h, ho, cost_ret)

        hold_rows.append({
            'seed': int(seed),
            'w_hgb': float(th_best['w_hgb']),
            'w_rf': float(th_best['w_rf']),
            'w_lr': float(th_best['w_lr']),
            'gate_thr': float(th_best['gate_thr']),
            'inf_min': float(th_best['inf_min']),
            'q_conf_min': float(th_best['q_conf_min']),
            'override_rate': float(th_best['override_rate']),
            'cal_score': float(th_best['score']),
            **{f'meta_{k}': v for k, v in m_meta_h.items()},
        })

        seed_objects[int(seed)] = {
            'models': models,
            'thresholds': th_best,
            'p_hold': p_hold,
            'a_hold': a_meta_h,
        }

    hold_seed_df = pd.DataFrame(hold_rows).sort_values('cal_score', ascending=False)
    hold_seed_df.to_csv(OUT_HOLDOUT_BY_SEED_CSV, index=False)

    # Final model selection uses only calibration score (no holdout metric).
    selected_seed = int(hold_seed_df.iloc[0]['seed'])

    sel = seed_objects[selected_seed]
    a_meta_sel = sel['a_hold']
    p_meta_sel = sel['p_hold']
    th_sel = sel['thresholds']

    m_b_hold = evaluate_subset(base, b_h, ho, cost_ret)
    m_cql_hold = evaluate_subset(base, a_cql_h, ho, cost_ret)
    m_hyb_hold = evaluate_subset(base, a_hyb_h, ho, cost_ret)
    m_meta_hold = evaluate_subset(base, a_meta_sel, ho, cost_ret)

    wf_behavior = {
        'avg_net_return': float(wf_df['avg_behavior'].mean()),
        'cum_net_return': float(wf_df['cum_behavior'].mean()),
        'sharpe_ann': float(wf_df['sharpe_behavior'].mean()),
        'trade_rate': float(wf_df['trade_behavior'].mean()),
    }
    wf_cql = {
        'avg_net_return': float(wf_df['avg_cql'].mean()),
        'cum_net_return': float(wf_df['cum_cql'].mean()),
        'sharpe_ann': float(wf_df['sharpe_cql'].mean()),
        'trade_rate': float(np.nan),
    }
    wf_hybrid = {
        'avg_net_return': float(wf_df['avg_hybrid'].mean()),
        'cum_net_return': float(wf_df['cum_hybrid'].mean()),
        'sharpe_ann': float(wf_df['sharpe_hybrid'].mean()),
        'trade_rate': float(wf_df['trade_hybrid'].mean()),
    }
    wf_meta = {
        'avg_net_return': float(wf_df['avg_meta'].mean()),
        'cum_net_return': float(wf_df['cum_meta'].mean()),
        'sharpe_ann': float(wf_df['sharpe_meta'].mean()),
        'trade_rate': float(wf_df['trade_meta'].mean()),
    }

    cmp_rows = [
        {'split': 'walkforward_test_mean', 'policy': 'behavior', **wf_behavior},
        {'split': 'walkforward_test_mean', 'policy': 'cql', **wf_cql},
        {'split': 'walkforward_test_mean', 'policy': 'hybrid_advantage', **wf_hybrid},
        {'split': 'walkforward_test_mean', 'policy': 'meta_ensemble', **wf_meta},
        {'split': 'holdout_selected_seed', 'policy': 'behavior', **m_b_hold},
        {'split': 'holdout_selected_seed', 'policy': 'cql', **m_cql_hold},
        {'split': 'holdout_selected_seed', 'policy': 'hybrid_advantage', **m_hyb_hold},
        {'split': 'holdout_selected_seed', 'policy': 'meta_ensemble', **m_meta_hold},
    ]
    cmp_df = pd.DataFrame(cmp_rows)
    cmp_df.to_csv(OUT_COMPARE_CSV, index=False)

    pred = ho[['global_segment_id', 'timestamp', 'step', 'y_action', 'future_return_h']].copy()
    pred['action_behavior'] = b_h
    pred['action_cql'] = a_cql_h
    pred['action_hybrid'] = a_hyb_h
    pred['action_meta'] = a_meta_sel
    pred['meta_gate_prob'] = p_meta_sel
    pred['meta_changed_candidate'] = changed_h.astype(int)

    ret_hold = ho['future_return_h'].to_numpy(dtype=np.float32)
    for nm, a in [('behavior', b_h), ('cql', a_cql_h), ('hybrid', a_hyb_h), ('meta', a_meta_sel)]:
        pred[f'net_{nm}'] = compute_net(a, ret_hold, cost_ret)
        pred[f'cum_{nm}'] = pred[f'net_{nm}'].cumsum()

    pred.to_csv(OUT_HOLDOUT_PRED_CSV, index=False)

    payload = {
        'selected_seed_walkforward': selected_seed_wf,
        'selected_seed': selected_seed,
        'models': sel['models'],
        'thresholds': {
            'w_hgb': float(th_sel['w_hgb']),
            'w_rf': float(th_sel['w_rf']),
            'w_lr': float(th_sel['w_lr']),
            'gate_thr': float(th_sel['gate_thr']),
            'inf_min': float(th_sel['inf_min']),
            'q_conf_min': float(th_sel['q_conf_min']),
        },
        'meta_feature_cols': meta_feature_cols,
        'source_phase11_2_model': str(PHASE11_2_MODEL_CQL),
    }
    joblib.dump(payload, OUT_MODEL_PKL)

    plt.figure(figsize=(11, 4))
    plt.plot(pred['timestamp'], pred['cum_behavior'], label='behavior')
    plt.plot(pred['timestamp'], pred['cum_cql'], label='cql')
    plt.plot(pred['timestamp'], pred['cum_hybrid'], label='hybrid_advantage')
    plt.plot(pred['timestamp'], pred['cum_meta'], label=f'meta_ensemble_seed{selected_seed}', linewidth=2.0)
    plt.title('Phase11.5 holdout cumulative net return')
    plt.xlabel('timestamp')
    plt.ylabel('cumulative return')
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG_HOLDOUT_EQUITY, dpi=140)
    plt.close()

    plt.figure(figsize=(8, 4))
    data = [wf_df['cum_behavior'].to_numpy(), wf_df['cum_hybrid'].to_numpy(), wf_df['cum_meta'].to_numpy()]
    plt.boxplot(data, tick_labels=['behavior', 'hybrid', 'meta_ensemble'])
    plt.title('Phase11.5 walk-forward test cumulative return distribution')
    plt.ylabel('cum_net_return')
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_PNG_WF_BOX, dpi=140)
    plt.close()

    splot = seed_df.sort_values('seed').copy()
    plt.figure(figsize=(9, 4))
    x = np.arange(len(splot))
    wbar = 0.26
    plt.bar(x - wbar, splot['cum_behavior_mean'], width=wbar, label='behavior')
    plt.bar(x, splot['cum_hybrid_mean'], width=wbar, label='hybrid')
    plt.bar(x + wbar, splot['cum_meta_mean'], width=wbar, label='meta_ensemble')
    plt.xticks(x, splot['seed'].astype(str).tolist())
    plt.xlabel('seed')
    plt.ylabel('walk-forward mean cum return')
    plt.title('Phase11.5 seed stability (walk-forward)')
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG_WF_SEED, dpi=140)
    plt.close()

    phase11_2_summary = {}
    if PHASE11_2_SUMMARY.exists():
        phase11_2_summary = json.loads(PHASE11_2_SUMMARY.read_text())

    summary = {
        'symbol': getattr(cfg, 'symbol', 'GBPUSD'),
        'model_tag': 'phase11_5_meta_ensemble_walkforward',
        'description': 'Meta ensemble (HGB+RF+LR) with walk-forward + multi-seed temporal calibration',
        'seeds': SEEDS,
        'selected_seed': selected_seed,
        'weight_grid': WEIGHT_GRID,
        'walkforward': {
            'n_folds': int(len(folds)),
            'seed_summary_top': seed_df.head(5).to_dict(orient='records'),
            'mean_metrics': {
                'behavior': wf_behavior,
                'cql': wf_cql,
                'hybrid_advantage': wf_hybrid,
                'meta_ensemble': wf_meta,
            },
        },
        'holdout_metrics_selected_seed': {
            'behavior': m_b_hold,
            'cql': m_cql_hold,
            'hybrid_advantage': m_hyb_hold,
            'meta_ensemble': m_meta_hold,
        },
        'selected_thresholds': {
            'w_hgb': float(th_sel['w_hgb']),
            'w_rf': float(th_sel['w_rf']),
            'w_lr': float(th_sel['w_lr']),
            'gate_thr': float(th_sel['gate_thr']),
            'inf_min': float(th_sel['inf_min']),
            'q_conf_min': float(th_sel['q_conf_min']),
            'override_rate': float(th_sel['override_rate']),
        },
        'source': {
            'phase11_2_model': str(PHASE11_2_MODEL_CQL),
            'phase11_2_best_trial': phase11_2_summary.get('best_trial', {}),
            'phase11_2_hybrid_cfg': {
                'conf_thr': hyb_conf_thr,
                'adv_thr': hyb_adv_thr,
                'inf_min': hyb_inf_min,
            },
        },
        'dataset': {
            'n_windows': int(len(win)),
            'n_holdout': int(len(ho)),
            'windows_date_min': str(win['timestamp'].min()),
            'windows_date_max': str(win['timestamp'].max()),
            'holdout_date_min': str(ho['timestamp'].min()),
            'holdout_date_max': str(ho['timestamp'].max()),
        },
        'outputs': {
            'walkforward_rows_csv': str(OUT_WF_ROWS_CSV),
            'walkforward_seed_summary_csv': str(OUT_WF_SEED_SUMMARY_CSV),
            'holdout_by_seed_csv': str(OUT_HOLDOUT_BY_SEED_CSV),
            'compare_csv': str(OUT_COMPARE_CSV),
            'holdout_predictions_csv': str(OUT_HOLDOUT_PRED_CSV),
            'meta_features_csv': str(OUT_FEATURES_CSV),
            'selected_model_pkl': str(OUT_MODEL_PKL),
            'holdout_equity_png': str(OUT_PNG_HOLDOUT_EQUITY),
            'walkforward_boxplot_png': str(OUT_PNG_WF_BOX),
            'walkforward_seed_png': str(OUT_PNG_WF_SEED),
        },
    }

    OUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

    print('Saved summary:', OUT_SUMMARY_JSON)
    print('Selected seed:', selected_seed)
    print('Holdout metrics selected seed:', summary['holdout_metrics_selected_seed'])


if __name__ == '__main__':
    main()
