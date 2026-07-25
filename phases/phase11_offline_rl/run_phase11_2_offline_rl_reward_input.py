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
import torch

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


@dataclass
class TrialCfg:
    name: str
    hidden_dim: int
    cql_alpha: float
    bc_reg: float
    q_lr: float
    q_epochs: int
    turnover_penalty_mult: float
    reward_scale: float


TRIALS = [
    TrialCfg('t0_ref', 128, 1.20, 0.02, 8e-4, 50, 0.0, 1.0),
    TrialCfg('t1_turnover_light', 128, 1.00, 0.02, 8e-4, 55, 0.5, 1.0),
    TrialCfg('t2_turnover_med', 128, 1.20, 0.01, 8e-4, 55, 1.0, 1.0),
    TrialCfg('t3_wider', 160, 1.00, 0.01, 7e-4, 60, 0.5, 1.0),
    TrialCfg('t4_shaped_scale', 128, 1.40, 0.01, 7e-4, 60, 1.0, 1.5),
]


ROOT = Path(__file__).resolve().parents[2]
PHASE11_DIR = ROOT / 'phases/phase11_offline_rl'
PHASE11_BASE_SCRIPT = PHASE11_DIR / 'run_phase11_offline_rl_cql.py'
IN_DIR = ROOT / 'phases/phase10_decision_engine/output_phase10_2_strict_expanding_GBPUSD'
OUT_DIR = PHASE11_DIR / 'output_phase11_2_offline_rl_reward_input_GBPUSD'
OUT_DIR.mkdir(parents=True, exist_ok=True)

IN_WINDOWS = IN_DIR / 'GBPUSD_phase10_2_windows_predictions.csv'
IN_HOLDOUT = IN_DIR / 'GBPUSD_phase10_2_holdout_predictions.csv'

OUT_FEATURES_CSV = OUT_DIR / 'GBPUSD_phase11_2_features_schema.csv'
OUT_TRIALS_CSV = OUT_DIR / 'GBPUSD_phase11_2_trials.csv'
OUT_COMPARE_CSV = OUT_DIR / 'GBPUSD_phase11_2_policy_comparison.csv'
OUT_HOLDOUT_PRED_CSV = OUT_DIR / 'GBPUSD_phase11_2_holdout_predictions.csv'
OUT_HISTORY_CSV = OUT_DIR / 'GBPUSD_phase11_2_best_train_history.csv'
OUT_SUMMARY_JSON = OUT_DIR / 'GBPUSD_phase11_2_summary.json'
OUT_MODEL_BC = OUT_DIR / 'phase11_2_best_bc_model.pt'
OUT_MODEL_CQL = OUT_DIR / 'phase11_2_best_cql_model.pt'
OUT_PNG_EQUITY = OUT_DIR / 'GBPUSD_phase11_2_holdout_equity_compare.png'
OUT_PNG_LOSS = OUT_DIR / 'GBPUSD_phase11_2_train_losses.png'


ACTION_TO_IDX = {-1: 0, 0: 1, 1: 2}
IDX_TO_ACTION = {0: -1, 1: 0, 2: 1}


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location('phase11_base', str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot load module {path}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out['timestamp'], errors='coerce')

    hour = ts.dt.hour.fillna(0).astype(float)
    dow = ts.dt.dayofweek.fillna(0).astype(float)

    out['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
    out['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
    out['dow_sin'] = np.sin(2 * np.pi * dow / 7.0)
    out['dow_cos'] = np.cos(2 * np.pi * dow / 7.0)
    return out


def add_prob_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    p = out[['p_short', 'p_flat', 'p_long']].to_numpy(dtype=np.float32)
    p = np.clip(p, 1e-8, 1.0)

    out['prob_entropy_3'] = -(p * np.log(p)).sum(axis=1)
    out['p_margin_long_short'] = out['p_long'] - out['p_short']
    out['p_nonflat'] = out['p_long'] + out['p_short']

    top2 = np.sort(p, axis=1)[:, -2:]
    out['p_top_gap'] = top2[:, 1] - top2[:, 0]
    return out


def add_lag_features(df: pd.DataFrame, lag_cols: List[str], lags: Tuple[int, ...] = (1, 2, 3)) -> pd.DataFrame:
    out = df.sort_values(['global_segment_id', 'step', 'timestamp']).copy()
    g = out.groupby('global_segment_id', group_keys=False)

    for c in lag_cols:
        for L in lags:
            out[f'{c}_lag{L}'] = g[c].shift(L)

    out['pred_progress_diff1'] = out['pred_progress'] - out['pred_progress_lag1']
    out['pred_inf_prob_diff1'] = out['pred_inf_prob'] - out['pred_inf_prob_lag1']

    out = out.fillna(0.0)
    return out


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    out = add_time_features(out)
    out = add_prob_features(out)

    lag_cols = [
        'pred_inf_prob',
        'pred_progress',
        'pred_sub_conf',
        'pred_sub_entropy',
        'pred_conf_max',
        'p_short',
        'p_flat',
        'p_long',
        'p_margin_long_short',
        'p_nonflat',
        'p_top_gap',
        'prob_entropy_3',
    ]
    out = add_lag_features(out, lag_cols=lag_cols, lags=(1, 2, 3))

    # keep step and class as numeric features
    base_cols = [
        'pred_inf_prob',
        'pred_progress',
        'pred_sub_conf',
        'pred_sub_entropy',
        'pred_class',
        'pred_conf_max',
        'p_short',
        'p_flat',
        'p_long',
        'step',
        'hour_sin',
        'hour_cos',
        'dow_sin',
        'dow_cos',
        'p_margin_long_short',
        'p_nonflat',
        'p_top_gap',
        'prob_entropy_3',
        'pred_progress_diff1',
        'pred_inf_prob_diff1',
    ]

    lag_cols_final = [c for c in out.columns if c.endswith('_lag1') or c.endswith('_lag2') or c.endswith('_lag3')]
    feature_cols = base_cols + sorted(lag_cols_final)

    # sanitize
    out[feature_cols] = out[feature_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return out, feature_cols


def action_to_idx(actions: np.ndarray) -> np.ndarray:
    return np.array([ACTION_TO_IDX[int(a)] for a in actions], dtype=np.int64)


def idx_to_action(idxs: np.ndarray) -> np.ndarray:
    return np.array([IDX_TO_ACTION[int(i)] for i in idxs], dtype=np.int64)


def q_logits_np(base_mod, model, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        bs = 2048
        for i in range(0, len(x), bs):
            xb = torch.tensor(x[i : i + bs], dtype=torch.float32, device=device)
            q = model(xb)
            out.append(q.cpu().numpy())
    return np.concatenate(out, axis=0)


def hybrid_by_advantage(
    behavior_actions: np.ndarray,
    q_logits: np.ndarray,
    pred_inf_prob: np.ndarray,
    conf_thr: float,
    adv_thr: float,
    inf_min: float,
) -> np.ndarray:
    b_idx = action_to_idx(behavior_actions)

    m_idx = np.argmax(q_logits, axis=1)
    q_best = q_logits[np.arange(len(q_logits)), m_idx]
    q_beh = q_logits[np.arange(len(q_logits)), b_idx]
    adv = q_best - q_beh

    p = np.exp(q_logits - q_logits.max(axis=1, keepdims=True))
    p = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-8)
    conf = p.max(axis=1)

    m_actions = idx_to_action(m_idx)

    out = behavior_actions.copy()
    mask = (m_idx != b_idx) & (adv >= adv_thr) & (conf >= conf_thr) & (pred_inf_prob >= inf_min) & (m_actions != 0)
    out[mask] = m_actions[mask]
    return out.astype(np.int64)


def tune_adv_hybrid(base_mod, df_val: pd.DataFrame, behavior_actions: np.ndarray, q_logits_val: np.ndarray, cost_ret: float):
    y = df_val['y_action'].to_numpy(dtype=np.int64)
    ret = df_val['future_return_h'].to_numpy(dtype=np.float32)
    inf = df_val['pred_inf_prob'].to_numpy(dtype=np.float32)

    # derive data-driven advantage thresholds from quantiles
    b_idx = action_to_idx(behavior_actions)
    m_idx = np.argmax(q_logits_val, axis=1)
    adv = q_logits_val[np.arange(len(q_logits_val)), m_idx] - q_logits_val[np.arange(len(q_logits_val)), b_idx]
    adv_grid = [0.0] + [float(np.quantile(adv, q)) for q in [0.50, 0.60, 0.70, 0.80, 0.90]]

    best = None
    for conf in [0.35, 0.45, 0.55, 0.65, 0.75]:
        for inf_min in [0.30, 0.40, 0.50, 0.60]:
            for adv_thr in adv_grid:
                a = hybrid_by_advantage(behavior_actions, q_logits_val, inf, conf_thr=float(conf), adv_thr=float(adv_thr), inf_min=float(inf_min))
                m = base_mod.evaluate_actions(a, y, ret, cost_ret)
                score = base_mod.score_from_metrics(m)
                row = {
                    'conf_thr': float(conf),
                    'inf_min': float(inf_min),
                    'adv_thr': float(adv_thr),
                    'score': float(score),
                    **m,
                }
                if best is None or row['score'] > best['score']:
                    best = row

    return best


def build_transitions_custom(df: pd.DataFrame, feature_cols: List[str], turnover_penalty: float, reward_scale: float):
    s_list, a_list, r_list, ns_list, d_list = [], [], [], [], []
    zero_state = np.zeros((len(feature_cols),), dtype=np.float32)

    work = df.sort_values(['global_segment_id', 'step', 'timestamp'])

    for _seg, g in work.groupby('global_segment_id', sort=False):
        g = g.sort_values('step')

        x = g[feature_cols].to_numpy(dtype=np.float32)
        a_raw = g['pred_action'].to_numpy(dtype=np.int64)
        a = action_to_idx(a_raw)
        r_base = g['net_return'].to_numpy(dtype=np.float32)
        st = g['step'].to_numpy(dtype=np.int64)

        prev_a = np.roll(a_raw, 1)
        prev_a[0] = a_raw[0]
        turnover = (a_raw != prev_a).astype(np.float32)

        r = (r_base - turnover_penalty * turnover) * reward_scale

        n = len(g)
        for i in range(n):
            if i < n - 1 and st[i + 1] == st[i] + 1:
                ns = x[i + 1]
                done = 0.0
            else:
                ns = zero_state
                done = 1.0

            s_list.append(x[i])
            a_list.append(a[i])
            r_list.append(r[i])
            ns_list.append(ns)
            d_list.append(done)

    return (
        np.asarray(s_list, dtype=np.float32),
        np.asarray(a_list, dtype=np.int64),
        np.asarray(r_list, dtype=np.float32),
        np.asarray(ns_list, dtype=np.float32),
        np.asarray(d_list, dtype=np.float32),
    )


def main():
    base = load_module(PHASE11_BASE_SCRIPT)

    w = pd.read_csv(IN_WINDOWS)
    h = pd.read_csv(IN_HOLDOUT)
    w['timestamp'] = pd.to_datetime(w['timestamp'], errors='coerce')
    h['timestamp'] = pd.to_datetime(h['timestamp'], errors='coerce')

    w_feat, feature_cols = build_features(w)
    h_feat, _ = build_features(h)

    pd.DataFrame({'feature': feature_cols}).to_csv(OUT_FEATURES_CSV, index=False)

    win = w_feat.sort_values('timestamp').reset_index(drop=True)
    cut = int(len(win) * base.CFG.train_ratio)
    df_tr = win.iloc[:cut].copy()
    df_va = win.iloc[cut:].copy()
    df_ho = h_feat.sort_values('timestamp').reset_index(drop=True)

    x_tr_raw = df_tr[feature_cols].to_numpy(dtype=np.float32)
    x_va_raw = df_va[feature_cols].to_numpy(dtype=np.float32)
    x_ho_raw = df_ho[feature_cols].to_numpy(dtype=np.float32)

    x_tr, mu, sd = base.normalize_fit(x_tr_raw)
    x_va = base.normalize_apply(x_va_raw, mu, sd)
    x_ho = base.normalize_apply(x_ho_raw, mu, sd)

    a_tr = action_to_idx(df_tr['pred_action'].to_numpy(dtype=np.int64))
    a_va = action_to_idx(df_va['pred_action'].to_numpy(dtype=np.int64))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cost_ret = base.CFG.cost_pips * base.pip_size_for_symbol(base.CFG.symbol)

    trial_rows: List[Dict] = []
    best = None
    best_score = -np.inf

    for t in TRIALS:
        cfg = base.Config()
        cfg.hidden_dim = int(t.hidden_dim)
        cfg.cql_alpha = float(t.cql_alpha)
        cfg.bc_reg = float(t.bc_reg)
        cfg.q_lr = float(t.q_lr)
        cfg.q_epochs = int(t.q_epochs)

        turnover_penalty = float(t.turnover_penalty_mult) * cost_ret

        # transitions with shaped reward
        s_tr, at_tr, r_tr, ns_tr, d_tr = build_transitions_custom(df_tr, feature_cols, turnover_penalty, float(t.reward_scale))
        s_va, at_va, r_va, ns_va, d_va = build_transitions_custom(df_va, feature_cols, turnover_penalty, float(t.reward_scale))

        s_tr = base.normalize_apply(s_tr, mu, sd)
        ns_tr = base.normalize_apply(ns_tr, mu, sd)
        s_va = base.normalize_apply(s_va, mu, sd)
        ns_va = base.normalize_apply(ns_va, mu, sd)

        bc_model, bc_hist = base.train_bc(x_tr, a_tr, x_va, a_va, cfg, device)
        cql_model, cql_hist = base.train_cql(s_tr, at_tr, r_tr, ns_tr, d_tr, s_va, at_va, r_va, ns_va, d_va, cfg, device)

        # Calibrate single policies on validation
        bc_best = base.tune_policy(bc_model, x_va, df_va, cfg, device)
        cql_best = base.tune_policy(cql_model, x_va, df_va, cfg, device)

        # Policy actions
        b_va = df_va['pred_action'].to_numpy(dtype=np.int64)
        b_ho = df_ho['pred_action'].to_numpy(dtype=np.int64)

        a_bc_va = base.predict_policy(bc_model, x_va, temp=bc_best['temp'], conf_thr=bc_best['conf_thr'], device=device)
        a_bc_ho = base.predict_policy(bc_model, x_ho, temp=bc_best['temp'], conf_thr=bc_best['conf_thr'], device=device)

        a_cql_va = base.predict_policy(cql_model, x_va, temp=cql_best['temp'], conf_thr=cql_best['conf_thr'], device=device)
        a_cql_ho = base.predict_policy(cql_model, x_ho, temp=cql_best['temp'], conf_thr=cql_best['conf_thr'], device=device)

        # Advantage-based hybrid
        q_va = q_logits_np(base, cql_model, x_va, device=device)
        q_ho = q_logits_np(base, cql_model, x_ho, device=device)

        hyb_best = tune_adv_hybrid(base, df_va, behavior_actions=b_va, q_logits_val=q_va, cost_ret=cost_ret)

        a_hyb_va = hybrid_by_advantage(
            behavior_actions=b_va,
            q_logits=q_va,
            pred_inf_prob=df_va['pred_inf_prob'].to_numpy(dtype=np.float32),
            conf_thr=hyb_best['conf_thr'],
            adv_thr=hyb_best['adv_thr'],
            inf_min=hyb_best['inf_min'],
        )
        a_hyb_ho = hybrid_by_advantage(
            behavior_actions=b_ho,
            q_logits=q_ho,
            pred_inf_prob=df_ho['pred_inf_prob'].to_numpy(dtype=np.float32),
            conf_thr=hyb_best['conf_thr'],
            adv_thr=hyb_best['adv_thr'],
            inf_min=hyb_best['inf_min'],
        )

        y_va = df_va['y_action'].to_numpy(dtype=np.int64)
        rva = df_va['future_return_h'].to_numpy(dtype=np.float32)
        y_ho = df_ho['y_action'].to_numpy(dtype=np.int64)
        rho = df_ho['future_return_h'].to_numpy(dtype=np.float32)

        m_b_val = base.evaluate_actions(b_va, y_va, rva, cost_ret)
        m_b_ho = base.evaluate_actions(b_ho, y_ho, rho, cost_ret)
        m_bc_val = base.evaluate_actions(a_bc_va, y_va, rva, cost_ret)
        m_bc_ho = base.evaluate_actions(a_bc_ho, y_ho, rho, cost_ret)
        m_cql_val = base.evaluate_actions(a_cql_va, y_va, rva, cost_ret)
        m_cql_ho = base.evaluate_actions(a_cql_ho, y_ho, rho, cost_ret)
        m_hyb_val = base.evaluate_actions(a_hyb_va, y_va, rva, cost_ret)
        m_hyb_ho = base.evaluate_actions(a_hyb_ho, y_ho, rho, cost_ret)

        row = {
            'trial': t.name,
            'hidden_dim': t.hidden_dim,
            'cql_alpha': t.cql_alpha,
            'bc_reg': t.bc_reg,
            'q_lr': t.q_lr,
            'q_epochs': t.q_epochs,
            'turnover_penalty_mult': t.turnover_penalty_mult,
            'reward_scale': t.reward_scale,
            'val_score_hybrid': base.score_from_metrics(m_hyb_val),
            'val_avg_hybrid': m_hyb_val['avg_net_return'],
            'val_sharpe_hybrid': m_hyb_val['sharpe_ann'],
            'holdout_cum_hybrid': m_hyb_ho['cum_net_return'],
            'holdout_cum_behavior': m_b_ho['cum_net_return'],
            'holdout_cum_bc': m_bc_ho['cum_net_return'],
            'holdout_cum_cql': m_cql_ho['cum_net_return'],
            'hyb_conf_thr': hyb_best['conf_thr'],
            'hyb_adv_thr': hyb_best['adv_thr'],
            'hyb_inf_min': hyb_best['inf_min'],
        }
        trial_rows.append(row)

        if row['val_score_hybrid'] > best_score:
            best_score = float(row['val_score_hybrid'])
            best = {
                'trial_cfg': t,
                'cfg': cfg,
                'mu': mu,
                'sd': sd,
                'bc_model': bc_model,
                'cql_model': cql_model,
                'bc_best': bc_best,
                'cql_best': cql_best,
                'hyb_best': hyb_best,
                'metrics': {
                    'behavior_val': m_b_val,
                    'behavior_hold': m_b_ho,
                    'bc_val': m_bc_val,
                    'bc_hold': m_bc_ho,
                    'cql_val': m_cql_val,
                    'cql_hold': m_cql_ho,
                    'hyb_val': m_hyb_val,
                    'hyb_hold': m_hyb_ho,
                },
                'hist_df': pd.DataFrame(bc_hist + cql_hist),
                'actions_hold': {
                    'behavior': b_ho,
                    'bc': a_bc_ho,
                    'cql': a_cql_ho,
                    'hybrid': a_hyb_ho,
                },
                'df_hold': df_ho,
                'feature_cols': feature_cols,
            }

    trials_df = pd.DataFrame(trial_rows)
    trials_df.to_csv(OUT_TRIALS_CSV, index=False)

    # Save best model artifacts
    bt = best['trial_cfg']
    torch.save(
        {
            'state_dict': best['bc_model'].state_dict(),
            'feature_cols': best['feature_cols'],
            'mu': best['mu'],
            'sd': best['sd'],
            'calibration': {'temp': best['bc_best']['temp'], 'conf_thr': best['bc_best']['conf_thr']},
            'config': asdict(best['cfg']),
            'trial': asdict(bt),
        },
        OUT_MODEL_BC,
    )
    torch.save(
        {
            'state_dict': best['cql_model'].state_dict(),
            'feature_cols': best['feature_cols'],
            'mu': best['mu'],
            'sd': best['sd'],
            'calibration': {'temp': best['cql_best']['temp'], 'conf_thr': best['cql_best']['conf_thr']},
            'hybrid': {
                'conf_thr': best['hyb_best']['conf_thr'],
                'adv_thr': best['hyb_best']['adv_thr'],
                'inf_min': best['hyb_best']['inf_min'],
            },
            'config': asdict(best['cfg']),
            'trial': asdict(bt),
        },
        OUT_MODEL_CQL,
    )

    # Compare table
    rows = [
        {'split': 'validation', 'policy': 'behavior', **best['metrics']['behavior_val']},
        {'split': 'validation', 'policy': 'bc', **best['metrics']['bc_val']},
        {'split': 'validation', 'policy': 'cql', **best['metrics']['cql_val']},
        {'split': 'validation', 'policy': 'hybrid_advantage', **best['metrics']['hyb_val']},
        {'split': 'holdout', 'policy': 'behavior', **best['metrics']['behavior_hold']},
        {'split': 'holdout', 'policy': 'bc', **best['metrics']['bc_hold']},
        {'split': 'holdout', 'policy': 'cql', **best['metrics']['cql_hold']},
        {'split': 'holdout', 'policy': 'hybrid_advantage', **best['metrics']['hyb_hold']},
    ]
    cmp_df = pd.DataFrame(rows)
    cmp_df.to_csv(OUT_COMPARE_CSV, index=False)

    best['hist_df'].to_csv(OUT_HISTORY_CSV, index=False)

    # Holdout prediction records
    pred = best['df_hold'][['global_segment_id', 'timestamp', 'step', 'y_action', 'future_return_h']].copy()

    for name, act in best['actions_hold'].items():
        pred[f'action_{name}'] = act
        pred[f'net_{name}'] = pred[f'action_{name}'] * pred['future_return_h'] - (pred[f'action_{name}'] != 0).astype(float) * cost_ret
        pred[f'cum_{name}'] = pred[f'net_{name}'].cumsum()

    pred.to_csv(OUT_HOLDOUT_PRED_CSV, index=False)

    # plots
    plt.figure(figsize=(11, 4))
    for name in ['behavior', 'bc', 'cql', 'hybrid']:
        plt.plot(pred['timestamp'], pred[f'cum_{name}'], label=name)
    plt.title('Phase11.2 holdout cumulative net return')
    plt.xlabel('timestamp')
    plt.ylabel('cumulative return')
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG_EQUITY, dpi=140)
    plt.close()

    # loss plot for best
    h = best['hist_df']
    fig, ax = plt.subplots(figsize=(10, 4))
    hb = h[h['model'] == 'bc']
    hc = h[h['model'] == 'cql']
    if len(hb):
        ax.plot(hb['epoch'], hb['train_loss'], label='BC train')
        ax.plot(hb['epoch'], hb['val_loss'], label='BC val')
    if len(hc):
        ax.plot(hc['epoch'], hc['train_loss'], label='CQL train')
        ax.plot(hc['epoch'], hc['val_loss'], label='CQL val')
    ax.set_title('Phase11.2 best trial training losses')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_PNG_LOSS, dpi=140)
    plt.close(fig)

    summary = {
        'symbol': base.CFG.symbol,
        'model_tag': 'phase11_2_offline_rl_reward_input',
        'description': 'Offline RL improved with richer state input + shaped rewards + advantage-based hybrid policy',
        'best_trial': asdict(bt),
        'best_trial_row': trials_df.sort_values('val_score_hybrid', ascending=False).head(1).to_dict(orient='records')[0],
        'hybrid_calibration': best['hyb_best'],
        'validation_metrics': {
            'behavior': best['metrics']['behavior_val'],
            'bc': best['metrics']['bc_val'],
            'cql': best['metrics']['cql_val'],
            'hybrid_advantage': best['metrics']['hyb_val'],
        },
        'holdout_metrics': {
            'behavior': best['metrics']['behavior_hold'],
            'bc': best['metrics']['bc_hold'],
            'cql': best['metrics']['cql_hold'],
            'hybrid_advantage': best['metrics']['hyb_hold'],
        },
        'outputs': {
            'features_csv': str(OUT_FEATURES_CSV),
            'trials_csv': str(OUT_TRIALS_CSV),
            'compare_csv': str(OUT_COMPARE_CSV),
            'history_csv': str(OUT_HISTORY_CSV),
            'holdout_predictions_csv': str(OUT_HOLDOUT_PRED_CSV),
            'bc_model_pt': str(OUT_MODEL_BC),
            'cql_model_pt': str(OUT_MODEL_CQL),
            'equity_png': str(OUT_PNG_EQUITY),
            'loss_png': str(OUT_PNG_LOSS),
        },
    }

    OUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

    print('Saved summary:', OUT_SUMMARY_JSON)
    print('Best trial:', bt.name)
    print('Holdout metrics:', summary['holdout_metrics'])


if __name__ == '__main__':
    main()
