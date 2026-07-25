#!/usr/bin/env python3
import importlib.util
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

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


TRIALS = [
    TrialCfg('t0_baseline_like', 96, 1.20, 0.03, 8e-4, 45),
    TrialCfg('t1_less_conservative', 128, 0.80, 0.02, 1e-3, 50),
    TrialCfg('t2_more_conservative', 128, 1.60, 0.01, 8e-4, 50),
    TrialCfg('t3_wider_model', 160, 1.00, 0.01, 7e-4, 60),
]


ROOT = Path(__file__).resolve().parents[2]
PHASE11_DIR = ROOT / 'phases/phase11_offline_rl'
PHASE11_BASE_SCRIPT = PHASE11_DIR / 'run_phase11_offline_rl_cql.py'
IN_DIR = ROOT / 'phases/phase10_decision_engine/output_phase10_2_strict_expanding_GBPUSD'
OUT_DIR = PHASE11_DIR / 'output_phase11_1_offline_rl_improved_GBPUSD'
OUT_DIR.mkdir(parents=True, exist_ok=True)

IN_WINDOWS = IN_DIR / 'GBPUSD_phase10_2_windows_predictions.csv'
IN_HOLDOUT = IN_DIR / 'GBPUSD_phase10_2_holdout_predictions.csv'

OUT_TRIALS_CSV = OUT_DIR / 'GBPUSD_phase11_1_trials.csv'
OUT_COMPARE_CSV = OUT_DIR / 'GBPUSD_phase11_1_policy_comparison.csv'
OUT_HOLDOUT_PRED_CSV = OUT_DIR / 'GBPUSD_phase11_1_holdout_predictions.csv'
OUT_HISTORY_CSV = OUT_DIR / 'GBPUSD_phase11_1_best_train_history.csv'
OUT_SUMMARY_JSON = OUT_DIR / 'GBPUSD_phase11_1_summary.json'
OUT_MODEL_BC = OUT_DIR / 'phase11_1_best_bc_model.pt'
OUT_MODEL_CQL = OUT_DIR / 'phase11_1_best_cql_model.pt'
OUT_PNG_EQUITY = OUT_DIR / 'GBPUSD_phase11_1_holdout_equity_compare.png'


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location('phase11_base', str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot load module {path}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def hybrid_actions(
    behavior_actions: np.ndarray,
    model_actions: np.ndarray,
    model_conf: np.ndarray,
    pred_inf_prob: np.ndarray,
    override_conf: float,
    inf_min: float,
) -> np.ndarray:
    out = behavior_actions.copy()
    m = (model_conf >= override_conf) & (pred_inf_prob >= inf_min) & (model_actions != 0)
    out[m] = model_actions[m]
    return out.astype(np.int64)


def tune_hybrid(
    base_mod,
    df_val: pd.DataFrame,
    behavior_actions_val: np.ndarray,
    model_actions_val: np.ndarray,
    model_conf_val: np.ndarray,
    cost_ret: float,
):
    y = df_val['y_action'].to_numpy(dtype=np.int64)
    ret = df_val['future_return_h'].to_numpy(dtype=np.float32)
    inf = df_val['pred_inf_prob'].to_numpy(dtype=np.float32)

    best = None
    for oc in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        for im in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
            a = hybrid_actions(behavior_actions_val, model_actions_val, model_conf_val, inf, override_conf=float(oc), inf_min=float(im))
            m = base_mod.evaluate_actions(a, y, ret, cost_ret)
            score = base_mod.score_from_metrics(m)
            row = {'override_conf': float(oc), 'inf_min': float(im), 'score': float(score), **m}
            if best is None or row['score'] > best['score']:
                best = row
    return best


def model_action_conf(base_mod, model, x: np.ndarray, temp: float, device: torch.device):
    p = base_mod.model_probs(model, x, device=device, temp=float(temp))
    a = base_mod.choose_actions_from_probs(p, conf_thr=0.0)
    c = np.max(p, axis=1)
    return a, c, p


def main():
    base = load_module(PHASE11_BASE_SCRIPT)

    df_windows = pd.read_csv(IN_WINDOWS)
    df_holdout = pd.read_csv(IN_HOLDOUT)
    df_windows['timestamp'] = pd.to_datetime(df_windows['timestamp'], errors='coerce')
    df_holdout['timestamp'] = pd.to_datetime(df_holdout['timestamp'], errors='coerce')

    feature_cols = [
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
    ]

    win = df_windows.sort_values('timestamp').reset_index(drop=True)
    cut = int(len(win) * base.CFG.train_ratio)
    df_tr = win.iloc[:cut].copy()
    df_va = win.iloc[cut:].copy()

    x_tr_raw = df_tr[feature_cols].to_numpy(dtype=np.float32)
    x_va_raw = df_va[feature_cols].to_numpy(dtype=np.float32)
    x_ho_raw = df_holdout[feature_cols].to_numpy(dtype=np.float32)

    x_tr, mu, sd = base.normalize_fit(x_tr_raw)
    x_va = base.normalize_apply(x_va_raw, mu, sd)
    x_ho = base.normalize_apply(x_ho_raw, mu, sd)

    a_tr = base.action_to_idx(df_tr['pred_action'].to_numpy(dtype=np.int64))
    a_va = base.action_to_idx(df_va['pred_action'].to_numpy(dtype=np.int64))

    s_tr, at_tr, r_tr, ns_tr, d_tr = base.build_transitions(df_tr, feature_cols)
    s_va, at_va, r_va, ns_va, d_va = base.build_transitions(df_va, feature_cols)
    s_tr = base.normalize_apply(s_tr, mu, sd)
    ns_tr = base.normalize_apply(ns_tr, mu, sd)
    s_va = base.normalize_apply(s_va, mu, sd)
    ns_va = base.normalize_apply(ns_va, mu, sd)

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

        bc_model, bc_hist = base.train_bc(x_tr, a_tr, x_va, a_va, cfg, device)
        cql_model, cql_hist = base.train_cql(s_tr, at_tr, r_tr, ns_tr, d_tr, s_va, at_va, r_va, ns_va, d_va, cfg, device)

        bc_best = base.tune_policy(bc_model, x_va, df_va, cfg, device)
        cql_best = base.tune_policy(cql_model, x_va, df_va, cfg, device)

        # actions from calibrated policies
        a_bc_va = base.predict_policy(bc_model, x_va, temp=bc_best['temp'], conf_thr=bc_best['conf_thr'], device=device)
        a_bc_ho = base.predict_policy(bc_model, x_ho, temp=bc_best['temp'], conf_thr=bc_best['conf_thr'], device=device)

        a_cql_va = base.predict_policy(cql_model, x_va, temp=cql_best['temp'], conf_thr=cql_best['conf_thr'], device=device)
        a_cql_ho = base.predict_policy(cql_model, x_ho, temp=cql_best['temp'], conf_thr=cql_best['conf_thr'], device=device)

        # hybrid cql + behavior fallback
        a_cql_raw_va, cql_conf_va, _ = model_action_conf(base, cql_model, x_va, temp=cql_best['temp'], device=device)
        a_cql_raw_ho, cql_conf_ho, _ = model_action_conf(base, cql_model, x_ho, temp=cql_best['temp'], device=device)

        b_va = df_va['pred_action'].to_numpy(dtype=np.int64)
        b_ho = df_holdout['pred_action'].to_numpy(dtype=np.int64)

        hyb_best = tune_hybrid(base, df_va, b_va, a_cql_raw_va, cql_conf_va, cost_ret=cost_ret)

        a_hyb_va = hybrid_actions(
            behavior_actions=b_va,
            model_actions=a_cql_raw_va,
            model_conf=cql_conf_va,
            pred_inf_prob=df_va['pred_inf_prob'].to_numpy(dtype=np.float32),
            override_conf=hyb_best['override_conf'],
            inf_min=hyb_best['inf_min'],
        )
        a_hyb_ho = hybrid_actions(
            behavior_actions=b_ho,
            model_actions=a_cql_raw_ho,
            model_conf=cql_conf_ho,
            pred_inf_prob=df_holdout['pred_inf_prob'].to_numpy(dtype=np.float32),
            override_conf=hyb_best['override_conf'],
            inf_min=hyb_best['inf_min'],
        )

        y_va = df_va['y_action'].to_numpy(dtype=np.int64)
        rva = df_va['future_return_h'].to_numpy(dtype=np.float32)
        y_ho = df_holdout['y_action'].to_numpy(dtype=np.int64)
        rho = df_holdout['future_return_h'].to_numpy(dtype=np.float32)

        m_b_val = base.evaluate_actions(b_va, y_va, rva, cost_ret)
        m_b_hold = base.evaluate_actions(b_ho, y_ho, rho, cost_ret)

        m_bc_val = base.evaluate_actions(a_bc_va, y_va, rva, cost_ret)
        m_bc_hold = base.evaluate_actions(a_bc_ho, y_ho, rho, cost_ret)

        m_cql_val = base.evaluate_actions(a_cql_va, y_va, rva, cost_ret)
        m_cql_hold = base.evaluate_actions(a_cql_ho, y_ho, rho, cost_ret)

        m_hyb_val = base.evaluate_actions(a_hyb_va, y_va, rva, cost_ret)
        m_hyb_hold = base.evaluate_actions(a_hyb_ho, y_ho, rho, cost_ret)

        row = {
            'trial': t.name,
            'hidden_dim': t.hidden_dim,
            'cql_alpha': t.cql_alpha,
            'bc_reg': t.bc_reg,
            'q_lr': t.q_lr,
            'q_epochs': t.q_epochs,
            'val_score_hybrid': base.score_from_metrics(m_hyb_val),
            'val_avg_hybrid': m_hyb_val['avg_net_return'],
            'val_sharpe_hybrid': m_hyb_val['sharpe_ann'],
            'holdout_avg_hybrid': m_hyb_hold['avg_net_return'],
            'holdout_cum_hybrid': m_hyb_hold['cum_net_return'],
            'holdout_sharpe_hybrid': m_hyb_hold['sharpe_ann'],
            'holdout_cum_behavior': m_b_hold['cum_net_return'],
            'holdout_cum_bc': m_bc_hold['cum_net_return'],
            'holdout_cum_cql': m_cql_hold['cum_net_return'],
            'hybrid_override_conf': hyb_best['override_conf'],
            'hybrid_inf_min': hyb_best['inf_min'],
        }
        trial_rows.append(row)

        if best is None or row['val_score_hybrid'] > best_score:
            best_score = float(row['val_score_hybrid'])
            best = {
                'trial_cfg': t,
                'cfg': cfg,
                'bc_model': bc_model,
                'cql_model': cql_model,
                'bc_best': bc_best,
                'cql_best': cql_best,
                'hyb_best': hyb_best,
                'metrics': {
                    'behavior_val': m_b_val,
                    'behavior_hold': m_b_hold,
                    'bc_val': m_bc_val,
                    'bc_hold': m_bc_hold,
                    'cql_val': m_cql_val,
                    'cql_hold': m_cql_hold,
                    'hyb_val': m_hyb_val,
                    'hyb_hold': m_hyb_hold,
                },
                'hist_df': pd.DataFrame(bc_hist + cql_hist),
                'actions_holdout': {
                    'behavior': b_ho,
                    'bc': a_bc_ho,
                    'cql': a_cql_ho,
                    'hybrid': a_hyb_ho,
                },
            }

    trials_df = pd.DataFrame(trial_rows)
    trials_df.to_csv(OUT_TRIALS_CSV, index=False)

    # Save best models
    best_trial = best['trial_cfg']
    best_cfg = best['cfg']
    torch.save(
        {
            'state_dict': best['bc_model'].state_dict(),
            'feature_cols': feature_cols,
            'mu': mu,
            'sd': sd,
            'calibration': {'temp': best['bc_best']['temp'], 'conf_thr': best['bc_best']['conf_thr']},
            'config': asdict(best_cfg),
            'trial': asdict(best_trial),
        },
        OUT_MODEL_BC,
    )
    torch.save(
        {
            'state_dict': best['cql_model'].state_dict(),
            'feature_cols': feature_cols,
            'mu': mu,
            'sd': sd,
            'calibration': {'temp': best['cql_best']['temp'], 'conf_thr': best['cql_best']['conf_thr']},
            'hybrid': {'override_conf': best['hyb_best']['override_conf'], 'inf_min': best['hyb_best']['inf_min']},
            'config': asdict(best_cfg),
            'trial': asdict(best_trial),
        },
        OUT_MODEL_CQL,
    )

    # Save history of best trial
    best['hist_df'].to_csv(OUT_HISTORY_CSV, index=False)

    # Compare table for best
    rows = [
        {'split': 'validation', 'policy': 'behavior', **best['metrics']['behavior_val']},
        {'split': 'validation', 'policy': 'bc', **best['metrics']['bc_val']},
        {'split': 'validation', 'policy': 'cql', **best['metrics']['cql_val']},
        {'split': 'validation', 'policy': 'hybrid_cql_behavior', **best['metrics']['hyb_val']},
        {'split': 'holdout', 'policy': 'behavior', **best['metrics']['behavior_hold']},
        {'split': 'holdout', 'policy': 'bc', **best['metrics']['bc_hold']},
        {'split': 'holdout', 'policy': 'cql', **best['metrics']['cql_hold']},
        {'split': 'holdout', 'policy': 'hybrid_cql_behavior', **best['metrics']['hyb_hold']},
    ]
    cmp_df = pd.DataFrame(rows)
    cmp_df.to_csv(OUT_COMPARE_CSV, index=False)

    # Holdout prediction curves
    pred = df_holdout[['global_segment_id', 'timestamp', 'step', 'y_action', 'future_return_h']].copy()
    cost_ret = base.CFG.cost_pips * base.pip_size_for_symbol(base.CFG.symbol)

    for name, act in best['actions_holdout'].items():
        pred[f'action_{name}'] = act
        pred[f'net_{name}'] = pred[f'action_{name}'] * pred['future_return_h'] - (pred[f'action_{name}'] != 0).astype(float) * cost_ret
        pred[f'cum_{name}'] = pred[f'net_{name}'].cumsum()

    pred.to_csv(OUT_HOLDOUT_PRED_CSV, index=False)

    plt.figure(figsize=(11, 4))
    for name in ['behavior', 'bc', 'cql', 'hybrid']:
        plt.plot(pred['timestamp'], pred[f'cum_{name}'], label=name)
    plt.title('Phase11.1 holdout cumulative net return')
    plt.xlabel('timestamp')
    plt.ylabel('cumulative return')
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG_EQUITY, dpi=140)
    plt.close()

    summary = {
        'symbol': base.CFG.symbol,
        'model_tag': 'phase11_1_offline_rl_improved',
        'description': 'Offline RL improvement with trial search + hybrid CQL/behavior policy',
        'best_trial': asdict(best_trial),
        'best_trial_scores': trials_df.sort_values('val_score_hybrid', ascending=False).head(1).to_dict(orient='records')[0],
        'hybrid_calibration': best['hyb_best'],
        'validation_metrics': {
            'behavior': best['metrics']['behavior_val'],
            'bc': best['metrics']['bc_val'],
            'cql': best['metrics']['cql_val'],
            'hybrid_cql_behavior': best['metrics']['hyb_val'],
        },
        'holdout_metrics': {
            'behavior': best['metrics']['behavior_hold'],
            'bc': best['metrics']['bc_hold'],
            'cql': best['metrics']['cql_hold'],
            'hybrid_cql_behavior': best['metrics']['hyb_hold'],
        },
        'outputs': {
            'trials_csv': str(OUT_TRIALS_CSV),
            'compare_csv': str(OUT_COMPARE_CSV),
            'history_csv': str(OUT_HISTORY_CSV),
            'holdout_predictions_csv': str(OUT_HOLDOUT_PRED_CSV),
            'bc_model_pt': str(OUT_MODEL_BC),
            'cql_model_pt': str(OUT_MODEL_CQL),
            'equity_plot_png': str(OUT_PNG_EQUITY),
        },
    }

    OUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

    print('Saved summary:', OUT_SUMMARY_JSON)
    print('Best trial:', best_trial.name)
    print('Holdout metrics:', summary['holdout_metrics'])


if __name__ == '__main__':
    main()
