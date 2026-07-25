#!/usr/bin/env python3
import importlib.util
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


@dataclass
class Config:
    symbol: str = 'GBPUSD'

    train_ratio: float = 0.80

    # RL model
    gamma: float = 0.0
    batch_size: int = 512
    hidden_dim: int = 160
    q_lr: float = 8e-4
    q_weight_decay: float = 1e-4
    q_epochs: int = 55
    q_patience: int = 8
    q_target_tau: float = 0.02

    # Policy calibration
    temp_grid: Tuple[float, ...] = (0.8, 1.0, 1.2)
    conf_grid: Tuple[float, ...] = (0.34, 0.40, 0.46, 0.52, 0.58, 0.64, 0.70)
    max_lev_grid: Tuple[float, ...] = (1.0, 1.5, 2.0)

    # Trading cost
    cost_pips: float = 1.2


ROOT = Path(__file__).resolve().parents[2]
PHASE10_OUT_DIR = ROOT / 'phases/phase10_decision_engine/output_phase10_2_strict_expanding_GBPUSD'
PHASE11_DIR = ROOT / 'phases/phase11_offline_rl'
PHASE12_DIR = ROOT / 'phases/phase12_leverage_rl'

PHASE11_BASE_SCRIPT = PHASE11_DIR / 'run_phase11_offline_rl_cql.py'
PHASE11_2_SCRIPT = PHASE11_DIR / 'run_phase11_2_offline_rl_reward_input.py'

IN_WINDOWS = PHASE10_OUT_DIR / 'GBPUSD_phase10_2_windows_predictions.csv'
IN_HOLDOUT = PHASE10_OUT_DIR / 'GBPUSD_phase10_2_holdout_predictions.csv'

OUT_DIR = PHASE12_DIR / 'output_phase12_1_offline_rl_leverage_GBPUSD'
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_TRIALS_CSV = OUT_DIR / 'GBPUSD_phase12_1_trials.csv'
OUT_COMPARE_CSV = OUT_DIR / 'GBPUSD_phase12_1_policy_comparison.csv'
OUT_HOLDOUT_PRED_CSV = OUT_DIR / 'GBPUSD_phase12_1_holdout_predictions.csv'
OUT_HISTORY_CSV = OUT_DIR / 'GBPUSD_phase12_1_train_history.csv'
OUT_SUMMARY_JSON = OUT_DIR / 'GBPUSD_phase12_1_summary.json'
OUT_MODEL_PT = OUT_DIR / 'phase12_1_best_leverage_q_model.pt'
OUT_EQUITY_PNG = OUT_DIR / 'GBPUSD_phase12_1_holdout_equity_compare.png'
OUT_LOSS_PNG = OUT_DIR / 'GBPUSD_phase12_1_training_losses.png'
OUT_LEV_HIST_PNG = OUT_DIR / 'GBPUSD_phase12_1_holdout_leverage_hist.png'

CFG = Config()

ACTION_LEVELS = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float32)
TRIALS = [
    {'lev_penalty': 0.0, 'cost_mult': 0.5},
    {'lev_penalty': 0.0, 'cost_mult': 1.0},
    {'lev_penalty': 2e-5, 'cost_mult': 0.5},
    {'lev_penalty': 2e-5, 'cost_mult': 1.0},
    {'lev_penalty': 5e-5, 'cost_mult': 1.0},
]


class TransitionDataset(Dataset):
    def __init__(self, s: np.ndarray, a: np.ndarray, r: np.ndarray, ns: np.ndarray, done: np.ndarray):
        self.s = torch.tensor(s, dtype=torch.float32)
        self.a = torch.tensor(a, dtype=torch.long)
        self.r = torch.tensor(r, dtype=torch.float32)
        self.ns = torch.tensor(ns, dtype=torch.float32)
        self.done = torch.tensor(done, dtype=torch.float32)

    def __len__(self):
        return len(self.a)

    def __getitem__(self, idx):
        return self.s[idx], self.a[idx], self.r[idx], self.ns[idx], self.done[idx]


class QNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot load module {path}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def normalize_fit(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    xn = (x - mu) / sd
    xn = np.nan_to_num(xn, nan=0.0, posinf=0.0, neginf=0.0)
    return xn.astype(np.float32), mu.astype(np.float32), sd.astype(np.float32)


def normalize_apply(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    xn = (x - mu) / sd
    xn = np.nan_to_num(xn, nan=0.0, posinf=0.0, neginf=0.0)
    return xn.astype(np.float32)


def pip_size_for_symbol(symbol: str) -> float:
    return 0.01 if symbol.upper().endswith('JPY') else 0.0001


def level_to_action(levels: np.ndarray) -> np.ndarray:
    return np.sign(levels).astype(np.int64)


def evaluate_levels(levels: np.ndarray, y_true_action: np.ndarray, future_ret: np.ndarray, cost_ret: float) -> Dict[str, float]:
    action = level_to_action(levels)

    gross = levels * future_ret
    cost = np.abs(levels) * cost_ret
    net = gross - cost

    out = {}
    out['acc_action_3'] = float((action == y_true_action).mean())
    out['f1_macro_action_3'] = float(f1_score(y_true_action, action, labels=[-1, 0, 1], average='macro', zero_division=0))

    out['avg_net_return'] = float(np.mean(net))
    out['cum_net_return'] = float(np.sum(net))
    out['std_net_return'] = float(np.std(net))
    out['trade_rate'] = float(np.mean(levels != 0))

    eps = 1e-12
    out['sharpe_ann'] = float((out['avg_net_return'] / (out['std_net_return'] + eps)) * np.sqrt(24.0 * 252.0))

    traded = np.abs(levels) > 0
    out['win_rate_trades'] = float((net[traded] > 0).mean()) if traded.any() else 0.0
    out['avg_abs_lev'] = float(np.mean(np.abs(levels)))
    out['avg_abs_lev_on_trade'] = float(np.mean(np.abs(levels[traded]))) if traded.any() else 0.0
    return out


def score_from_metrics(m: Dict[str, float]) -> float:
    trade_pen = -5.0 if float(m['trade_rate']) < 0.05 else 0.0
    return float(m['avg_net_return']) * 1e4 + 0.10 * float(m['sharpe_ann']) + trade_pen


def build_full_action_transitions(
    df: pd.DataFrame,
    x_norm: np.ndarray,
    action_levels: np.ndarray,
    cost_ret: float,
    lev_penalty: float,
    cost_mult: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Build index aligned with sorted dataframe for contiguous next-state lookup.
    work = df.copy()
    work['__rowpos__'] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values(['global_segment_id', 'step', 'timestamp']).copy()
    src_idx = work['__rowpos__'].to_numpy(dtype=np.int64)
    x_work = x_norm[src_idx]
    work = work.drop(columns=['__rowpos__']).reset_index(drop=True)

    # map sorted rows to contiguous next-state pointer
    n = len(work)
    next_idx = np.full((n,), -1, dtype=np.int64)
    gs = work['global_segment_id'].astype(str).to_numpy()
    st = pd.to_numeric(work['step'], errors='coerce').fillna(-1).astype(np.int64).to_numpy()

    for i in range(n - 1):
        if gs[i + 1] == gs[i] and st[i + 1] == st[i] + 1:
            next_idx[i] = i + 1

    zero_state = np.zeros((x_work.shape[1],), dtype=np.float32)
    future_ret = work['future_return_h'].to_numpy(dtype=np.float32)

    s_list, a_list, r_list, ns_list, d_list = [], [], [], [], []

    for i in range(n):
        s = x_work[i]

        if next_idx[i] >= 0:
            ns = x_work[next_idx[i]]
            done = 0.0
        else:
            ns = zero_state
            done = 1.0

        ret = float(future_ret[i])

        for a_idx, lev in enumerate(action_levels):
            # cost scales with leverage magnitude
            reward = lev * ret - abs(float(lev)) * cost_ret * float(cost_mult) - lev_penalty * (max(abs(float(lev)) - 1.0, 0.0) ** 2)

            s_list.append(s)
            a_list.append(a_idx)
            r_list.append(reward)
            ns_list.append(ns)
            d_list.append(done)

    return (
        np.asarray(s_list, dtype=np.float32),
        np.asarray(a_list, dtype=np.int64),
        np.asarray(r_list, dtype=np.float32),
        np.asarray(ns_list, dtype=np.float32),
        np.asarray(d_list, dtype=np.float32),
    )


def train_q(
    s_train: np.ndarray,
    a_train: np.ndarray,
    r_train: np.ndarray,
    ns_train: np.ndarray,
    d_train: np.ndarray,
    s_val: np.ndarray,
    a_val: np.ndarray,
    r_val: np.ndarray,
    ns_val: np.ndarray,
    d_val: np.ndarray,
    cfg: Config,
    device: torch.device,
):
    out_dim = len(ACTION_LEVELS)
    q = QNet(in_dim=s_train.shape[1], hidden_dim=cfg.hidden_dim, out_dim=out_dim).to(device)
    q_tgt = QNet(in_dim=s_train.shape[1], hidden_dim=cfg.hidden_dim, out_dim=out_dim).to(device)
    q_tgt.load_state_dict(q.state_dict())

    ds_train = TransitionDataset(s_train, a_train, r_train, ns_train, d_train)
    ds_val = TransitionDataset(s_val, a_val, r_val, ns_val, d_val)
    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False)

    opt = torch.optim.AdamW(q.parameters(), lr=cfg.q_lr, weight_decay=cfg.q_weight_decay)

    best_state = None
    best_val = np.inf
    wait = 0
    hist = []

    for epoch in range(1, cfg.q_epochs + 1):
        q.train()
        tr_loss = []

        for sb, ab, rb, nsb, db in dl_train:
            sb = sb.to(device)
            ab = ab.to(device)
            rb = rb.to(device)
            nsb = nsb.to(device)
            db = db.to(device)

            q_logits = q(sb)
            q_sa = q_logits.gather(1, ab.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                q_next = q_tgt(nsb).max(dim=1).values
                y = rb + cfg.gamma * (1.0 - db) * q_next

            loss = F.mse_loss(q_sa, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 1.0)
            opt.step()

            with torch.no_grad():
                for p_tgt, p in zip(q_tgt.parameters(), q.parameters()):
                    p_tgt.data.mul_(1.0 - cfg.q_target_tau).add_(cfg.q_target_tau * p.data)

            tr_loss.append(float(loss.detach().cpu()))

        q.eval()
        va_loss = []
        with torch.no_grad():
            for sb, ab, rb, nsb, db in dl_val:
                sb = sb.to(device)
                ab = ab.to(device)
                rb = rb.to(device)
                nsb = nsb.to(device)
                db = db.to(device)

                q_logits = q(sb)
                q_sa = q_logits.gather(1, ab.unsqueeze(1)).squeeze(1)
                q_next = q_tgt(nsb).max(dim=1).values
                y = rb + cfg.gamma * (1.0 - db) * q_next
                loss = F.mse_loss(q_sa, y)
                va_loss.append(float(loss.detach().cpu()))

        tr = float(np.mean(tr_loss))
        va = float(np.mean(va_loss))
        hist.append({'epoch': epoch, 'train_loss': tr, 'val_loss': va})

        if va < best_val - 1e-5:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in q.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= cfg.q_patience:
                break

    if best_state is not None:
        q.load_state_dict(best_state)

    return q, hist


def q_logits_np(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        bs = 2048
        for i in range(0, len(x), bs):
            xb = torch.tensor(x[i:i + bs], dtype=torch.float32, device=device)
            logits = model(xb)
            out.append(logits.cpu().numpy())
    return np.concatenate(out, axis=0)


def apply_hybrid_leverage_from_logits(
    q_logits: np.ndarray,
    behavior_actions: np.ndarray,
    temp: float,
    conf_thr: float,
    max_lev: float,
) -> np.ndarray:
    out = np.zeros((len(q_logits),), dtype=np.float32)
    zero_idx = int(np.where(ACTION_LEVELS == 0.0)[0][0])

    for i in range(len(q_logits)):
        b = int(behavior_actions[i])
        if b == 0:
            out[i] = 0.0
            continue

        mask_sign = np.sign(ACTION_LEVELS) == float(b)
        mask_lev = np.abs(ACTION_LEVELS) <= float(max_lev)
        cand = np.where(mask_sign & mask_lev)[0]
        cand = np.unique(np.append(cand, zero_idx))

        logits = q_logits[i, cand] / max(float(temp), 1e-6)
        logits = logits - np.max(logits)
        p = np.exp(logits)
        p = p / max(float(np.sum(p)), 1e-8)

        j = int(np.argmax(p))
        lev = float(ACTION_LEVELS[cand[j]])
        conf = float(p[j])

        # fallback to baseline 1x behavior direction on low confidence
        if conf < float(conf_thr):
            lev = float(b)

        out[i] = lev

    return out.astype(np.float32)


def tune_policy(
    model: nn.Module,
    x_val: np.ndarray,
    df_val: pd.DataFrame,
    behavior_actions_val: np.ndarray,
    cfg: Config,
    device: torch.device,
    cost_ret: float,
):
    y_true = df_val['y_action'].to_numpy(dtype=np.int64)
    ret = df_val['future_return_h'].to_numpy(dtype=np.float32)
    q_val = q_logits_np(model, x_val, device=device)

    best = None
    for temp in cfg.temp_grid:
        for conf in cfg.conf_grid:
            for max_lev in cfg.max_lev_grid:
                lev = apply_hybrid_leverage_from_logits(
                    q_logits=q_val,
                    behavior_actions=behavior_actions_val,
                    temp=float(temp),
                    conf_thr=float(conf),
                    max_lev=float(max_lev),
                )
                m = evaluate_levels(lev, y_true, ret, cost_ret=cost_ret)
                score = score_from_metrics(m)
                row = {
                    'temp': float(temp),
                    'conf_thr': float(conf),
                    'max_lev': float(max_lev),
                    'score': float(score),
                    **m,
                }
                if best is None or row['score'] > best['score']:
                    best = row
    return best


def main():
    base = load_module(PHASE11_BASE_SCRIPT, name='phase11_base_for_phase12')
    p112 = load_module(PHASE11_2_SCRIPT, name='phase11_2_for_phase12')

    w = pd.read_csv(IN_WINDOWS)
    h = pd.read_csv(IN_HOLDOUT)
    w['timestamp'] = pd.to_datetime(w['timestamp'], errors='coerce')
    h['timestamp'] = pd.to_datetime(h['timestamp'], errors='coerce')

    w_feat, feature_cols = p112.build_features(w)
    h_feat, _ = p112.build_features(h)

    win = w_feat.sort_values('timestamp').reset_index(drop=True)
    hold = h_feat.sort_values('timestamp').reset_index(drop=True)

    cut = int(len(win) * CFG.train_ratio)
    df_tr = win.iloc[:cut].copy()
    df_va = win.iloc[cut:].copy()
    df_ho = hold.copy()

    x_tr_raw = df_tr[feature_cols].to_numpy(dtype=np.float32)
    x_va_raw = df_va[feature_cols].to_numpy(dtype=np.float32)
    x_ho_raw = df_ho[feature_cols].to_numpy(dtype=np.float32)

    x_tr, mu, sd = normalize_fit(x_tr_raw)
    x_va = normalize_apply(x_va_raw, mu, sd)
    x_ho = normalize_apply(x_ho_raw, mu, sd)

    # metrics inputs
    y_va = df_va['y_action'].to_numpy(dtype=np.int64)
    ret_va = df_va['future_return_h'].to_numpy(dtype=np.float32)
    y_ho = df_ho['y_action'].to_numpy(dtype=np.int64)
    ret_ho = df_ho['future_return_h'].to_numpy(dtype=np.float32)
    b_va = df_va['pred_action'].to_numpy(dtype=np.int64)
    b_ho = df_ho['pred_action'].to_numpy(dtype=np.int64)

    cost_ret = CFG.cost_pips * pip_size_for_symbol(CFG.symbol)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trial_rows: List[Dict] = []
    best = None
    best_score = -np.inf

    for t in TRIALS:
        lev_penalty = float(t['lev_penalty'])
        cost_mult = float(t['cost_mult'])
        s_tr, a_tr, r_tr, ns_tr, d_tr = build_full_action_transitions(
            df=df_tr,
            x_norm=x_tr,
            action_levels=ACTION_LEVELS,
            cost_ret=cost_ret,
            lev_penalty=lev_penalty,
            cost_mult=cost_mult,
        )
        s_va, a_va, r_va, ns_va, d_va = build_full_action_transitions(
            df=df_va,
            x_norm=x_va,
            action_levels=ACTION_LEVELS,
            cost_ret=cost_ret,
            lev_penalty=lev_penalty,
            cost_mult=cost_mult,
        )

        model, hist = train_q(
            s_train=s_tr,
            a_train=a_tr,
            r_train=r_tr,
            ns_train=ns_tr,
            d_train=d_tr,
            s_val=s_va,
            a_val=a_va,
            r_val=r_va,
            ns_val=ns_va,
            d_val=d_va,
            cfg=CFG,
            device=device,
        )

        pol_best = tune_policy(
            model=model,
            x_val=x_va,
            df_val=df_va,
            behavior_actions_val=b_va,
            cfg=CFG,
            device=device,
            cost_ret=cost_ret,
        )

        q_va = q_logits_np(model, x_va, device=device)
        q_ho = q_logits_np(model, x_ho, device=device)

        lev_va = apply_hybrid_leverage_from_logits(
            q_logits=q_va,
            behavior_actions=b_va,
            temp=float(pol_best['temp']),
            conf_thr=float(pol_best['conf_thr']),
            max_lev=float(pol_best['max_lev']),
        )
        lev_ho = apply_hybrid_leverage_from_logits(
            q_logits=q_ho,
            behavior_actions=b_ho,
            temp=float(pol_best['temp']),
            conf_thr=float(pol_best['conf_thr']),
            max_lev=float(pol_best['max_lev']),
        )

        m_q_va = evaluate_levels(lev_va, y_va, ret_va, cost_ret=cost_ret)
        m_q_ho = evaluate_levels(lev_ho, y_ho, ret_ho, cost_ret=cost_ret)

        lev_b_va = b_va.astype(np.float32)
        lev_b_ho = b_ho.astype(np.float32)
        lev_b2_va = 2.0 * lev_b_va
        lev_b2_ho = 2.0 * lev_b_ho

        m_b_va = evaluate_levels(lev_b_va, y_va, ret_va, cost_ret=cost_ret)
        m_b_ho = evaluate_levels(lev_b_ho, y_ho, ret_ho, cost_ret=cost_ret)
        m_b2_va = evaluate_levels(lev_b2_va, y_va, ret_va, cost_ret=cost_ret)
        m_b2_ho = evaluate_levels(lev_b2_ho, y_ho, ret_ho, cost_ret=cost_ret)

        row = {
            'lev_penalty': float(lev_penalty),
            'cost_mult': float(cost_mult),
            'val_score_q': float(score_from_metrics(m_q_va)),
            'val_avg_q': float(m_q_va['avg_net_return']),
            'val_sharpe_q': float(m_q_va['sharpe_ann']),
            'holdout_cum_q': float(m_q_ho['cum_net_return']),
            'holdout_cum_behavior_1x': float(m_b_ho['cum_net_return']),
            'holdout_cum_behavior_2x': float(m_b2_ho['cum_net_return']),
            'best_temp': float(pol_best['temp']),
            'best_conf_thr': float(pol_best['conf_thr']),
            'best_max_lev': float(pol_best['max_lev']),
        }
        trial_rows.append(row)

        score = row['val_score_q']
        if score > best_score:
            best_score = score
            best = {
                'lev_penalty': float(lev_penalty),
                'cost_mult': float(cost_mult),
                'model': model,
                'hist': pd.DataFrame(hist),
                'policy': pol_best,
                'metrics': {
                    'b_val': m_b_va,
                    'b_hold': m_b_ho,
                    'b2_val': m_b2_va,
                    'b2_hold': m_b2_ho,
                    'q_val': m_q_va,
                    'q_hold': m_q_ho,
                },
                'levels_hold': {
                    'behavior_1x': lev_b_ho,
                    'behavior_2x': lev_b2_ho,
                    'q_leverage': lev_ho,
                },
                'mu': mu,
                'sd': sd,
                'feature_cols': feature_cols,
            }

    trials_df = pd.DataFrame(trial_rows).sort_values('val_score_q', ascending=False)
    trials_df.to_csv(OUT_TRIALS_CSV, index=False)

    # Save model
    torch.save(
        {
            'state_dict': best['model'].state_dict(),
            'feature_cols': best['feature_cols'],
            'mu': best['mu'],
            'sd': best['sd'],
            'action_levels': ACTION_LEVELS.tolist(),
            'policy': {
                'temp': float(best['policy']['temp']),
                'conf_thr': float(best['policy']['conf_thr']),
                'max_lev': float(best['policy']['max_lev']),
                'lev_penalty': float(best['lev_penalty']),
                'cost_mult': float(best['cost_mult']),
            },
            'config': asdict(CFG),
        },
        OUT_MODEL_PT,
    )

    # Compare table
    cmp_rows = [
        {'split': 'validation', 'policy': 'behavior_1x', **best['metrics']['b_val']},
        {'split': 'validation', 'policy': 'behavior_2x', **best['metrics']['b2_val']},
        {'split': 'validation', 'policy': 'q_leverage', **best['metrics']['q_val']},
        {'split': 'holdout', 'policy': 'behavior_1x', **best['metrics']['b_hold']},
        {'split': 'holdout', 'policy': 'behavior_2x', **best['metrics']['b2_hold']},
        {'split': 'holdout', 'policy': 'q_leverage', **best['metrics']['q_hold']},
    ]
    cmp_df = pd.DataFrame(cmp_rows)
    cmp_df.to_csv(OUT_COMPARE_CSV, index=False)

    # holdout predictions
    pred = df_ho[['global_segment_id', 'timestamp', 'step', 'y_action', 'future_return_h']].copy()
    ret = pred['future_return_h'].to_numpy(dtype=np.float32)

    for name, lev in best['levels_hold'].items():
        pred[f'lev_{name}'] = lev
        pred[f'action_{name}'] = np.sign(lev).astype(np.int64)
        pred[f'net_{name}'] = lev * ret - np.abs(lev) * cost_ret
        pred[f'cum_{name}'] = pred[f'net_{name}'].cumsum()

    pred.to_csv(OUT_HOLDOUT_PRED_CSV, index=False)

    # plots
    plt.figure(figsize=(11, 4))
    for name in ['behavior_1x', 'behavior_2x', 'q_leverage']:
        plt.plot(pred['timestamp'], pred[f'cum_{name}'], label=name)
    plt.title('Phase12.1 holdout cumulative net return (leverage-aware)')
    plt.xlabel('timestamp')
    plt.ylabel('cumulative return')
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_EQUITY_PNG, dpi=140)
    plt.close()

    hist = best['hist']
    hist.to_csv(OUT_HISTORY_CSV, index=False)

    plt.figure(figsize=(10, 4))
    plt.plot(hist['epoch'], hist['train_loss'], label='train')
    plt.plot(hist['epoch'], hist['val_loss'], label='val')
    plt.title('Phase12.1 Q training loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_LOSS_PNG, dpi=140)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.hist(best['levels_hold']['q_leverage'], bins=[-2.25, -1.75, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75, 2.25])
    plt.title('Phase12.1 holdout leverage distribution (q_leverage)')
    plt.xlabel('leverage level')
    plt.ylabel('count')
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_LEV_HIST_PNG, dpi=140)
    plt.close()

    summary = {
        'symbol': CFG.symbol,
        'model_tag': 'phase12_1_offline_rl_leverage',
        'description': 'Offline RL with leverage-aware action space and risk calibration',
        'action_levels': ACTION_LEVELS.tolist(),
        'cost_pips': CFG.cost_pips,
        'dataset': {
            'n_windows_total': int(len(win)),
            'n_windows_train': int(len(df_tr)),
            'n_windows_val': int(len(df_va)),
            'n_holdout': int(len(df_ho)),
            'windows_date_min': str(win['timestamp'].min()),
            'windows_date_max': str(win['timestamp'].max()),
            'holdout_date_min': str(df_ho['timestamp'].min()),
            'holdout_date_max': str(df_ho['timestamp'].max()),
        },
        'best_trial': trials_df.head(1).to_dict(orient='records')[0],
        'best_policy': {
            'temp': float(best['policy']['temp']),
            'conf_thr': float(best['policy']['conf_thr']),
            'max_lev': float(best['policy']['max_lev']),
            'lev_penalty': float(best['lev_penalty']),
            'cost_mult': float(best['cost_mult']),
        },
        'validation_metrics': {
            'behavior_1x': best['metrics']['b_val'],
            'behavior_2x': best['metrics']['b2_val'],
            'q_leverage': best['metrics']['q_val'],
        },
        'holdout_metrics': {
            'behavior_1x': best['metrics']['b_hold'],
            'behavior_2x': best['metrics']['b2_hold'],
            'q_leverage': best['metrics']['q_hold'],
        },
        'outputs': {
            'trials_csv': str(OUT_TRIALS_CSV),
            'compare_csv': str(OUT_COMPARE_CSV),
            'holdout_predictions_csv': str(OUT_HOLDOUT_PRED_CSV),
            'history_csv': str(OUT_HISTORY_CSV),
            'model_pt': str(OUT_MODEL_PT),
            'equity_png': str(OUT_EQUITY_PNG),
            'loss_png': str(OUT_LOSS_PNG),
            'lev_hist_png': str(OUT_LEV_HIST_PNG),
        },
    }

    OUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

    print('Saved summary:', OUT_SUMMARY_JSON)
    print('Best policy:', summary['best_policy'])
    print('Holdout metrics:', summary['holdout_metrics'])


if __name__ == '__main__':
    main()
