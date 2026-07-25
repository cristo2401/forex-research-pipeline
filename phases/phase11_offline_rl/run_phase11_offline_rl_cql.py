#!/usr/bin/env python3
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

    # Data split (inside pre-holdout windows)
    train_ratio: float = 0.80

    # RL
    gamma: float = 0.97
    batch_size: int = 256
    hidden_dim: int = 96

    # BC training
    bc_lr: float = 1e-3
    bc_weight_decay: float = 1e-4
    bc_epochs: int = 30
    bc_patience: int = 5

    # CQL-lite training
    q_lr: float = 8e-4
    q_weight_decay: float = 1e-4
    q_epochs: int = 45
    q_patience: int = 7
    q_target_tau: float = 0.02
    cql_alpha: float = 1.2
    bc_reg: float = 0.03

    # Policy calibration
    temp_grid: Tuple[float, ...] = (0.7, 1.0, 1.3)
    conf_grid: Tuple[float, ...] = (0.34, 0.40, 0.46, 0.52, 0.58, 0.64, 0.70)

    # Trading cost
    cost_pips: float = 1.2


CFG = Config()
ROOT = Path(__file__).resolve().parents[2]
IN_DIR = ROOT / 'phases/phase10_decision_engine/output_phase10_2_strict_expanding_GBPUSD'
OUT_DIR = ROOT / 'phases/phase11_offline_rl/output_phase11_offline_rl_cql_GBPUSD'
OUT_DIR.mkdir(parents=True, exist_ok=True)

IN_WINDOWS = IN_DIR / 'GBPUSD_phase10_2_windows_predictions.csv'
IN_HOLDOUT = IN_DIR / 'GBPUSD_phase10_2_holdout_predictions.csv'

OUT_HISTORY_CSV = OUT_DIR / 'GBPUSD_phase11_train_history.csv'
OUT_COMPARE_CSV = OUT_DIR / 'GBPUSD_phase11_policy_comparison.csv'
OUT_HOLDOUT_PRED_CSV = OUT_DIR / 'GBPUSD_phase11_holdout_policy_predictions.csv'
OUT_SUMMARY_JSON = OUT_DIR / 'GBPUSD_phase11_summary.json'
OUT_MODEL_BC = OUT_DIR / 'phase11_bc_model.pt'
OUT_MODEL_CQL = OUT_DIR / 'phase11_cql_model.pt'
OUT_PNG_LOSS = OUT_DIR / 'GBPUSD_phase11_training_losses.png'
OUT_PNG_EQUITY = OUT_DIR / 'GBPUSD_phase11_holdout_equity_compare.png'


ACTION_TO_IDX = {-1: 0, 0: 1, 1: 2}
IDX_TO_ACTION = {0: -1, 1: 0, 2: 1}


class BehaviorDataset(Dataset):
    def __init__(self, x: np.ndarray, a: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.a = torch.tensor(a, dtype=torch.long)

    def __len__(self):
        return len(self.a)

    def __getitem__(self, idx):
        return self.x[idx], self.a[idx]


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


class MLPPolicy(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int = 3):
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


def pip_size_for_symbol(symbol: str) -> float:
    return 0.01 if symbol.upper().endswith('JPY') else 0.0001


def action_to_idx(actions: np.ndarray) -> np.ndarray:
    return np.array([ACTION_TO_IDX[int(a)] for a in actions], dtype=np.int64)


def idx_to_action(idxs: np.ndarray) -> np.ndarray:
    return np.array([IDX_TO_ACTION[int(i)] for i in idxs], dtype=np.int64)


def score_from_metrics(m: Dict[str, float]) -> float:
    return float(m['avg_net_return']) * 1e4 + 0.10 * float(m['sharpe_ann'])


def evaluate_actions(actions: np.ndarray, y_true: np.ndarray, future_ret: np.ndarray, cost_ret: float) -> Dict[str, float]:
    out = {}
    out['acc_action_3'] = float((actions == y_true).mean())
    out['f1_macro_action_3'] = float(f1_score(y_true, actions, labels=[-1, 0, 1], average='macro', zero_division=0))

    gross = actions * future_ret
    trade = (actions != 0).astype(np.float32)
    net = gross - trade * cost_ret

    out['avg_net_return'] = float(np.mean(net))
    out['cum_net_return'] = float(np.sum(net))
    out['std_net_return'] = float(np.std(net))
    out['trade_rate'] = float(np.mean(trade))

    eps = 1e-12
    out['sharpe_ann'] = float((out['avg_net_return'] / (out['std_net_return'] + eps)) * np.sqrt(24.0 * 252.0))

    traded = trade > 0
    out['win_rate_trades'] = float((net[traded] > 0).mean()) if traded.any() else 0.0
    return out


def build_transitions(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    s_list, a_list, r_list, ns_list, d_list = [], [], [], [], []
    zero_state = np.zeros((len(feature_cols),), dtype=np.float32)

    work = df.sort_values(['global_segment_id', 'step', 'timestamp'])

    for _seg, g in work.groupby('global_segment_id', sort=False):
        g = g.sort_values('step')
        x = g[feature_cols].to_numpy(dtype=np.float32)
        a = action_to_idx(g['pred_action'].to_numpy(dtype=np.int64))
        r = g['net_return'].to_numpy(dtype=np.float32)
        st = g['step'].to_numpy(dtype=np.int64)

        n = len(g)
        for i in range(n):
            s = x[i]
            ai = a[i]
            ri = r[i]

            if i < n - 1 and st[i + 1] == st[i] + 1:
                ns = x[i + 1]
                done = 0.0
            else:
                ns = zero_state
                done = 1.0

            s_list.append(s)
            a_list.append(ai)
            r_list.append(ri)
            ns_list.append(ns)
            d_list.append(done)

    return (
        np.asarray(s_list, dtype=np.float32),
        np.asarray(a_list, dtype=np.int64),
        np.asarray(r_list, dtype=np.float32),
        np.asarray(ns_list, dtype=np.float32),
        np.asarray(d_list, dtype=np.float32),
    )


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


def train_bc(
    x_train: np.ndarray,
    a_train: np.ndarray,
    x_val: np.ndarray,
    a_val: np.ndarray,
    cfg: Config,
    device: torch.device,
):
    model = MLPPolicy(in_dim=x_train.shape[1], hidden_dim=cfg.hidden_dim).to(device)

    ds_train = BehaviorDataset(x_train, a_train)
    ds_val = BehaviorDataset(x_val, a_val)
    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False)

    # inverse frequency weights for class imbalance
    cnt = np.bincount(a_train, minlength=3).astype(float)
    cnt = np.maximum(cnt, 1.0)
    cw = cnt.sum() / (3.0 * cnt)
    cw = torch.tensor(cw, dtype=torch.float32, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.bc_lr, weight_decay=cfg.bc_weight_decay)

    best_state = None
    best_val = np.inf
    wait = 0
    hist = []

    for epoch in range(1, cfg.bc_epochs + 1):
        model.train()
        tr_loss = []
        for xb, ab in dl_train:
            xb = xb.to(device)
            ab = ab.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, ab, weight=cw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss.append(float(loss.detach().cpu()))

        model.eval()
        va_loss = []
        with torch.no_grad():
            for xb, ab in dl_val:
                xb = xb.to(device)
                ab = ab.to(device)
                logits = model(xb)
                loss = F.cross_entropy(logits, ab, weight=cw)
                va_loss.append(float(loss.detach().cpu()))

        tr = float(np.mean(tr_loss))
        va = float(np.mean(va_loss))
        hist.append({'model': 'bc', 'epoch': epoch, 'train_loss': tr, 'val_loss': va})

        if va < best_val - 1e-5:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= cfg.bc_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, hist


def train_cql(
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
    q = MLPPolicy(in_dim=s_train.shape[1], hidden_dim=cfg.hidden_dim).to(device)
    q_tgt = MLPPolicy(in_dim=s_train.shape[1], hidden_dim=cfg.hidden_dim).to(device)
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
        tr_td, tr_cql, tr_bc, tr_tot = [], [], [], []

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

            td_loss = F.mse_loss(q_sa, y)
            cql_loss = torch.logsumexp(q_logits, dim=1).mean() - q_sa.mean()
            bc_loss = F.cross_entropy(q_logits, ab)

            loss = td_loss + cfg.cql_alpha * cql_loss + cfg.bc_reg * bc_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 1.0)
            opt.step()

            # soft update target
            with torch.no_grad():
                for p_tgt, p in zip(q_tgt.parameters(), q.parameters()):
                    p_tgt.data.mul_(1.0 - cfg.q_target_tau).add_(cfg.q_target_tau * p.data)

            tr_td.append(float(td_loss.detach().cpu()))
            tr_cql.append(float(cql_loss.detach().cpu()))
            tr_bc.append(float(bc_loss.detach().cpu()))
            tr_tot.append(float(loss.detach().cpu()))

        q.eval()
        va_tot = []
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

                td_loss = F.mse_loss(q_sa, y)
                cql_loss = torch.logsumexp(q_logits, dim=1).mean() - q_sa.mean()
                bc_loss = F.cross_entropy(q_logits, ab)
                loss = td_loss + cfg.cql_alpha * cql_loss + cfg.bc_reg * bc_loss
                va_tot.append(float(loss.detach().cpu()))

        row = {
            'model': 'cql',
            'epoch': epoch,
            'train_td': float(np.mean(tr_td)),
            'train_cql': float(np.mean(tr_cql)),
            'train_bc': float(np.mean(tr_bc)),
            'train_loss': float(np.mean(tr_tot)),
            'val_loss': float(np.mean(va_tot)),
        }
        hist.append(row)

        if row['val_loss'] < best_val - 1e-5:
            best_val = row['val_loss']
            best_state = {k: v.detach().cpu().clone() for k, v in q.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= cfg.q_patience:
                break

    if best_state is not None:
        q.load_state_dict(best_state)

    return q, hist


def model_probs(model: nn.Module, x: np.ndarray, device: torch.device, temp: float = 1.0) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        bs = 2048
        for i in range(0, len(x), bs):
            xb = torch.tensor(x[i : i + bs], dtype=torch.float32, device=device)
            logits = model(xb) / max(temp, 1e-6)
            p = torch.softmax(logits, dim=1)
            out.append(p.cpu().numpy())
    return np.concatenate(out, axis=0)


def choose_actions_from_probs(p: np.ndarray, conf_thr: float) -> np.ndarray:
    idx = np.argmax(p, axis=1)
    conf = np.max(p, axis=1)
    actions = idx_to_action(idx)
    actions = np.where(conf >= conf_thr, actions, 0)
    return actions.astype(np.int64)


def tune_policy(model: nn.Module, x_val: np.ndarray, df_val: pd.DataFrame, cfg: Config, device: torch.device) -> Dict[str, float]:
    cost_ret = cfg.cost_pips * pip_size_for_symbol(cfg.symbol)

    y_true = df_val['y_action'].to_numpy(dtype=np.int64)
    future_ret = df_val['future_return_h'].to_numpy(dtype=np.float32)

    best = None
    for temp in cfg.temp_grid:
        p = model_probs(model, x_val, device=device, temp=float(temp))
        for conf in cfg.conf_grid:
            a = choose_actions_from_probs(p, conf_thr=float(conf))
            m = evaluate_actions(a, y_true, future_ret, cost_ret)
            score = score_from_metrics(m)
            row = {'temp': float(temp), 'conf_thr': float(conf), 'score': float(score), **m}
            if best is None or row['score'] > best['score']:
                best = row

    return best


def predict_policy(model: nn.Module, x: np.ndarray, temp: float, conf_thr: float, device: torch.device) -> np.ndarray:
    p = model_probs(model, x, device=device, temp=float(temp))
    return choose_actions_from_probs(p, conf_thr=float(conf_thr))


def main():
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

    # time split inside windows for training/validation
    win = df_windows.sort_values('timestamp').reset_index(drop=True)
    cut = int(len(win) * CFG.train_ratio)
    df_tr = win.iloc[:cut].copy()
    df_va = win.iloc[cut:].copy()

    x_tr_raw = df_tr[feature_cols].to_numpy(dtype=np.float32)
    x_va_raw = df_va[feature_cols].to_numpy(dtype=np.float32)
    x_ho_raw = df_holdout[feature_cols].to_numpy(dtype=np.float32)

    x_tr, mu, sd = normalize_fit(x_tr_raw)
    x_va = normalize_apply(x_va_raw, mu, sd)
    x_ho = normalize_apply(x_ho_raw, mu, sd)

    a_tr = action_to_idx(df_tr['pred_action'].to_numpy(dtype=np.int64))
    a_va = action_to_idx(df_va['pred_action'].to_numpy(dtype=np.int64))

    # transitions for offline RL training
    s_tr, at_tr, r_tr, ns_tr, d_tr = build_transitions(df_tr, feature_cols)
    s_va, at_va, r_va, ns_va, d_va = build_transitions(df_va, feature_cols)

    s_tr = normalize_apply(s_tr, mu, sd)
    ns_tr = normalize_apply(ns_tr, mu, sd)
    s_va = normalize_apply(s_va, mu, sd)
    ns_va = normalize_apply(ns_va, mu, sd)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    bc_model, bc_hist = train_bc(x_tr, a_tr, x_va, a_va, CFG, device)
    cql_model, cql_hist = train_cql(s_tr, at_tr, r_tr, ns_tr, d_tr, s_va, at_va, r_va, ns_va, d_va, CFG, device)

    # calibrate on validation only
    bc_best = tune_policy(bc_model, x_va, df_va, CFG, device)
    cql_best = tune_policy(cql_model, x_va, df_va, CFG, device)

    cost_ret = CFG.cost_pips * pip_size_for_symbol(CFG.symbol)

    # baseline behavior on holdout
    y_true_ho = df_holdout['y_action'].to_numpy(dtype=np.int64)
    ret_ho = df_holdout['future_return_h'].to_numpy(dtype=np.float32)
    a_behavior_ho = df_holdout['pred_action'].to_numpy(dtype=np.int64)
    behavior_holdout = evaluate_actions(a_behavior_ho, y_true_ho, ret_ho, cost_ret)

    # BC policy holdout
    a_bc_ho = predict_policy(bc_model, x_ho, temp=bc_best['temp'], conf_thr=bc_best['conf_thr'], device=device)
    bc_holdout = evaluate_actions(a_bc_ho, y_true_ho, ret_ho, cost_ret)

    # CQL policy holdout
    a_cql_ho = predict_policy(cql_model, x_ho, temp=cql_best['temp'], conf_thr=cql_best['conf_thr'], device=device)
    cql_holdout = evaluate_actions(a_cql_ho, y_true_ho, ret_ho, cost_ret)

    # validation scores (for transparency)
    y_true_va = df_va['y_action'].to_numpy(dtype=np.int64)
    ret_va = df_va['future_return_h'].to_numpy(dtype=np.float32)
    behavior_val = evaluate_actions(df_va['pred_action'].to_numpy(dtype=np.int64), y_true_va, ret_va, cost_ret)
    a_bc_va = predict_policy(bc_model, x_va, temp=bc_best['temp'], conf_thr=bc_best['conf_thr'], device=device)
    bc_val = evaluate_actions(a_bc_va, y_true_va, ret_va, cost_ret)
    a_cql_va = predict_policy(cql_model, x_va, temp=cql_best['temp'], conf_thr=cql_best['conf_thr'], device=device)
    cql_val = evaluate_actions(a_cql_va, y_true_va, ret_va, cost_ret)

    # save model artifacts
    torch.save(
        {
            'state_dict': bc_model.state_dict(),
            'feature_cols': feature_cols,
            'mu': mu,
            'sd': sd,
            'calibration': {'temp': bc_best['temp'], 'conf_thr': bc_best['conf_thr']},
            'config': asdict(CFG),
        },
        OUT_MODEL_BC,
    )

    torch.save(
        {
            'state_dict': cql_model.state_dict(),
            'feature_cols': feature_cols,
            'mu': mu,
            'sd': sd,
            'calibration': {'temp': cql_best['temp'], 'conf_thr': cql_best['conf_thr']},
            'config': asdict(CFG),
        },
        OUT_MODEL_CQL,
    )

    # holdout predictions table
    pred = df_holdout[['global_segment_id', 'timestamp', 'step', 'y_action', 'future_return_h']].copy()
    pred['action_behavior'] = a_behavior_ho
    pred['action_bc'] = a_bc_ho
    pred['action_cql'] = a_cql_ho
    pred['net_behavior'] = pred['action_behavior'] * pred['future_return_h'] - (pred['action_behavior'] != 0).astype(float) * cost_ret
    pred['net_bc'] = pred['action_bc'] * pred['future_return_h'] - (pred['action_bc'] != 0).astype(float) * cost_ret
    pred['net_cql'] = pred['action_cql'] * pred['future_return_h'] - (pred['action_cql'] != 0).astype(float) * cost_ret
    pred['cum_behavior'] = pred['net_behavior'].cumsum()
    pred['cum_bc'] = pred['net_bc'].cumsum()
    pred['cum_cql'] = pred['net_cql'].cumsum()
    pred.to_csv(OUT_HOLDOUT_PRED_CSV, index=False)

    # comparison table
    rows = [
        {'split': 'validation', 'policy': 'behavior', **behavior_val},
        {'split': 'validation', 'policy': 'bc', **bc_val},
        {'split': 'validation', 'policy': 'cql', **cql_val},
        {'split': 'holdout', 'policy': 'behavior', **behavior_holdout},
        {'split': 'holdout', 'policy': 'bc', **bc_holdout},
        {'split': 'holdout', 'policy': 'cql', **cql_holdout},
    ]
    cmp_df = pd.DataFrame(rows)
    cmp_df.to_csv(OUT_COMPARE_CSV, index=False)

    # training history
    hist_df = pd.DataFrame(bc_hist + cql_hist)
    hist_df.to_csv(OUT_HISTORY_CSV, index=False)

    # plots
    fig, ax = plt.subplots(figsize=(10, 4))
    hbc = hist_df[hist_df['model'] == 'bc']
    hcq = hist_df[hist_df['model'] == 'cql']

    if len(hbc):
        ax.plot(hbc['epoch'], hbc['train_loss'], label='BC train')
        ax.plot(hbc['epoch'], hbc['val_loss'], label='BC val')
    if len(hcq):
        ax.plot(hcq['epoch'], hcq['train_loss'], label='CQL train')
        ax.plot(hcq['epoch'], hcq['val_loss'], label='CQL val')

    ax.set_title('Phase11 training losses')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_PNG_LOSS, dpi=140)
    plt.close(fig)

    plt.figure(figsize=(11, 4))
    plt.plot(pred['timestamp'], pred['cum_behavior'], label='behavior')
    plt.plot(pred['timestamp'], pred['cum_bc'], label='bc')
    plt.plot(pred['timestamp'], pred['cum_cql'], label='cql')
    plt.title('Phase11 holdout cumulative net return')
    plt.xlabel('timestamp')
    plt.ylabel('cumulative return')
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG_EQUITY, dpi=140)
    plt.close()

    summary = {
        'symbol': CFG.symbol,
        'model_tag': 'phase11_offline_rl_cql',
        'description': 'Offline RL (CQL-lite) and Behavior Cloning over leak-safe phase10.2 data',
        'config': asdict(CFG),
        'dataset': {
            'n_windows_total': int(len(df_windows)),
            'n_windows_train': int(len(df_tr)),
            'n_windows_val': int(len(df_va)),
            'n_holdout': int(len(df_holdout)),
            'windows_date_min': str(df_windows['timestamp'].min()),
            'windows_date_max': str(df_windows['timestamp'].max()),
            'holdout_date_min': str(df_holdout['timestamp'].min()),
            'holdout_date_max': str(df_holdout['timestamp'].max()),
        },
        'calibration': {'bc': bc_best, 'cql': cql_best},
        'validation_metrics': {'behavior': behavior_val, 'bc': bc_val, 'cql': cql_val},
        'holdout_metrics': {'behavior': behavior_holdout, 'bc': bc_holdout, 'cql': cql_holdout},
        'outputs': {
            'history_csv': str(OUT_HISTORY_CSV),
            'compare_csv': str(OUT_COMPARE_CSV),
            'holdout_predictions_csv': str(OUT_HOLDOUT_PRED_CSV),
            'bc_model_pt': str(OUT_MODEL_BC),
            'cql_model_pt': str(OUT_MODEL_CQL),
            'loss_plot_png': str(OUT_PNG_LOSS),
            'equity_plot_png': str(OUT_PNG_EQUITY),
        },
    }

    OUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

    print('Saved summary:', OUT_SUMMARY_JSON)
    print('Holdout metrics:')
    print(summary['holdout_metrics'])


if __name__ == '__main__':
    main()
