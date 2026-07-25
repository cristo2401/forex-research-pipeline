#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import random
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
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class Config:
    phase16_4_run_dir: str
    out_dir: str

    min_train_windows: int = 1200
    val_windows: int = 128
    test_windows: int = 256
    step_windows: int = 256

    d_model: int = 192
    nhead: int = 6
    num_layers: int = 3
    ff_dim: int = 384
    dropout: float = 0.1

    batch_size: int = 128
    imitation_epochs: int = 8
    rl_epochs: int = 18
    patience: int = 6
    lr: float = 3e-4
    weight_decay: float = 1e-5

    max_gross_leverage: float = 2.0
    target_gross_leverage: float = 0.9
    min_target_mult: float = 0.35
    max_target_mult: float = 1.45
    weight_temperature: float = 1.0

    loss_w_ret: float = 1.0
    loss_w_risk: float = 0.30
    loss_w_excess: float = 1.0
    loss_w_exposure: float = 0.12
    loss_w_l1: float = 0.002
    loss_w_turnover: float = 0.06
    loss_w_bc_anchor: float = 0.18
    loss_w_value: float = 0.15
    loss_w_gate: float = 0.10

    temporal_weight_min: float = 0.6
    temporal_weight_max: float = 1.6

    cost_per_turnover: float = 5e-5
    turnover_smooth_alpha: float = 0.7

    # Safety shield in evaluation
    shield_dd_limit: float = -0.08
    shield_risk_mult: float = 0.55

    ensemble_seeds: str = '42,43'
    gate_min_excess_sharpe: float = 0.05
    gate_min_excess_cum: float = 0.0

    seed: int = 42
    device: str = 'auto'


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger('phase17_hierarchical_gym')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    fh = logging.FileHandler(log_path, mode='w')
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class HierarchicalPolicyTransformer(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, n_assets: int, d_model: int, nhead: int, num_layers: int, ff_dim: int, dropout: float):
        super().__init__()
        self.seq_len = int(seq_len)
        self.n_assets = int(n_assets)

        self.in_proj = nn.Linear(int(input_dim), int(d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, int(seq_len), int(d_model)))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(nhead),
            dim_feedforward=int(ff_dim),
            dropout=float(dropout),
            batch_first=True,
            norm_first=True,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))
        self.norm = nn.LayerNorm(int(d_model))

        hid = int(d_model) * 2
        self.fuse = nn.Sequential(
            nn.Linear(hid, int(d_model)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.LayerNorm(int(d_model)),
        )

        # Hierarchical heads:
        # 1) gate: activate/deactivate symbol
        # 2) direction/intensity: signed symbol preference
        # 3) budget: gross leverage multiplier
        # 4) value head: critic baseline for RL-like objective
        self.gate_head = nn.Linear(int(d_model), int(n_assets))
        self.dir_head = nn.Linear(int(d_model), int(n_assets))
        self.ret_head = nn.Linear(int(d_model), int(n_assets))
        self.risk_head = nn.Linear(int(d_model), int(n_assets))
        self.budget_head = nn.Linear(int(d_model), 1)
        self.value_head = nn.Linear(int(d_model), 1)

        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        _, l, _ = x.shape
        if l > self.seq_len:
            raise ValueError(f'L={l} > seq_len={self.seq_len}')

        h = self.in_proj(x) + self.pos_emb[:, :l, :]
        h = self.encoder(h)
        h = self.norm(h)

        z_last = h[:, -1, :]
        z_mean = h.mean(dim=1)
        z = self.fuse(torch.cat([z_last, z_mean], dim=1))

        gate_logits = self.gate_head(z)
        dir_raw = self.dir_head(z)
        budget_raw = self.budget_head(z)

        return {
            'gate_logits': gate_logits,
            'dir_raw': dir_raw,
            'ret': self.ret_head(z),
            'risk_raw': self.risk_head(z),
            'budget_raw': budget_raw,
            'value': self.value_head(z).squeeze(-1),
        }


def parse_seed_list(s: str) -> List[int]:
    vals = []
    for x in str(s).split(','):
        x = x.strip()
        if not x:
            continue
        vals.append(int(x))
    if not vals:
        vals = [42]
    return vals


def build_equal_weight_torch(mask: torch.Tensor, target_gross: float) -> torch.Tensor:
    k = torch.sum(mask > 0.5, dim=1, keepdim=True).float()
    k = torch.clamp(k, min=1.0)
    return (mask / k) * float(target_gross)


def normalize_weights_np(raw: np.ndarray, mask: np.ndarray, target_gross: float, max_gross: float, eps: float = 1e-12) -> np.ndarray:
    w = np.tanh(raw) * mask
    gross = np.sum(np.abs(w), axis=1, keepdims=True)
    active = np.sum(mask > 0.5, axis=1, keepdims=True)

    fallback = np.zeros_like(w)
    ok = active.squeeze(-1) > 0
    fallback[ok] = (mask[ok] / np.maximum(active[ok], 1.0)) * float(target_gross)

    low = gross.squeeze(-1) < 1e-6
    if np.any(low):
        w[low] = fallback[low]
        gross = np.sum(np.abs(w), axis=1, keepdims=True)

    tgt = np.where(active > 0, float(target_gross), 0.0)
    w = w * (tgt / np.maximum(gross, eps))

    gross2 = np.sum(np.abs(w), axis=1, keepdims=True)
    cap = np.where(gross2 > float(max_gross), float(max_gross) / np.maximum(gross2, eps), 1.0)
    w = w * cap
    return w.astype(np.float32)


def build_teacher_weights_from_features(
    x_last_raw: np.ndarray,
    feature_cols: List[str],
    symbols: List[str],
    y_mask: np.ndarray,
    target_gross: float,
    max_gross: float,
) -> np.ndarray:
    n = x_last_raw.shape[0]
    a = len(symbols)

    feat_index = {c: i for i, c in enumerate(feature_cols)}

    def col(token: str, symbol: str) -> np.ndarray | None:
        c = f'{token}__{symbol}'
        idx = feat_index.get(c)
        if idx is None:
            return None
        return x_last_raw[:, idx]

    raw = np.zeros((n, a), dtype=np.float32)

    for j, s in enumerate(symbols):
        score = col('score_ls_lag1', s)
        pred_action = col('pred_action_lag1', s)
        conf = col('pred_conf_max_lag1', s)
        nonflat = col('nonflat_score_lag1', s)
        age = col('state_age_steps_lag1', s)

        sig = np.zeros((n,), dtype=np.float32)
        if score is not None:
            sig += 0.70 * score.astype(np.float32)
        if pred_action is not None:
            sig += 0.30 * pred_action.astype(np.float32)

        if conf is not None:
            c = np.clip(conf.astype(np.float32), 0.0, 1.0)
            sig = sig * (0.25 + 0.75 * c)

        if nonflat is not None:
            nf = np.clip(np.abs(nonflat.astype(np.float32)), 0.0, 2.0) / 2.0
            sig = sig * (0.20 + 0.80 * nf)

        if age is not None:
            # stale signal decay
            ag = np.clip(age.astype(np.float32), 0.0, 96.0)
            decay = np.exp(-ag / 32.0).astype(np.float32)
            sig = sig * (0.4 + 0.6 * decay)

        raw[:, j] = sig

    return normalize_weights_np(raw=raw, mask=y_mask, target_gross=float(target_gross), max_gross=float(max_gross))


def build_hier_weights(
    gate_logits: torch.Tensor,
    dir_raw: torch.Tensor,
    mask: torch.Tensor,
    budget_raw: torch.Tensor,
    cfg: Config,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gate = torch.sigmoid(gate_logits)
    direction = torch.tanh(dir_raw / max(float(cfg.weight_temperature), 1e-6))
    raw = (gate * direction) * mask

    gross_raw = torch.sum(torch.abs(raw), dim=1, keepdim=True)
    low = gross_raw < 1e-4
    if torch.any(low):
        fallback = build_equal_weight_torch(mask=mask, target_gross=float(cfg.target_gross_leverage))
        raw = torch.where(low, fallback, raw)
        gross_raw = torch.sum(torch.abs(raw), dim=1, keepdim=True)

    active = torch.sum(mask > 0.5, dim=1, keepdim=True).float()
    budget = torch.sigmoid(budget_raw)
    target_mult = float(cfg.min_target_mult) + (float(cfg.max_target_mult) - float(cfg.min_target_mult)) * budget
    target = torch.where(active > 0, target_mult * float(cfg.target_gross_leverage), torch.zeros_like(active))

    w = raw * (target / (gross_raw + 1e-12))
    gross = torch.sum(torch.abs(w), dim=1, keepdim=True)
    cap = torch.where(gross > float(cfg.max_gross_leverage), float(cfg.max_gross_leverage) / (gross + 1e-12), torch.ones_like(gross))
    w = w * cap
    return w, gate, budget.squeeze(-1)


def compute_imitation_losses(
    out: Dict[str, torch.Tensor],
    y_mask: torch.Tensor,
    y_teacher: torch.Tensor,
    t_weight: torch.Tensor,
    cfg: Config,
) -> Dict[str, torch.Tensor]:
    w, gate, budget = build_hier_weights(
        gate_logits=out['gate_logits'],
        dir_raw=out['dir_raw'],
        mask=y_mask,
        budget_raw=out['budget_raw'],
        cfg=cfg,
    )

    tw = t_weight.view(-1, 1)
    den = torch.clamp(torch.sum(y_mask * tw), min=1.0)

    per_bc = torch.nn.functional.smooth_l1_loss(w, y_teacher, reduction='none')
    loss_bc = torch.sum(per_bc * y_mask * tw) / den

    gate_target = (torch.abs(y_teacher) > 1e-5).float()
    per_gate = torch.nn.functional.binary_cross_entropy_with_logits(out['gate_logits'], gate_target, reduction='none')
    loss_gate = torch.sum(per_gate * y_mask * tw) / den

    gross_teacher = torch.sum(torch.abs(y_teacher), dim=1)
    gross_ref = torch.clamp(gross_teacher / max(float(cfg.target_gross_leverage), 1e-6), 0.0, 1.0)
    loss_budget = torch.mean((budget - gross_ref) ** 2)

    loss_total = loss_bc + float(cfg.loss_w_gate) * loss_gate + 0.20 * loss_budget

    return {
        'loss_total': loss_total,
        'loss_bc': loss_bc,
        'loss_gate': loss_gate,
        'loss_budget': loss_budget,
        'w': w,
    }


def compute_rl_losses(
    out: Dict[str, torch.Tensor],
    y_ret1: torch.Tensor,
    y_mask: torch.Tensor,
    y_teacher: torch.Tensor,
    t_weight: torch.Tensor,
    cfg: Config,
) -> Dict[str, torch.Tensor]:
    ret_pred = out['ret']
    risk_pred = torch.nn.functional.softplus(out['risk_raw'])

    tw = t_weight.view(-1, 1)
    den = torch.clamp(torch.sum(y_mask * tw), min=1.0)

    per_ret = torch.nn.functional.smooth_l1_loss(ret_pred, y_ret1, reduction='none')
    per_risk = torch.nn.functional.smooth_l1_loss(risk_pred, torch.abs(y_ret1), reduction='none')
    loss_ret = torch.sum(per_ret * y_mask * tw) / den
    loss_risk = torch.sum(per_risk * y_mask * tw) / den

    w, gate, _ = build_hier_weights(
        gate_logits=out['gate_logits'],
        dir_raw=out['dir_raw'],
        mask=y_mask,
        budget_raw=out['budget_raw'],
        cfg=cfg,
    )

    # pseudo-environment reward (gym-like one-step):
    # reward = portfolio return - turnover cost - weak l1 friction
    port_ret = torch.sum(w * y_ret1, dim=1)

    w_eq = build_equal_weight_torch(mask=y_mask, target_gross=float(cfg.target_gross_leverage))
    eq_ret = torch.sum(w_eq * y_ret1, dim=1)

    if len(w) > 1:
        turn = torch.mean(torch.abs(w[1:] - w[:-1]), dim=1)
        turn = torch.cat([torch.mean(torch.abs(w[:1]), dim=1), turn], dim=0)
    else:
        turn = torch.mean(torch.abs(w), dim=1)

    reward = port_ret - float(cfg.cost_per_turnover) * turn
    excess = reward - eq_ret

    wt1 = t_weight
    den_t = torch.clamp(torch.sum(wt1), min=1.0)

    loss_excess = -torch.sum(excess * wt1) / den_t

    gross = torch.sum(torch.abs(w), dim=1)
    active = torch.sum(y_mask > 0.5, dim=1).float()
    tgt = torch.where(active > 0, torch.full_like(active, float(cfg.target_gross_leverage)), torch.zeros_like(active))
    loss_exposure = torch.sum(((gross - tgt) ** 2) * wt1) / den_t

    loss_l1 = torch.sum(torch.mean(torch.abs(w), dim=1) * wt1) / den_t
    loss_turn = torch.sum(turn * wt1) / den_t

    # keep policy close to strong lag1 teacher to reduce instability
    per_bc = torch.nn.functional.smooth_l1_loss(w, y_teacher, reduction='none')
    loss_bc_anchor = torch.sum(per_bc * y_mask * tw) / den

    gate_target = (torch.abs(y_teacher) > 1e-5).float()
    per_gate = torch.nn.functional.binary_cross_entropy_with_logits(out['gate_logits'], gate_target, reduction='none')
    loss_gate = torch.sum(per_gate * y_mask * tw) / den

    # value approximates excess reward
    value = out['value']
    loss_value = torch.mean((value - excess.detach()) ** 2)

    loss_total = (
        float(cfg.loss_w_ret) * loss_ret
        + float(cfg.loss_w_risk) * loss_risk
        + float(cfg.loss_w_excess) * loss_excess
        + float(cfg.loss_w_exposure) * loss_exposure
        + float(cfg.loss_w_l1) * loss_l1
        + float(cfg.loss_w_turnover) * loss_turn
        + float(cfg.loss_w_bc_anchor) * loss_bc_anchor
        + float(cfg.loss_w_value) * loss_value
        + float(cfg.loss_w_gate) * loss_gate
    )

    return {
        'loss_total': loss_total,
        'loss_ret': loss_ret,
        'loss_risk': loss_risk,
        'loss_excess': loss_excess,
        'loss_exposure': loss_exposure,
        'loss_l1': loss_l1,
        'loss_turn': loss_turn,
        'loss_bc_anchor': loss_bc_anchor,
        'loss_gate': loss_gate,
        'loss_value': loss_value,
        'w': w,
        'reward_mean': torch.mean(reward),
        'excess_mean': torch.mean(excess),
        'gate_mean': torch.mean(gate),
    }


def run_epoch_imitation(model: HierarchicalPolicyTransformer, dl: DataLoader, device: torch.device, cfg: Config, train: bool, optimizer: torch.optim.Optimizer | None = None) -> Dict[str, float]:
    model.train() if train else model.eval()
    rows: List[Dict[str, float]] = []

    for xb, yb, mb, tw, yteach in dl:
        xb = xb.to(device)
        mb = mb.to(device)
        tw = tw.to(device)
        yteach = yteach.to(device)

        if train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            out = model(xb)
            losses = compute_imitation_losses(out=out, y_mask=mb, y_teacher=yteach, t_weight=tw, cfg=cfg)
            loss = losses['loss_total']
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        w = losses['w'].detach()
        gross = torch.sum(torch.abs(w), dim=1)
        rows.append(
            {
                'loss_total': float(losses['loss_total'].detach().item()),
                'loss_bc': float(losses['loss_bc'].detach().item()),
                'loss_gate': float(losses['loss_gate'].detach().item()),
                'loss_budget': float(losses['loss_budget'].detach().item()),
                'gross_mean': float(gross.mean().item()),
            }
        )

    if not rows:
        return {k: float('nan') for k in ['loss_total', 'loss_bc', 'loss_gate', 'loss_budget', 'gross_mean']}
    d = pd.DataFrame(rows)
    return {c: float(d[c].mean()) for c in d.columns}


def run_epoch_rl(model: HierarchicalPolicyTransformer, dl: DataLoader, device: torch.device, cfg: Config, train: bool, optimizer: torch.optim.Optimizer | None = None) -> Dict[str, float]:
    model.train() if train else model.eval()
    rows: List[Dict[str, float]] = []

    for xb, yb, mb, tw, yteach in dl:
        xb = xb.to(device)
        yb = yb.to(device)
        mb = mb.to(device)
        tw = tw.to(device)
        yteach = yteach.to(device)

        if train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            out = model(xb)
            losses = compute_rl_losses(out=out, y_ret1=yb, y_mask=mb, y_teacher=yteach, t_weight=tw, cfg=cfg)
            loss = losses['loss_total']
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        w = losses['w'].detach()
        realized = torch.sum(w * yb.detach(), dim=1)
        gross = torch.sum(torch.abs(w), dim=1)
        rows.append(
            {
                'loss_total': float(losses['loss_total'].detach().item()),
                'loss_ret': float(losses['loss_ret'].detach().item()),
                'loss_risk': float(losses['loss_risk'].detach().item()),
                'loss_excess': float(losses['loss_excess'].detach().item()),
                'loss_exposure': float(losses['loss_exposure'].detach().item()),
                'loss_l1': float(losses['loss_l1'].detach().item()),
                'loss_turn': float(losses['loss_turn'].detach().item()),
                'loss_bc_anchor': float(losses['loss_bc_anchor'].detach().item()),
                'loss_gate': float(losses['loss_gate'].detach().item()),
                'loss_value': float(losses['loss_value'].detach().item()),
                'reward_mean': float(losses['reward_mean'].detach().item()),
                'excess_mean': float(losses['excess_mean'].detach().item()),
                'gate_mean': float(losses['gate_mean'].detach().item()),
                'realized_port_mean': float(realized.mean().item()),
                'gross_mean': float(gross.mean().item()),
            }
        )

    if not rows:
        keys = ['loss_total', 'loss_ret', 'loss_risk', 'loss_excess', 'loss_exposure', 'loss_l1', 'loss_turn', 'loss_bc_anchor', 'loss_gate', 'loss_value', 'reward_mean', 'excess_mean', 'gate_mean', 'realized_port_mean', 'gross_mean']
        return {k: float('nan') for k in keys}

    d = pd.DataFrame(rows)
    return {c: float(d[c].mean()) for c in d.columns}


def infer_raw_weights(model: HierarchicalPolicyTransformer, x: np.ndarray, y_mask: np.ndarray, device: torch.device, batch_size: int, cfg: Config) -> np.ndarray:
    ds = TensorDataset(torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y_mask.astype(np.float32)))
    dl = DataLoader(ds, batch_size=int(batch_size), shuffle=False, drop_last=False)

    model.eval()
    out_raw = []
    with torch.no_grad():
        for xb, mb in dl:
            xb = xb.to(device)
            mb = mb.to(device)
            out = model(xb)
            w, _, _ = build_hier_weights(
                gate_logits=out['gate_logits'],
                dir_raw=out['dir_raw'],
                mask=mb,
                budget_raw=out['budget_raw'],
                cfg=cfg,
            )
            out_raw.append(w.detach().cpu().numpy())
    return np.concatenate(out_raw, axis=0).astype(np.float32)


def compute_drawdown(equity_curve: np.ndarray) -> float:
    if len(equity_curve) == 0:
        return float('nan')
    peak = np.maximum.accumulate(equity_curve)
    dd = equity_curve / np.maximum(peak, 1e-12) - 1.0
    return float(np.min(dd))


def backtest_from_weights(
    y_ret1: np.ndarray,
    y_mask: np.ndarray,
    w_raw: np.ndarray,
    timestamps: np.ndarray,
    cost_per_turnover: float,
    cost_mult: float,
    smooth_alpha: float,
    max_gross: float,
    shield_dd_limit: float,
    shield_risk_mult: float,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    n = len(y_ret1)
    n_assets = y_ret1.shape[1]
    prev_exec = np.zeros((n_assets,), dtype=np.float32)

    rows = []
    a = float(np.clip(smooth_alpha, 0.0, 1.0))
    equity = 1.0
    peak = 1.0

    for i in range(n):
        w_tgt = (w_raw[i] * y_mask[i]).astype(np.float32)
        w_exec = ((1.0 - a) * prev_exec + a * w_tgt).astype(np.float32)

        gross0 = float(np.sum(np.abs(w_exec)))
        if gross0 > float(max_gross) and gross0 > 1e-12:
            w_exec *= float(max_gross) / gross0

        dd_prev = (equity / max(peak, 1e-12)) - 1.0
        shield_on = 0
        if dd_prev <= float(shield_dd_limit):
            w_exec *= float(shield_risk_mult)
            shield_on = 1

        turnover = float(np.sum(np.abs(w_exec - prev_exec)))
        gross = float(np.sum(w_exec * y_ret1[i]))
        cost = float(cost_per_turnover * cost_mult * turnover)
        net = gross - cost

        equity *= (1.0 + net)
        peak = max(peak, equity)

        rows.append(
            {
                'timestamp_end': pd.Timestamp(timestamps[i]),
                'gross_ret': gross,
                'net_ret': net,
                'turnover': turnover,
                'cost': cost,
                'gross_lev': float(np.sum(np.abs(w_exec))),
                'active_assets': int(np.sum(y_mask[i] > 0.5)),
                'shield_on': int(shield_on),
                'dd_prev': float(dd_prev),
            }
        )
        prev_exec = w_exec

    df = pd.DataFrame(rows).sort_values('timestamp_end').reset_index(drop=True)
    df['equity'] = (1.0 + df['net_ret']).cumprod()

    r = df['net_ret'].to_numpy(dtype=float)
    mu = float(np.mean(r)) if len(r) else float('nan')
    sd = float(np.std(r)) if len(r) else float('nan')
    sharpe = float(mu / sd * np.sqrt(24 * 252)) if (len(r) and sd > 1e-12) else float('nan')

    metrics = {
        'n_steps': int(len(df)),
        'mean_net_ret': mu,
        'std_net_ret': sd,
        'sharpe_ann': sharpe,
        'cum_net_return': float(df['equity'].iloc[-1] - 1.0) if len(df) else float('nan'),
        'max_drawdown': compute_drawdown(df['equity'].to_numpy(dtype=float)),
        'trade_rate': float((df['turnover'] > 1e-6).mean()) if len(df) else float('nan'),
        'avg_turnover': float(df['turnover'].mean()) if len(df) else float('nan'),
        'avg_gross_lev': float(df['gross_lev'].mean()) if len(df) else float('nan'),
        'avg_active_assets': float(df['active_assets'].mean()) if len(df) else float('nan'),
        'shield_rate': float(df['shield_on'].mean()) if len(df) else float('nan'),
    }
    return df, metrics


def baseline_equal_weight(y_mask: np.ndarray, max_gross: float) -> np.ndarray:
    n, a = y_mask.shape
    out = np.zeros((n, a), dtype=np.float32)
    for i in range(n):
        m = y_mask[i] > 0.5
        k = int(np.sum(m))
        if k == 0:
            continue
        out[i, m] = float(max_gross) / float(k)
    return out


def standardize_windows(x: np.ndarray, idx_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tr = x[idx_train]
    mu = tr.reshape(-1, tr.shape[-1]).mean(axis=0)
    sd = tr.reshape(-1, tr.shape[-1]).std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    xz = (x - mu[None, None, :]) / sd[None, None, :]
    return xz.astype(np.float32), mu.astype(np.float32), sd.astype(np.float32)


def make_temporal_weights(n: int, wmin: float, wmax: float) -> np.ndarray:
    if n <= 1:
        return np.array([float(wmax)], dtype=np.float32)
    return np.linspace(float(wmin), float(wmax), n, dtype=np.float32)


def make_walkforward_folds(n: int, min_train: int, val_w: int, test_w: int, step_w: int) -> List[Dict[str, int]]:
    folds = []
    train_end = int(min_train)
    fid = 1
    while True:
        tr0, tr1 = 0, int(train_end)
        v0, v1 = tr1, tr1 + int(val_w)
        t0, t1 = v1, v1 + int(test_w)
        if t1 > n:
            break
        folds.append(
            {
                'fold_id': int(fid),
                'train_start': int(tr0),
                'train_end': int(tr1),
                'val_start': int(v0),
                'val_end': int(v1),
                'test_start': int(t0),
                'test_end': int(t1),
            }
        )
        train_end += int(step_w)
        fid += 1
    return folds


def aggregate_policy_metrics(df: pd.DataFrame) -> Dict[str, float]:
    if len(df) == 0:
        return {
            'n_steps': 0,
            'mean_net_ret': float('nan'),
            'std_net_ret': float('nan'),
            'sharpe_ann': float('nan'),
            'cum_net_return': float('nan'),
            'max_drawdown': float('nan'),
            'trade_rate': float('nan'),
            'avg_turnover': float('nan'),
            'avg_gross_lev': float('nan'),
            'avg_active_assets': float('nan'),
            'shield_rate': float('nan'),
        }
    d = df.sort_values('timestamp_end').reset_index(drop=True).copy()
    d['equity'] = (1.0 + d['net_ret']).cumprod()
    r = d['net_ret'].to_numpy(dtype=float)
    mu = float(np.mean(r))
    sd = float(np.std(r))
    sharpe = float(mu / sd * np.sqrt(24 * 252)) if sd > 1e-12 else float('nan')
    return {
        'n_steps': int(len(d)),
        'mean_net_ret': mu,
        'std_net_ret': sd,
        'sharpe_ann': sharpe,
        'cum_net_return': float(d['equity'].iloc[-1] - 1.0),
        'max_drawdown': compute_drawdown(d['equity'].to_numpy(dtype=float)),
        'trade_rate': float((d['turnover'] > 1e-6).mean()),
        'avg_turnover': float(d['turnover'].mean()),
        'avg_gross_lev': float(d['gross_lev'].mean()),
        'avg_active_assets': float(d['active_assets'].mean()),
        'shield_rate': float(d['shield_on'].mean()) if 'shield_on' in d.columns else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description='Phase17.5 - Walk-forward jerárquico tipo gym (imitación + RL + gating + shield).')
    ap.add_argument('--phase16_4_run_dir', required=True, type=str)
    ap.add_argument('--out_dir', default='', type=str)

    ap.add_argument('--min_train_windows', type=int, default=1200)
    ap.add_argument('--val_windows', type=int, default=128)
    ap.add_argument('--test_windows', type=int, default=256)
    ap.add_argument('--step_windows', type=int, default=256)

    ap.add_argument('--d_model', type=int, default=192)
    ap.add_argument('--nhead', type=int, default=6)
    ap.add_argument('--num_layers', type=int, default=3)
    ap.add_argument('--ff_dim', type=int, default=384)
    ap.add_argument('--dropout', type=float, default=0.1)

    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--imitation_epochs', type=int, default=8)
    ap.add_argument('--rl_epochs', type=int, default=18)
    ap.add_argument('--patience', type=int, default=6)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--weight_decay', type=float, default=1e-5)

    ap.add_argument('--max_gross_leverage', type=float, default=2.0)
    ap.add_argument('--target_gross_leverage', type=float, default=0.9)
    ap.add_argument('--min_target_mult', type=float, default=0.35)
    ap.add_argument('--max_target_mult', type=float, default=1.45)
    ap.add_argument('--weight_temperature', type=float, default=1.0)

    ap.add_argument('--loss_w_ret', type=float, default=1.0)
    ap.add_argument('--loss_w_risk', type=float, default=0.30)
    ap.add_argument('--loss_w_excess', type=float, default=1.0)
    ap.add_argument('--loss_w_exposure', type=float, default=0.12)
    ap.add_argument('--loss_w_l1', type=float, default=0.002)
    ap.add_argument('--loss_w_turnover', type=float, default=0.06)
    ap.add_argument('--loss_w_bc_anchor', type=float, default=0.18)
    ap.add_argument('--loss_w_value', type=float, default=0.15)
    ap.add_argument('--loss_w_gate', type=float, default=0.10)

    ap.add_argument('--temporal_weight_min', type=float, default=0.6)
    ap.add_argument('--temporal_weight_max', type=float, default=1.6)

    ap.add_argument('--cost_per_turnover', type=float, default=5e-5)
    ap.add_argument('--turnover_smooth_alpha', type=float, default=0.7)

    ap.add_argument('--shield_dd_limit', type=float, default=-0.08)
    ap.add_argument('--shield_risk_mult', type=float, default=0.55)

    ap.add_argument('--ensemble_seeds', type=str, default='42,43')
    ap.add_argument('--gate_min_excess_sharpe', type=float, default=0.05)
    ap.add_argument('--gate_min_excess_cum', type=float, default=0.0)

    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = ap.parse_args()

    p16 = Path(args.phase16_4_run_dir).resolve()
    if not p16.exists():
        raise FileNotFoundError(p16)

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (p16 / 'phase17_hierarchical_gym')
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config(
        phase16_4_run_dir=str(p16),
        out_dir=str(out_dir),
        min_train_windows=int(args.min_train_windows),
        val_windows=int(args.val_windows),
        test_windows=int(args.test_windows),
        step_windows=int(args.step_windows),
        d_model=int(args.d_model),
        nhead=int(args.nhead),
        num_layers=int(args.num_layers),
        ff_dim=int(args.ff_dim),
        dropout=float(args.dropout),
        batch_size=int(args.batch_size),
        imitation_epochs=int(args.imitation_epochs),
        rl_epochs=int(args.rl_epochs),
        patience=int(args.patience),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        max_gross_leverage=float(args.max_gross_leverage),
        target_gross_leverage=float(args.target_gross_leverage),
        min_target_mult=float(args.min_target_mult),
        max_target_mult=float(args.max_target_mult),
        weight_temperature=float(args.weight_temperature),
        loss_w_ret=float(args.loss_w_ret),
        loss_w_risk=float(args.loss_w_risk),
        loss_w_excess=float(args.loss_w_excess),
        loss_w_exposure=float(args.loss_w_exposure),
        loss_w_l1=float(args.loss_w_l1),
        loss_w_turnover=float(args.loss_w_turnover),
        loss_w_bc_anchor=float(args.loss_w_bc_anchor),
        loss_w_value=float(args.loss_w_value),
        loss_w_gate=float(args.loss_w_gate),
        temporal_weight_min=float(args.temporal_weight_min),
        temporal_weight_max=float(args.temporal_weight_max),
        cost_per_turnover=float(args.cost_per_turnover),
        turnover_smooth_alpha=float(args.turnover_smooth_alpha),
        shield_dd_limit=float(args.shield_dd_limit),
        shield_risk_mult=float(args.shield_risk_mult),
        ensemble_seeds=str(args.ensemble_seeds),
        gate_min_excess_sharpe=float(args.gate_min_excess_sharpe),
        gate_min_excess_cum=float(args.gate_min_excess_cum),
        seed=int(args.seed),
        device=str(args.device),
    )

    logger = setup_logger(out_dir / 'phase17_5.log')
    logger.info('Config: %s', cfg)

    set_seed(int(cfg.seed))

    p_npz = p16 / 'phase16_4_windows.npz'
    p_meta = p16 / 'phase16_4_windows_meta.parquet'
    p_summary = p16 / 'phase16_4_summary.json'
    p_feat = p16 / 'phase16_4_feature_columns.txt'
    if not p_npz.exists() or not p_meta.exists() or not p_summary.exists() or not p_feat.exists():
        raise FileNotFoundError('Missing phase16.4 artifacts.')

    data = np.load(p_npz)
    x = data['x'].astype(np.float32)
    y_ret1 = data['y_ret1'].astype(np.float32)
    y_mask = data['y_mask'].astype(np.float32) if 'y_mask' in data.files else np.ones_like(y_ret1, dtype=np.float32)

    meta = pd.read_parquet(p_meta).copy()
    meta['timestamp_end'] = pd.to_datetime(meta['timestamp_end'], utc=True)

    s16 = json.loads(p_summary.read_text(encoding='utf-8'))
    symbols = list(s16.get('symbols', [f'S{i}' for i in range(y_ret1.shape[1])]))
    if len(symbols) != y_ret1.shape[1]:
        symbols = symbols[:y_ret1.shape[1]] + [f'S{i}' for i in range(len(symbols), y_ret1.shape[1])]

    feature_cols = [ln.strip() for ln in p_feat.read_text(encoding='utf-8').splitlines() if ln.strip()]
    if len(feature_cols) != x.shape[2]:
        raise RuntimeError(f'Feature column count mismatch: file={len(feature_cols)} vs x={x.shape[2]}')

    order = np.argsort(meta['timestamp_end'].to_numpy())
    x = x[order]
    y_ret1 = y_ret1[order]
    y_mask = y_mask[order]
    meta = meta.iloc[order].reset_index(drop=True)

    n = len(x)
    folds = make_walkforward_folds(
        n=n,
        min_train=int(cfg.min_train_windows),
        val_w=int(cfg.val_windows),
        test_w=int(cfg.test_windows),
        step_w=int(cfg.step_windows),
    )
    if not folds:
        raise RuntimeError('No folds generated. Reduce min_train/val/test windows.')

    seed_list = parse_seed_list(cfg.ensemble_seeds)
    logger.info('Folds=%s | Ensemble seeds=%s', len(folds), seed_list)

    pd.DataFrame(folds).to_csv(out_dir / 'phase17_5_fold_ranges.csv', index=False)

    device = torch.device('cuda' if (cfg.device == 'auto' and torch.cuda.is_available()) else cfg.device if cfg.device != 'auto' else 'cpu')
    logger.info('Device=%s', device)

    fold_metric_rows = []
    history_rows = []
    steps_all = []
    gate_rows = []

    for f in folds:
        fold_id = int(f['fold_id'])
        i_tr = np.arange(f['train_start'], f['train_end'])
        i_va = np.arange(f['val_start'], f['val_end'])
        i_te = np.arange(f['test_start'], f['test_end'])

        xz, _, _ = standardize_windows(x, idx_train=i_tr)

        x_last = x[:, -1, :]
        teacher_all = build_teacher_weights_from_features(
            x_last_raw=x_last,
            feature_cols=feature_cols,
            symbols=symbols,
            y_mask=y_mask,
            target_gross=float(cfg.target_gross_leverage),
            max_gross=float(cfg.max_gross_leverage),
        )

        # temporal recency reweight
        tw_tr = make_temporal_weights(len(i_tr), cfg.temporal_weight_min, cfg.temporal_weight_max)
        tw_va = make_temporal_weights(len(i_va), cfg.temporal_weight_min, cfg.temporal_weight_max)

        ds_tr = TensorDataset(
            torch.from_numpy(xz[i_tr]),
            torch.from_numpy(y_ret1[i_tr]),
            torch.from_numpy(y_mask[i_tr]),
            torch.from_numpy(tw_tr),
            torch.from_numpy(teacher_all[i_tr]),
        )
        ds_va = TensorDataset(
            torch.from_numpy(xz[i_va]),
            torch.from_numpy(y_ret1[i_va]),
            torch.from_numpy(y_mask[i_va]),
            torch.from_numpy(tw_va),
            torch.from_numpy(teacher_all[i_va]),
        )

        dl_tr = DataLoader(ds_tr, batch_size=int(cfg.batch_size), shuffle=False, drop_last=False)
        dl_va = DataLoader(ds_va, batch_size=int(cfg.batch_size), shuffle=False, drop_last=False)

        val_weights_ensemble = []
        test_weights_ensemble = []

        for seed_i in seed_list:
            set_seed(int(seed_i))

            model = HierarchicalPolicyTransformer(
                input_dim=xz.shape[2],
                seq_len=xz.shape[1],
                n_assets=y_ret1.shape[1],
                d_model=int(cfg.d_model),
                nhead=int(cfg.nhead),
                num_layers=int(cfg.num_layers),
                ff_dim=int(cfg.ff_dim),
                dropout=float(cfg.dropout),
            ).to(device)

            opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

            # Stage A: imitation warm-start
            best_im_val = float('inf')
            best_state_im = None
            bad = 0
            for ep in range(1, int(cfg.imitation_epochs) + 1):
                tr = run_epoch_imitation(model=model, dl=dl_tr, device=device, cfg=cfg, train=True, optimizer=opt)
                va = run_epoch_imitation(model=model, dl=dl_va, device=device, cfg=cfg, train=False, optimizer=None)

                history_rows.append(
                    {
                        'fold_id': fold_id,
                        'seed': int(seed_i),
                        'stage': 'imitation',
                        'epoch': ep,
                        **{f'train_{k}': v for k, v in tr.items()},
                        **{f'val_{k}': v for k, v in va.items()},
                    }
                )

                if va['loss_total'] < best_im_val:
                    best_im_val = va['loss_total']
                    best_state_im = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                    if bad >= int(cfg.patience):
                        break

            if best_state_im is not None:
                model.load_state_dict(best_state_im)

            # Stage B: RL fine-tune with early stop on val RL loss
            best_rl_val = float('inf')
            best_state_rl = None
            bad = 0
            for ep in range(1, int(cfg.rl_epochs) + 1):
                tr = run_epoch_rl(model=model, dl=dl_tr, device=device, cfg=cfg, train=True, optimizer=opt)
                va = run_epoch_rl(model=model, dl=dl_va, device=device, cfg=cfg, train=False, optimizer=None)

                history_rows.append(
                    {
                        'fold_id': fold_id,
                        'seed': int(seed_i),
                        'stage': 'rl',
                        'epoch': ep,
                        **{f'train_{k}': v for k, v in tr.items()},
                        **{f'val_{k}': v for k, v in va.items()},
                    }
                )

                if va['loss_total'] < best_rl_val:
                    best_rl_val = va['loss_total']
                    best_state_rl = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                    if bad >= int(cfg.patience):
                        break

            if best_state_rl is not None:
                model.load_state_dict(best_state_rl)

            val_weights_ensemble.append(
                infer_raw_weights(
                    model=model,
                    x=xz[i_va],
                    y_mask=y_mask[i_va],
                    device=device,
                    batch_size=int(cfg.batch_size),
                    cfg=cfg,
                )
            )
            test_weights_ensemble.append(
                infer_raw_weights(
                    model=model,
                    x=xz[i_te],
                    y_mask=y_mask[i_te],
                    device=device,
                    batch_size=int(cfg.batch_size),
                    cfg=cfg,
                )
            )

        w_val_model = np.mean(np.stack(val_weights_ensemble, axis=0), axis=0)
        w_test_model = np.mean(np.stack(test_weights_ensemble, axis=0), axis=0)

        w_val_eq_target = baseline_equal_weight(y_mask=y_mask[i_va], max_gross=float(cfg.target_gross_leverage))
        w_val_eq_max = baseline_equal_weight(y_mask=y_mask[i_va], max_gross=float(cfg.max_gross_leverage))
        w_val_teacher = teacher_all[i_va]

        _, m_val_model = backtest_from_weights(
            y_ret1=y_ret1[i_va],
            y_mask=y_mask[i_va],
            w_raw=w_val_model,
            timestamps=meta['timestamp_end'].to_numpy()[i_va],
            cost_per_turnover=float(cfg.cost_per_turnover),
            cost_mult=1.0,
            smooth_alpha=float(cfg.turnover_smooth_alpha),
            max_gross=float(cfg.max_gross_leverage),
            shield_dd_limit=float(cfg.shield_dd_limit),
            shield_risk_mult=float(cfg.shield_risk_mult),
        )
        _, m_val_eq = backtest_from_weights(
            y_ret1=y_ret1[i_va],
            y_mask=y_mask[i_va],
            w_raw=w_val_eq_target,
            timestamps=meta['timestamp_end'].to_numpy()[i_va],
            cost_per_turnover=float(cfg.cost_per_turnover),
            cost_mult=1.0,
            smooth_alpha=float(cfg.turnover_smooth_alpha),
            max_gross=float(cfg.max_gross_leverage),
            shield_dd_limit=float(cfg.shield_dd_limit),
            shield_risk_mult=float(cfg.shield_risk_mult),
        )

        gate_on = bool(
            (m_val_model['sharpe_ann'] >= m_val_eq['sharpe_ann'] + float(cfg.gate_min_excess_sharpe))
            and (m_val_model['cum_net_return'] >= m_val_eq['cum_net_return'] + float(cfg.gate_min_excess_cum))
            and (m_val_model['mean_net_ret'] > 0.0)
        )

        gate_rows.append(
            {
                'fold_id': fold_id,
                'gate_on': int(gate_on),
                'val_model_sharpe_x1': float(m_val_model['sharpe_ann']),
                'val_eq_target_sharpe_x1': float(m_val_eq['sharpe_ann']),
                'val_model_cum_x1': float(m_val_model['cum_net_return']),
                'val_eq_target_cum_x1': float(m_val_eq['cum_net_return']),
            }
        )

        w_test_eq_target = baseline_equal_weight(y_mask=y_mask[i_te], max_gross=float(cfg.target_gross_leverage))
        w_test_eq_max = baseline_equal_weight(y_mask=y_mask[i_te], max_gross=float(cfg.max_gross_leverage))
        w_test_teacher = teacher_all[i_te]
        w_test_gated = w_test_model if gate_on else w_test_eq_target

        ts_te = meta['timestamp_end'].to_numpy()[i_te]
        policies = [
            ('model_hier', w_test_model),
            ('model_hier_gated', w_test_gated),
            ('teacher_lag1', w_test_teacher),
            ('equal_weight_target', w_test_eq_target),
            ('equal_weight_max', w_test_eq_max),
        ]

        for mult in [1.0, 2.0, 3.0]:
            for pol, wpol in policies:
                step_df, m = backtest_from_weights(
                    y_ret1=y_ret1[i_te],
                    y_mask=y_mask[i_te],
                    w_raw=wpol,
                    timestamps=ts_te,
                    cost_per_turnover=float(cfg.cost_per_turnover),
                    cost_mult=float(mult),
                    smooth_alpha=float(cfg.turnover_smooth_alpha),
                    max_gross=float(cfg.max_gross_leverage),
                    shield_dd_limit=float(cfg.shield_dd_limit),
                    shield_risk_mult=float(cfg.shield_risk_mult),
                )
                step_df['fold_id'] = fold_id
                step_df['policy'] = pol
                step_df['cost_mult'] = float(mult)
                step_df['gate_on'] = int(gate_on)
                steps_all.append(step_df)

                fold_metric_rows.append(
                    {
                        'fold_id': fold_id,
                        'policy': pol,
                        'cost_mult': float(mult),
                        'gate_on': int(gate_on),
                        **m,
                        'train_n': int(len(i_tr)),
                        'val_n': int(len(i_va)),
                        'test_n': int(len(i_te)),
                        'test_start_ts': str(meta['timestamp_end'].iloc[i_te[0]]),
                        'test_end_ts': str(meta['timestamp_end'].iloc[i_te[-1]]),
                    }
                )

        logger.info(
            'fold=%s done | gate=%s | test %s -> %s',
            fold_id,
            int(gate_on),
            meta['timestamp_end'].iloc[i_te[0]],
            meta['timestamp_end'].iloc[i_te[-1]],
        )

    hist_df = pd.DataFrame(history_rows)
    hist_df.to_csv(out_dir / 'phase17_5_train_history.csv', index=False)

    fold_metrics = pd.DataFrame(fold_metric_rows)
    fold_metrics.to_csv(out_dir / 'phase17_5_fold_metrics.csv', index=False)

    gate_df = pd.DataFrame(gate_rows)
    gate_df.to_csv(out_dir / 'phase17_5_gate_decisions.csv', index=False)

    steps_df = pd.concat(steps_all, axis=0, ignore_index=True)
    steps_df = steps_df.sort_values(['timestamp_end', 'fold_id', 'policy', 'cost_mult']).reset_index(drop=True)
    steps_df.to_parquet(out_dir / 'phase17_5_test_steps.parquet', index=False)

    pooled_rows = []
    for policy in ['model_hier', 'model_hier_gated', 'teacher_lag1', 'equal_weight_target', 'equal_weight_max']:
        for mult in [1.0, 2.0, 3.0]:
            g = steps_df[(steps_df['policy'] == policy) & (np.isclose(steps_df['cost_mult'], float(mult)))].copy()
            m = aggregate_policy_metrics(g)
            pooled_rows.append({'policy': policy, 'cost_mult': float(mult), **m})

    pooled_df = pd.DataFrame(pooled_rows)
    pooled_df.to_csv(out_dir / 'phase17_5_pooled_metrics.csv', index=False)

    plt.figure(figsize=(11, 5))
    for policy in ['equal_weight_target', 'equal_weight_max', 'teacher_lag1', 'model_hier', 'model_hier_gated']:
        for mult in [1.0, 2.0, 3.0]:
            g = steps_df[(steps_df['policy'] == policy) & (np.isclose(steps_df['cost_mult'], float(mult)))].copy()
            if len(g) == 0:
                continue
            g = g.sort_values('timestamp_end').reset_index(drop=True)
            g['equity'] = (1.0 + g['net_ret']).cumprod()
            plt.plot(g['timestamp_end'], g['equity'], label=f'{policy}_x{int(mult)}')
    plt.title('Phase17.5 pooled walk-forward test equity')
    plt.xlabel('timestamp')
    plt.ylabel('equity')
    plt.legend(ncol=5, fontsize=7)
    plt.tight_layout()
    plt.savefig(out_dir / 'phase17_5_pooled_test_equity.png', dpi=140)
    plt.close()

    def get_metric(policy: str, mult: float, col: str) -> float:
        s = pooled_df[(pooled_df['policy'] == policy) & (np.isclose(pooled_df['cost_mult'], float(mult)))]
        return float(s.iloc[0][col]) if len(s) else float('nan')

    comparisons = {}
    for mult in [1.0, 2.0, 3.0]:
        comparisons[f'x{int(mult)}'] = {
            'model_hier_sharpe': get_metric('model_hier', mult, 'sharpe_ann'),
            'model_hier_gated_sharpe': get_metric('model_hier_gated', mult, 'sharpe_ann'),
            'teacher_lag1_sharpe': get_metric('teacher_lag1', mult, 'sharpe_ann'),
            'eq_target_sharpe': get_metric('equal_weight_target', mult, 'sharpe_ann'),
            'eq_max_sharpe': get_metric('equal_weight_max', mult, 'sharpe_ann'),
            'model_hier_cum': get_metric('model_hier', mult, 'cum_net_return'),
            'model_hier_gated_cum': get_metric('model_hier_gated', mult, 'cum_net_return'),
            'teacher_lag1_cum': get_metric('teacher_lag1', mult, 'cum_net_return'),
            'eq_target_cum': get_metric('equal_weight_target', mult, 'cum_net_return'),
            'eq_max_cum': get_metric('equal_weight_max', mult, 'cum_net_return'),
        }

    summary = {
        'status': 'SUCCESS',
        'config': asdict(cfg),
        'n_total_windows': int(n),
        'n_folds': int(len(folds)),
        'symbols': symbols,
        'folds': folds,
        'gate_on_rate': float(gate_df['gate_on'].mean()) if len(gate_df) else 0.0,
        'pooled_metrics': pooled_rows,
        'comparisons': comparisons,
        'artifacts': {
            'fold_ranges_csv': str(out_dir / 'phase17_5_fold_ranges.csv'),
            'train_history_csv': str(out_dir / 'phase17_5_train_history.csv'),
            'fold_metrics_csv': str(out_dir / 'phase17_5_fold_metrics.csv'),
            'gate_decisions_csv': str(out_dir / 'phase17_5_gate_decisions.csv'),
            'test_steps_parquet': str(out_dir / 'phase17_5_test_steps.parquet'),
            'pooled_metrics_csv': str(out_dir / 'phase17_5_pooled_metrics.csv'),
            'pooled_test_equity_png': str(out_dir / 'phase17_5_pooled_test_equity.png'),
            'log': str(out_dir / 'phase17_5.log'),
        },
    }

    (out_dir / 'phase17_5_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    logger.info('SUCCESS: %s', out_dir / 'phase17_5_summary.json')


if __name__ == '__main__':
    main()
