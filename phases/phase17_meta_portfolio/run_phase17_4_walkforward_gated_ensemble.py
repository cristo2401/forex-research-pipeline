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
    epochs: int = 24
    patience: int = 6
    lr: float = 3e-4
    weight_decay: float = 1e-5

    max_gross_leverage: float = 2.0
    target_gross_leverage: float = 0.9
    weight_temperature: float = 1.0

    loss_w_ret: float = 1.0
    loss_w_risk: float = 0.3
    loss_w_port: float = 1.0
    loss_w_excess: float = 1.0
    loss_w_exposure: float = 0.15
    loss_w_l1: float = 0.003
    loss_w_turnover: float = 0.05

    temporal_weight_min: float = 0.6
    temporal_weight_max: float = 1.6

    cost_per_turnover: float = 5e-5
    turnover_smooth_alpha: float = 0.7

    ensemble_seeds: str = '42,43'
    gate_min_excess_sharpe: float = 0.10
    gate_min_excess_cum: float = 0.0

    seed: int = 42
    device: str = 'auto'


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger('phase17_walkforward_gated_ensemble')
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


class PortfolioTransformer(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, n_assets: int, d_model: int, nhead: int, num_layers: int, ff_dim: int, dropout: float):
        super().__init__()
        self.seq_len = int(seq_len)
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

        self.ret_head = nn.Linear(int(d_model), int(n_assets))
        self.risk_head = nn.Linear(int(d_model), int(n_assets))
        self.weight_head = nn.Linear(int(d_model), int(n_assets))

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

        return {
            'ret': self.ret_head(z),
            'risk_raw': self.risk_head(z),
            'score': self.weight_head(z),
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


def build_weights(score: torch.Tensor, mask: torch.Tensor, temperature: float, target_gross: float, max_gross: float) -> torch.Tensor:
    raw = torch.tanh(score / max(float(temperature), 1e-6))
    raw = raw * mask

    gross_raw = torch.sum(torch.abs(raw), dim=1, keepdim=True)
    low = gross_raw < 1e-4
    if torch.any(low):
        fallback = build_equal_weight_torch(mask, target_gross=float(target_gross))
        raw = torch.where(low, fallback, raw)
        gross_raw = torch.sum(torch.abs(raw), dim=1, keepdim=True)

    active = torch.sum(mask > 0.5, dim=1, keepdim=True).float()
    target = torch.where(active > 0, torch.full_like(active, float(target_gross)), torch.zeros_like(active))
    w = raw * (target / (gross_raw + 1e-12))

    gross = torch.sum(torch.abs(w), dim=1, keepdim=True)
    cap = torch.where(gross > float(max_gross), float(max_gross) / (gross + 1e-12), torch.ones_like(gross))
    return w * cap


def compute_losses(
    out: Dict[str, torch.Tensor],
    y_ret1: torch.Tensor,
    y_mask: torch.Tensor,
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

    w = build_weights(
        score=out['score'],
        mask=y_mask,
        temperature=float(cfg.weight_temperature),
        target_gross=float(cfg.target_gross_leverage),
        max_gross=float(cfg.max_gross_leverage),
    )

    port_ret = torch.sum(w * y_ret1, dim=1)
    w_eq = build_equal_weight_torch(mask=y_mask, target_gross=float(cfg.target_gross_leverage))
    eq_ret = torch.sum(w_eq * y_ret1, dim=1)

    wt1 = t_weight
    den_t = torch.clamp(torch.sum(wt1), min=1.0)
    loss_port = -torch.sum(port_ret * wt1) / den_t
    loss_excess = -torch.sum((port_ret - eq_ret) * wt1) / den_t

    gross = torch.sum(torch.abs(w), dim=1)
    active = torch.sum(y_mask > 0.5, dim=1).float()
    target = torch.where(active > 0, torch.full_like(active, float(cfg.target_gross_leverage)), torch.zeros_like(active))
    loss_exposure = torch.sum(((gross - target) ** 2) * wt1) / den_t

    loss_l1 = torch.sum(torch.mean(torch.abs(w), dim=1) * wt1) / den_t

    # turnover proxy in temporal order within batch
    if len(w) > 1:
        turn = torch.mean(torch.abs(w[1:] - w[:-1]), dim=1)
        wt_turn = wt1[1:]
        loss_turn = torch.sum(turn * wt_turn) / torch.clamp(torch.sum(wt_turn), min=1.0)
    else:
        loss_turn = torch.zeros((), device=w.device)

    loss_total = (
        float(cfg.loss_w_ret) * loss_ret
        + float(cfg.loss_w_risk) * loss_risk
        + float(cfg.loss_w_port) * loss_port
        + float(cfg.loss_w_excess) * loss_excess
        + float(cfg.loss_w_exposure) * loss_exposure
        + float(cfg.loss_w_l1) * loss_l1
        + float(cfg.loss_w_turnover) * loss_turn
    )

    return {
        'loss_total': loss_total,
        'loss_ret': loss_ret,
        'loss_risk': loss_risk,
        'loss_port': loss_port,
        'loss_excess': loss_excess,
        'loss_exposure': loss_exposure,
        'loss_l1': loss_l1,
        'loss_turn': loss_turn,
        'w': w,
    }


def run_epoch(model: PortfolioTransformer, dl: DataLoader, device: torch.device, cfg: Config, train: bool, optimizer: torch.optim.Optimizer | None = None) -> Dict[str, float]:
    model.train() if train else model.eval()
    rows: List[Dict[str, float]] = []

    for xb, yb, mb, tw in dl:
        xb = xb.to(device)
        yb = yb.to(device)
        mb = mb.to(device)
        tw = tw.to(device)

        if train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            out = model(xb)
            losses = compute_losses(out=out, y_ret1=yb, y_mask=mb, t_weight=tw, cfg=cfg)
            loss = losses['loss_total']
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        w = losses['w'].detach()
        realized = torch.sum(w * yb.detach(), dim=1)
        gross = torch.sum(torch.abs(w), dim=1)
        active_assets = torch.sum(mb.detach() > 0.5, dim=1).float()

        rows.append(
            {
                'loss_total': float(losses['loss_total'].detach().item()),
                'loss_ret': float(losses['loss_ret'].detach().item()),
                'loss_risk': float(losses['loss_risk'].detach().item()),
                'loss_port': float(losses['loss_port'].detach().item()),
                'loss_excess': float(losses['loss_excess'].detach().item()),
                'loss_exposure': float(losses['loss_exposure'].detach().item()),
                'loss_l1': float(losses['loss_l1'].detach().item()),
                'loss_turn': float(losses['loss_turn'].detach().item()),
                'realized_port_mean': float(realized.mean().item()),
                'gross_mean': float(gross.mean().item()),
                'active_assets_mean': float(active_assets.mean().item()),
            }
        )

    if not rows:
        keys = ['loss_total', 'loss_ret', 'loss_risk', 'loss_port', 'loss_excess', 'loss_exposure', 'loss_l1', 'loss_turn', 'realized_port_mean', 'gross_mean', 'active_assets_mean']
        return {k: float('nan') for k in keys}

    d = pd.DataFrame(rows)
    return {c: float(d[c].mean()) for c in d.columns}


def infer_scores(model: PortfolioTransformer, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    ds = TensorDataset(torch.from_numpy(x.astype(np.float32)))
    dl = DataLoader(ds, batch_size=int(batch_size), shuffle=False, drop_last=False)

    model.eval()
    all_sc = []
    with torch.no_grad():
        for (xb,) in dl:
            xb = xb.to(device)
            out = model(xb)
            all_sc.append(out['score'].detach().cpu().numpy())
    return np.concatenate(all_sc, axis=0)


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
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    n = len(y_ret1)
    n_assets = y_ret1.shape[1]
    prev_exec = np.zeros((n_assets,), dtype=np.float32)

    rows = []
    a = float(np.clip(smooth_alpha, 0.0, 1.0))
    for i in range(n):
        w_tgt = (w_raw[i] * y_mask[i]).astype(np.float32)
        w_exec = ((1.0 - a) * prev_exec + a * w_tgt).astype(np.float32)

        turnover = float(np.sum(np.abs(w_exec - prev_exec)))
        gross = float(np.sum(w_exec * y_ret1[i]))
        cost = float(cost_per_turnover * cost_mult * turnover)
        net = gross - cost

        rows.append(
            {
                'timestamp_end': pd.Timestamp(timestamps[i]),
                'gross_ret': gross,
                'net_ret': net,
                'turnover': turnover,
                'cost': cost,
                'gross_lev': float(np.sum(np.abs(w_exec))),
                'active_assets': int(np.sum(y_mask[i] > 0.5)),
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
    }


def main():
    ap = argparse.ArgumentParser(description='Phase17.4 - Walk-forward con ensemble + gating + temporal reweight + turnover control.')
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
    ap.add_argument('--epochs', type=int, default=24)
    ap.add_argument('--patience', type=int, default=6)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--weight_decay', type=float, default=1e-5)

    ap.add_argument('--max_gross_leverage', type=float, default=2.0)
    ap.add_argument('--target_gross_leverage', type=float, default=0.9)
    ap.add_argument('--weight_temperature', type=float, default=1.0)

    ap.add_argument('--loss_w_ret', type=float, default=1.0)
    ap.add_argument('--loss_w_risk', type=float, default=0.3)
    ap.add_argument('--loss_w_port', type=float, default=1.0)
    ap.add_argument('--loss_w_excess', type=float, default=1.0)
    ap.add_argument('--loss_w_exposure', type=float, default=0.15)
    ap.add_argument('--loss_w_l1', type=float, default=0.003)
    ap.add_argument('--loss_w_turnover', type=float, default=0.05)

    ap.add_argument('--temporal_weight_min', type=float, default=0.6)
    ap.add_argument('--temporal_weight_max', type=float, default=1.6)

    ap.add_argument('--cost_per_turnover', type=float, default=5e-5)
    ap.add_argument('--turnover_smooth_alpha', type=float, default=0.7)

    ap.add_argument('--ensemble_seeds', type=str, default='42,43')
    ap.add_argument('--gate_min_excess_sharpe', type=float, default=0.10)
    ap.add_argument('--gate_min_excess_cum', type=float, default=0.0)

    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = ap.parse_args()

    p16 = Path(args.phase16_4_run_dir).resolve()
    if not p16.exists():
        raise FileNotFoundError(p16)

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (p16 / 'phase17_walkforward_gated_ensemble')
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
        epochs=int(args.epochs),
        patience=int(args.patience),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        max_gross_leverage=float(args.max_gross_leverage),
        target_gross_leverage=float(args.target_gross_leverage),
        weight_temperature=float(args.weight_temperature),
        loss_w_ret=float(args.loss_w_ret),
        loss_w_risk=float(args.loss_w_risk),
        loss_w_port=float(args.loss_w_port),
        loss_w_excess=float(args.loss_w_excess),
        loss_w_exposure=float(args.loss_w_exposure),
        loss_w_l1=float(args.loss_w_l1),
        loss_w_turnover=float(args.loss_w_turnover),
        temporal_weight_min=float(args.temporal_weight_min),
        temporal_weight_max=float(args.temporal_weight_max),
        cost_per_turnover=float(args.cost_per_turnover),
        turnover_smooth_alpha=float(args.turnover_smooth_alpha),
        ensemble_seeds=str(args.ensemble_seeds),
        gate_min_excess_sharpe=float(args.gate_min_excess_sharpe),
        gate_min_excess_cum=float(args.gate_min_excess_cum),
        seed=int(args.seed),
        device=str(args.device),
    )

    logger = setup_logger(out_dir / 'phase17_4.log')
    logger.info('Config: %s', cfg)

    set_seed(int(cfg.seed))

    p_npz = p16 / 'phase16_4_windows.npz'
    p_meta = p16 / 'phase16_4_windows_meta.parquet'
    p_summary = p16 / 'phase16_4_summary.json'
    if not p_npz.exists() or not p_meta.exists() or not p_summary.exists():
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

    pd.DataFrame(folds).to_csv(out_dir / 'phase17_4_fold_ranges.csv', index=False)

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

        # temporal weights increase with recency inside each split
        tw_tr = make_temporal_weights(len(i_tr), cfg.temporal_weight_min, cfg.temporal_weight_max)
        tw_va = make_temporal_weights(len(i_va), cfg.temporal_weight_min, cfg.temporal_weight_max)

        ds_tr = TensorDataset(torch.from_numpy(xz[i_tr]), torch.from_numpy(y_ret1[i_tr]), torch.from_numpy(y_mask[i_tr]), torch.from_numpy(tw_tr))
        ds_va = TensorDataset(torch.from_numpy(xz[i_va]), torch.from_numpy(y_ret1[i_va]), torch.from_numpy(y_mask[i_va]), torch.from_numpy(tw_va))

        dl_tr = DataLoader(ds_tr, batch_size=int(cfg.batch_size), shuffle=False, drop_last=False)
        dl_va = DataLoader(ds_va, batch_size=int(cfg.batch_size), shuffle=False, drop_last=False)

        val_scores_ensemble = []
        test_scores_ensemble = []

        for seed_i in seed_list:
            set_seed(int(seed_i))

            model = PortfolioTransformer(
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

            best_val = float('inf')
            best_state = None
            bad = 0

            for ep in range(1, int(cfg.epochs) + 1):
                tr = run_epoch(model=model, dl=dl_tr, device=device, cfg=cfg, train=True, optimizer=opt)
                va = run_epoch(model=model, dl=dl_va, device=device, cfg=cfg, train=False, optimizer=None)

                history_rows.append(
                    {
                        'fold_id': fold_id,
                        'seed': int(seed_i),
                        'epoch': ep,
                        **{f'train_{k}': v for k, v in tr.items()},
                        **{f'val_{k}': v for k, v in va.items()},
                    }
                )

                if va['loss_total'] < best_val:
                    best_val = va['loss_total']
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                    if bad >= int(cfg.patience):
                        break

            if best_state is not None:
                model.load_state_dict(best_state)

            val_scores_ensemble.append(
                infer_scores(
                    model=model,
                    x=xz[i_va],
                    device=device,
                    batch_size=int(cfg.batch_size),
                )
            )
            test_scores_ensemble.append(
                infer_scores(
                    model=model,
                    x=xz[i_te],
                    device=device,
                    batch_size=int(cfg.batch_size),
                )
            )

        sc_val = np.mean(np.stack(val_scores_ensemble, axis=0), axis=0)
        sc_test = np.mean(np.stack(test_scores_ensemble, axis=0), axis=0)

        w_val_model = normalize_weights_np(
            raw=sc_val / max(float(cfg.weight_temperature), 1e-6),
            mask=y_mask[i_va],
            target_gross=float(cfg.target_gross_leverage),
            max_gross=float(cfg.max_gross_leverage),
        )
        w_test_model = normalize_weights_np(
            raw=sc_test / max(float(cfg.weight_temperature), 1e-6),
            mask=y_mask[i_te],
            target_gross=float(cfg.target_gross_leverage),
            max_gross=float(cfg.max_gross_leverage),
        )

        w_val_eq_target = baseline_equal_weight(y_mask=y_mask[i_va], max_gross=float(cfg.target_gross_leverage))
        w_val_eq_max = baseline_equal_weight(y_mask=y_mask[i_va], max_gross=float(cfg.max_gross_leverage))

        # Gate by VAL excess over equal_weight_target in x1.
        _, m_val_model = backtest_from_weights(
            y_ret1=y_ret1[i_va],
            y_mask=y_mask[i_va],
            w_raw=w_val_model,
            timestamps=meta['timestamp_end'].to_numpy()[i_va],
            cost_per_turnover=float(cfg.cost_per_turnover),
            cost_mult=1.0,
            smooth_alpha=float(cfg.turnover_smooth_alpha),
        )
        _, m_val_eq = backtest_from_weights(
            y_ret1=y_ret1[i_va],
            y_mask=y_mask[i_va],
            w_raw=w_val_eq_target,
            timestamps=meta['timestamp_end'].to_numpy()[i_va],
            cost_per_turnover=float(cfg.cost_per_turnover),
            cost_mult=1.0,
            smooth_alpha=float(cfg.turnover_smooth_alpha),
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
        w_test_gated = w_test_model if gate_on else w_test_eq_target

        ts_te = meta['timestamp_end'].to_numpy()[i_te]
        policies = [
            ('model_ensemble', w_test_model),
            ('model_gated', w_test_gated),
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
    hist_df.to_csv(out_dir / 'phase17_4_train_history.csv', index=False)

    fold_metrics = pd.DataFrame(fold_metric_rows)
    fold_metrics.to_csv(out_dir / 'phase17_4_fold_metrics.csv', index=False)

    gate_df = pd.DataFrame(gate_rows)
    gate_df.to_csv(out_dir / 'phase17_4_gate_decisions.csv', index=False)

    steps_df = pd.concat(steps_all, axis=0, ignore_index=True)
    steps_df = steps_df.sort_values(['timestamp_end', 'fold_id', 'policy', 'cost_mult']).reset_index(drop=True)
    steps_df.to_parquet(out_dir / 'phase17_4_test_steps.parquet', index=False)

    pooled_rows = []
    for policy in ['model_ensemble', 'model_gated', 'equal_weight_target', 'equal_weight_max']:
        for mult in [1.0, 2.0, 3.0]:
            g = steps_df[(steps_df['policy'] == policy) & (np.isclose(steps_df['cost_mult'], float(mult)))].copy()
            m = aggregate_policy_metrics(g)
            pooled_rows.append({'policy': policy, 'cost_mult': float(mult), **m})

    pooled_df = pd.DataFrame(pooled_rows)
    pooled_df.to_csv(out_dir / 'phase17_4_pooled_metrics.csv', index=False)

    plt.figure(figsize=(11, 5))
    for policy in ['equal_weight_target', 'equal_weight_max', 'model_ensemble', 'model_gated']:
        for mult in [1.0, 2.0, 3.0]:
            g = steps_df[(steps_df['policy'] == policy) & (np.isclose(steps_df['cost_mult'], float(mult)))].copy()
            if len(g) == 0:
                continue
            g = g.sort_values('timestamp_end').reset_index(drop=True)
            g['equity'] = (1.0 + g['net_ret']).cumprod()
            plt.plot(g['timestamp_end'], g['equity'], label=f'{policy}_x{int(mult)}')
    plt.title('Phase17.4 pooled walk-forward test equity')
    plt.xlabel('timestamp')
    plt.ylabel('equity')
    plt.legend(ncol=4, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / 'phase17_4_pooled_test_equity.png', dpi=140)
    plt.close()

    def get_metric(policy: str, mult: float, col: str) -> float:
        s = pooled_df[(pooled_df['policy'] == policy) & (np.isclose(pooled_df['cost_mult'], float(mult)))]
        return float(s.iloc[0][col]) if len(s) else float('nan')

    comparisons = {}
    for mult in [1.0, 2.0, 3.0]:
        comparisons[f'x{int(mult)}'] = {
            'model_ensemble_sharpe': get_metric('model_ensemble', mult, 'sharpe_ann'),
            'model_gated_sharpe': get_metric('model_gated', mult, 'sharpe_ann'),
            'eq_target_sharpe': get_metric('equal_weight_target', mult, 'sharpe_ann'),
            'eq_max_sharpe': get_metric('equal_weight_max', mult, 'sharpe_ann'),
            'model_ensemble_cum': get_metric('model_ensemble', mult, 'cum_net_return'),
            'model_gated_cum': get_metric('model_gated', mult, 'cum_net_return'),
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
            'fold_ranges_csv': str(out_dir / 'phase17_4_fold_ranges.csv'),
            'train_history_csv': str(out_dir / 'phase17_4_train_history.csv'),
            'fold_metrics_csv': str(out_dir / 'phase17_4_fold_metrics.csv'),
            'gate_decisions_csv': str(out_dir / 'phase17_4_gate_decisions.csv'),
            'test_steps_parquet': str(out_dir / 'phase17_4_test_steps.parquet'),
            'pooled_metrics_csv': str(out_dir / 'phase17_4_pooled_metrics.csv'),
            'pooled_test_equity_png': str(out_dir / 'phase17_4_pooled_test_equity.png'),
            'log': str(out_dir / 'phase17_4.log'),
        },
    }

    (out_dir / 'phase17_4_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    logger.info('SUCCESS: %s', out_dir / 'phase17_4_summary.json')


if __name__ == '__main__':
    main()
