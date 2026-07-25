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
    epochs: int = 35
    patience: int = 7
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

    cost_per_turnover: float = 5e-5
    seed: int = 42
    device: str = 'auto'


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger('phase17_walkforward')
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
        b, l, _ = x.shape
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


def build_equal_weight_torch(mask: torch.Tensor, target_gross: float) -> torch.Tensor:
    k = torch.sum(mask > 0.5, dim=1, keepdim=True).float()
    k = torch.clamp(k, min=1.0)
    return (mask / k) * float(target_gross)


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


def compute_losses(out: Dict[str, torch.Tensor], y_ret1: torch.Tensor, y_mask: torch.Tensor, cfg: Config) -> Dict[str, torch.Tensor]:
    ret_pred = out['ret']
    risk_pred = torch.nn.functional.softplus(out['risk_raw'])

    den = torch.clamp(y_mask.sum(), min=1.0)
    loss_ret = torch.sum(torch.nn.functional.smooth_l1_loss(ret_pred, y_ret1, reduction='none') * y_mask) / den
    loss_risk = torch.sum(torch.nn.functional.smooth_l1_loss(risk_pred, torch.abs(y_ret1), reduction='none') * y_mask) / den

    w = build_weights(
        score=out['score'],
        mask=y_mask,
        temperature=float(cfg.weight_temperature),
        target_gross=float(cfg.target_gross_leverage),
        max_gross=float(cfg.max_gross_leverage),
    )
    port_ret = torch.sum(w * y_ret1, dim=1)
    loss_port = -torch.mean(port_ret)

    w_eq = build_equal_weight_torch(mask=y_mask, target_gross=float(cfg.target_gross_leverage))
    eq_ret = torch.sum(w_eq * y_ret1, dim=1)
    loss_excess = -torch.mean(port_ret - eq_ret)

    gross = torch.sum(torch.abs(w), dim=1)
    active = torch.sum(y_mask > 0.5, dim=1).float()
    target = torch.where(active > 0, torch.full_like(active, float(cfg.target_gross_leverage)), torch.zeros_like(active))
    loss_exposure = torch.mean((gross - target) ** 2)

    loss_l1 = torch.mean(torch.abs(w))

    loss_total = (
        float(cfg.loss_w_ret) * loss_ret
        + float(cfg.loss_w_risk) * loss_risk
        + float(cfg.loss_w_port) * loss_port
        + float(cfg.loss_w_excess) * loss_excess
        + float(cfg.loss_w_exposure) * loss_exposure
        + float(cfg.loss_w_l1) * loss_l1
    )

    return {
        'loss_total': loss_total,
        'loss_ret': loss_ret,
        'loss_risk': loss_risk,
        'loss_port': loss_port,
        'loss_excess': loss_excess,
        'loss_exposure': loss_exposure,
        'loss_l1': loss_l1,
        'w': w,
    }


def run_epoch(model: PortfolioTransformer, dl: DataLoader, device: torch.device, cfg: Config, train: bool, optimizer: torch.optim.Optimizer | None = None) -> Dict[str, float]:
    model.train() if train else model.eval()
    rows: List[Dict[str, float]] = []

    for xb, yb, mb in dl:
        xb = xb.to(device)
        yb = yb.to(device)
        mb = mb.to(device)

        if train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            out = model(xb)
            losses = compute_losses(out=out, y_ret1=yb, y_mask=mb, cfg=cfg)
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
                'realized_port_mean': float(realized.mean().item()),
                'gross_mean': float(gross.mean().item()),
                'active_assets_mean': float(active_assets.mean().item()),
            }
        )

    if not rows:
        return {k: float('nan') for k in ['loss_total', 'loss_ret', 'loss_risk', 'loss_port', 'loss_excess', 'loss_exposure', 'loss_l1', 'realized_port_mean', 'gross_mean', 'active_assets_mean']}

    d = pd.DataFrame(rows)
    return {c: float(d[c].mean()) for c in d.columns}


def infer_weights(model: PortfolioTransformer, x: np.ndarray, y_mask: np.ndarray, cfg: Config, device: torch.device, batch_size: int) -> np.ndarray:
    ds = TensorDataset(torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y_mask.astype(np.float32)))
    dl = DataLoader(ds, batch_size=int(batch_size), shuffle=False, drop_last=False)

    model.eval()
    all_w = []
    with torch.no_grad():
        for xb, mb in dl:
            xb = xb.to(device)
            mb = mb.to(device)
            out = model(xb)
            w = build_weights(
                score=out['score'],
                mask=mb,
                temperature=float(cfg.weight_temperature),
                target_gross=float(cfg.target_gross_leverage),
                max_gross=float(cfg.max_gross_leverage),
            )
            all_w.append(w.detach().cpu().numpy())
    return np.concatenate(all_w, axis=0)


def compute_drawdown(equity_curve: np.ndarray) -> float:
    if len(equity_curve) == 0:
        return float('nan')
    peak = np.maximum.accumulate(equity_curve)
    dd = equity_curve / np.maximum(peak, 1e-12) - 1.0
    return float(np.min(dd))


def backtest_from_weights(
    y_ret1: np.ndarray,
    y_mask: np.ndarray,
    w: np.ndarray,
    timestamps: np.ndarray,
    cost_per_turnover: float,
    cost_mult: float,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    n = len(y_ret1)
    n_assets = y_ret1.shape[1]
    prev = np.zeros((n_assets,), dtype=np.float32)

    rows = []
    for i in range(n):
        ww = w[i] * y_mask[i]
        turnover = float(np.sum(np.abs(ww - prev)))
        gross = float(np.sum(ww * y_ret1[i]))
        cost = float(cost_per_turnover * cost_mult * turnover)
        net = gross - cost
        rows.append(
            {
                'timestamp_end': pd.Timestamp(timestamps[i]),
                'gross_ret': gross,
                'net_ret': net,
                'turnover': turnover,
                'cost': cost,
                'gross_lev': float(np.sum(np.abs(ww))),
                'active_assets': int(np.sum(y_mask[i] > 0.5)),
            }
        )
        prev = ww

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


def make_walkforward_folds(n: int, min_train: int, val_w: int, test_w: int, step_w: int) -> List[Dict[str, int]]:
    folds = []
    train_end = int(min_train)
    fold_id = 1
    while True:
        tr0, tr1 = 0, int(train_end)
        v0, v1 = tr1, tr1 + int(val_w)
        t0, t1 = v1, v1 + int(test_w)
        if t1 > n:
            break
        folds.append(
            {
                'fold_id': int(fold_id),
                'train_start': int(tr0),
                'train_end': int(tr1),
                'val_start': int(v0),
                'val_end': int(v1),
                'test_start': int(t0),
                'test_end': int(t1),
            }
        )
        train_end += int(step_w)
        fold_id += 1
    return folds


def aggregate_policy_metrics(df: pd.DataFrame) -> Dict[str, float]:
    if len(df) == 0:
        return {'n_steps': 0, 'sharpe_ann': float('nan'), 'cum_net_return': float('nan'), 'max_drawdown': float('nan'), 'avg_turnover': float('nan'), 'avg_gross_lev': float('nan')}
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
    ap = argparse.ArgumentParser(description='Phase17.3 - Walk-forward validation del meta-portafolio (retrain por fold temporal).')
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
    ap.add_argument('--epochs', type=int, default=35)
    ap.add_argument('--patience', type=int, default=7)
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

    ap.add_argument('--cost_per_turnover', type=float, default=5e-5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = ap.parse_args()

    p16 = Path(args.phase16_4_run_dir).resolve()
    if not p16.exists():
        raise FileNotFoundError(p16)

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (p16 / 'phase17_walkforward')
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
        cost_per_turnover=float(args.cost_per_turnover),
        seed=int(args.seed),
        device=str(args.device),
    )

    logger = setup_logger(out_dir / 'phase17_3.log')
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

    pd.DataFrame(folds).to_csv(out_dir / 'phase17_3_fold_ranges.csv', index=False)
    logger.info('Generated folds=%s over windows=%s', len(folds), n)

    device = torch.device('cuda' if (cfg.device == 'auto' and torch.cuda.is_available()) else cfg.device if cfg.device != 'auto' else 'cpu')
    logger.info('Device=%s', device)

    fold_metric_rows = []
    history_rows = []
    steps_all = []

    for f in folds:
        fold_id = int(f['fold_id'])
        i_tr = np.arange(f['train_start'], f['train_end'])
        i_va = np.arange(f['val_start'], f['val_end'])
        i_te = np.arange(f['test_start'], f['test_end'])

        xz, _, _ = standardize_windows(x, idx_train=i_tr)

        ds_tr = TensorDataset(torch.from_numpy(xz[i_tr]), torch.from_numpy(y_ret1[i_tr]), torch.from_numpy(y_mask[i_tr]))
        ds_va = TensorDataset(torch.from_numpy(xz[i_va]), torch.from_numpy(y_ret1[i_va]), torch.from_numpy(y_mask[i_va]))

        dl_tr = DataLoader(ds_tr, batch_size=int(cfg.batch_size), shuffle=False, drop_last=False)
        dl_va = DataLoader(ds_va, batch_size=int(cfg.batch_size), shuffle=False, drop_last=False)

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

            row_hist = {
                'fold_id': fold_id,
                'epoch': ep,
                **{f'train_{k}': v for k, v in tr.items()},
                **{f'val_{k}': v for k, v in va.items()},
            }
            history_rows.append(row_hist)

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

        # Test inference.
        w_model = infer_weights(
            model=model,
            x=xz[i_te],
            y_mask=y_mask[i_te],
            cfg=cfg,
            device=device,
            batch_size=int(cfg.batch_size),
        )
        w_eq_target = baseline_equal_weight(y_mask=y_mask[i_te], max_gross=float(cfg.target_gross_leverage))
        w_eq_max = baseline_equal_weight(y_mask=y_mask[i_te], max_gross=float(cfg.max_gross_leverage))

        ts_te = meta['timestamp_end'].to_numpy()[i_te]

        for mult in [1.0, 2.0, 3.0]:
            for policy, w in [('model', w_model), ('equal_weight_target', w_eq_target), ('equal_weight_max', w_eq_max)]:
                step_df, m = backtest_from_weights(
                    y_ret1=y_ret1[i_te],
                    y_mask=y_mask[i_te],
                    w=w,
                    timestamps=ts_te,
                    cost_per_turnover=float(cfg.cost_per_turnover),
                    cost_mult=float(mult),
                )
                step_df['fold_id'] = fold_id
                step_df['policy'] = policy
                step_df['cost_mult'] = float(mult)
                steps_all.append(step_df)

                fold_metric_rows.append(
                    {
                        'fold_id': fold_id,
                        'policy': policy,
                        'cost_mult': float(mult),
                        **m,
                        'train_n': int(len(i_tr)),
                        'val_n': int(len(i_va)),
                        'test_n': int(len(i_te)),
                        'train_start_ts': str(meta['timestamp_end'].iloc[i_tr[0]]),
                        'train_end_ts': str(meta['timestamp_end'].iloc[i_tr[-1]]),
                        'test_start_ts': str(meta['timestamp_end'].iloc[i_te[0]]),
                        'test_end_ts': str(meta['timestamp_end'].iloc[i_te[-1]]),
                    }
                )

        logger.info(
            'fold=%s done | test %s->%s | best_val=%.6g',
            fold_id,
            meta['timestamp_end'].iloc[i_te[0]],
            meta['timestamp_end'].iloc[i_te[-1]],
            best_val,
        )

    hist_df = pd.DataFrame(history_rows)
    hist_df.to_csv(out_dir / 'phase17_3_train_history.csv', index=False)

    fold_metrics = pd.DataFrame(fold_metric_rows)
    fold_metrics.to_csv(out_dir / 'phase17_3_fold_metrics.csv', index=False)

    steps_df = pd.concat(steps_all, axis=0, ignore_index=True)
    steps_df = steps_df.sort_values(['timestamp_end', 'fold_id', 'policy', 'cost_mult']).reset_index(drop=True)
    steps_df.to_parquet(out_dir / 'phase17_3_test_steps.parquet', index=False)

    # Aggregate pooled test metrics per policy-cost.
    pooled_rows = []
    for policy in ['model', 'equal_weight_target', 'equal_weight_max']:
        for mult in [1.0, 2.0, 3.0]:
            g = steps_df[(steps_df['policy'] == policy) & (np.isclose(steps_df['cost_mult'], float(mult)))].copy()
            m = aggregate_policy_metrics(g)
            pooled_rows.append({'policy': policy, 'cost_mult': float(mult), **m})

    pooled_df = pd.DataFrame(pooled_rows)
    pooled_df.to_csv(out_dir / 'phase17_3_pooled_metrics.csv', index=False)

    # Plot pooled test equity.
    plt.figure(figsize=(11, 5))
    for policy in ['equal_weight_target', 'equal_weight_max', 'model']:
        for mult in [1.0, 2.0, 3.0]:
            g = steps_df[(steps_df['policy'] == policy) & (np.isclose(steps_df['cost_mult'], float(mult)))].copy()
            if len(g) == 0:
                continue
            g = g.sort_values('timestamp_end').reset_index(drop=True)
            g['equity'] = (1.0 + g['net_ret']).cumprod()
            plt.plot(g['timestamp_end'], g['equity'], label=f'{policy}_x{int(mult)}')
    plt.title('Phase17.3 pooled walk-forward test equity')
    plt.xlabel('timestamp')
    plt.ylabel('equity')
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / 'phase17_3_pooled_test_equity.png', dpi=140)
    plt.close()

    # Relative gain table model vs baselines.
    def get_metric(policy: str, mult: float, col: str) -> float:
        s = pooled_df[(pooled_df['policy'] == policy) & (np.isclose(pooled_df['cost_mult'], float(mult)))]
        return float(s.iloc[0][col]) if len(s) else float('nan')

    comparisons = {}
    for mult in [1.0, 2.0, 3.0]:
        comparisons[f'x{int(mult)}'] = {
            'model_sharpe': get_metric('model', mult, 'sharpe_ann'),
            'eq_target_sharpe': get_metric('equal_weight_target', mult, 'sharpe_ann'),
            'eq_max_sharpe': get_metric('equal_weight_max', mult, 'sharpe_ann'),
            'model_cum': get_metric('model', mult, 'cum_net_return'),
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
        'pooled_metrics': pooled_rows,
        'comparisons': comparisons,
        'artifacts': {
            'fold_ranges_csv': str(out_dir / 'phase17_3_fold_ranges.csv'),
            'train_history_csv': str(out_dir / 'phase17_3_train_history.csv'),
            'fold_metrics_csv': str(out_dir / 'phase17_3_fold_metrics.csv'),
            'test_steps_parquet': str(out_dir / 'phase17_3_test_steps.parquet'),
            'pooled_metrics_csv': str(out_dir / 'phase17_3_pooled_metrics.csv'),
            'pooled_test_equity_png': str(out_dir / 'phase17_3_pooled_test_equity.png'),
            'log': str(out_dir / 'phase17_3.log'),
        },
    }

    (out_dir / 'phase17_3_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    logger.info('SUCCESS: %s', out_dir / 'phase17_3_summary.json')


if __name__ == '__main__':
    main()
