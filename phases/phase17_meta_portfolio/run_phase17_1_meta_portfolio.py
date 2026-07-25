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
    d_model: int = 192
    nhead: int = 6
    num_layers: int = 3
    ff_dim: int = 384
    dropout: float = 0.1
    batch_size: int = 128
    epochs: int = 40
    lr: float = 4e-4
    weight_decay: float = 1e-5
    max_gross_leverage: float = 2.0
    weight_temperature: float = 1.0
    loss_w_ret: float = 1.0
    loss_w_risk: float = 0.3
    loss_w_port: float = 0.8
    loss_w_l1: float = 0.02
    cost_per_turnover: float = 5e-5
    seed: int = 42
    device: str = 'auto'


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger('phase17_meta_portfolio')
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
        self.input_dim = int(input_dim)
        self.seq_len = int(seq_len)
        self.n_assets = int(n_assets)
        self.d_model = int(d_model)

        self.in_proj = nn.Linear(self.input_dim, self.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.seq_len, self.d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(nhead),
            dim_feedforward=int(ff_dim),
            dropout=float(dropout),
            batch_first=True,
            norm_first=True,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))
        self.norm = nn.LayerNorm(self.d_model)

        hid = self.d_model * 2
        self.fuse = nn.Sequential(
            nn.Linear(hid, self.d_model),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.LayerNorm(self.d_model),
        )

        self.ret_head = nn.Linear(self.d_model, self.n_assets)
        self.risk_head = nn.Linear(self.d_model, self.n_assets)
        self.weight_head = nn.Linear(self.d_model, self.n_assets)

        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: [B, L, F]
        b, l, _ = x.shape
        if l > self.seq_len:
            raise ValueError(f'L={l} > seq_len={self.seq_len}')
        h = self.in_proj(x) + self.pos_emb[:, :l, :]
        h = self.encoder(h)
        h = self.norm(h)

        z_last = h[:, -1, :]
        z_mean = h.mean(dim=1)
        z = self.fuse(torch.cat([z_last, z_mean], dim=1))

        ret = self.ret_head(z)
        risk_raw = self.risk_head(z)
        score = self.weight_head(z)
        return {
            'ret': ret,
            'risk_raw': risk_raw,
            'score': score,
        }


def build_weights(score: torch.Tensor, mask: torch.Tensor, temperature: float, max_gross: float) -> torch.Tensor:
    s = torch.tanh(score / max(float(temperature), 1e-6))
    s = s * mask
    gross = torch.sum(torch.abs(s), dim=1, keepdim=True)
    scale = torch.where(gross > float(max_gross), float(max_gross) / (gross + 1e-12), torch.ones_like(gross))
    return s * scale


def compute_losses(
    out: Dict[str, torch.Tensor],
    y_ret1: torch.Tensor,
    y_mask: torch.Tensor,
    cfg: Config,
) -> Dict[str, torch.Tensor]:
    ret_pred = out['ret']
    risk_pred = torch.nn.functional.softplus(out['risk_raw'])

    den = torch.clamp(y_mask.sum(), min=1.0)
    loss_ret = torch.sum(torch.nn.functional.smooth_l1_loss(ret_pred, y_ret1, reduction='none') * y_mask) / den
    loss_risk = torch.sum(torch.nn.functional.smooth_l1_loss(risk_pred, torch.abs(y_ret1), reduction='none') * y_mask) / den

    w = build_weights(
        score=out['score'],
        mask=y_mask,
        temperature=float(cfg.weight_temperature),
        max_gross=float(cfg.max_gross_leverage),
    )
    port_ret = torch.sum(w * y_ret1, dim=1)
    loss_port = -torch.mean(port_ret)
    loss_l1 = torch.mean(torch.abs(w))

    loss_total = (
        float(cfg.loss_w_ret) * loss_ret
        + float(cfg.loss_w_risk) * loss_risk
        + float(cfg.loss_w_port) * loss_port
        + float(cfg.loss_w_l1) * loss_l1
    )
    return {
        'loss_total': loss_total,
        'loss_ret': loss_ret,
        'loss_risk': loss_risk,
        'loss_port': loss_port,
        'loss_l1': loss_l1,
        'w': w,
    }


def run_epoch(model: PortfolioTransformer, dl: DataLoader, device: torch.device, cfg: Config, train: bool, optimizer: torch.optim.Optimizer | None = None) -> Dict[str, float]:
    if train:
        model.train()
    else:
        model.eval()

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
            losses = compute_losses(out, y_ret1=yb, y_mask=mb, cfg=cfg)
            loss = losses['loss_total']
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        w = losses['w'].detach()
        yb_np = yb.detach()
        mb_np = mb.detach()
        w_np = w.detach()
        realized = torch.sum(w_np * yb_np, dim=1)
        gross = torch.sum(torch.abs(w_np), dim=1)
        active_assets = torch.sum(mb_np > 0.5, dim=1).float()

        rows.append(
            {
                'loss_total': float(losses['loss_total'].detach().item()),
                'loss_ret': float(losses['loss_ret'].detach().item()),
                'loss_risk': float(losses['loss_risk'].detach().item()),
                'loss_port': float(losses['loss_port'].detach().item()),
                'loss_l1': float(losses['loss_l1'].detach().item()),
                'realized_port_mean': float(realized.mean().item()),
                'gross_mean': float(gross.mean().item()),
                'active_assets_mean': float(active_assets.mean().item()),
            }
        )

    if not rows:
        return {k: float('nan') for k in ['loss_total', 'loss_ret', 'loss_risk', 'loss_port', 'loss_l1', 'realized_port_mean', 'gross_mean', 'active_assets_mean']}

    df = pd.DataFrame(rows)
    return {c: float(df[c].mean()) for c in df.columns}


def infer_outputs(model: PortfolioTransformer, x: np.ndarray, y_mask: np.ndarray, cfg: Config, device: torch.device, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ds = TensorDataset(torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y_mask.astype(np.float32)))
    dl = DataLoader(ds, batch_size=int(batch_size), shuffle=False, drop_last=False)

    model.eval()
    all_w = []
    all_ret = []
    all_risk = []

    with torch.no_grad():
        for xb, mb in dl:
            xb = xb.to(device)
            mb = mb.to(device)
            out = model(xb)
            w = build_weights(out['score'], mb, float(cfg.weight_temperature), float(cfg.max_gross_leverage))
            ret = out['ret']
            risk = torch.nn.functional.softplus(out['risk_raw'])
            all_w.append(w.detach().cpu().numpy())
            all_ret.append(ret.detach().cpu().numpy())
            all_risk.append(risk.detach().cpu().numpy())

    return (
        np.concatenate(all_w, axis=0),
        np.concatenate(all_ret, axis=0),
        np.concatenate(all_risk, axis=0),
    )


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
    split: np.ndarray,
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
                'split': str(split[i]),
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


def main():
    ap = argparse.ArgumentParser(description='Phase17.1 - Meta-modelo de portafolio multi-divisa sobre dataset Phase16.4 (mask-aware, leak-safe).')
    ap.add_argument('--phase16_4_run_dir', required=True, type=str)
    ap.add_argument('--out_dir', default='', type=str)

    ap.add_argument('--d_model', type=int, default=192)
    ap.add_argument('--nhead', type=int, default=6)
    ap.add_argument('--num_layers', type=int, default=3)
    ap.add_argument('--ff_dim', type=int, default=384)
    ap.add_argument('--dropout', type=float, default=0.1)

    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--lr', type=float, default=4e-4)
    ap.add_argument('--weight_decay', type=float, default=1e-5)

    ap.add_argument('--max_gross_leverage', type=float, default=2.0)
    ap.add_argument('--weight_temperature', type=float, default=1.0)

    ap.add_argument('--loss_w_ret', type=float, default=1.0)
    ap.add_argument('--loss_w_risk', type=float, default=0.3)
    ap.add_argument('--loss_w_port', type=float, default=0.8)
    ap.add_argument('--loss_w_l1', type=float, default=0.02)

    ap.add_argument('--cost_per_turnover', type=float, default=5e-5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = ap.parse_args()

    p16 = Path(args.phase16_4_run_dir).resolve()
    if not p16.exists():
        raise FileNotFoundError(p16)

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (p16 / 'phase17_meta_portfolio')
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config(
        phase16_4_run_dir=str(p16),
        out_dir=str(out_dir),
        d_model=int(args.d_model),
        nhead=int(args.nhead),
        num_layers=int(args.num_layers),
        ff_dim=int(args.ff_dim),
        dropout=float(args.dropout),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        max_gross_leverage=float(args.max_gross_leverage),
        weight_temperature=float(args.weight_temperature),
        loss_w_ret=float(args.loss_w_ret),
        loss_w_risk=float(args.loss_w_risk),
        loss_w_port=float(args.loss_w_port),
        loss_w_l1=float(args.loss_w_l1),
        cost_per_turnover=float(args.cost_per_turnover),
        seed=int(args.seed),
        device=str(args.device),
    )

    logger = setup_logger(out_dir / 'phase17_1.log')
    logger.info('Config: %s', cfg)

    set_seed(int(cfg.seed))

    p_npz = p16 / 'phase16_4_windows.npz'
    p_meta = p16 / 'phase16_4_windows_meta.parquet'
    p_summary = p16 / 'phase16_4_summary.json'
    if not p_npz.exists() or not p_meta.exists() or not p_summary.exists():
        raise FileNotFoundError('Missing phase16.4 artifacts (windows.npz/meta.parquet/summary.json).')

    data = np.load(p_npz)
    x = data['x'].astype(np.float32)
    y_ret1 = data['y_ret1'].astype(np.float32)
    y_mask = data['y_mask'].astype(np.float32) if 'y_mask' in data.files else np.ones_like(y_ret1, dtype=np.float32)

    meta = pd.read_parquet(p_meta)
    if len(meta) != len(x):
        raise RuntimeError(f'meta rows ({len(meta)}) != windows ({len(x)})')

    s16 = json.loads(p_summary.read_text(encoding='utf-8'))
    symbols = list(s16.get('symbols', [f'S{i}' for i in range(y_ret1.shape[1])]))
    if len(symbols) != y_ret1.shape[1]:
        symbols = symbols[:y_ret1.shape[1]]
        if len(symbols) < y_ret1.shape[1]:
            symbols += [f'S{i}' for i in range(len(symbols), y_ret1.shape[1])]

    split = meta['split'].astype(str).to_numpy()
    idx_train = np.where(split == 'train')[0]
    idx_val = np.where(split == 'val')[0]
    idx_test = np.where(split == 'test')[0]

    if len(idx_train) < 100 or len(idx_val) < 30 or len(idx_test) < 30:
        raise RuntimeError(f'Not enough windows train/val/test: {len(idx_train)}/{len(idx_val)}/{len(idx_test)}')

    xz, mu, sd = standardize_windows(x, idx_train=idx_train)

    device = torch.device('cuda' if (cfg.device == 'auto' and torch.cuda.is_available()) else cfg.device if cfg.device != 'auto' else 'cpu')
    logger.info('Device=%s | windows=%s train=%s val=%s test=%s', device, len(xz), len(idx_train), len(idx_val), len(idx_test))

    def make_loader(indices: np.ndarray, shuffle: bool) -> DataLoader:
        ds = TensorDataset(
            torch.from_numpy(xz[indices]),
            torch.from_numpy(y_ret1[indices]),
            torch.from_numpy(y_mask[indices]),
        )
        return DataLoader(ds, batch_size=int(cfg.batch_size), shuffle=shuffle, drop_last=False)

    dl_train = make_loader(idx_train, shuffle=True)
    dl_val = make_loader(idx_val, shuffle=False)

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

    hist_rows = []
    best_val = float('inf')
    best_state = None
    patience = 8
    bad = 0

    for ep in range(1, int(cfg.epochs) + 1):
        tr = run_epoch(model, dl_train, device=device, cfg=cfg, train=True, optimizer=opt)
        va = run_epoch(model, dl_val, device=device, cfg=cfg, train=False, optimizer=None)
        row = {
            'epoch': ep,
            **{f'train_{k}': v for k, v in tr.items()},
            **{f'val_{k}': v for k, v in va.items()},
        }
        hist_rows.append(row)

        logger.info(
            'epoch=%s train_loss=%.6f val_loss=%.6f train_port=%.6g val_port=%.6g val_gross=%.4f',
            ep,
            row['train_loss_total'],
            row['val_loss_total'],
            row['train_realized_port_mean'],
            row['val_realized_port_mean'],
            row['val_gross_mean'],
        )

        if row['val_loss_total'] < best_val:
            best_val = row['val_loss_total']
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                logger.info('Early stopping at epoch=%s', ep)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    hist = pd.DataFrame(hist_rows)
    hist.to_csv(out_dir / 'phase17_1_train_history.csv', index=False)

    w_all, ret_pred_all, risk_pred_all = infer_outputs(
        model=model,
        x=xz,
        y_mask=y_mask,
        cfg=cfg,
        device=device,
        batch_size=int(cfg.batch_size),
    )

    pred_df = meta[['timestamp_end', 'split']].copy()
    pred_df['timestamp_end'] = pd.to_datetime(pred_df['timestamp_end'], utc=True)
    for j, s in enumerate(symbols):
        pred_df[f'w_{s}'] = w_all[:, j]
        pred_df[f'y_ret1_{s}'] = y_ret1[:, j]
        pred_df[f'y_mask_{s}'] = y_mask[:, j]
        pred_df[f'ret_pred_{s}'] = ret_pred_all[:, j]
        pred_df[f'risk_pred_{s}'] = risk_pred_all[:, j]

    bt_rows = []
    bt_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}

    eq_weights = baseline_equal_weight(y_mask=y_mask, max_gross=float(cfg.max_gross_leverage))

    for split_name in ['train', 'val', 'test']:
        idx = np.where(split == split_name)[0]
        if len(idx) == 0:
            continue

        bt_metrics[split_name] = {}

        for mult in [1.0, 2.0, 3.0]:
            bt_df, m = backtest_from_weights(
                y_ret1=y_ret1[idx],
                y_mask=y_mask[idx],
                w=w_all[idx],
                timestamps=pred_df['timestamp_end'].to_numpy()[idx],
                split=pred_df['split'].to_numpy()[idx],
                cost_per_turnover=float(cfg.cost_per_turnover),
                cost_mult=float(mult),
            )
            bt_df['policy'] = 'model'
            bt_df['cost_mult'] = float(mult)
            bt_rows.append(bt_df)
            bt_metrics[split_name][f'model_x{int(mult)}'] = m

            btb_df, mb = backtest_from_weights(
                y_ret1=y_ret1[idx],
                y_mask=y_mask[idx],
                w=eq_weights[idx],
                timestamps=pred_df['timestamp_end'].to_numpy()[idx],
                split=pred_df['split'].to_numpy()[idx],
                cost_per_turnover=float(cfg.cost_per_turnover),
                cost_mult=float(mult),
            )
            btb_df['policy'] = 'equal_weight'
            btb_df['cost_mult'] = float(mult)
            bt_rows.append(btb_df)
            bt_metrics[split_name][f'equal_weight_x{int(mult)}'] = mb

    bt_all = pd.concat(bt_rows, axis=0, ignore_index=True)
    bt_all.to_parquet(out_dir / 'phase17_1_backtest_steps.parquet', index=False)

    pred_df.to_parquet(out_dir / 'phase17_1_predictions.parquet', index=False)

    # Visuals on TEST split.
    t = bt_all[(bt_all['split'] == 'test') & (bt_all['policy'].isin(['model', 'equal_weight']))].copy()
    plt.figure(figsize=(10, 5))
    for pol in ['equal_weight', 'model']:
        for mult in [1.0, 2.0, 3.0]:
            g = t[(t['policy'] == pol) & (np.isclose(t['cost_mult'], mult))].sort_values('timestamp_end')
            if len(g) == 0:
                continue
            plt.plot(g['timestamp_end'], g['equity'], label=f'{pol}_x{int(mult)}')
    plt.title('Phase17.1 test equity (net)')
    plt.xlabel('timestamp')
    plt.ylabel('equity')
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / 'phase17_1_test_equity.png', dpi=140)
    plt.close()

    plt.figure(figsize=(10, 5))
    if len(hist):
        plt.plot(hist['epoch'], hist['train_loss_total'], label='train_loss')
        plt.plot(hist['epoch'], hist['val_loss_total'], label='val_loss')
    plt.title('Phase17.1 training loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'phase17_1_loss.png', dpi=140)
    plt.close()

    # Compact metrics table.
    metric_rows = []
    for split_name, d in bt_metrics.items():
        for policy_cost, m in d.items():
            row = {'split': split_name, 'policy_cost': policy_cost}
            row.update(m)
            metric_rows.append(row)
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(out_dir / 'phase17_1_metrics.csv', index=False)

    summary = {
        'status': 'SUCCESS',
        'config': asdict(cfg),
        'data': {
            'n_windows': int(len(xz)),
            'seq_len': int(xz.shape[1]),
            'n_features': int(xz.shape[2]),
            'n_assets': int(y_ret1.shape[1]),
            'symbols': symbols,
            'split_counts': {
                'train': int(len(idx_train)),
                'val': int(len(idx_val)),
                'test': int(len(idx_test)),
            },
            'mask_mean': float(y_mask.mean()),
        },
        'best_val_loss': float(best_val),
        'metrics': bt_metrics,
        'artifacts': {
            'train_history_csv': str(out_dir / 'phase17_1_train_history.csv'),
            'predictions_parquet': str(out_dir / 'phase17_1_predictions.parquet'),
            'backtest_steps_parquet': str(out_dir / 'phase17_1_backtest_steps.parquet'),
            'metrics_csv': str(out_dir / 'phase17_1_metrics.csv'),
            'test_equity_png': str(out_dir / 'phase17_1_test_equity.png'),
            'loss_png': str(out_dir / 'phase17_1_loss.png'),
            'log': str(out_dir / 'phase17_1.log'),
            'model_pt': str(out_dir / 'phase17_1_model.pt'),
        },
    }

    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'config': asdict(cfg),
            'symbols': symbols,
            'input_dim': int(xz.shape[2]),
            'seq_len': int(xz.shape[1]),
            'n_assets': int(y_ret1.shape[1]),
            'scaler_mu': mu,
            'scaler_sd': sd,
        },
        out_dir / 'phase17_1_model.pt',
    )

    (out_dir / 'phase17_1_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    logger.info('SUCCESS: %s', out_dir / 'phase17_1_summary.json')


if __name__ == '__main__':
    main()
