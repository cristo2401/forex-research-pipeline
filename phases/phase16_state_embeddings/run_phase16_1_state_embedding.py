#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import math
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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset


STATE_BASE_COLS: Tuple[str, ...] = (
    'cluster_id',
    'is_informative_cluster',
    'selection_score',
    'pred_inf_prob',
    'pred_progress',
    'pred_sub_conf',
    'pred_sub_entropy',
    'pred_conf_max',
    'p_short',
    'p_flat',
    'p_long',
    'pred_action',
    'hour_sin',
    'hour_cos',
    'dow_sin',
    'dow_cos',
    'prob_entropy_3',
    'p_margin_long_short',
    'p_nonflat',
    'p_top_gap',
    'pred_progress_diff1',
    'pred_inf_prob_diff1',
    'lev_phase12_2_frozen',
    'action_phase12_2_frozen',
    'score_ls',
    'nonflat_score',
    'realized_return_1bar',
    'lev_phase14_hardened',
    'action_phase14_hardened',
    'phase14_trade_event',
    'phase14_turnover_event',
    'phase14_kill_locked',
    'phase14_day_drawdown',
    'phase14_trades_in_hour',
)


@dataclass
class Config:
    phase15_run_dir: str
    out_dir: str
    universe: str = 'selected7'  # selected7 | all10
    window_len: int = 48
    window_stride: int = 1
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    batch_size: int = 256
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-5
    d_model: int = 96
    nhead: int = 4
    num_layers: int = 2
    ff_dim: int = 192
    dropout: float = 0.1
    mask_prob: float = 0.15
    k_regimes: int = 8
    include_symbol_onehot: bool = False
    scaler_mode: str = 'per_symbol'  # per_symbol | global
    cluster_fit_max_per_symbol: int = 3000
    cluster_on_l2_normalized_embeddings: bool = True
    seed: int = 42
    device: str = 'auto'  # auto | cpu | cuda


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger('phase16_state_embedding')
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


class StateTransformerAE(nn.Module):
    def __init__(self, in_dim: int, d_model: int, nhead: int, num_layers: int, ff_dim: int, dropout: float, max_len: int):
        super().__init__()
        self.in_dim = int(in_dim)
        self.d_model = int(d_model)
        self.max_len = int(max_len)

        self.input_proj = nn.Linear(self.in_dim, self.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.max_len, self.d_model))

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
        self.recon_head = nn.Linear(self.d_model, self.in_dim)
        self.norm = nn.LayerNorm(self.d_model)

        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, mask_prob: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, L, F]
        b, l, _ = x.shape
        if l > self.max_len:
            raise ValueError(f'Input length {l} > max_len {self.max_len}')

        if mask_prob > 0:
            mask = (torch.rand((b, l, 1), device=x.device) < float(mask_prob)).float()
            # Keep at least one masked token per sample.
            force_idx = torch.randint(0, l, (b,), device=x.device)
            mask[torch.arange(b, device=x.device), force_idx, 0] = 1.0
            x_in = x * (1.0 - mask)
        else:
            mask = torch.zeros((b, l, 1), device=x.device)
            x_in = x

        h = self.input_proj(x_in) + self.pos_emb[:, :l, :]
        h = self.encoder(h)
        h = self.norm(h)

        recon = self.recon_head(h)
        emb = h.mean(dim=1)
        return recon, emb, mask



def select_symbols(phase15_dir: Path, universe: str) -> pd.DataFrame:
    if universe == 'selected7':
        src = phase15_dir / 'phase15_selected7.csv'
    else:
        src = phase15_dir / 'phase15_symbol_results.csv'
    df = pd.read_csv(src)
    df = df[df['status'] == 'SUCCESS'].copy()
    need = ['symbol', 'phase14_run_dir', 'phase13_run_dir']
    for c in need:
        if c not in df.columns:
            raise RuntimeError(f'Missing column {c} in {src}')
    return df[['symbol', 'phase14_run_dir', 'phase13_run_dir']].drop_duplicates('symbol').reset_index(drop=True)


def load_state_df(phase14_run_dir: Path, symbol: str) -> pd.DataFrame:
    p = phase14_run_dir / 'phase14_policy_strict_predictions.parquet'
    if not p.exists():
        raise FileNotFoundError(f'Missing {p}')
    df = pd.read_parquet(p)
    if 'timestamp' not in df.columns:
        raise RuntimeError(f'{p} missing timestamp')

    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    df = df.dropna(subset=['timestamp']).sort_values('timestamp').drop_duplicates('timestamp', keep='last').reset_index(drop=True)
    df['symbol'] = symbol
    return df


def build_feature_frame(df: pd.DataFrame, symbols_all: List[str], include_symbol_onehot: bool) -> Tuple[pd.DataFrame, List[str]]:
    x = df.copy()

    missing = [c for c in STATE_BASE_COLS if c not in x.columns]
    if missing:
        raise RuntimeError(f"Missing required columns for state_t: {missing}")

    # Keep only causal/current state columns.
    keep = ['symbol', 'timestamp'] + list(STATE_BASE_COLS)
    x = x[keep].copy()

    # Cluster one-hot (kept causal, avoids fake ordinal relation).
    cl = x['cluster_id'].fillna(-1).astype(int)
    cl_oh = pd.get_dummies(cl, prefix='cluster')

    feat = x.drop(columns=['symbol', 'timestamp']).copy()
    parts = [feat, cl_oh]
    if include_symbol_onehot:
        # Optional: this can leak symbol identity into clusters, so default is OFF.
        sym_oh = pd.get_dummies(x['symbol'], prefix='sym')
        for s in symbols_all:
            col = f'sym_{s}'
            if col not in sym_oh.columns:
                sym_oh[col] = 0
        sym_oh = sym_oh[[f'sym_{s}' for s in symbols_all]]
        parts.append(sym_oh)
    feat = pd.concat(parts, axis=1)

    for c in feat.columns:
        feat[c] = pd.to_numeric(feat[c], errors='coerce')
    feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    out = pd.concat([x[['symbol', 'timestamp']], feat], axis=1)
    feature_cols = [c for c in out.columns if c not in ('symbol', 'timestamp')]
    return out, feature_cols


def compute_scalers_from_train_rows(
    df_feat: pd.DataFrame,
    feature_cols: List[str],
    train_mask: np.ndarray,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    if mode not in ('global', 'per_symbol'):
        raise ValueError(f'Invalid scaler mode={mode}')

    # Always keep global stats for diagnostics/model card.
    arr_all = df_feat.loc[train_mask, feature_cols].to_numpy(dtype=np.float32)
    mu_global = np.mean(arr_all, axis=0)
    sd_global = np.std(arr_all, axis=0)
    sd_global = np.where(sd_global < 1e-6, 1.0, sd_global)

    mu_by_symbol: Dict[str, np.ndarray] = {}
    sd_by_symbol: Dict[str, np.ndarray] = {}

    if mode == 'per_symbol':
        for symbol, g in df_feat.groupby('symbol', sort=True):
            idx = g.index.to_numpy()
            idx_train = idx[train_mask[idx]]
            if len(idx_train) == 0:
                arr = g[feature_cols].to_numpy(dtype=np.float32)
            else:
                arr = df_feat.loc[idx_train, feature_cols].to_numpy(dtype=np.float32)
            mu = np.mean(arr, axis=0)
            sd = np.std(arr, axis=0)
            sd = np.where(sd < 1e-6, 1.0, sd)
            mu_by_symbol[str(symbol)] = mu.astype(np.float32)
            sd_by_symbol[str(symbol)] = sd.astype(np.float32)

    return mu_global.astype(np.float32), sd_global.astype(np.float32), mu_by_symbol, sd_by_symbol


def apply_scaler(
    df_feat: pd.DataFrame,
    feature_cols: List[str],
    mode: str,
    mu_global: np.ndarray,
    sd_global: np.ndarray,
    mu_by_symbol: Dict[str, np.ndarray],
    sd_by_symbol: Dict[str, np.ndarray],
) -> pd.DataFrame:
    out = df_feat.copy()
    out[feature_cols] = out[feature_cols].astype(np.float32)
    if mode == 'global':
        out[feature_cols] = (out[feature_cols].to_numpy(dtype=np.float32) - mu_global) / sd_global
        return out

    for symbol, g in out.groupby('symbol', sort=True):
        idx = g.index
        mu = mu_by_symbol.get(str(symbol), mu_global)
        sd = sd_by_symbol.get(str(symbol), sd_global)
        arr = g[feature_cols].to_numpy(dtype=np.float32)
        out.loc[idx, feature_cols] = (arr - mu) / sd
    return out


def split_bounds(n: int, train_ratio: float, val_ratio: float) -> Tuple[int, int]:
    tr = int(max(1, min(n - 2, math.floor(n * train_ratio))))
    va = int(max(tr + 1, min(n - 1, math.floor(n * (train_ratio + val_ratio)))))
    return tr, va


def build_windows(
    df_feat: pd.DataFrame,
    feature_cols: List[str],
    window_len: int,
    stride: int,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, Dict[str, int]], np.ndarray, np.ndarray]:
    arrays: List[np.ndarray] = []
    metas: List[dict] = []
    split_stats: Dict[str, Dict[str, int]] = {}

    # Row-level train mask for scaler fit (strict temporal by symbol).
    train_mask = np.zeros(len(df_feat), dtype=bool)

    for symbol, g0 in df_feat.groupby('symbol', sort=True):
        g = g0.sort_values('timestamp').copy()
        n = len(g)
        if n <= window_len:
            continue

        tr_end, va_end = split_bounds(n, train_ratio, val_ratio)
        idx_sorted = g.index.to_numpy()
        train_mask[idx_sorted[:tr_end]] = True

        x = g[feature_cols].to_numpy(dtype=np.float32)
        ts = pd.to_datetime(g['timestamp'], utc=True)

        cnt = {'train': 0, 'val': 0, 'test': 0}
        for end in range(window_len - 1, n, max(1, stride)):
            start = end - window_len + 1
            if end < tr_end:
                split = 'train'
            elif end < va_end:
                split = 'val'
            else:
                split = 'test'

            arrays.append(x[start:end + 1])
            metas.append({
                'symbol': symbol,
                'split': split,
                'idx_start': int(start),
                'idx_end': int(end),
                'timestamp_start': ts.iloc[start],
                'timestamp_end': ts.iloc[end],
            })
            cnt[split] += 1

        split_stats[symbol] = cnt

    if not arrays:
        raise RuntimeError('No windows were generated. Check window_len/stride/data size.')

    x_all = np.stack(arrays).astype(np.float32)
    meta_df = pd.DataFrame(metas)
    return x_all, meta_df, split_stats, train_mask, np.arange(len(df_feat))


def select_balanced_kmeans_fit_indices(
    meta_df: pd.DataFrame,
    idx_train: np.ndarray,
    seed: int,
    max_per_symbol: int,
) -> np.ndarray:
    train_meta = meta_df.iloc[idx_train].copy()
    counts = train_meta['symbol'].value_counts()
    if len(counts) <= 1:
        return idx_train
    n_per_symbol = int(min(int(counts.min()), int(max_per_symbol)))
    if n_per_symbol <= 0:
        return idx_train

    rng = np.random.default_rng(int(seed))
    selected = []
    for symbol, g in train_meta.groupby('symbol', sort=True):
        choices = g.index.to_numpy()
        take = min(n_per_symbol, len(choices))
        pick = rng.choice(choices, size=take, replace=False)
        selected.extend(pick.tolist())
    selected = np.array(sorted(selected), dtype=np.int64)
    return selected


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n


def compute_regime_symbol_purity(meta_df: pd.DataFrame, regime_id: np.ndarray) -> Tuple[float, Dict[int, float]]:
    d = meta_df[['symbol']].copy()
    d['regime_id'] = regime_id.astype(int)
    c = d.groupby(['regime_id', 'symbol']).size().rename('n').reset_index()
    total = c.groupby('regime_id')['n'].sum().rename('tot').reset_index()
    c = c.merge(total, on='regime_id', how='left')
    c['share'] = c['n'] / c['tot']
    best = c.groupby('regime_id')['share'].max()
    weights = total.set_index('regime_id')['tot']
    purity = float((best * weights.loc[best.index]).sum() / weights.loc[best.index].sum())
    per_regime = {int(k): float(v) for k, v in best.to_dict().items()}
    return purity, per_regime


def train_autoencoder(
    model: StateTransformerAE,
    x_train: np.ndarray,
    x_val: np.ndarray,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    mask_prob: float,
    device: torch.device,
    logger: logging.Logger,
) -> pd.DataFrame:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    ds_train = TensorDataset(torch.from_numpy(x_train))
    ds_val = TensorDataset(torch.from_numpy(x_val))

    dl_train = DataLoader(ds_train, batch_size=int(batch_size), shuffle=True, drop_last=False)
    dl_val = DataLoader(ds_val, batch_size=int(batch_size), shuffle=False, drop_last=False)

    hist = []
    best_state = None
    best_val = float('inf')
    patience = 5
    bad = 0

    for ep in range(1, int(epochs) + 1):
        model.train()
        t_losses = []
        for (xb,) in dl_train:
            xb = xb.to(device)
            rec, _, mask = model(xb, mask_prob=float(mask_prob))
            if float(mask_prob) > 0:
                denom = torch.clamp(mask.sum(), min=1.0)
                loss = (((rec - xb) ** 2) * mask).sum() / denom
            else:
                loss = torch.mean((rec - xb) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            t_losses.append(float(loss.item()))

        model.eval()
        v_losses = []
        with torch.no_grad():
            for (xb,) in dl_val:
                xb = xb.to(device)
                rec, _, _ = model(xb, mask_prob=0.0)
                loss = torch.mean((rec - xb) ** 2)
                v_losses.append(float(loss.item()))

        tr = float(np.mean(t_losses)) if t_losses else float('nan')
        va = float(np.mean(v_losses)) if v_losses else float('nan')
        hist.append({'epoch': ep, 'train_loss': tr, 'val_loss': va})
        logger.info('epoch=%s train_loss=%.6f val_loss=%.6f', ep, tr, va)

        if va < best_val:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                logger.info('early stopping at epoch=%s (patience=%s)', ep, patience)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return pd.DataFrame(hist)


def infer_embeddings(model: StateTransformerAE, x_all: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    ds = TensorDataset(torch.from_numpy(x_all))
    dl = DataLoader(ds, batch_size=int(batch_size), shuffle=False, drop_last=False)
    model.eval()
    out = []
    with torch.no_grad():
        for (xb,) in dl:
            xb = xb.to(device)
            _, emb, _ = model(xb, mask_prob=0.0)
            out.append(emb.detach().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description='Phase16 - Estado derivado -> ventanas -> embeddings (Transformer auto-supervisado, leak-safe).')
    ap.add_argument('--phase15_run_dir', required=True, type=str)
    ap.add_argument('--out_dir', required=False, type=str, default='')
    ap.add_argument('--universe', choices=['selected7', 'all10'], default='selected7')
    ap.add_argument('--window_len', type=int, default=48)
    ap.add_argument('--window_stride', type=int, default=1)
    ap.add_argument('--train_ratio', type=float, default=0.70)
    ap.add_argument('--val_ratio', type=float, default=0.15)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=1e-5)
    ap.add_argument('--d_model', type=int, default=96)
    ap.add_argument('--nhead', type=int, default=4)
    ap.add_argument('--num_layers', type=int, default=2)
    ap.add_argument('--ff_dim', type=int, default=192)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--mask_prob', type=float, default=0.15)
    ap.add_argument('--k_regimes', type=int, default=8)
    ap.add_argument('--include_symbol_onehot', type=int, choices=[0, 1], default=0)
    ap.add_argument('--scaler_mode', type=str, choices=['global', 'per_symbol'], default='per_symbol')
    ap.add_argument('--cluster_fit_max_per_symbol', type=int, default=3000)
    ap.add_argument('--cluster_on_l2_normalized_embeddings', type=int, choices=[0, 1], default=1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda'], default='auto')
    args = ap.parse_args()

    phase15_dir = Path(args.phase15_run_dir).resolve()
    if not phase15_dir.exists():
        raise FileNotFoundError(phase15_dir)

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (phase15_dir / 'phase16_state_embeddings')
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(out_dir / 'phase16.log')
    cfg = Config(
        phase15_run_dir=str(phase15_dir),
        out_dir=str(out_dir),
        universe=args.universe,
        window_len=int(args.window_len),
        window_stride=int(args.window_stride),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        d_model=int(args.d_model),
        nhead=int(args.nhead),
        num_layers=int(args.num_layers),
        ff_dim=int(args.ff_dim),
        dropout=float(args.dropout),
        mask_prob=float(args.mask_prob),
        k_regimes=int(args.k_regimes),
        include_symbol_onehot=bool(int(args.include_symbol_onehot)),
        scaler_mode=str(args.scaler_mode),
        cluster_fit_max_per_symbol=int(args.cluster_fit_max_per_symbol),
        cluster_on_l2_normalized_embeddings=bool(int(args.cluster_on_l2_normalized_embeddings)),
        seed=int(args.seed),
        device=args.device,
    )

    set_seed(cfg.seed)
    device = torch.device('cuda' if (cfg.device == 'auto' and torch.cuda.is_available()) else cfg.device if cfg.device != 'auto' else 'cpu')
    logger.info('device=%s', device)

    sym_df = select_symbols(phase15_dir, cfg.universe)
    symbols = sorted(sym_df['symbol'].tolist())
    logger.info('universe=%s n_symbols=%s symbols=%s', cfg.universe, len(symbols), symbols)

    state_frames = []
    for _, row in sym_df.iterrows():
        symbol = str(row['symbol'])
        p14 = Path(row['phase14_run_dir']).resolve()
        g = load_state_df(p14, symbol)
        state_frames.append(g)
        logger.info('loaded symbol=%s rows=%s from=%s', symbol, len(g), p14)

    state_all = pd.concat(state_frames, axis=0, ignore_index=True)
    feat_df, feature_cols = build_feature_frame(
        state_all,
        symbols_all=symbols,
        include_symbol_onehot=cfg.include_symbol_onehot,
    )

    # Build temporal windows first to preserve strictly causal split by symbol.
    _, meta_df, split_stats, train_mask_rows, _ = build_windows(
        feat_df,
        feature_cols,
        window_len=cfg.window_len,
        stride=cfg.window_stride,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
    )

    # Row-level scaler fit using only training rows (global or per-symbol).
    mu_global, sd_global, mu_by_symbol, sd_by_symbol = compute_scalers_from_train_rows(
        feat_df,
        feature_cols,
        train_mask_rows,
        mode=cfg.scaler_mode,
    )
    feat_std = apply_scaler(
        feat_df,
        feature_cols,
        mode=cfg.scaler_mode,
        mu_global=mu_global,
        sd_global=sd_global,
        mu_by_symbol=mu_by_symbol,
        sd_by_symbol=sd_by_symbol,
    )

    # Rebuild windows on standardized features (same window indexing logic).
    x_std, meta_df_std, split_stats_std, _, _ = build_windows(
        feat_std,
        feature_cols,
        window_len=cfg.window_len,
        stride=cfg.window_stride,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
    )
    # Sanity: ensure same metadata order.
    if len(meta_df) != len(meta_df_std) or not (meta_df['symbol'].values == meta_df_std['symbol'].values).all():
        raise RuntimeError('Window metadata mismatch after standardization rebuild.')

    split = meta_df['split'].to_numpy()
    idx_train = np.where(split == 'train')[0]
    idx_val = np.where(split == 'val')[0]
    idx_test = np.where(split == 'test')[0]

    if len(idx_train) == 0 or len(idx_val) == 0:
        raise RuntimeError('Insufficient windows for train/val. Reduce window_len or adjust splits.')

    x_train = x_std[idx_train]
    x_val = x_std[idx_val]

    logger.info('windows total=%s train=%s val=%s test=%s', len(x_std), len(idx_train), len(idx_val), len(idx_test))

    model = StateTransformerAE(
        in_dim=len(feature_cols),
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        ff_dim=cfg.ff_dim,
        dropout=cfg.dropout,
        max_len=cfg.window_len,
    )

    hist_df = train_autoencoder(
        model=model,
        x_train=x_train,
        x_val=x_val,
        batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        mask_prob=cfg.mask_prob,
        device=device,
        logger=logger,
    )

    emb = infer_embeddings(model, x_std, batch_size=cfg.batch_size, device=device)
    emb_dim = emb.shape[1]

    # Clustering over train embeddings only; use balanced sample by symbol to reduce collapse.
    emb_for_cluster = l2_normalize_rows(emb) if cfg.cluster_on_l2_normalized_embeddings else emb
    idx_fit = select_balanced_kmeans_fit_indices(
        meta_df=meta_df,
        idx_train=idx_train,
        seed=cfg.seed,
        max_per_symbol=cfg.cluster_fit_max_per_symbol,
    )
    kmeans = KMeans(n_clusters=int(cfg.k_regimes), random_state=int(cfg.seed), n_init='auto')
    kmeans.fit(emb_for_cluster[idx_fit])
    regime_id = kmeans.predict(emb_for_cluster)

    regime_symbol_purity, regime_symbol_purity_by_regime = compute_regime_symbol_purity(meta_df=meta_df, regime_id=regime_id)

    emb_cols = [f'emb_{i:03d}' for i in range(emb_dim)]
    emb_df = meta_df.copy()
    emb_df['regime_id'] = regime_id.astype(int)
    for i, c in enumerate(emb_cols):
        emb_df[c] = emb[:, i]

    # PCA preview.
    pca = PCA(n_components=2, random_state=cfg.seed)
    p2 = pca.fit_transform(emb[idx_train])
    vis = emb_df.iloc[idx_train].copy()
    vis['pc1'] = p2[:, 0]
    vis['pc2'] = p2[:, 1]

    plt.figure(figsize=(10, 5))
    for rid in sorted(vis['regime_id'].unique()):
        g = vis[vis['regime_id'] == rid]
        plt.scatter(g['pc1'], g['pc2'], s=8, alpha=0.6, label=f'r{rid}')
    plt.title('Phase16 train embeddings PCA (by regime_id)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(loc='best', fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / 'phase16_embeddings_pca_train.png', dpi=140)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(hist_df['epoch'], hist_df['train_loss'], label='train')
    plt.plot(hist_df['epoch'], hist_df['val_loss'], label='val')
    plt.title('Phase16 Transformer AE loss')
    plt.xlabel('epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'phase16_training_loss.png', dpi=140)
    plt.close()

    # Save artifacts.
    feat_df.to_parquet(out_dir / 'phase16_state_rows_raw.parquet', index=False)
    feat_std.to_parquet(out_dir / 'phase16_state_rows_std.parquet', index=False)
    meta_df.to_parquet(out_dir / 'phase16_windows_meta.parquet', index=False)
    emb_df.to_parquet(out_dir / 'phase16_embeddings.parquet', index=False)
    hist_df.to_csv(out_dir / 'phase16_training_history.csv', index=False)

    np.savez_compressed(
        out_dir / 'phase16_embeddings.npz',
        embeddings=emb.astype(np.float32),
        split=split.astype('S8'),
        regime_id=regime_id.astype(np.int32),
    )

    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'feature_cols': feature_cols,
            'symbols': symbols,
            'window_len': cfg.window_len,
            'mean_global': mu_global,
            'std_global': sd_global,
            'mean_by_symbol': mu_by_symbol,
            'std_by_symbol': sd_by_symbol,
            'scaler_mode': cfg.scaler_mode,
            'kmeans_centers': kmeans.cluster_centers_,
            'kmeans_n_clusters': int(cfg.k_regimes),
            'config': asdict(cfg),
        },
        out_dir / 'phase16_embedding_model.pt',
    )

    split_count = emb_df.groupby(['symbol', 'split']).size().reset_index(name='n_windows')
    split_count.to_csv(out_dir / 'phase16_windows_split_count.csv', index=False)

    summary = {
        'status': 'SUCCESS',
        'phase15_run_dir': str(phase15_dir),
        'out_dir': str(out_dir),
        'universe': cfg.universe,
        'symbols': symbols,
        'n_symbols': len(symbols),
        'n_rows_state': int(len(feat_df)),
        'n_features': int(len(feature_cols)),
        'n_windows_total': int(len(meta_df)),
        'n_windows_train': int(len(idx_train)),
        'n_windows_val': int(len(idx_val)),
        'n_windows_test': int(len(idx_test)),
        'window_len': int(cfg.window_len),
        'window_stride': int(cfg.window_stride),
        'embedding_dim': int(emb_dim),
        'k_regimes': int(cfg.k_regimes),
        'regime_symbol_purity': float(regime_symbol_purity),
        'regime_symbol_purity_by_regime': regime_symbol_purity_by_regime,
        'kmeans_fit_n': int(len(idx_fit)),
        'split_stats_by_symbol': split_stats_std,
        'final_train_loss': float(hist_df['train_loss'].iloc[-1]),
        'final_val_loss': float(hist_df['val_loss'].iloc[-1]),
        'best_val_loss': float(hist_df['val_loss'].min()),
        'device': str(device),
        'anti_leak': {
            'scaler_fit_only_train_rows': True,
            'temporal_split_by_symbol': True,
            'future_columns_used': False,
        },
        'domain_generalization': {
            'include_symbol_onehot': bool(cfg.include_symbol_onehot),
            'scaler_mode': cfg.scaler_mode,
            'cluster_fit_balanced_by_symbol': True,
            'cluster_fit_max_per_symbol': int(cfg.cluster_fit_max_per_symbol),
            'cluster_on_l2_normalized_embeddings': bool(cfg.cluster_on_l2_normalized_embeddings),
        },
        'artifacts': {
            'state_rows_raw_parquet': str(out_dir / 'phase16_state_rows_raw.parquet'),
            'state_rows_std_parquet': str(out_dir / 'phase16_state_rows_std.parquet'),
            'windows_meta_parquet': str(out_dir / 'phase16_windows_meta.parquet'),
            'embeddings_parquet': str(out_dir / 'phase16_embeddings.parquet'),
            'embeddings_npz': str(out_dir / 'phase16_embeddings.npz'),
            'model_pt': str(out_dir / 'phase16_embedding_model.pt'),
            'training_history_csv': str(out_dir / 'phase16_training_history.csv'),
            'windows_split_count_csv': str(out_dir / 'phase16_windows_split_count.csv'),
            'pca_png': str(out_dir / 'phase16_embeddings_pca_train.png'),
            'loss_png': str(out_dir / 'phase16_training_loss.png'),
            'log': str(out_dir / 'phase16.log'),
        },
    }
    (out_dir / 'phase16_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    logger.info('SUCCESS: %s', out_dir / 'phase16_summary.json')


if __name__ == '__main__':
    main()
