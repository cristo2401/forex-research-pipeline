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
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class Config:
    phase16_2_run_dir: str
    out_dir: str
    seq_len: int = 32
    d_model: int = 96
    nhead: int = 4
    num_layers: int = 2
    ff_dim: int = 192
    dropout: float = 0.1
    batch_size: int = 512
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-5
    k_context: int = 6
    seed: int = 42
    device: str = 'auto'


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger('phase16_context_transformer')
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


class ContextTokenTransformer(nn.Module):
    def __init__(self, vocab_size: int, seq_len: int, d_model: int, nhead: int, num_layers: int, ff_dim: int, dropout: float):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.seq_len = int(seq_len)
        self.d_model = int(d_model)

        self.tok_emb = nn.Embedding(self.vocab_size, self.d_model)
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
        self.head = nn.Linear(self.d_model, self.vocab_size)

        causal = torch.triu(torch.ones(self.seq_len, self.seq_len), diagonal=1).bool()
        self.register_buffer('causal_mask', causal, persistent=False)
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # tokens: [B, L]
        b, l = tokens.shape
        if l > self.seq_len:
            raise ValueError(f'Input L={l} > seq_len={self.seq_len}')

        x = self.tok_emb(tokens) + self.pos_emb[:, :l, :]
        h = self.encoder(x, mask=self.causal_mask[:l, :l])
        h = self.norm(h)
        ctx = h[:, -1, :]  # contextual embedding at prediction time
        logits = self.head(ctx)
        return logits, ctx


def build_sequences(df: pd.DataFrame, seq_len: int) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rows_tokens: List[np.ndarray] = []
    rows_targets: List[int] = []
    rows_meta: List[dict] = []

    for (symbol, split), g in df.groupby(['symbol', 'split'], sort=True):
        g = g.sort_values('timestamp_end').reset_index(drop=True)
        tok = g['cluster_id'].to_numpy(dtype=np.int64)
        ts = pd.to_datetime(g['timestamp_end'], utc=True)
        n = len(g)
        if n <= seq_len:
            continue

        for i in range(seq_len, n):
            start = i - seq_len
            end = i
            rows_tokens.append(tok[start:end])
            rows_targets.append(int(tok[i]))
            rows_meta.append({
                'symbol': str(symbol),
                'split': str(split),
                'timestamp_end': ts.iloc[i],
                'source_idx_start': int(start),
                'source_idx_end': int(i),
                'target_cluster': int(tok[i]),
            })

    if not rows_tokens:
        raise RuntimeError('No sequences generated. Reduce seq_len or check data.')

    x = np.stack(rows_tokens).astype(np.int64)
    y = np.array(rows_targets, dtype=np.int64)
    meta = pd.DataFrame(rows_meta)
    return x, y, meta


def eval_epoch(model: ContextTokenTransformer, dl: DataLoader, device: torch.device, criterion: nn.Module) -> Dict[str, float]:
    model.eval()
    losses = []
    accs = []
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            pred = torch.argmax(logits, dim=1)
            acc = (pred == yb).float().mean()
            losses.append(float(loss.item()))
            accs.append(float(acc.item()))
    return {
        'loss': float(np.mean(losses)) if losses else float('nan'),
        'acc': float(np.mean(accs)) if accs else float('nan'),
    }


def train_model(
    model: ContextTokenTransformer,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: Config,
    device: torch.device,
    logger: logging.Logger,
) -> pd.DataFrame:
    ds_train = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    ds_val = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))

    dl_train = DataLoader(ds_train, batch_size=int(cfg.batch_size), shuffle=True, drop_last=False)
    dl_val = DataLoader(ds_val, batch_size=int(cfg.batch_size), shuffle=False, drop_last=False)

    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    criterion = nn.CrossEntropyLoss()

    hist = []
    best_val = float('inf')
    best_state = None
    patience = 4
    bad = 0

    for ep in range(1, int(cfg.epochs) + 1):
        model.train()
        tr_losses = []
        tr_accs = []

        for xb, yb in dl_train:
            xb = xb.to(device)
            yb = yb.to(device)
            logits, _ = model(xb)
            loss = criterion(logits, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            pred = torch.argmax(logits, dim=1)
            acc = (pred == yb).float().mean()
            tr_losses.append(float(loss.item()))
            tr_accs.append(float(acc.item()))

        val_stats = eval_epoch(model, dl_val, device, criterion)
        row = {
            'epoch': ep,
            'train_loss': float(np.mean(tr_losses)) if tr_losses else float('nan'),
            'train_acc': float(np.mean(tr_accs)) if tr_accs else float('nan'),
            'val_loss': float(val_stats['loss']),
            'val_acc': float(val_stats['acc']),
        }
        hist.append(row)
        logger.info(
            'epoch=%s train_loss=%.6f train_acc=%.4f val_loss=%.6f val_acc=%.4f',
            ep, row['train_loss'], row['train_acc'], row['val_loss'], row['val_acc'],
        )

        if row['val_loss'] < best_val:
            best_val = row['val_loss']
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                logger.info('early stopping at epoch=%s', ep)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return pd.DataFrame(hist)


def infer_context(model: ContextTokenTransformer, x: np.ndarray, batch_size: int, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ds = TensorDataset(torch.from_numpy(x))
    dl = DataLoader(ds, batch_size=int(batch_size), shuffle=False, drop_last=False)

    model.eval()
    all_ctx = []
    all_pred = []
    all_conf = []

    with torch.no_grad():
        for (xb,) in dl:
            xb = xb.to(device)
            logits, ctx = model(xb)
            prob = torch.softmax(logits, dim=1)
            conf, pred = torch.max(prob, dim=1)
            all_ctx.append(ctx.detach().cpu().numpy())
            all_pred.append(pred.detach().cpu().numpy())
            all_conf.append(conf.detach().cpu().numpy())

    return (
        np.concatenate(all_ctx).astype(np.float32),
        np.concatenate(all_pred).astype(np.int64),
        np.concatenate(all_conf).astype(np.float32),
    )


def weighted_purity(df: pd.DataFrame, cluster_col: str, symbol_col: str = 'symbol') -> float:
    c = df.groupby([cluster_col, symbol_col]).size().rename('n').reset_index()
    t = c.groupby(cluster_col)['n'].sum().rename('tot').reset_index()
    c = c.merge(t, on=cluster_col, how='left')
    c['share'] = c['n'] / c['tot']
    mx = c.groupby(cluster_col)['share'].max()
    w = t.set_index(cluster_col)['tot']
    return float((mx * w.loc[mx.index]).sum() / w.loc[mx.index].sum())


def main():
    ap = argparse.ArgumentParser(description='Phase16.3 - Transformer contextual sobre secuencia de clusters (next-token + context embeddings).')
    ap.add_argument('--phase16_2_run_dir', required=True, type=str)
    ap.add_argument('--out_dir', default='', type=str)
    ap.add_argument('--seq_len', type=int, default=32)
    ap.add_argument('--d_model', type=int, default=96)
    ap.add_argument('--nhead', type=int, default=4)
    ap.add_argument('--num_layers', type=int, default=2)
    ap.add_argument('--ff_dim', type=int, default=192)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--batch_size', type=int, default=512)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=1e-5)
    ap.add_argument('--k_context', type=int, default=6)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = ap.parse_args()

    cfg = Config(
        phase16_2_run_dir=str(Path(args.phase16_2_run_dir).resolve()),
        out_dir=str(Path(args.out_dir).resolve()) if args.out_dir else '',
        seq_len=int(args.seq_len),
        d_model=int(args.d_model),
        nhead=int(args.nhead),
        num_layers=int(args.num_layers),
        ff_dim=int(args.ff_dim),
        dropout=float(args.dropout),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        k_context=int(args.k_context),
        seed=int(args.seed),
        device=str(args.device),
    )

    set_seed(cfg.seed)

    phase16_2_dir = Path(cfg.phase16_2_run_dir)
    if not phase16_2_dir.exists():
        raise FileNotFoundError(phase16_2_dir)

    out_dir = Path(cfg.out_dir) if cfg.out_dir else (phase16_2_dir / 'phase16_3_context_transformer')
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(out_dir / 'phase16_3.log')

    inp = phase16_2_dir / 'phase16_2_cluster_assignments.parquet'
    if not inp.exists():
        raise FileNotFoundError(inp)

    d = pd.read_parquet(inp)
    required = ['symbol', 'split', 'timestamp_end', 'cluster_id']
    miss = [c for c in required if c not in d.columns]
    if miss:
        raise RuntimeError(f'Missing required columns: {miss}')

    d['timestamp_end'] = pd.to_datetime(d['timestamp_end'], utc=True, errors='coerce')
    d = d.dropna(subset=['timestamp_end']).copy()
    d = d.sort_values(['symbol', 'split', 'timestamp_end']).reset_index(drop=True)

    vocab_size = int(d['cluster_id'].max()) + 1
    logger.info('vocab_size=%s rows=%s symbols=%s', vocab_size, len(d), sorted(d['symbol'].unique().tolist()))

    x, y, meta = build_sequences(d, seq_len=cfg.seq_len)
    split = meta['split'].astype(str).to_numpy()
    idx_train = np.where(split == 'train')[0]
    idx_val = np.where(split == 'val')[0]
    idx_test = np.where(split == 'test')[0]

    if len(idx_train) < 100:
        raise RuntimeError('Too few train sequences')

    x_train, y_train = x[idx_train], y[idx_train]
    x_val, y_val = x[idx_val], y[idx_val]
    x_test, y_test = x[idx_test], y[idx_test]

    device = torch.device('cuda' if (cfg.device == 'auto' and torch.cuda.is_available()) else cfg.device if cfg.device != 'auto' else 'cpu')
    logger.info('device=%s | sequences total=%s train=%s val=%s test=%s', device, len(x), len(x_train), len(x_val), len(x_test))

    model = ContextTokenTransformer(
        vocab_size=vocab_size,
        seq_len=cfg.seq_len,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        ff_dim=cfg.ff_dim,
        dropout=cfg.dropout,
    )

    hist = train_model(model, x_train, y_train, x_val, y_val, cfg=cfg, device=device, logger=logger)

    # Final eval on all splits.
    crit = nn.CrossEntropyLoss()
    dl_train = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)), batch_size=cfg.batch_size)
    dl_val = DataLoader(TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)), batch_size=cfg.batch_size)
    dl_test = DataLoader(TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test)), batch_size=cfg.batch_size)
    m_train = eval_epoch(model, dl_train, device=device, criterion=crit)
    m_val = eval_epoch(model, dl_val, device=device, criterion=crit)
    m_test = eval_epoch(model, dl_test, device=device, criterion=crit)

    # Inference contextual embeddings for every sequence.
    ctx, pred_next, pred_conf = infer_context(model, x=x, batch_size=cfg.batch_size, device=device)
    ctx_dim = int(ctx.shape[1])

    ctx_df = meta.copy()
    ctx_df['target_cluster_true'] = y.astype(np.int64)
    ctx_df['target_cluster_pred'] = pred_next.astype(np.int64)
    ctx_df['pred_conf'] = pred_conf.astype(np.float32)
    ctx_df['next_token_hit'] = (ctx_df['target_cluster_true'] == ctx_df['target_cluster_pred']).astype(np.int8)
    ctx_cols = pd.DataFrame(
        ctx,
        columns=[f'ctx_{j:03d}' for j in range(ctx_dim)],
        index=ctx_df.index,
    )
    ctx_df = pd.concat([ctx_df, ctx_cols], axis=1)

    # Context regimes on train contextual embeddings only.
    kctx = int(min(max(2, cfg.k_context), len(np.unique(ctx_df.loc[idx_train, 'target_cluster_true'])) * 4))
    km = KMeans(n_clusters=kctx, random_state=cfg.seed, n_init='auto')
    km.fit(ctx[idx_train])
    ctx_regime = km.predict(ctx)
    ctx_df['context_regime_id'] = ctx_regime.astype(np.int32)

    purity_context_symbol = weighted_purity(ctx_df[['symbol', 'context_regime_id']], cluster_col='context_regime_id')

    # Save artifacts.
    hist.to_csv(out_dir / 'phase16_3_train_history.csv', index=False)
    ctx_df.to_parquet(out_dir / 'phase16_3_context_embeddings.parquet', index=False)

    # Quick visualizations.
    plt.figure(figsize=(8, 4))
    plt.plot(hist['epoch'], hist['train_loss'], label='train_loss')
    plt.plot(hist['epoch'], hist['val_loss'], label='val_loss')
    plt.title('Phase16.3 next-token loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'phase16_3_loss.png', dpi=140)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(hist['epoch'], hist['train_acc'], label='train_acc')
    plt.plot(hist['epoch'], hist['val_acc'], label='val_acc')
    plt.title('Phase16.3 next-token accuracy')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'phase16_3_acc.png', dpi=140)
    plt.close()

    split_acc = ctx_df.groupby('split')['next_token_hit'].mean().rename('acc').reset_index()
    split_acc.to_csv(out_dir / 'phase16_3_next_token_acc_by_split.csv', index=False)

    mix = ctx_df.groupby(['context_regime_id', 'symbol']).size().rename('n').reset_index()
    mix['regime_total'] = mix.groupby('context_regime_id')['n'].transform('sum')
    mix['symbol_share_in_regime'] = mix['n'] / mix['regime_total']
    mix.to_csv(out_dir / 'phase16_3_context_regime_symbol_mix.csv', index=False)

    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'config': asdict(cfg),
            'vocab_size': int(vocab_size),
            'seq_len': int(cfg.seq_len),
            'context_dim': int(ctx_dim),
            'k_context': int(kctx),
            'kmeans_centers': km.cluster_centers_,
        },
        out_dir / 'phase16_3_context_transformer.pt',
    )

    summary = {
        'status': 'SUCCESS',
        'phase16_2_run_dir': str(phase16_2_dir),
        'out_dir': str(out_dir),
        'vocab_size': int(vocab_size),
        'seq_len': int(cfg.seq_len),
        'n_sequences_total': int(len(x)),
        'n_train': int(len(x_train)),
        'n_val': int(len(x_val)),
        'n_test': int(len(x_test)),
        'context_dim': int(ctx_dim),
        'k_context': int(kctx),
        'next_token_metrics': {
            'train': {'loss': float(m_train['loss']), 'acc': float(m_train['acc'])},
            'val': {'loss': float(m_val['loss']), 'acc': float(m_val['acc'])},
            'test': {'loss': float(m_test['loss']), 'acc': float(m_test['acc'])},
        },
        'context_regime_symbol_purity_weighted': float(purity_context_symbol),
        'anti_leak': {
            'sequence_build_within_symbol_and_split': True,
            'no_cross_split_sequences': True,
            'train_val_test_temporal_from_phase16': True,
        },
        'artifacts': {
            'context_embeddings_parquet': str(out_dir / 'phase16_3_context_embeddings.parquet'),
            'train_history_csv': str(out_dir / 'phase16_3_train_history.csv'),
            'next_token_acc_by_split_csv': str(out_dir / 'phase16_3_next_token_acc_by_split.csv'),
            'context_regime_symbol_mix_csv': str(out_dir / 'phase16_3_context_regime_symbol_mix.csv'),
            'model_pt': str(out_dir / 'phase16_3_context_transformer.pt'),
            'loss_png': str(out_dir / 'phase16_3_loss.png'),
            'acc_png': str(out_dir / 'phase16_3_acc.png'),
            'log': str(out_dir / 'phase16_3.log'),
        },
    }
    (out_dir / 'phase16_3_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    logger.info('SUCCESS: %s', out_dir / 'phase16_3_summary.json')


if __name__ == '__main__':
    main()
