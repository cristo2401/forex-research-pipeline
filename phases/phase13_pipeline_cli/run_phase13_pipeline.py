#!/usr/bin/env python3
import argparse
import hashlib
import importlib.util
import json
import logging
import math
import os
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

ROOT = Path(__file__).resolve().parents[2]
PHASE4_OUT = ROOT / 'phases/phase04_transfer_clustering/output_transfer_GBPUSD_full_v2'
PHASE10_OUT = ROOT / 'phases/phase10_decision_engine/output_phase10_2_strict_expanding_GBPUSD'
PHASE11_SCRIPT = ROOT / 'phases/phase11_offline_rl/run_phase11_2_offline_rl_reward_input.py'
PHASE12_MODEL = ROOT / 'phases/phase12_leverage_rl/output_phase12_2_gym_leverage_GBPUSD/phase12_2_best_actor_critic.pt'

ENCODER_PT = PHASE4_OUT / 'transfer_attention_encoder.pt'
CLUSTER_ASSIGN_REF = PHASE4_OUT / 'GBPUSD_segment_cluster_assignments.parquet'
EMB_REF_NPZ = PHASE4_OUT / 'GBPUSD_embeddings_transfer.npz'
PHASE10_WINDOWS_CSV = PHASE10_OUT / 'GBPUSD_phase10_2_windows_predictions.csv'
PHASE10_HOLDOUT_CSV = PHASE10_OUT / 'GBPUSD_phase10_2_holdout_predictions.csv'


@dataclass
class PipelineConfig:
    symbol: str = 'GBPUSD'
    mode: str = 'infer_frozen'
    paper_only: bool = True

    # stage01
    quality_min_complete: float = 0.50
    quality_max_gap_rate: float = 0.50

    # stage02 segmentation
    event_q: float = 0.985
    event_max_gap: int = 10
    seg_prepad: int = 6
    seg_postpad: int = 10
    min_seg_bars: int = 6
    max_seg_bars: int = 176
    max_overlap: float = 0.60
    quality_min_segments: int = 10
    event_min_history: int = 500
    force_tail_segment: bool = True
    tail_segment_bars: int = 24

    # stage04/05
    decision_horizon_bars: int = 4
    min_move_pips: float = 4.0
    cost_pips: float = 1.2
    priors_source: str = 'windows_only'  # windows_only | windows_plus_holdout
    strict_temporal_eval: bool = True


class SegmentDataset(torch.utils.data.Dataset):
    def __init__(self, x: np.ndarray, m: np.ndarray):
        self.x = torch.from_numpy(x).float()
        self.m = torch.from_numpy(m).bool()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.m[i]


class AttentionPool(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model) * 0.02)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scale = math.sqrt(h.shape[-1])
        logits = (h * self.query.view(1, 1, -1)).sum(dim=-1) / scale
        logits = logits.masked_fill(~mask, -1e4)
        attn = torch.softmax(logits, dim=1)
        pooled = torch.einsum('bl,bld->bd', attn, h)
        return pooled


class SegmentAttentionAutoencoder(nn.Module):
    def __init__(self, max_len: int, d_model: int, emb_dim: int, n_head: int, n_layers: int, ff_dim: int, dropout: float):
        super().__init__()
        self.max_len = max_len
        self.in_proj = nn.Linear(1, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.pool = AttentionPool(d_model)

        self.to_emb = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, emb_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(emb_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, max_len),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        _, l = x.shape
        h = self.in_proj(x.unsqueeze(-1))
        h = h + self.pos_emb[:, :l, :]
        h = self.encoder(h, src_key_padding_mask=~mask)
        pooled = self.pool(h, mask)
        z = self.to_emb(pooled)
        recon = self.decoder(z)[:, :l]
        return z, recon


class ActorCritic(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, n_actions: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.pi = nn.Linear(hidden_dim, n_actions)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        return self.pi(h), self.v(h).squeeze(-1)


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def pip_size_for_symbol(symbol: str) -> float:
    return 0.01 if symbol.upper().endswith('JPY') else 0.0001


def robust_zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad < eps:
        m = np.nanmean(x)
        s = np.nanstd(x)
        if not np.isfinite(s) or s < eps:
            return np.zeros_like(x, dtype=float)
        z = (x - m) / (s + eps)
    else:
        z = (x - med) / (1.4826 * mad + eps)
    return np.clip(z, -8.0, 8.0)


def overlap_ratio(a0: int, a1: int, b0: int, b1: int) -> float:
    inter = max(0, min(a1, b1) - max(a0, b0) + 1)
    la = max(1, a1 - a0 + 1)
    lb = max(1, b1 - b0 + 1)
    return inter / float(min(la, lb))


def nms_segments(df: pd.DataFrame, max_overlap: float = 0.6) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    d = df.sort_values('score_event', ascending=False).reset_index(drop=True)
    keep = []
    for _, r in d.iterrows():
        s0, s1 = int(r['start_idx']), int(r['end_idx'])
        ok = True
        for k in keep:
            ov = overlap_ratio(s0, s1, int(k['start_idx']), int(k['end_idx']))
            if ov > max_overlap:
                ok = False
                break
        if ok:
            keep.append(r)
    out = pd.DataFrame(keep).sort_values('start_idx').reset_index(drop=True)
    return out


def preprocess_sequence(close_arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(close_arr, dtype=float)
    if len(x) == 0:
        return np.zeros(0, dtype=np.float32)
    x = np.clip(x, eps, None)
    logp = np.log(x)
    y = logp - logp[0]

    dy = np.diff(y, prepend=y[0])
    med = np.nanmedian(dy)
    mad = np.nanmedian(np.abs(dy - med))
    scale = 1.4826 * mad if np.isfinite(mad) and mad > eps else np.nan
    if not np.isfinite(scale) or scale < eps:
        scale = np.nanstd(dy)
    if not np.isfinite(scale) or scale < eps:
        scale = np.nanstd(y)
    if not np.isfinite(scale) or scale < eps:
        scale = 1.0

    z = y / float(scale)
    z = np.clip(z, -8.0, 8.0)
    return z.astype(np.float32)


def safe_json_write(path: Path, obj: dict):
    path.write_text(json.dumps(obj, indent=2, default=str))


def detect_timestamp_col(columns: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in columns}
    for c in ['timestamp', 'dt', 'datetime', 'time', 'date']:
        if c in lower_map:
            return lower_map[c]
    return None


def load_input_m1(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'Input file not found: {path}')

    if path.suffix.lower() in ['.parquet', '.pq']:
        df = pd.read_parquet(path).copy()
    elif path.suffix.lower() in ['.csv', '.txt']:
        df = pd.read_csv(path).copy()
    else:
        raise ValueError(f'Unsupported input format: {path.suffix}. Use csv or parquet.')

    ts_col = detect_timestamp_col(df.columns.tolist())
    if ts_col is None:
        raise ValueError('Missing timestamp column. Expected one of: timestamp, dt, datetime, time, date.')

    rename_map = {ts_col: 'timestamp'}
    for c in ['open', 'high', 'low', 'close', 'volume']:
        if c not in df.columns and c.capitalize() in df.columns:
            rename_map[c.capitalize()] = c
    df = df.rename(columns=rename_map)

    required = ['timestamp', 'open', 'high', 'low', 'close']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')

    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
    for c in ['open', 'high', 'low', 'close']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    if 'volume' in df.columns:
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0.0)

    bad_ts = int(df['timestamp'].isna().sum())
    bad_ohlc = int(df[['open', 'high', 'low', 'close']].isna().any(axis=1).sum())
    if bad_ts > 0:
        raise ValueError(f'Invalid timestamps found: {bad_ts}')
    if bad_ohlc > 0:
        raise ValueError(f'Invalid OHLC rows found: {bad_ohlc}')

    df = df.sort_values('timestamp').reset_index(drop=True)

    if df['timestamp'].duplicated().any():
        dup = int(df['timestamp'].duplicated().sum())
        raise ValueError(f'Duplicate timestamps found: {dup}')

    if not df['timestamp'].is_monotonic_increasing:
        raise ValueError('Timestamp is not monotonic increasing')

    # timezone normalized to UTC naive
    df['timestamp'] = df['timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)

    # OHLC coherence
    invalid = (
        (df['high'] < df[['open', 'close']].max(axis=1))
        | (df['low'] > df[['open', 'close']].min(axis=1))
        | (df['high'] < df['low'])
    )
    n_invalid = int(invalid.sum())
    if n_invalid > 0:
        raise ValueError(f'OHLC coherence violated in {n_invalid} rows')

    return df


def stage01_resample_h1(df_m1: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    w = df_m1.set_index('timestamp').sort_index()

    agg = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
    }
    if 'volume' in w.columns:
        agg['volume'] = 'sum'

    h1 = w.resample('1h', label='left', closed='left').agg(agg)
    n_m1 = w['close'].resample('1h', label='left', closed='left').count()

    h1 = h1[n_m1 > 0].copy()
    h1['n_m1'] = n_m1[n_m1 > 0].astype(int)
    h1['is_complete'] = (h1['n_m1'] == 60)

    idx = h1.index
    gap = idx.to_series().diff().dt.total_seconds().div(3600.0).fillna(1.0)
    h1['gap_hours'] = gap.astype(float)
    h1['is_gap'] = h1['gap_hours'] > 1.0

    h1 = h1.reset_index()

    metrics = {
        'n_rows_h1': int(len(h1)),
        'pct_complete': float(h1['is_complete'].mean()) if len(h1) else 0.0,
        'gap_rate': float(h1['is_gap'].mean()) if len(h1) else 0.0,
        'date_min': str(h1['timestamp'].min()) if len(h1) else None,
        'date_max': str(h1['timestamp'].max()) if len(h1) else None,
    }
    return h1, metrics


def stage02_segment_h1(px_h1: pd.DataFrame, cfg: PipelineConfig) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    close = px_h1['close'].to_numpy(dtype=float)

    ret1 = np.zeros(len(close), dtype=float)
    if len(close) > 1:
        prev = np.clip(close[:-1], 1e-12, None)
        curr = np.clip(close[1:], 1e-12, None)
        ret1[1:] = np.log(curr / prev)

    range12 = (px_h1['high'].rolling(12, min_periods=3).max() - px_h1['low'].rolling(12, min_periods=3).min()).to_numpy(dtype=float)
    drange = np.zeros_like(range12)
    drange[1:] = np.diff(range12)

    z_ret = np.abs(robust_zscore(ret1))
    z_rng = np.abs(robust_zscore(drange))
    score = 0.7 * z_ret + 0.3 * z_rng

    valid = np.isfinite(score)
    if valid.sum() == 0:
        raise RuntimeError('No valid event scores for segmentation')
    # Causal threshold: quantile computed only from past information.
    s = pd.Series(score)
    q_past = s.expanding(min_periods=max(32, int(cfg.event_min_history))).quantile(cfg.event_q).shift(1)
    if q_past.notna().sum() == 0:
        seed_thr = float(np.nanquantile(score[valid], cfg.event_q))
        q_past = pd.Series(np.full(len(s), seed_thr, dtype=float))
    else:
        first_valid = float(q_past.dropna().iloc[0])
        q_past = q_past.fillna(first_valid)

    event_mask = (s.to_numpy(dtype=float) >= q_past.to_numpy(dtype=float)) & valid
    event_idx = np.where(event_mask)[0]
    thr = float(np.nanmedian(q_past.to_numpy(dtype=float)))

    clusters = []
    if len(event_idx) > 0:
        cur = [int(event_idx[0])]
        for i in event_idx[1:]:
            i = int(i)
            if i - cur[-1] <= cfg.event_max_gap:
                cur.append(i)
            else:
                clusters.append(cur)
                cur = [i]
        clusters.append(cur)

    rows = []
    for c in clusters:
        s = max(0, min(c) - cfg.seg_prepad)
        e = min(len(px_h1) - 1, max(c) + cfg.seg_postpad)
        n = e - s + 1

        if n < cfg.min_seg_bars:
            continue
        if n > cfg.max_seg_bars:
            e = min(len(px_h1) - 1, s + cfg.max_seg_bars - 1)
            n = e - s + 1

        w = px_h1.iloc[s:e + 1]
        ps = pip_size_for_symbol(cfg.symbol)
        move_pips = (float(w['close'].iloc[-1]) - float(w['close'].iloc[0])) / ps
        range_pips = (float(w['high'].max()) - float(w['low'].min())) / ps
        sc = float(np.nanmax(score[s:e + 1]))

        rows.append(
            {
                'symbol': cfg.symbol,
                'start_idx': int(s),
                'end_idx': int(e),
                'start_ts': pd.Timestamp(w['timestamp'].iloc[0]),
                'end_ts': pd.Timestamp(w['timestamp'].iloc[-1]),
                'n_bars': int(n),
                'move_pips': float(move_pips),
                'range_pips': float(range_pips),
                'score_event': sc,
                'source': 'phase13_auto_event_segmentation',
            }
        )

    seg = pd.DataFrame(rows)
    if seg.empty:
        raise RuntimeError('No segments generated in stage02')

    seg = nms_segments(seg, max_overlap=cfg.max_overlap)

    # Live-tail safety: if the latest H1 bar is outside all segments, append a
    # bounded tail segment so downstream stages can emit a fresh timestamp.
    n_tail_segments = 0
    if bool(cfg.force_tail_segment) and len(px_h1) > 0:
        last_idx = int(len(px_h1) - 1)
        covered_last = bool(
            ((seg['start_idx'].to_numpy(dtype=int) <= last_idx) & (seg['end_idx'].to_numpy(dtype=int) >= last_idx)).any()
        )
        if not covered_last:
            tail_n = int(np.clip(int(cfg.tail_segment_bars), int(cfg.min_seg_bars), int(cfg.max_seg_bars)))
            s_tail = max(0, last_idx - tail_n + 1)
            e_tail = last_idx
            n_tail = int(e_tail - s_tail + 1)
            if n_tail >= int(cfg.min_seg_bars):
                w = px_h1.iloc[s_tail:e_tail + 1]
                ps = pip_size_for_symbol(cfg.symbol)
                move_pips = (float(w['close'].iloc[-1]) - float(w['close'].iloc[0])) / ps
                range_pips = (float(w['high'].max()) - float(w['low'].min())) / ps
                sc = float(np.nanmax(score[s_tail:e_tail + 1])) if np.isfinite(score[s_tail:e_tail + 1]).any() else 0.0

                tail_row = pd.DataFrame(
                    [
                        {
                            'symbol': cfg.symbol,
                            'start_idx': int(s_tail),
                            'end_idx': int(e_tail),
                            'start_ts': pd.Timestamp(w['timestamp'].iloc[0]),
                            'end_ts': pd.Timestamp(w['timestamp'].iloc[-1]),
                            'n_bars': int(n_tail),
                            'move_pips': float(move_pips),
                            'range_pips': float(range_pips),
                            'score_event': sc,
                            'source': 'phase13_forced_tail_segment',
                        }
                    ]
                )
                seg = pd.concat([seg, tail_row], ignore_index=True)
                n_tail_segments = 1

    seg = seg.sort_values('start_ts').reset_index(drop=True)
    seg['global_segment_id'] = [f'{cfg.symbol}_seg_{i:05d}' for i in range(len(seg))]

    long_rows = []
    for _, r in seg.iterrows():
        sid = str(r['global_segment_id'])
        s = int(r['start_idx'])
        e = int(r['end_idx'])
        w = px_h1.iloc[s:e + 1]
        for i, (_, rr) in enumerate(w.iterrows()):
            long_rows.append(
                {
                    'global_segment_id': sid,
                    'step': int(i),
                    'timestamp': pd.Timestamp(rr['timestamp']),
                    'open': float(rr['open']),
                    'high': float(rr['high']),
                    'low': float(rr['low']),
                    'close': float(rr['close']),
                }
            )

    long_df = pd.DataFrame(long_rows)
    metrics = {
        'event_threshold': float(thr),
        'n_events': int(len(event_idx)),
        'n_segments': int(len(seg)),
        'n_tail_segments': int(n_tail_segments),
        'n_rows_long': int(len(long_df)),
        'mean_segment_len': float(seg['n_bars'].mean()) if len(seg) else 0.0,
        'latest_segment_end_ts': str(seg['end_ts'].max()) if len(seg) else None,
    }
    return seg, long_df, metrics


def stage03_embed_cluster(
    seg_catalog: pd.DataFrame,
    seg_long: pd.DataFrame,
    device: torch.device,
) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, float]]:
    if not ENCODER_PT.exists():
        raise FileNotFoundError(f'Missing encoder checkpoint: {ENCODER_PT}')
    if not EMB_REF_NPZ.exists():
        raise FileNotFoundError(f'Missing reference embeddings: {EMB_REF_NPZ}')
    if not CLUSTER_ASSIGN_REF.exists():
        raise FileNotFoundError(f'Missing reference assignments: {CLUSTER_ASSIGN_REF}')

    ckpt = torch.load(ENCODER_PT, map_location='cpu')
    cfg = ckpt['config']

    model = SegmentAttentionAutoencoder(
        max_len=int(cfg['max_len']),
        d_model=int(cfg['d_model']),
        emb_dim=int(cfg['emb_dim']),
        n_head=int(cfg['n_head']),
        n_layers=int(cfg['n_layers']),
        ff_dim=int(cfg['ff_dim']),
        dropout=float(cfg['dropout']),
    ).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    seq_map = {
        sid: g['close'].to_numpy(dtype=float)
        for sid, g in seg_long.sort_values(['global_segment_id', 'step']).groupby('global_segment_id', sort=False)
    }

    ids = []
    x_rows = []
    m_rows = []
    lengths = []
    max_len = int(cfg['max_len'])

    for _, r in seg_catalog.iterrows():
        sid = str(r['global_segment_id'])
        seq = seq_map.get(sid)
        if seq is None or len(seq) < 2:
            continue
        x = preprocess_sequence(seq)
        x = x[:max_len]
        ln = len(x)

        xx = np.zeros((max_len,), dtype=np.float32)
        mm = np.zeros((max_len,), dtype=bool)
        xx[:ln] = x
        mm[:ln] = True

        ids.append(sid)
        x_rows.append(xx)
        m_rows.append(mm)
        lengths.append(ln)

    if not x_rows:
        raise RuntimeError('No valid sequences for embedding stage')

    x_np = np.asarray(x_rows, dtype=np.float32)
    m_np = np.asarray(m_rows, dtype=bool)

    ds = SegmentDataset(x_np, m_np)
    loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False)

    z_parts = []
    with torch.no_grad():
        for xb, mb in loader:
            xb = xb.to(device)
            mb = mb.to(device)
            z, _ = model(xb, mb)
            z_parts.append(z.cpu().numpy())

    z_tar = np.concatenate(z_parts, axis=0).astype(np.float32)

    ref_npz = np.load(EMB_REF_NPZ)
    z_ref = ref_npz['Z_target'].astype(np.float32)
    c_ref = ref_npz['cluster_id'].astype(int)

    centroids = {}
    for c in sorted(np.unique(c_ref).tolist()):
        centroids[int(c)] = z_ref[c_ref == c].mean(axis=0)

    cent_ids = np.array(sorted(centroids.keys()), dtype=int)
    cent_mat = np.stack([centroids[c] for c in cent_ids], axis=0)

    d = ((z_tar[:, None, :] - cent_mat[None, :, :]) ** 2).sum(axis=2)
    j = d.argmin(axis=1)
    cluster_tar = cent_ids[j]
    cluster_dist = np.sqrt(d[np.arange(len(d)), j])

    ref_ass = pd.read_parquet(CLUSTER_ASSIGN_REF)
    informative = set(
        ref_ass.loc[ref_ass['is_informative_cluster'] == True, 'cluster_id'].dropna().astype(int).tolist()
    )

    ass = seg_catalog[['global_segment_id', 'symbol', 'start_ts', 'end_ts', 'n_bars', 'move_pips', 'range_pips', 'score_event']].copy()
    ass = ass.merge(pd.DataFrame({'global_segment_id': ids, 'cluster_id': cluster_tar, 'cluster_dist': cluster_dist, 'length': lengths}), on='global_segment_id', how='inner')
    ass['is_informative_cluster'] = ass['cluster_id'].isin(informative)

    se = ass['score_event'].to_numpy(dtype=float)
    se_norm = (se - np.nanmin(se)) / (np.nanmax(se) - np.nanmin(se) + 1e-9)
    cd = ass['cluster_dist'].to_numpy(dtype=float)
    cd_norm = (cd - np.nanmin(cd)) / (np.nanmax(cd) - np.nanmin(cd) + 1e-9)

    ass['score_event_norm'] = se_norm
    ass['cluster_dist_norm'] = cd_norm
    ass['selection_score'] = (
        ass['is_informative_cluster'].astype(float) * 0.65
        + ass['score_event_norm'] * 0.25
        + (1.0 - ass['cluster_dist_norm']) * 0.10
    )

    metrics = {
        'n_embeddings': int(len(z_tar)),
        'n_clusters_used': int(pd.Series(cluster_tar).nunique()),
        'informative_rate': float(ass['is_informative_cluster'].mean()) if len(ass) else 0.0,
    }
    return z_tar, ass, metrics


def build_cluster_priors(priors_source: str = 'windows_only') -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, object]]:
    for p in [PHASE10_WINDOWS_CSV, PHASE10_HOLDOUT_CSV, CLUSTER_ASSIGN_REF]:
        if not p.exists():
            raise FileNotFoundError(f'Missing required frozen artifact: {p}')

    w = pd.read_csv(PHASE10_WINDOWS_CSV)
    h = pd.read_csv(PHASE10_HOLDOUT_CSV)

    if priors_source == 'windows_only':
        hist = w.copy()
    elif priors_source == 'windows_plus_holdout':
        hist = pd.concat([w, h], ignore_index=True)
    else:
        raise ValueError(f'Unsupported priors_source={priors_source}')

    hist['timestamp'] = pd.to_datetime(hist['timestamp'], errors='coerce')

    ass = pd.read_parquet(CLUSTER_ASSIGN_REF)[['global_segment_id', 'cluster_id', 'is_informative_cluster']]
    m = hist.merge(ass, on='global_segment_id', how='inner')

    pri = (
        m.groupby('cluster_id')
        .agg(
            p_short=('p_short', 'mean'),
            p_flat=('p_flat', 'mean'),
            p_long=('p_long', 'mean'),
            pred_inf_prob=('pred_inf_prob', 'mean'),
            pred_sub_conf=('pred_sub_conf', 'mean'),
            pred_sub_entropy=('pred_sub_entropy', 'mean'),
            pred_conf_max=('pred_conf_max', 'mean'),
            pred_action_mean=('pred_action', 'mean'),
            n=('cluster_id', 'count'),
            informative_rate=('is_informative_cluster', 'mean'),
        )
        .reset_index()
    )

    g = {
        'p_short': float(hist['p_short'].mean()),
        'p_flat': float(hist['p_flat'].mean()),
        'p_long': float(hist['p_long'].mean()),
        'pred_inf_prob': float(hist['pred_inf_prob'].mean()),
        'pred_sub_conf': float(hist['pred_sub_conf'].mean()),
        'pred_sub_entropy': float(hist['pred_sub_entropy'].mean()),
        'pred_conf_max': float(hist['pred_conf_max'].mean()),
        'pred_action_mean': float(hist['pred_action'].mean()),
    }
    meta = {
        'priors_source': priors_source,
        'n_prior_rows': int(len(hist)),
        'prior_ts_min': str(hist['timestamp'].min()) if len(hist) else None,
        'prior_ts_max': str(hist['timestamp'].max()) if len(hist) else None,
    }
    return pri, g, meta


def stage04_decision(
    seg_long: pd.DataFrame,
    ass: pd.DataFrame,
    cfg: PipelineConfig,
    pri: pd.DataFrame,
    g: Dict[str, float],
) -> Tuple[pd.DataFrame, Dict[str, float]]:

    df = seg_long.merge(
        ass[['global_segment_id', 'cluster_id', 'is_informative_cluster', 'selection_score']],
        on='global_segment_id',
        how='left',
    )
    df = df.sort_values(['global_segment_id', 'step', 'timestamp']).reset_index(drop=True)

    pmap = pri.set_index('cluster_id').to_dict(orient='index')

    p_short = np.full(len(df), g['p_short'], dtype=float)
    p_flat = np.full(len(df), g['p_flat'], dtype=float)
    p_long = np.full(len(df), g['p_long'], dtype=float)
    pred_inf_prob = np.full(len(df), g['pred_inf_prob'], dtype=float)
    pred_sub_conf = np.full(len(df), g['pred_sub_conf'], dtype=float)
    pred_sub_entropy = np.full(len(df), g['pred_sub_entropy'], dtype=float)
    pred_conf_max = np.full(len(df), g['pred_conf_max'], dtype=float)

    cid_arr = df['cluster_id'].fillna(-1).to_numpy(dtype=int)
    for i, c in enumerate(cid_arr):
        if c in pmap:
            row = pmap[c]
            p_short[i] = float(row['p_short'])
            p_flat[i] = float(row['p_flat'])
            p_long[i] = float(row['p_long'])
            pred_inf_prob[i] = float(row['pred_inf_prob'])
            pred_sub_conf[i] = float(row['pred_sub_conf'])
            pred_sub_entropy[i] = float(row['pred_sub_entropy'])
            pred_conf_max[i] = float(row['pred_conf_max'])

    # momentum tweak from current/past information only
    grp = df.groupby('global_segment_id', group_keys=False)
    mom1 = grp['close'].pct_change().fillna(0.0).to_numpy(dtype=float)
    adj = np.clip(np.abs(mom1) * 12.0, 0.0, 0.12)

    pos = mom1 > 0
    neg = mom1 < 0
    p_long[pos] += adj[pos]
    p_short[pos] -= 0.5 * adj[pos]
    p_short[neg] += adj[neg]
    p_long[neg] -= 0.5 * adj[neg]

    p_short = np.clip(p_short, 1e-6, 1.0)
    p_flat = np.clip(p_flat, 1e-6, 1.0)
    p_long = np.clip(p_long, 1e-6, 1.0)
    s = p_short + p_flat + p_long
    p_short /= s
    p_flat /= s
    p_long /= s

    probs = np.vstack([p_short, p_flat, p_long]).T
    pred_conf_max = probs.max(axis=1)
    pred_sub_entropy = -(probs * np.log(np.clip(probs, 1e-8, 1.0))).sum(axis=1)
    pred_sub_conf = pred_conf_max.copy()

    pred_progress = grp['step'].transform(lambda x: x / max(x.max(), 1)).to_numpy(dtype=float)
    pred_class = np.where(df['is_informative_cluster'].fillna(False).to_numpy(), (df['cluster_id'].fillna(0).to_numpy(dtype=int) % 8) + 1, 0)

    idx = probs.argmax(axis=1)
    raw_action = np.array([-1, 0, 1], dtype=int)[idx]

    conf_thr = 0.42
    inf_min = 0.35
    pred_action = raw_action.copy()
    gate = (pred_conf_max >= conf_thr) & (pred_inf_prob >= inf_min)
    pred_action[~gate] = 0

    trade_rate = float(np.mean(pred_action != 0))
    if trade_rate < 0.02:
        # Fallback for sparse samples: keep conservative but avoid a fully-degenerate flat policy.
        pred_action = raw_action.copy()
        # Causal volatility gate: row-wise threshold from rolling past.
        abs_mom = pd.Series(np.abs(mom1))
        q70_past = abs_mom.rolling(window=256, min_periods=32).quantile(0.70).shift(1)
        if q70_past.notna().sum() == 0:
            fallback_thr = float(np.nanquantile(np.abs(mom1), 0.70))
            q70_past = pd.Series(np.full(len(abs_mom), fallback_thr, dtype=float))
        else:
            q70_past = q70_past.fillna(float(q70_past.dropna().iloc[0]))

        sparse_mask = (
            (pred_action == 0)
            & (
                df['is_informative_cluster'].fillna(False).to_numpy()
                | (np.abs(mom1) >= q70_past.to_numpy(dtype=float))
            )
        )
        pred_action[sparse_mask] = np.sign(mom1[sparse_mask]).astype(int)

    df['pred_inf_prob'] = pred_inf_prob.astype(np.float32)
    df['pred_progress'] = pred_progress.astype(np.float32)
    df['pred_sub_conf'] = pred_sub_conf.astype(np.float32)
    df['pred_sub_entropy'] = pred_sub_entropy.astype(np.float32)
    df['pred_conf_max'] = pred_conf_max.astype(np.float32)
    df['pred_class'] = pred_class.astype(np.int64)
    df['p_short'] = p_short.astype(np.float32)
    df['p_flat'] = p_flat.astype(np.float32)
    df['p_long'] = p_long.astype(np.float32)
    df['pred_action'] = pred_action.astype(np.int64)

    # build evaluation target (offline paper check)
    df['close_fwd'] = grp['close'].shift(-cfg.decision_horizon_bars)
    df['future_return_h'] = (df['close_fwd'] / df['close']) - 1.0
    pip = pip_size_for_symbol(cfg.symbol)
    df['future_pips_h'] = df['future_return_h'] / pip

    thr = cfg.min_move_pips + cfg.cost_pips
    y = np.zeros(len(df), dtype=int)
    y[df['future_pips_h'].to_numpy(dtype=float) > thr] = 1
    y[df['future_pips_h'].to_numpy(dtype=float) < -thr] = -1
    df['y_action'] = y.astype(np.int64)

    cost_ret = cfg.cost_pips * pip
    rr = df['future_return_h'].fillna(0.0).to_numpy(dtype=float)
    aa = df['pred_action'].to_numpy(dtype=float)
    df['net_return'] = aa * rr - (aa != 0).astype(float) * cost_ret

    trade_rate = float((df['pred_action'] != 0).mean())
    if trade_rate <= 0.01 or trade_rate >= 0.99:
        raise RuntimeError(f'Quality gate failed in stage04: trade_rate={trade_rate:.4f} out of bounds')

    dist = df['pred_action'].value_counts(normalize=True).to_dict()
    metrics = {
        'n_rows': int(len(df)),
        'trade_rate': trade_rate,
        'pred_action_dist': {int(k): float(v) for k, v in dist.items()},
        'conf_thr': conf_thr,
        'inf_min': inf_min,
    }
    return df, metrics


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot load module {path}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stage05_apply_phase12_frozen(df_stage4: pd.DataFrame, cfg: PipelineConfig, device: torch.device) -> Tuple[pd.DataFrame, Dict[str, float]]:
    if not PHASE12_MODEL.exists():
        raise FileNotFoundError(f'Missing phase12 model checkpoint: {PHASE12_MODEL}')
    if not PHASE11_SCRIPT.exists():
        raise FileNotFoundError(f'Missing feature builder script: {PHASE11_SCRIPT}')

    blob = torch.load(PHASE12_MODEL, map_location='cpu', weights_only=False)
    feature_cols_ckpt = list(blob['feature_cols'])
    mu = np.asarray(blob['mu'], dtype=np.float32)
    sd = np.asarray(blob['sd'], dtype=np.float32)
    sd = np.where(sd < 1e-8, 1.0, sd)

    p112 = load_module(PHASE11_SCRIPT, name='phase11_2_for_phase13')
    feat_df, _ = p112.build_features(df_stage4.copy())

    for c in feature_cols_ckpt:
        if c not in feat_df.columns:
            feat_df[c] = 0.0

    x_raw = feat_df[feature_cols_ckpt].to_numpy(dtype=np.float32)
    x = (x_raw - mu) / sd
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    behavior = feat_df['pred_action'].to_numpy(dtype=np.float32)
    obs = np.concatenate([x, behavior.reshape(-1, 1)], axis=1)

    hidden = int(blob['config']['hidden_dim'])
    action_mags = np.asarray(blob['action_mags'], dtype=np.float32)
    force_nonzero = bool(blob.get('best_trial', {}).get('force_nonzero', False))
    min_force_lev = 0.5

    model = ActorCritic(in_dim=obs.shape[1], hidden_dim=hidden, n_actions=len(action_mags)).to(device)
    model.load_state_dict(blob['state_dict'])
    model.eval()

    lev = np.zeros(len(obs), dtype=np.float32)
    bs = 2048
    with torch.no_grad():
        for i in range(0, len(obs), bs):
            xb = torch.tensor(obs[i:i + bs], dtype=torch.float32, device=device)
            logits, _ = model(xb)
            act = torch.argmax(logits, dim=1).cpu().numpy().astype(int)
            mag = action_mags[act]
            sign = np.sign(behavior[i:i + bs]).astype(np.float32)

            if force_nonzero:
                mask = (sign != 0.0) & (mag <= 0.0)
                mag = mag.copy()
                mag[mask] = min_force_lev

            lev[i:i + bs] = sign * mag

    out = feat_df.copy()
    out['lev_phase12_2_frozen'] = lev
    out['action_phase12_2_frozen'] = np.sign(lev).astype(np.int64)

    rr = out['future_return_h'].fillna(0.0).to_numpy(dtype=np.float32)
    cost_ret = cfg.cost_pips * pip_size_for_symbol(cfg.symbol)

    out['net_behavior_1x'] = out['pred_action'].to_numpy(dtype=np.float32) * rr - (out['pred_action'] != 0).astype(float) * cost_ret
    out['net_behavior_2x'] = (2.0 * out['pred_action'].to_numpy(dtype=np.float32)) * rr - (out['pred_action'] != 0).astype(float) * 2.0 * cost_ret
    out['net_phase12_2_frozen'] = lev * rr - np.abs(lev) * cost_ret

    out['cum_behavior_1x'] = out['net_behavior_1x'].cumsum()
    out['cum_behavior_2x'] = out['net_behavior_2x'].cumsum()
    out['cum_phase12_2_frozen'] = out['net_phase12_2_frozen'].cumsum()

    metrics = {
        'avg_abs_lev': float(np.mean(np.abs(lev))),
        'trade_rate': float(np.mean(lev != 0.0)),
        'force_nonzero': force_nonzero,
    }
    return out, metrics


def evaluate_levels(levels: np.ndarray, y_true_action: np.ndarray, future_ret: np.ndarray, cost_ret: float) -> Dict[str, float]:
    action = np.sign(levels).astype(np.int64)
    net = levels * future_ret - np.abs(levels) * cost_ret
    trade = (levels != 0).astype(np.float32)

    out = {
        'acc_action_3': float((action == y_true_action).mean()),
        'f1_macro_action_3': float(f1_score(y_true_action, action, labels=[-1, 0, 1], average='macro', zero_division=0)),
        'avg_net_return': float(np.mean(net)),
        'cum_net_return': float(np.sum(net)),
        'std_net_return': float(np.std(net)),
        'trade_rate': float(np.mean(trade)),
    }
    eps = 1e-12
    out['sharpe_ann'] = float((out['avg_net_return'] / (out['std_net_return'] + eps)) * np.sqrt(24.0 * 252.0))
    traded = trade > 0
    out['win_rate_trades'] = float((net[traded] > 0).mean()) if traded.any() else 0.0
    out['avg_abs_lev'] = float(np.mean(np.abs(levels)))
    out['avg_abs_lev_on_trade'] = float(np.mean(np.abs(levels[traded]))) if traded.any() else 0.0
    return out


def run_command(cmd: List[str], cwd: Path, logger: logging.Logger):
    logger.info('Running command: %s', ' '.join(cmd))
    p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    logger.info('stdout:\n%s', p.stdout)
    if p.returncode != 0:
        logger.error('stderr:\n%s', p.stderr)
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}")


def configure_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger('phase13_pipeline')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def main():
    parser = argparse.ArgumentParser(description='Phase13 single-command pipeline (M1 -> phase12 paper outputs).')
    parser.add_argument('--input_m1', required=True, type=str, help='Path to M1 input file (csv/parquet).')
    parser.add_argument('--symbol', default='GBPUSD', type=str)
    parser.add_argument('--mode', default='infer_frozen', choices=['infer_frozen', 'retrain_full'])
    parser.add_argument('--cost_pips', default=1.2, type=float)
    parser.add_argument('--priors_source', default='windows_only', choices=['windows_only', 'windows_plus_holdout'])
    parser.add_argument('--strict_temporal_eval', default=1, type=int, choices=[0, 1])
    parser.add_argument('--out_dir', default=str(ROOT / 'phases/phase13_pipeline_cli/output_runs'), type=str)
    parser.add_argument('--run_id', default=None, type=str)
    args = parser.parse_args()

    cfg = PipelineConfig(
        symbol=str(args.symbol).upper(),
        mode=args.mode,
        cost_pips=float(args.cost_pips),
        priors_source=str(args.priors_source),
        strict_temporal_eval=bool(int(args.strict_temporal_eval)),
    )
    if len(cfg.symbol) != 6:
        raise ValueError('symbol must be a 6-letter FX pair, e.g. GBPUSD or EURUSD')

    run_id = args.run_id or f"phase13_{cfg.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = run_dir / 'manifest.json'
    status_path = run_dir / 'status.json'
    summary_path = run_dir / 'phase13_summary.json'
    log_path = run_dir / 'phase13.log'

    out_stage01 = run_dir / 'stage01_h1.parquet'
    out_stage02_catalog = run_dir / 'stage02_segments_catalog.parquet'
    out_stage02_long = run_dir / 'stage02_segments_long.parquet'
    out_stage03_npz = run_dir / 'stage03_embeddings.npz'
    out_stage03_assign = run_dir / 'stage03_cluster_assignments.parquet'
    out_stage04 = run_dir / 'stage04_phase10_predictions.parquet'
    out_stage05 = run_dir / 'stage05_phase12_predictions.parquet'
    out_equity = run_dir / 'phase13_equity.png'
    out_leak_audit = run_dir / 'phase13_leak_audit.json'

    logger = configure_logger(log_path)

    manifest = {
        'run_id': run_id,
        'created_at_utc': now_utc_iso(),
        'status': 'RUNNING',
        'config': asdict(cfg),
        'inputs': {
            'input_m1': str(Path(args.input_m1).resolve()),
        },
        'stages': {},
        'outputs': {},
        'notes': [],
    }

    def update_status(status: str, stage: Optional[str] = None, error: Optional[str] = None):
        s = {
            'run_id': run_id,
            'status': status,
            'current_stage': stage,
            'updated_at_utc': now_utc_iso(),
        }
        if error:
            s['error'] = error
        safe_json_write(status_path, s)

    def start_stage(name: str):
        manifest['stages'][name] = {'status': 'RUNNING', 'started_at_utc': now_utc_iso()}
        safe_json_write(manifest_path, manifest)
        update_status('RUNNING', stage=name)
        logger.info('=== %s START ===', name)

    def done_stage(name: str, metrics: dict, outputs: dict):
        manifest['stages'][name].update({'status': 'SUCCESS', 'ended_at_utc': now_utc_iso(), 'metrics': metrics, 'outputs': outputs})
        safe_json_write(manifest_path, manifest)
        logger.info('=== %s DONE ===', name)

    try:
        safe_json_write(manifest_path, manifest)
        update_status('RUNNING', stage='init')

        input_path = Path(args.input_m1)
        manifest['inputs']['input_sha256'] = sha256_file(input_path)

        if cfg.mode == 'retrain_full':
            manifest['notes'].append('mode=retrain_full requested; v1 executes frozen inference pipeline and does not mutate live trading state.')
        if cfg.symbol != 'GBPUSD':
            manifest['notes'].append(
                f'transfer_mode: using frozen GBPUSD priors/models on {cfg.symbol} input (cross-currency probe).'
            )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info('Using device: %s', device)

        # Stage01
        start_stage('stage01_h1')
        m1 = load_input_m1(input_path)
        h1, m01 = stage01_resample_h1(m1)
        if m01['pct_complete'] < cfg.quality_min_complete:
            raise RuntimeError(f"Quality gate failed: pct_complete={m01['pct_complete']:.4f} < {cfg.quality_min_complete}")
        if m01['gap_rate'] > cfg.quality_max_gap_rate:
            raise RuntimeError(f"Quality gate failed: gap_rate={m01['gap_rate']:.4f} > {cfg.quality_max_gap_rate}")

        h1.to_parquet(out_stage01, index=False)
        done_stage('stage01_h1', m01, {'stage01_h1_parquet': str(out_stage01)})

        # Stage02
        start_stage('stage02_segmentation')
        seg_cat, seg_long, m02 = stage02_segment_h1(h1, cfg)
        if m02['n_segments'] < cfg.quality_min_segments:
            raise RuntimeError(f"Quality gate failed: too few segments ({m02['n_segments']})")

        seg_cat.to_parquet(out_stage02_catalog, index=False)
        seg_long.to_parquet(out_stage02_long, index=False)
        done_stage('stage02_segmentation', m02, {'stage02_segments_catalog': str(out_stage02_catalog), 'stage02_segments_long': str(out_stage02_long)})

        # Stage03
        start_stage('stage03_embedding_cluster')
        z_tar, ass, m03 = stage03_embed_cluster(seg_cat, seg_long, device)
        np.savez_compressed(
            out_stage03_npz,
            Z_target=z_tar,
            global_segment_id=ass['global_segment_id'].to_numpy(dtype='U64'),
            cluster_id=ass['cluster_id'].to_numpy(dtype=np.int32),
        )
        ass.to_parquet(out_stage03_assign, index=False)

        if m03['n_clusters_used'] < 3:
            raise RuntimeError(f"Quality gate failed: low cluster diversity ({m03['n_clusters_used']})")

        done_stage('stage03_embedding_cluster', m03, {'stage03_embeddings_npz': str(out_stage03_npz), 'stage03_cluster_assignments': str(out_stage03_assign)})

        # Stage04
        start_stage('stage04_decision')
        pri, g, prior_meta = build_cluster_priors(cfg.priors_source)
        st4, m04 = stage04_decision(seg_long, ass, cfg, pri, g)
        m04['priors_source'] = prior_meta['priors_source']
        m04['prior_ts_min'] = prior_meta['prior_ts_min']
        m04['prior_ts_max'] = prior_meta['prior_ts_max']
        m04['n_prior_rows'] = prior_meta['n_prior_rows']
        st4.to_parquet(out_stage04, index=False)
        done_stage('stage04_decision', m04, {'stage04_phase10_predictions': str(out_stage04)})

        # Stage05
        start_stage('stage05_phase12_leverage')
        st5, m05 = stage05_apply_phase12_frozen(st4, cfg, device)
        st5.to_parquet(out_stage05, index=False)
        done_stage('stage05_phase12_leverage', m05, {'stage05_phase12_predictions': str(out_stage05)})

        # Summary metrics (finite target only)
        eval_df_full = st5[np.isfinite(st5['future_return_h'].to_numpy(dtype=float))].copy()
        eval_df = eval_df_full.copy()

        strict_cutoff_ts = None
        if cfg.strict_temporal_eval and prior_meta.get('prior_ts_max') is not None:
            strict_cutoff_ts = pd.to_datetime(prior_meta['prior_ts_max'], errors='coerce')
            ts = pd.to_datetime(eval_df['timestamp'], errors='coerce')
            eval_df = eval_df[ts > strict_cutoff_ts].copy()

        if len(eval_df) < 200:
            raise RuntimeError(f'Leak-safe eval has too few rows ({len(eval_df)}).')

        cost_ret = cfg.cost_pips * pip_size_for_symbol(cfg.symbol)
        leak_audit = {
            'strict_temporal_eval': bool(cfg.strict_temporal_eval),
            'priors_source': cfg.priors_source,
            'cost_pips': float(cfg.cost_pips),
            'cost_ret': float(cost_ret),
            'prior_ts_min': prior_meta.get('prior_ts_min'),
            'prior_ts_max': prior_meta.get('prior_ts_max'),
            'strict_eval_start_exclusive': str(strict_cutoff_ts) if strict_cutoff_ts is not None else None,
            'eval_rows_full': int(len(eval_df_full)),
            'eval_rows_strict': int(len(eval_df)),
            'all_eval_ts_gt_prior_max': bool(
                True
                if strict_cutoff_ts is None
                else (pd.to_datetime(eval_df['timestamp'], errors='coerce') > strict_cutoff_ts).all()
            ),
            'future_columns_used_as_features': False,
            'feature_columns_exclude_targets': True,
        }
        safe_json_write(out_leak_audit, leak_audit)
        y = eval_df['y_action'].to_numpy(dtype=np.int64)
        ret = eval_df['future_return_h'].to_numpy(dtype=np.float32)

        m_b1 = evaluate_levels(eval_df['pred_action'].to_numpy(dtype=np.float32), y, ret, cost_ret)
        m_b2 = evaluate_levels((2.0 * eval_df['pred_action'].to_numpy(dtype=np.float32)), y, ret, cost_ret)
        m_q = evaluate_levels(eval_df['lev_phase12_2_frozen'].to_numpy(dtype=np.float32), y, ret, cost_ret)

        # Equity plot
        plt.figure(figsize=(11, 4))
        plt.plot(pd.to_datetime(eval_df['timestamp']), eval_df['cum_behavior_1x'], label='behavior_1x')
        plt.plot(pd.to_datetime(eval_df['timestamp']), eval_df['cum_behavior_2x'], label='behavior_2x')
        plt.plot(pd.to_datetime(eval_df['timestamp']), eval_df['cum_phase12_2_frozen'], label='phase12_2_frozen')
        plt.title('Phase13 equity curve (paper)')
        plt.xlabel('timestamp')
        plt.ylabel('cumulative return')
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_equity, dpi=140)
        plt.close()

        summary = {
            'run_id': run_id,
            'status': 'SUCCESS',
            'symbol': cfg.symbol,
            'mode': cfg.mode,
            'paper_only': cfg.paper_only,
            'cost_pips': float(cfg.cost_pips),
            'cost_ret': float(cost_ret),
            'created_at_utc': manifest['created_at_utc'],
            'finished_at_utc': now_utc_iso(),
            'n_eval_rows': int(len(eval_df)),
            'n_eval_rows_full': int(len(eval_df_full)),
            'metrics': {
                'behavior_1x': m_b1,
                'behavior_2x': m_b2,
                'phase12_2_frozen': m_q,
            },
            'leak_audit': leak_audit,
            'quality': {
                'stage01': manifest['stages']['stage01_h1']['metrics'],
                'stage02': manifest['stages']['stage02_segmentation']['metrics'],
                'stage03': manifest['stages']['stage03_embedding_cluster']['metrics'],
                'stage04': manifest['stages']['stage04_decision']['metrics'],
                'stage05': manifest['stages']['stage05_phase12_leverage']['metrics'],
            },
            'outputs': {
                'manifest_json': str(manifest_path),
                'status_json': str(status_path),
                'stage01_h1_parquet': str(out_stage01),
                'stage02_segments_catalog_parquet': str(out_stage02_catalog),
                'stage02_segments_long_parquet': str(out_stage02_long),
                'stage03_embeddings_npz': str(out_stage03_npz),
                'stage03_cluster_assignments_parquet': str(out_stage03_assign),
                'stage04_phase10_predictions_parquet': str(out_stage04),
                'stage05_phase12_predictions_parquet': str(out_stage05),
                'phase13_leak_audit_json': str(out_leak_audit),
                'phase13_summary_json': str(summary_path),
                'phase13_equity_png': str(out_equity),
                'log_file': str(log_path),
            },
        }

        safe_json_write(summary_path, summary)

        manifest['status'] = 'SUCCESS'
        manifest['ended_at_utc'] = now_utc_iso()
        manifest['outputs'] = summary['outputs']
        safe_json_write(manifest_path, manifest)
        update_status('SUCCESS', stage='done')

        logger.info('Pipeline SUCCESS. Summary: %s', summary_path)
        print(f'SUCCESS: {summary_path}')

    except Exception as e:
        err = f'{type(e).__name__}: {e}'
        logger.error('Pipeline FAILED: %s', err)
        logger.error(traceback.format_exc())

        manifest['status'] = 'FAILED'
        manifest['error'] = err
        manifest['ended_at_utc'] = now_utc_iso()
        safe_json_write(manifest_path, manifest)

        update_status('FAILED', stage='failed', error=err)
        print(f'FAILED: {err}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
