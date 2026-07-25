#!/usr/bin/env python3
import json
from dataclasses import dataclass, asdict
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


@dataclass
class Config:
    symbol: str = 'GBPUSD'

    # phase9-like model
    max_seq_len: int = 64
    in_dim: int = 4
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.10

    batch_size_phase9: int = 256
    batch_size_infer: int = 1024
    num_workers: int = 0

    lr_phase9: float = 1e-3
    wd_phase9: float = 1e-4
    min_lr_phase9: float = 1e-5
    epochs_phase9: int = 10
    patience_phase9: int = 3

    # losses
    w_inf: float = 1.0
    w_sub: float = 1.1
    w_reg: float = 0.9
    pos_scale: float = 1.0
    label_smoothing_sub: float = 0.03

    # decision target and costs
    horizon_bars: int = 4
    min_move_pips: float = 4.0
    cost_pips: float = 1.2

    # walk-forward + leakage controls
    train_min_ratio: float = 0.45
    test_ratio: float = 0.10
    max_folds: int = 5
    calib_ratio: float = 0.15
    min_train_rows: int = 3000
    min_test_rows: int = 700
    embargo_bars: int = 16


CFG = Config()
ROOT = Path(__file__).resolve().parents[2]
PHASE9_DIR = ROOT / 'phases/phase09_multitask_patterns'
PHASE4_DIR = ROOT / 'phases/phase04_transfer_clustering'
OUT_DIR = ROOT / 'phases/phase10_decision_engine/output_phase10_1_leak_safe_GBPUSD'
OUT_DIR.mkdir(parents=True, exist_ok=True)

IN_SAMPLES_CSV = PHASE9_DIR / 'output_phase9_3_search_best_GBPUSD/GBPUSD_phase9_3_samples.csv'
IN_SEG_LONG = PHASE4_DIR / 'output_transfer_GBPUSD_full_v2/GBPUSD_segments_long.parquet'
IN_MODEL_PT = PHASE9_DIR / 'output_phase9_3_search_best_GBPUSD/phase9_3_best_model.pt'

OUT_MASTER_CSV = OUT_DIR / 'GBPUSD_phase10_1_master_dataset.csv'
OUT_FOLDS_CSV = OUT_DIR / 'GBPUSD_phase10_1_walkforward_folds.csv'
OUT_PHASE9_FOLD_CSV = OUT_DIR / 'GBPUSD_phase10_1_phase9_fold_metrics.csv'
OUT_METRICS_CSV = OUT_DIR / 'GBPUSD_phase10_1_walkforward_metrics.csv'
OUT_PRED_CSV = OUT_DIR / 'GBPUSD_phase10_1_fold_predictions.csv'
OUT_SUMMARY_JSON = OUT_DIR / 'GBPUSD_phase10_1_summary.json'
OUT_PNG_CUM = OUT_DIR / 'GBPUSD_phase10_1_equity_walkforward.png'
OUT_PNG_FOLD = OUT_DIR / 'GBPUSD_phase10_1_fold_sharpe_and_return.png'
OUT_PNG_ACTION = OUT_DIR / 'GBPUSD_phase10_1_action_distribution.png'


class PatternDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        lengths: np.ndarray,
        y_class: np.ndarray,
        y_inf: np.ndarray,
        y_sub: np.ndarray,
        y_prog: np.ndarray,
        reg_mask: np.ndarray,
    ):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.lengths = torch.tensor(lengths, dtype=torch.long)
        self.y_class = torch.tensor(y_class, dtype=torch.long)
        self.y_inf = torch.tensor(y_inf, dtype=torch.float32)
        self.y_sub = torch.tensor(y_sub, dtype=torch.long)
        self.y_prog = torch.tensor(y_prog, dtype=torch.float32)
        self.reg_mask = torch.tensor(reg_mask, dtype=torch.bool)

    def __len__(self):
        return len(self.y_class)

    def __getitem__(self, idx):
        return (
            self.x[idx],
            self.lengths[idx],
            self.y_class[idx],
            self.y_inf[idx],
            self.y_sub[idx],
            self.y_prog[idx],
            self.reg_mask[idx],
        )


class GatedMultiTaskTransformer(nn.Module):
    def __init__(self, cfg: Config, n_subclasses: int):
        super().__init__()
        self.in_proj = nn.Linear(cfg.in_dim, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_seq_len, cfg.d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=4 * cfg.d_model,
            dropout=cfg.dropout,
            batch_first=True,
            activation='gelu',
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.ln = nn.LayerNorm(cfg.d_model)

        self.inf_head = nn.Linear(cfg.d_model, 1)
        self.sub_head = nn.Linear(cfg.d_model, n_subclasses)
        self.reg_head = nn.Linear(cfg.d_model, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        b, l, _ = x.shape
        pos_ids = torch.arange(l, device=x.device).unsqueeze(0).expand(b, l)
        z = self.in_proj(x) + self.pos(pos_ids)

        left_pad = (l - lengths).unsqueeze(1)
        idx = torch.arange(l, device=x.device).unsqueeze(0)
        pad_mask = idx < left_pad

        h = self.enc(z, src_key_padding_mask=pad_mask)
        pooled = self.ln(h[:, -1, :])

        inf_logit = self.inf_head(pooled).squeeze(1)
        sub_logits = self.sub_head(pooled)
        prog = torch.sigmoid(self.reg_head(pooled).squeeze(1))
        return inf_logit, sub_logits, prog


def pip_size_for_symbol(symbol: str) -> float:
    return 0.01 if symbol.upper().endswith('JPY') else 0.0001


def decode_class(p_inf: np.ndarray, sub_logits: np.ndarray, thr: float) -> np.ndarray:
    sub_pred = sub_logits.argmax(axis=1)
    return np.where(p_inf >= thr, sub_pred + 1, 0).astype(np.int64)


def compute_phase9_metrics(
    y_true_class: np.ndarray,
    y_pred_class: np.ndarray,
    y_true_prog: np.ndarray,
    y_pred_prog: np.ndarray,
    reg_mask: np.ndarray,
    n_classes: int,
) -> Dict[str, float]:
    out = {}
    out['acc'] = float((y_true_class == y_pred_class).mean())
    labels_all = list(range(n_classes))
    out['f1_macro_all'] = float(f1_score(y_true_class, y_pred_class, labels=labels_all, average='macro', zero_division=0))

    inf = y_true_class > 0
    if inf.any():
        labels_inf = sorted(list(set(y_true_class[inf].tolist())))
        out['f1_macro_informative'] = float(f1_score(y_true_class[inf], y_pred_class[inf], labels=labels_inf, average='macro', zero_division=0))
    else:
        out['f1_macro_informative'] = np.nan

    out['recall_class0'] = float((y_pred_class[y_true_class == 0] == 0).mean()) if (y_true_class == 0).any() else np.nan

    m = reg_mask.astype(bool)
    if m.any():
        err = np.abs(y_true_prog[m] - y_pred_prog[m])
        out['mae_progress'] = float(err.mean())
        out['rmse_progress'] = float(np.sqrt(np.mean((y_true_prog[m] - y_pred_prog[m]) ** 2)))
        early = m & (y_true_prog < 0.5)
        out['mae_progress_early'] = float(np.abs(y_true_prog[early] - y_pred_prog[early]).mean()) if early.any() else np.nan
    else:
        out['mae_progress'] = np.nan
        out['rmse_progress'] = np.nan
        out['mae_progress_early'] = np.nan

    return out


def phase9_objective(m: Dict[str, float]) -> float:
    mae_early = float(m['mae_progress_early']) if not np.isnan(m['mae_progress_early']) else float(m['mae_progress'])
    return 1.0 * float(m['f1_macro_all']) + 0.3 * float(m['f1_macro_informative']) + 0.15 * float(m['recall_class0']) - 0.75 * mae_early


def sub_class_weights(y_sub_train: np.ndarray, y_inf_train: np.ndarray, n_subclasses: int) -> torch.Tensor:
    ys = y_sub_train[y_inf_train == 1]
    cnt = np.bincount(ys, minlength=n_subclasses).astype(float)
    cnt = np.maximum(cnt, 1.0)
    w = cnt.sum() / (n_subclasses * cnt)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def run_phase9_epoch(model, loader, pos_weight_inf, sub_weight, cfg: Config, device, optimizer=None):
    train = optimizer is not None
    model.train(train)

    y_true_class, p_inf_list, sub_logits_list = [], [], []
    y_true_prog, y_pred_prog, reg_masks = [], [], []

    losses = []

    for xb, lb, yb_class, yb_inf, yb_sub, yb_prog, yb_mask in loader:
        xb = xb.to(device, non_blocking=True)
        lb = lb.to(device, non_blocking=True)
        yb_class = yb_class.to(device, non_blocking=True)
        yb_inf = yb_inf.to(device, non_blocking=True)
        yb_sub = yb_sub.to(device, non_blocking=True)
        yb_prog = yb_prog.to(device, non_blocking=True)
        yb_mask = yb_mask.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            inf_logit, sub_logits, prog = model(xb, lb)

            inf_loss = F.binary_cross_entropy_with_logits(inf_logit, yb_inf, pos_weight=pos_weight_inf)
            if yb_mask.any():
                sub_loss = F.cross_entropy(
                    sub_logits[yb_mask],
                    yb_sub[yb_mask],
                    weight=sub_weight,
                    label_smoothing=cfg.label_smoothing_sub,
                )
                reg_loss = F.smooth_l1_loss(prog[yb_mask], yb_prog[yb_mask])
            else:
                sub_loss = torch.tensor(0.0, device=device)
                reg_loss = torch.tensor(0.0, device=device)

            loss = cfg.w_inf * inf_loss + cfg.w_sub * sub_loss + cfg.w_reg * reg_loss

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        losses.append(float(loss.detach().cpu()))

        y_true_class.append(yb_class.detach().cpu().numpy())
        p_inf_list.append(torch.sigmoid(inf_logit).detach().cpu().numpy())
        sub_logits_list.append(sub_logits.detach().cpu().numpy())
        y_true_prog.append(yb_prog.detach().cpu().numpy())
        y_pred_prog.append(prog.detach().cpu().numpy())
        reg_masks.append(yb_mask.detach().cpu().numpy().astype(np.int64))

    y_true_class = np.concatenate(y_true_class)
    p_inf = np.concatenate(p_inf_list)
    sub_logits = np.concatenate(sub_logits_list)
    y_true_prog = np.concatenate(y_true_prog)
    y_pred_prog = np.concatenate(y_pred_prog)
    reg_masks = np.concatenate(reg_masks)

    return {
        'loss': float(np.mean(losses)),
        'y_true_class': y_true_class,
        'p_inf': p_inf,
        'sub_logits': sub_logits,
        'y_true_prog': y_true_prog,
        'y_pred_prog': y_pred_prog,
        'reg_masks': reg_masks,
    }


def tune_phase9_threshold(v: Dict, n_classes: int) -> Tuple[float, Dict[str, float]]:
    best = None
    for thr in np.linspace(0.25, 0.75, 21):
        y_pred = decode_class(v['p_inf'], v['sub_logits'], float(thr))
        m = compute_phase9_metrics(v['y_true_class'], y_pred, v['y_true_prog'], v['y_pred_prog'], v['reg_masks'], n_classes)
        score = phase9_objective(m)
        row = {'thr': float(thr), 'score': float(score), **m}
        if best is None or row['score'] > best['score']:
            best = row
    return float(best['thr']), best


def train_phase9_fold(
    cfg: Config,
    x: np.ndarray,
    lengths: np.ndarray,
    y_class: np.ndarray,
    y_inf: np.ndarray,
    y_sub: np.ndarray,
    y_prog: np.ndarray,
    reg_mask: np.ndarray,
    train_idx: np.ndarray,
) -> Tuple[nn.Module, float, Dict[str, float]]:
    train_idx = np.asarray(train_idx, dtype=np.int64)
    val_n = max(500, int(0.15 * len(train_idx)))
    if val_n >= len(train_idx):
        val_n = max(100, len(train_idx) // 5)

    fit_idx = train_idx[:-val_n]
    val_idx = train_idx[-val_n:]

    n_subclasses = int(y_sub[y_inf == 1].max()) + 1
    n_classes = int(y_class.max()) + 1

    ds_fit = PatternDataset(x[fit_idx], lengths[fit_idx], y_class[fit_idx], y_inf[fit_idx], y_sub[fit_idx], y_prog[fit_idx], reg_mask[fit_idx])
    ds_val = PatternDataset(x[val_idx], lengths[val_idx], y_class[val_idx], y_inf[val_idx], y_sub[val_idx], y_prog[val_idx], reg_mask[val_idx])

    dl_fit = DataLoader(ds_fit, batch_size=cfg.batch_size_phase9, shuffle=True, num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())
    dl_val = DataLoader(ds_val, batch_size=cfg.batch_size_phase9, shuffle=False, num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GatedMultiTaskTransformer(cfg=cfg, n_subclasses=n_subclasses).to(device)

    n_pos = float(y_inf[fit_idx].sum())
    n_neg = float(len(fit_idx) - n_pos)
    base_pos_weight = max(n_neg / max(n_pos, 1.0), 1.0)
    pos_weight_inf = torch.tensor([base_pos_weight * cfg.pos_scale], dtype=torch.float32, device=device)
    sub_weight = sub_class_weights(y_sub[fit_idx], y_inf[fit_idx], n_subclasses).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr_phase9, weight_decay=cfg.wd_phase9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, min_lr=cfg.min_lr_phase9)

    best_score = -np.inf
    best_thr = 0.55
    best_state = None
    best_val_metrics = None
    wait = 0

    for _epoch in range(1, cfg.epochs_phase9 + 1):
        _ = run_phase9_epoch(model, dl_fit, pos_weight_inf, sub_weight, cfg, device, optimizer=optimizer)
        va = run_phase9_epoch(model, dl_val, pos_weight_inf, sub_weight, cfg, device, optimizer=None)

        thr, vm = tune_phase9_threshold(va, n_classes=n_classes)
        scheduler.step(float(vm['score']))

        if vm['score'] > best_score + 1e-5:
            best_score = float(vm['score'])
            best_thr = float(thr)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_val_metrics = vm
            wait = 0
        else:
            wait += 1
            if wait >= cfg.patience_phase9:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_thr, {'best_score': best_score, **(best_val_metrics or {})}


def infer_phase9_fold(model: nn.Module, cfg: Config, x: np.ndarray, lengths: np.ndarray, thr: float) -> Dict[str, np.ndarray]:
    model.eval()
    device = next(model.parameters()).device

    p_inf_all, pred_class_all, pred_prog_all = [], [], []
    sub_conf_all, sub_entropy_all = [], []

    with torch.no_grad():
        for i in range(0, len(x), cfg.batch_size_infer):
            xb = torch.tensor(x[i:i + cfg.batch_size_infer], dtype=torch.float32, device=device)
            lb = torch.tensor(lengths[i:i + cfg.batch_size_infer], dtype=torch.long, device=device)

            inf_logit, sub_logits, prog = model(xb, lb)
            p_inf = torch.sigmoid(inf_logit)
            p_sub = torch.softmax(sub_logits, dim=1)

            sub_conf, sub_idx = torch.max(p_sub, dim=1)
            entropy = -(p_sub * torch.log(torch.clamp(p_sub, min=1e-8))).sum(dim=1)
            y_pred = torch.where(p_inf >= thr, sub_idx + 1, torch.zeros_like(sub_idx))

            p_inf_all.append(p_inf.cpu().numpy())
            pred_class_all.append(y_pred.cpu().numpy())
            pred_prog_all.append(prog.cpu().numpy())
            sub_conf_all.append(sub_conf.cpu().numpy())
            sub_entropy_all.append(entropy.cpu().numpy())

    return {
        'pred_inf_prob': np.concatenate(p_inf_all).astype(np.float32),
        'pred_class': np.concatenate(pred_class_all).astype(np.int64),
        'pred_progress': np.concatenate(pred_prog_all).astype(np.float32),
        'pred_sub_conf': np.concatenate(sub_conf_all).astype(np.float32),
        'pred_sub_entropy': np.concatenate(sub_entropy_all).astype(np.float32),
    }


def class_weights(y: np.ndarray) -> Dict[int, float]:
    uniq, cnt = np.unique(y, return_counts=True)
    d = {int(u): float(c) for u, c in zip(uniq, cnt)}
    total = float(sum(d.values()))
    n_cls = len(d)
    return {k: total / (n_cls * v) for k, v in d.items()}


def apply_policy(proba: np.ndarray, conf_thr: float, inf_prob: np.ndarray, inf_min: float) -> np.ndarray:
    pred_idx = np.argmax(proba, axis=1)
    conf = np.max(proba, axis=1)
    action = np.where(pred_idx == 0, -1, np.where(pred_idx == 2, 1, 0))
    action = np.where((conf >= conf_thr) & (inf_prob >= inf_min), action, 0)
    return action.astype(int)


def evaluate_trading(actions: np.ndarray, future_ret: np.ndarray, cost_ret: float) -> Dict[str, float]:
    gross = actions * future_ret
    trade = (actions != 0).astype(np.float32)
    net = gross - trade * cost_ret

    eps = 1e-12
    avg = float(np.mean(net))
    std = float(np.std(net))
    sharpe = float((avg / (std + eps)) * np.sqrt(24.0 * 252.0))

    traded = trade > 0
    win_rate = float((net[traded] > 0).mean()) if traded.any() else 0.0

    return {
        'avg_net_return': avg,
        'cum_net_return': float(np.sum(net)),
        'std_net_return': std,
        'sharpe_ann': sharpe,
        'trade_rate': float(trade.mean()),
        'win_rate_trades': win_rate,
    }


def tune_policy_thresholds(calib_df: pd.DataFrame, proba: np.ndarray, cfg: Config) -> Tuple[float, float, Dict[str, float]]:
    pip = pip_size_for_symbol(cfg.symbol)
    cost_ret = cfg.cost_pips * pip

    best = None
    for conf_thr in np.linspace(0.40, 0.75, 8):
        for inf_min in np.linspace(0.30, 0.70, 9):
            a = apply_policy(proba, float(conf_thr), calib_df['pred_inf_prob'].to_numpy(), float(inf_min))
            m = evaluate_trading(a, calib_df['future_return_h'].to_numpy(), cost_ret)
            score = m['avg_net_return'] * 1e4 + 0.12 * m['sharpe_ann']
            row = {'conf_thr': float(conf_thr), 'inf_min': float(inf_min), 'score': float(score), **m}
            if best is None or row['score'] > best['score']:
                best = row

    return float(best['conf_thr']), float(best['inf_min']), best


def make_walkforward_folds(df: pd.DataFrame, cfg: Config) -> List[Dict]:
    n = len(df)
    train_min = int(n * cfg.train_min_ratio)
    test_size = int(n * cfg.test_ratio)

    folds = []
    for k in range(cfg.max_folds):
        train_end = train_min + k * test_size
        test_start = train_end + cfg.embargo_bars
        test_end = min(test_start + test_size, n)

        if test_end - test_start < cfg.min_test_rows:
            break

        purge_end = max(cfg.min_train_rows, train_end - cfg.horizon_bars - cfg.embargo_bars)
        if purge_end < cfg.min_train_rows:
            continue

        folds.append(
            {
                'fold': k + 1,
                'train_start': 0,
                'train_end': purge_end,
                'test_start': test_start,
                'test_end': test_end,
            }
        )

    return folds


def load_master(cfg: Config) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    samples = pd.read_csv(IN_SAMPLES_CSV)
    samples['timestamp'] = pd.to_datetime(samples['timestamp'], errors='coerce')

    seg_long = pd.read_parquet(IN_SEG_LONG)
    seg_long['timestamp'] = pd.to_datetime(seg_long['timestamp'], errors='coerce')

    # Build sequence tensor input like phase9
    seg_long = seg_long.sort_values(['global_segment_id', 'step'])
    norm_by_segment: Dict[str, np.ndarray] = {}
    for seg_id, g in seg_long.groupby('global_segment_id', sort=False):
        arr = g[['open', 'high', 'low', 'close']].to_numpy(dtype=np.float32)
        base = float(arr[0, 3])
        base = base if abs(base) > 1e-8 else 1.0
        norm_by_segment[seg_id] = (arr / base) - 1.0

    x = np.zeros((len(samples), cfg.max_seq_len, cfg.in_dim), dtype=np.float32)
    lengths = np.ones(len(samples), dtype=np.int64)

    for i, row in samples[['global_segment_id', 'step']].iterrows():
        seg_id = row['global_segment_id']
        step = int(row['step'])
        arr = norm_by_segment[seg_id]

        start = max(0, step - cfg.max_seq_len + 1)
        seq = arr[start:step + 1]
        ln = seq.shape[0]

        x[i, -ln:] = seq
        lengths[i] = ln

    # Merge OHLC and make market features leak-safe by segment group
    ohlc = seg_long[['global_segment_id', 'step', 'open', 'high', 'low', 'close']].copy()
    m = samples.merge(ohlc, on=['global_segment_id', 'step'], how='left')
    m = m.sort_values(['global_segment_id', 'step']).reset_index(drop=True)
    m['master_idx'] = np.arange(len(m), dtype=np.int64)

    g = m.groupby('global_segment_id', group_keys=False)
    ret_1 = g['close'].pct_change()
    m['ret_1'] = ret_1.fillna(0.0)
    m['ret_2'] = g['close'].pct_change(2).fillna(0.0)
    m['ret_4'] = g['close'].pct_change(4).fillna(0.0)
    m['vol_4'] = (
        ret_1.groupby(m['global_segment_id'])
        .rolling(4, min_periods=1)
        .std()
        .reset_index(level=0, drop=True)
        .fillna(0.0)
    )

    m['range_1'] = (m['high'] / m['low'] - 1.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    m['body_1'] = ((m['close'] - m['open']) / m['open'].replace(0.0, np.nan)).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    seg_len = g['step'].transform('max') + 1
    m['segment_len'] = seg_len.astype(np.float32)
    m['step_frac'] = (m['step'] / np.maximum(seg_len - 1, 1)).astype(np.float32)

    # decision target
    m['close_fwd'] = g['close'].shift(-cfg.horizon_bars)
    m['future_return_h'] = (m['close_fwd'] / m['close']) - 1.0

    pip = pip_size_for_symbol(cfg.symbol)
    m['future_pips_h'] = m['future_return_h'] / pip
    thr = cfg.min_move_pips + cfg.cost_pips

    m['y_action'] = 0
    m.loc[m['future_pips_h'] > thr, 'y_action'] = 1
    m.loc[m['future_pips_h'] < -thr, 'y_action'] = -1
    m['y_action_idx'] = m['y_action'].map({-1: 0, 0: 1, 1: 2}).astype('Int64')

    # drop rows without forward target
    m = m[np.isfinite(m['future_return_h'])].copy()
    m['y_action_idx'] = m['y_action_idx'].astype(int)

    # sort by time for walk-forward
    m = m.sort_values('timestamp').reset_index(drop=True)

    return m, x, lengths


def run_leak_safe_phase10_1(cfg: Config):
    master, x_all, lengths_all = load_master(cfg)

    # labels aligned to original samples index (same indexing as x_all/lengths_all)
    samples_full = pd.read_csv(IN_SAMPLES_CSV)
    y_class_all = samples_full['y_class'].to_numpy(dtype=np.int64)
    y_inf_all = samples_full['y_inf'].to_numpy(dtype=np.int64)
    y_sub_all = samples_full['y_sub'].to_numpy(dtype=np.int64)
    y_prog_all = samples_full['y_progress'].to_numpy(dtype=np.float32)
    reg_mask_all = samples_full['reg_mask'].to_numpy(dtype=np.int64)

    folds = make_walkforward_folds(master, cfg)
    if not folds:
        raise RuntimeError('No valid leak-safe folds.')

    folds_df = pd.DataFrame(folds)
    phase9_rows = []
    metrics_rows = []
    pred_rows = []

    feature_cols = [
        'pred_inf_prob', 'pred_progress', 'pred_sub_conf', 'pred_sub_entropy', 'pred_class',
        'ret_1', 'ret_2', 'ret_4', 'vol_4', 'range_1', 'body_1', 'segment_len', 'step_frac',
    ]

    pip = pip_size_for_symbol(cfg.symbol)
    cost_ret = cfg.cost_pips * pip

    for fold in folds:
        tr = master.iloc[fold['train_start']:fold['train_end']].copy()
        te = master.iloc[fold['test_start']:fold['test_end']].copy()

        train_master_idx = tr['master_idx'].to_numpy(dtype=np.int64)

        # Train phase9-like model only on this fold's past
        model9, thr9, m9 = train_phase9_fold(
            cfg=cfg,
            x=x_all,
            lengths=lengths_all,
            y_class=y_class_all,
            y_inf=y_inf_all,
            y_sub=y_sub_all,
            y_prog=y_prog_all,
            reg_mask=reg_mask_all,
            train_idx=train_master_idx,
        )

        # infer phase9 signals for this fold train and test (no future leakage)
        tr_idx = tr['master_idx'].to_numpy(dtype=np.int64)
        te_idx = te['master_idx'].to_numpy(dtype=np.int64)

        tr_sig = infer_phase9_fold(model9, cfg, x_all[tr_idx], lengths_all[tr_idx], thr9)
        te_sig = infer_phase9_fold(model9, cfg, x_all[te_idx], lengths_all[te_idx], thr9)

        for k in tr_sig:
            tr[k] = tr_sig[k]
            te[k] = te_sig[k]

        # decision model split: fit/calib from train
        calib_n = max(300, int(len(tr) * cfg.calib_ratio))
        if calib_n >= len(tr):
            calib_n = max(100, len(tr) // 5)

        fit = tr.iloc[:-calib_n].copy()
        calib = tr.iloc[-calib_n:].copy()

        x_fit = fit[feature_cols].to_numpy(dtype=np.float32)
        y_fit = fit['y_action_idx'].to_numpy(dtype=np.int64)
        cw = class_weights(y_fit)
        sw = np.array([cw[int(v)] for v in y_fit], dtype=np.float32)

        clf = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=6,
            max_iter=280,
            min_samples_leaf=80,
            l2_regularization=1e-3,
            random_state=SEED,
        )
        clf.fit(x_fit, y_fit, sample_weight=sw)

        x_cal = calib[feature_cols].to_numpy(dtype=np.float32)
        p_cal = clf.predict_proba(x_cal)
        conf_thr, inf_min, cal_best = tune_policy_thresholds(calib, p_cal, cfg)

        # retrain on full train
        x_tr = tr[feature_cols].to_numpy(dtype=np.float32)
        y_tr = tr['y_action_idx'].to_numpy(dtype=np.int64)
        cw2 = class_weights(y_tr)
        sw2 = np.array([cw2[int(v)] for v in y_tr], dtype=np.float32)
        clf.fit(x_tr, y_tr, sample_weight=sw2)

        x_te = te[feature_cols].to_numpy(dtype=np.float32)
        p_te = clf.predict_proba(x_te)
        a_te = apply_policy(p_te, conf_thr, te['pred_inf_prob'].to_numpy(), inf_min)

        y_true = te['y_action'].to_numpy(dtype=np.int64)
        acc = float((y_true == a_te).mean())
        f1m = float(f1_score(y_true, a_te, labels=[-1, 0, 1], average='macro', zero_division=0))

        trade_m = evaluate_trading(a_te, te['future_return_h'].to_numpy(), cost_ret)

        metrics_rows.append(
            {
                'fold': int(fold['fold']),
                'train_rows': int(len(tr)),
                'test_rows': int(len(te)),
                'timestamp_test_start': str(te['timestamp'].iloc[0]),
                'timestamp_test_end': str(te['timestamp'].iloc[-1]),
                'phase9_thr': float(thr9),
                'policy_conf_thr': float(conf_thr),
                'policy_inf_min': float(inf_min),
                'policy_calib_score': float(cal_best['score']),
                'acc_action_3': acc,
                'f1_macro_action_3': f1m,
                **trade_m,
            }
        )

        phase9_rows.append({'fold': int(fold['fold']), **m9})

        te_out = te[[
            'global_segment_id', 'timestamp', 'step', 'y_action', 'future_return_h', 'future_pips_h',
            'pred_inf_prob', 'pred_progress', 'pred_sub_conf', 'pred_sub_entropy', 'pred_class'
        ]].copy()
        te_out['fold'] = int(fold['fold'])
        te_out['pred_action'] = a_te.astype(int)
        te_out['pred_conf_max'] = np.max(p_te, axis=1)
        te_out['p_short'] = p_te[:, 0]
        te_out['p_flat'] = p_te[:, 1]
        te_out['p_long'] = p_te[:, 2]
        te_out['net_return'] = te_out['pred_action'] * te_out['future_return_h'] - (te_out['pred_action'] != 0).astype(float) * cost_ret
        pred_rows.append(te_out)

    phase9_df = pd.DataFrame(phase9_rows)
    metrics_df = pd.DataFrame(metrics_rows)
    preds_df = pd.concat(pred_rows, axis=0, ignore_index=True)

    return master, folds_df, phase9_df, metrics_df, preds_df


def make_plots(metrics_df: pd.DataFrame, preds_df: pd.DataFrame):
    p = preds_df.sort_values(['timestamp', 'fold']).copy()
    p['cum_net'] = p['net_return'].cumsum()

    plt.figure(figsize=(11, 4))
    plt.plot(pd.to_datetime(p['timestamp']), p['cum_net'], linewidth=1.5)
    plt.title('Phase10.1 Leak-safe - Cumulative Net Return (walk-forward test)')
    plt.xlabel('timestamp')
    plt.ylabel('cumulative return')
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_PNG_CUM, dpi=140)
    plt.close()

    fig, ax1 = plt.subplots(figsize=(10, 4))
    x = metrics_df['fold'].astype(int).to_numpy()
    ax1.bar(x - 0.15, metrics_df['cum_net_return'], width=0.3)
    ax1.set_xlabel('fold')
    ax1.set_ylabel('cum net return')

    ax2 = ax1.twinx()
    ax2.plot(x + 0.15, metrics_df['sharpe_ann'], marker='o', linewidth=1.6, color='tab:red')
    ax2.set_ylabel('sharpe_ann')

    ax1.set_title('Phase10.1 Leak-safe - Fold Returns and Sharpe')
    ax1.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT_PNG_FOLD, dpi=140)
    plt.close(fig)

    vc = preds_df['pred_action'].value_counts().sort_index()
    labels = [str(i) for i in vc.index.tolist()]
    plt.figure(figsize=(6.5, 4))
    plt.bar(labels, vc.values)
    plt.title('Phase10.1 Leak-safe - Predicted Action Distribution')
    plt.xlabel('action (-1 short, 0 flat, 1 long)')
    plt.ylabel('count')
    plt.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_PNG_ACTION, dpi=140)
    plt.close()


def main():
    model_blob = torch.load(IN_MODEL_PT, map_location='cpu')
    cfg_dict = model_blob.get('config', {})
    CFG.d_model = int(cfg_dict.get('d_model', CFG.d_model))
    CFG.n_heads = int(cfg_dict.get('n_heads', CFG.n_heads))
    CFG.n_layers = int(cfg_dict.get('n_layers', CFG.n_layers))
    CFG.dropout = float(cfg_dict.get('dropout', CFG.dropout))

    master, folds_df, phase9_df, metrics_df, preds_df = run_leak_safe_phase10_1(CFG)

    master.to_csv(OUT_MASTER_CSV, index=False)
    folds_df.to_csv(OUT_FOLDS_CSV, index=False)
    phase9_df.to_csv(OUT_PHASE9_FOLD_CSV, index=False)
    metrics_df.to_csv(OUT_METRICS_CSV, index=False)
    preds_df.to_csv(OUT_PRED_CSV, index=False)

    make_plots(metrics_df, preds_df)

    summary = {
        'symbol': CFG.symbol,
        'model_tag': 'phase10_1_decision_pre_rl_leak_safe',
        'description': 'Leak-safe walk-forward: phase9 model retrained per fold on past only + policy model',
        'config': asdict(CFG),
        'dataset': {
            'n_rows': int(len(master)),
            'date_min': str(master['timestamp'].min()),
            'date_max': str(master['timestamp'].max()),
            'target_action_dist': master['y_action'].value_counts().sort_index().to_dict(),
        },
        'walkforward': {
            'n_folds': int(len(metrics_df)),
            'mean_metrics': metrics_df.mean(numeric_only=True).to_dict(),
            'std_metrics': metrics_df.std(numeric_only=True).to_dict(),
        },
        'phase9_per_fold': phase9_df.to_dict(orient='records'),
        'outputs': {
            'master_csv': str(OUT_MASTER_CSV),
            'folds_csv': str(OUT_FOLDS_CSV),
            'phase9_fold_metrics_csv': str(OUT_PHASE9_FOLD_CSV),
            'walkforward_metrics_csv': str(OUT_METRICS_CSV),
            'predictions_csv': str(OUT_PRED_CSV),
            'plot_cumulative_png': str(OUT_PNG_CUM),
            'plot_fold_png': str(OUT_PNG_FOLD),
            'plot_action_png': str(OUT_PNG_ACTION),
        },
    }

    OUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

    print('Saved summary:', OUT_SUMMARY_JSON)
    print('Walk-forward mean metrics:')
    print(metrics_df.mean(numeric_only=True).to_dict())

    return summary, metrics_df


if __name__ == '__main__':
    main()
