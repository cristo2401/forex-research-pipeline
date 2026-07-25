#!/usr/bin/env python3
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
    top_k_accuracy_score,
)
from torch.utils.data import DataLoader, Dataset

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

ROOT = Path(__file__).resolve().parents[2]
IN_TOKENS_PQ = ROOT / "phases/phase06_context_regimes/output_context_words_GBPUSD_v6_4_hybrid_balanced_from_v5_1/GBPUSD_context_words_hybrid_balanced_v6_4.parquet"

OUT_DIR = ROOT / "phases/phase08_two_stage_detector/output_phase8_two_stage_GBPUSD"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_SAMPLES_CSV = OUT_DIR / "GBPUSD_phase8_samples.csv"
OUT_FOLD_METRICS_CSV = OUT_DIR / "GBPUSD_phase8_fold_metrics.csv"
OUT_HIST_CHANGE_CSV = OUT_DIR / "GBPUSD_phase8_history_change_detector.csv"
OUT_HIST_TOKEN_CSV = OUT_DIR / "GBPUSD_phase8_history_change_token.csv"
OUT_PRED_MODEL_CSV = OUT_DIR / "GBPUSD_phase8_predictions_model.csv"
OUT_PRED_BASE_CSV = OUT_DIR / "GBPUSD_phase8_predictions_baseline.csv"
OUT_MODEL_PT = OUT_DIR / "phase8_two_stage_models.pt"
OUT_SUMMARY_JSON = OUT_DIR / "GBPUSD_phase8_summary.json"


@dataclass
class Config:
    symbol: str = "GBPUSD"
    seq_len: int = 16

    # shared model
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.10

    # training
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 20
    early_stop_patience: int = 5
    min_lr: float = 1e-5
    num_workers: int = 0

    # change detector
    focal_gamma: float = 2.0
    token_change_weight: float = 2.0

    # fold splits (walk-forward)
    folds: Tuple[Tuple[float, float, float], ...] = (
        (0.55, 0.10, 0.10),
        (0.65, 0.10, 0.10),
        (0.75, 0.10, 0.10),
    )


CFG = Config()


class SeqChangeDataset(Dataset):
    def __init__(self, x: np.ndarray, y_change: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.long)
        self.y = torch.tensor(y_change, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class SeqTokenDataset(Dataset):
    def __init__(self, x: np.ndarray, y_token: np.ndarray, sample_weight: np.ndarray | None = None):
        self.x = torch.tensor(x, dtype=torch.long)
        self.y = torch.tensor(y_token, dtype=torch.long)
        if sample_weight is None:
            sample_weight = np.ones(len(y_token), dtype=np.float32)
        self.w = torch.tensor(sample_weight, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.w[idx]


class TokenTransformer(nn.Module):
    def __init__(self, n_vocab: int, cfg: Config, n_out: int):
        super().__init__()
        self.tok = nn.Embedding(n_vocab, cfg.d_model)
        self.pos = nn.Embedding(cfg.seq_len, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=4 * cfg.d_model,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.ln = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l = x.shape
        pos = torch.arange(l, device=x.device).unsqueeze(0).expand(b, l)
        z = self.tok(x) + self.pos(pos)
        mask = torch.triu(torch.ones(l, l, device=x.device), diagonal=1).bool()
        z = self.enc(z, mask=mask)
        z = self.ln(z[:, -1, :])
        return self.head(z)


def build_samples(df: pd.DataFrame, cfg: Config):
    ids = df["word_hybrid_balanced_id"].astype(int).to_numpy()
    ts = pd.to_datetime(df["end_ts"], errors="coerce").to_numpy(dtype="datetime64[ns]")
    id_to_token = {
        int(i): str(t)
        for i, t in zip(df["word_hybrid_balanced_id"], df["word_hybrid_balanced_token"])
    }

    x_rows: List[np.ndarray] = []
    y_rows: List[int] = []
    last_rows: List[int] = []
    ts_rows: List[np.datetime64] = []
    change_rows: List[int] = []

    for i in range(cfg.seq_len, len(df)):
        x = ids[i - cfg.seq_len : i]
        y = int(ids[i])
        last = int(x[-1])
        x_rows.append(x)
        y_rows.append(y)
        last_rows.append(last)
        ts_rows.append(ts[i])
        change_rows.append(1 if y != last else 0)

    x = np.asarray(x_rows, dtype=np.int64)
    y = np.asarray(y_rows, dtype=np.int64)
    last = np.asarray(last_rows, dtype=np.int64)
    ts_rows = np.asarray(ts_rows)
    is_change = np.asarray(change_rows, dtype=np.int64)

    n_vocab = int(ids.max()) + 1
    for i in range(n_vocab):
        if i not in id_to_token:
            id_to_token[i] = f"w{i:02d}"

    samples_df = pd.DataFrame(
        {
            "end_ts": pd.to_datetime(ts_rows),
            "last_token_id": last,
            "target_token_id": y,
            "is_change": is_change,
        }
    )
    samples_df["last_token"] = samples_df["last_token_id"].map(lambda z: id_to_token[int(z)])
    samples_df["target_token"] = samples_df["target_token_id"].map(lambda z: id_to_token[int(z)])
    samples_df.to_csv(OUT_SAMPLES_CSV, index=False)

    return x, y, last, ts_rows, is_change, n_vocab, id_to_token


def build_fold_indices(n: int, cfg: Config):
    idx = np.arange(n)
    folds = []
    for fold_id, (tr_r, va_r, te_r) in enumerate(cfg.folds, start=1):
        a = int(n * tr_r)
        b = int(n * (tr_r + va_r))
        c = int(n * (tr_r + va_r + te_r))
        if a < 100 or b - a < 50 or c - b < 50:
            continue
        folds.append((fold_id, idx[:a], idx[a:b], idx[b:c]))
    return folds


def class_weights(y_train: np.ndarray, n_vocab: int) -> torch.Tensor:
    cnt = np.bincount(y_train, minlength=n_vocab).astype(float)
    cnt = np.maximum(cnt, 1.0)
    w = cnt.sum() / (n_vocab * cnt)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def focal_bce_with_logits(logits, targets, pos_weight, gamma: float):
    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=pos_weight,
    )
    prob = torch.sigmoid(logits)
    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    mod = (1.0 - p_t).pow(gamma)
    return (mod * bce).mean()


def run_epoch_change(model, loader, device, cfg: Config, pos_weight=None, optimizer=None):
    train = optimizer is not None
    model.train(train)
    losses = []
    ys = []
    probs = []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            logits = model(xb).squeeze(-1)
            if pos_weight is None:
                loss = F.binary_cross_entropy_with_logits(logits, yb)
            else:
                loss = focal_bce_with_logits(logits, yb, pos_weight=pos_weight, gamma=cfg.focal_gamma)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        losses.append(float(loss.detach().cpu()))
        ys.append(yb.detach().cpu().numpy())
        probs.append(torch.sigmoid(logits).detach().cpu().numpy())

    y_true = np.concatenate(ys) if ys else np.array([], dtype=float)
    p_change = np.concatenate(probs) if probs else np.array([], dtype=float)
    return {
        "loss": float(np.mean(losses)) if losses else np.nan,
        "y_true": y_true,
        "p_change": p_change,
    }


def run_epoch_token(model, loader, class_weight, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    losses = []
    ys = []
    probs = []

    for xb, yb, wb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        wb = wb.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            logits = model(xb)
            loss_vec = F.cross_entropy(logits, yb, weight=class_weight, reduction="none")
            loss = (loss_vec * wb).mean()
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        losses.append(float(loss.detach().cpu()))
        ys.append(yb.detach().cpu().numpy())
        probs.append(torch.softmax(logits, dim=1).detach().cpu().numpy())

    y_true = np.concatenate(ys) if ys else np.array([], dtype=int)
    p_tok = np.concatenate(probs) if probs else np.zeros((0, 0), dtype=float)
    return {
        "loss": float(np.mean(losses)) if losses else np.nan,
        "y_true": y_true,
        "p_tok": p_tok,
    }


def tune_threshold_f1(y_true: np.ndarray, prob: np.ndarray):
    yb = y_true.astype(int)
    best = {"thr": 0.5, "f1": -1.0, "precision": np.nan, "recall": np.nan}
    for thr in np.linspace(0.05, 0.95, 19):
        pred = (prob >= thr).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(yb, pred, average="binary", zero_division=0)
        cand = {"thr": float(thr), "f1": float(f1), "precision": float(p), "recall": float(r)}
        if (cand["f1"] > best["f1"]) or (cand["f1"] == best["f1"] and cand["recall"] > best["recall"]):
            best = cand
    return best


def safe_binary_auc(y_true: np.ndarray, prob: np.ndarray):
    yb = y_true.astype(int)
    if len(np.unique(yb)) < 2:
        return np.nan, np.nan
    return float(roc_auc_score(yb, prob)), float(average_precision_score(yb, prob))


def topk_metrics(y_true: np.ndarray, probs: np.ndarray, last_token: np.ndarray, n_vocab: int) -> Dict[str, float]:
    labels = list(range(n_vocab))
    out = {}
    for k in [1, 3, 5]:
        kk = min(k, n_vocab)
        out[f"top{k}_all"] = float(top_k_accuracy_score(y_true, probs, k=kk, labels=labels))

    m = y_true != last_token
    out["n_change"] = int(m.sum())
    out["change_ratio"] = float(m.mean()) if len(m) else np.nan

    if m.any():
        for k in [1, 3, 5]:
            kk = min(k, n_vocab)
            out[f"top{k}_change"] = float(top_k_accuracy_score(y_true[m], probs[m], k=kk, labels=labels))
    else:
        out["top1_change"] = np.nan
        out["top3_change"] = np.nan
        out["top5_change"] = np.nan
    return out


def baseline_markov(last_train: np.ndarray, y_train: np.ndarray, last_eval: np.ndarray, n_vocab: int):
    cnt = np.ones((n_vocab, n_vocab), dtype=float)
    for a, b in zip(last_train, y_train):
        cnt[int(a), int(b)] += 1.0
    p = cnt / cnt.sum(axis=1, keepdims=True)
    return np.vstack([p[int(a)] for a in last_eval])


def predict_token_probs(model, x: np.ndarray, batch_size: int, device: torch.device):
    model.eval()
    out = []
    dl = DataLoader(
        SeqTokenDataset(x, np.zeros(len(x), dtype=np.int64), sample_weight=np.ones(len(x), dtype=np.float32)),
        batch_size=batch_size,
        shuffle=False,
    )
    with torch.no_grad():
        for xb, _, _ in dl:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            out.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.vstack(out)


def predict_change_probs(model, x: np.ndarray, batch_size: int, device: torch.device):
    model.eval()
    out = []
    dl = DataLoader(SeqChangeDataset(x, np.zeros(len(x), dtype=np.float32)), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for xb, _ in dl:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb).squeeze(-1)
            out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out)


def suppress_last_and_normalize(p_tok: np.ndarray, last_token: np.ndarray, n_vocab: int):
    p = p_tok.copy()
    idx = np.arange(len(p))
    p[idx, last_token] = 0.0
    s = p.sum(axis=1, keepdims=True)
    zero = (s[:, 0] <= 1e-12)
    if np.any(zero):
        p[zero, :] = 1.0 / max(n_vocab - 1, 1)
        p[zero, last_token[zero]] = 0.0
        s = p.sum(axis=1, keepdims=True)
    p = p / np.clip(s, 1e-12, None)
    return p


def combine_hard_gate(p_change: np.ndarray, p_tok_change: np.ndarray, last_token: np.ndarray, thr: float, n_vocab: int):
    n = len(p_change)
    out = np.zeros((n, n_vocab), dtype=float)
    gate = p_change >= thr
    idx = np.arange(n)

    out[~gate, last_token[~gate]] = 1.0
    out[gate] = p_tok_change[gate]

    # keep strict normalization
    s = out.sum(axis=1, keepdims=True)
    out = out / np.clip(s, 1e-12, None)
    return out, gate.astype(int)


def combine_soft_mixture(
    p_change: np.ndarray,
    p_tok_change: np.ndarray,
    last_token: np.ndarray,
    alpha: float,
    n_vocab: int,
):
    n = len(p_change)
    w = np.clip(alpha * p_change, 0.0, 1.0)
    out = p_tok_change * w[:, None]
    out[np.arange(n), last_token] += (1.0 - w)
    s = out.sum(axis=1, keepdims=True)
    out = out / np.clip(s, 1e-12, None)
    pred_change = (w >= 0.5).astype(int)
    return out, pred_change, w


def choose_final_alpha(
    y_val: np.ndarray,
    last_val: np.ndarray,
    p_change_val: np.ndarray,
    p_tok_change_val: np.ndarray,
    baseline_top1_all_val: float,
    n_vocab: int,
):
    best_feasible = {"alpha": 1.0, "score": -1.0, "top1_all": np.nan, "top3_change": np.nan}
    best_fallback = {"alpha": 1.0, "score": -1.0, "top1_all": np.nan, "top3_change": np.nan}
    min_top1 = max(0.0, baseline_top1_all_val - 0.05)

    for alpha in [0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0, 4.0]:
        p_final, _, _ = combine_soft_mixture(
            p_change_val, p_tok_change_val, last_val, float(alpha), n_vocab
        )
        m = topk_metrics(y_val, p_final, last_val, n_vocab)
        top1_all = m["top1_all"]
        top3_change = m.get("top3_change", np.nan)
        if np.isnan(top3_change):
            top3_change = 0.0
        # feasible path keeps global accuracy close to baseline while improving change ranking
        score_feasible = float(top3_change + 0.05 * top1_all)
        if top1_all >= min_top1 and score_feasible > best_feasible["score"]:
            best_feasible = {
                "alpha": float(alpha),
                "score": score_feasible,
                "top1_all": float(top1_all),
                "top3_change": float(top3_change),
            }

        # fallback if no feasible threshold exists
        score_fallback = float(top3_change + 0.02 * top1_all)
        if score_fallback > best_fallback["score"]:
            best_fallback = {
                "alpha": float(alpha),
                "score": score_fallback,
                "top1_all": float(top1_all),
                "top3_change": float(top3_change),
            }

    if best_feasible["score"] > -1.0:
        return best_feasible
    return best_fallback


def train_fold(
    fold_id: int,
    x: np.ndarray,
    y: np.ndarray,
    last: np.ndarray,
    ts: np.ndarray,
    is_change: np.ndarray,
    n_vocab: int,
    id_to_token: Dict[int, str],
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    te_idx: np.ndarray,
    cfg: Config,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # stage 1: change detector
    change_model = TokenTransformer(n_vocab=n_vocab, cfg=cfg, n_out=1).to(device)

    y_tr_change = is_change[tr_idx].astype(np.float32)
    y_va_change = is_change[va_idx].astype(np.float32)

    n_pos = float(y_tr_change.sum())
    n_neg = float(len(y_tr_change) - n_pos)
    pos_weight_val = max(n_neg / max(n_pos, 1.0), 1.0)
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)

    dl_tr_change = DataLoader(
        SeqChangeDataset(x[tr_idx], y_tr_change),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    dl_va_change = DataLoader(
        SeqChangeDataset(x[va_idx], y_va_change),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    opt_ch = torch.optim.AdamW(change_model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sch_ch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_ch, mode="max", factor=0.5, patience=2, min_lr=cfg.min_lr
    )

    hist_change = []
    best_state_change = None
    best_val_f1 = -1.0
    wait = 0

    for epoch in range(1, cfg.max_epochs + 1):
        tr = run_epoch_change(change_model, dl_tr_change, device, cfg, pos_weight=pos_weight, optimizer=opt_ch)
        va = run_epoch_change(change_model, dl_va_change, device, cfg, pos_weight=pos_weight, optimizer=None)
        thr_info = tune_threshold_f1(va["y_true"], va["p_change"])
        va_pred = (va["p_change"] >= thr_info["thr"]).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            va["y_true"].astype(int), va_pred, average="binary", zero_division=0
        )
        sch_ch.step(f1)

        hist_change.append(
            {
                "fold_id": fold_id,
                "epoch": epoch,
                "lr": float(opt_ch.param_groups[0]["lr"]),
                "train_loss": tr["loss"],
                "val_loss": va["loss"],
                "val_thr": float(thr_info["thr"]),
                "val_precision": float(p),
                "val_recall": float(r),
                "val_f1": float(f1),
            }
        )

        if f1 > best_val_f1 + 1e-4:
            best_val_f1 = f1
            wait = 0
            best_state_change = {k: v.detach().cpu().clone() for k, v in change_model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.early_stop_patience:
                break

    if best_state_change is not None:
        change_model.load_state_dict(best_state_change)

    # stage 2: token classifier over all samples, with extra weight on change events
    token_model = TokenTransformer(n_vocab=n_vocab, cfg=cfg, n_out=n_vocab).to(device)

    tr_token_idx = tr_idx
    va_token_idx = va_idx

    tr_token_w = 1.0 + cfg.token_change_weight * is_change[tr_token_idx].astype(np.float32)
    va_token_w = 1.0 + cfg.token_change_weight * is_change[va_token_idx].astype(np.float32)

    dl_tr_tok = DataLoader(
        SeqTokenDataset(x[tr_token_idx], y[tr_token_idx], sample_weight=tr_token_w),
        batch_size=min(cfg.batch_size, max(len(tr_token_idx), 1)),
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    dl_va_tok = DataLoader(
        SeqTokenDataset(x[va_token_idx], y[va_token_idx], sample_weight=va_token_w),
        batch_size=min(cfg.batch_size, max(len(va_token_idx), 1)),
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    ce_w = class_weights(y[tr_token_idx], n_vocab).to(device)
    opt_tok = torch.optim.AdamW(token_model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sch_tok = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_tok, mode="max", factor=0.5, patience=2, min_lr=cfg.min_lr
    )

    hist_token = []
    best_state_tok = None
    best_val_top3_change = -1.0
    wait = 0

    for epoch in range(1, cfg.max_epochs + 1):
        tr_tok = run_epoch_token(token_model, dl_tr_tok, ce_w, device, optimizer=opt_tok)
        va_tok = run_epoch_token(token_model, dl_va_tok, ce_w, device, optimizer=None)
        m_va = topk_metrics(y[va_token_idx], va_tok["p_tok"], last[va_token_idx], n_vocab)
        val_score = m_va["top3_change"] if not np.isnan(m_va["top3_change"]) else m_va["top3_all"]

        sch_tok.step(val_score)

        hist_token.append(
            {
                "fold_id": fold_id,
                "epoch": epoch,
                "lr": float(opt_tok.param_groups[0]["lr"]),
                "train_loss": tr_tok["loss"],
                "val_loss": float(va_tok["loss"]),
                "val_top3_change_only": float(val_score),
                "n_train_change": int((is_change[tr_token_idx] == 1).sum()),
                "n_val_change": int((is_change[va_token_idx] == 1).sum()),
            }
        )

        if val_score > best_val_top3_change + 1e-4:
            best_val_top3_change = val_score
            wait = 0
            best_state_tok = {k: v.detach().cpu().clone() for k, v in token_model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.early_stop_patience:
                break

    if best_state_tok is not None:
        token_model.load_state_dict(best_state_tok)

    # tune hard gate threshold on validation
    p_change_val = predict_change_probs(change_model, x[va_idx], batch_size=cfg.batch_size, device=device)
    p_change_test = predict_change_probs(change_model, x[te_idx], batch_size=cfg.batch_size, device=device)

    p_tok_val = predict_token_probs(token_model, x[va_idx], batch_size=cfg.batch_size, device=device)
    p_tok_test = predict_token_probs(token_model, x[te_idx], batch_size=cfg.batch_size, device=device)

    p_tok_val = suppress_last_and_normalize(p_tok_val, last[va_idx], n_vocab)
    p_tok_test = suppress_last_and_normalize(p_tok_test, last[te_idx], n_vocab)

    det_thr = tune_threshold_f1(is_change[va_idx], p_change_val)

    p_base_val = baseline_markov(last[tr_idx], y[tr_idx], last[va_idx], n_vocab)
    m_base_val = topk_metrics(y[va_idx], p_base_val, last[va_idx], n_vocab)

    final_alpha = choose_final_alpha(
        y_val=y[va_idx],
        last_val=last[va_idx],
        p_change_val=p_change_val,
        p_tok_change_val=p_tok_val,
        baseline_top1_all_val=float(m_base_val["top1_all"]),
        n_vocab=n_vocab,
    )

    p_final_test, det_pred, det_weight = combine_soft_mixture(
        p_change=p_change_test,
        p_tok_change=p_tok_test,
        last_token=last[te_idx],
        alpha=final_alpha["alpha"],
        n_vocab=n_vocab,
    )

    # baseline from Markov transition
    p_base_test = baseline_markov(last[tr_idx], y[tr_idx], last[te_idx], n_vocab)

    # token metrics
    m_model = topk_metrics(y[te_idx], p_final_test, last[te_idx], n_vocab)
    m_base = topk_metrics(y[te_idx], p_base_test, last[te_idx], n_vocab)

    # change detector metrics (model)
    det_true = is_change[te_idx].astype(int)
    det_pred = (p_change_test >= det_thr["thr"]).astype(int)
    det_p, det_r, det_f1, _ = precision_recall_fscore_support(det_true, det_pred, average="binary", zero_division=0)
    det_roc, det_pr = safe_binary_auc(det_true, p_change_test)

    # change detector baseline from markov "not staying" probability
    p_change_base = 1.0 - p_base_test[np.arange(len(te_idx)), last[te_idx]]
    base_thr = tune_threshold_f1(is_change[va_idx], 1.0 - p_base_val[np.arange(len(va_idx)), last[va_idx]])
    det_base = (p_change_base >= base_thr["thr"]).astype(int)
    b_p, b_r, b_f1, _ = precision_recall_fscore_support(det_true, det_base, average="binary", zero_division=0)

    # per-sample predictions
    topk = min(5, n_vocab)
    ord_model = np.argsort(-p_final_test, axis=1)[:, :topk]
    ord_base = np.argsort(-p_base_test, axis=1)[:, :topk]

    out_common = pd.DataFrame(
        {
            "fold_id": fold_id,
            "end_ts": pd.to_datetime(ts[te_idx]),
            "last_token_id": last[te_idx],
            "target_token_id": y[te_idx],
            "is_change": is_change[te_idx].astype(int),
        }
    )

    out_model = out_common.copy()
    out_model["p_change"] = p_change_test
    out_model["mix_weight"] = det_weight
    out_model["pred_change"] = det_pred
    out_model["detector_threshold"] = float(det_thr["thr"])
    out_model["mix_alpha"] = float(final_alpha["alpha"])

    out_base = out_common.copy()
    out_base["p_change_markov"] = p_change_base
    out_base["pred_change_markov"] = det_base
    out_base["markov_change_threshold"] = float(base_thr["thr"])

    for j in range(topk):
        out_model[f"pred_token_id_top{j+1}"] = ord_model[:, j]
        out_base[f"pred_token_id_top{j+1}"] = ord_base[:, j]

    for col in ["last_token_id", "target_token_id"] + [f"pred_token_id_top{j+1}" for j in range(topk)]:
        out_model[col.replace("_id", "")] = out_model[col].map(lambda z: id_to_token[int(z)])
        out_base[col.replace("_id", "")] = out_base[col].map(lambda z: id_to_token[int(z)])

    fold_metric = {
        "fold_id": fold_id,
        "n_train": int(len(tr_idx)),
        "n_val": int(len(va_idx)),
        "n_test": int(len(te_idx)),
        "n_train_change": int((is_change[tr_idx] == 1).sum()),
        "n_val_change": int((is_change[va_idx] == 1).sum()),
        "n_test_change": int((is_change[te_idx] == 1).sum()),
        "detector_threshold": float(det_thr["thr"]),
        "final_mix_alpha": float(final_alpha["alpha"]),
        "baseline_change_threshold": float(base_thr["thr"]),
        "baseline_top1_all": m_base["top1_all"],
        "baseline_top3_all": m_base["top3_all"],
        "baseline_top5_all": m_base["top5_all"],
        "baseline_top1_change": m_base["top1_change"],
        "baseline_top3_change": m_base["top3_change"],
        "baseline_top5_change": m_base["top5_change"],
        "model_top1_all": m_model["top1_all"],
        "model_top3_all": m_model["top3_all"],
        "model_top5_all": m_model["top5_all"],
        "model_top1_change": m_model["top1_change"],
        "model_top3_change": m_model["top3_change"],
        "model_top5_change": m_model["top5_change"],
        "model_detector_precision": float(det_p),
        "model_detector_recall": float(det_r),
        "model_detector_f1": float(det_f1),
        "model_detector_roc_auc": float(det_roc) if not np.isnan(det_roc) else np.nan,
        "model_detector_pr_auc": float(det_pr) if not np.isnan(det_pr) else np.nan,
        "baseline_detector_precision": float(b_p),
        "baseline_detector_recall": float(b_r),
        "baseline_detector_f1": float(b_f1),
    }

    model_bundle = {
        "change_state_dict": change_model.state_dict(),
        "token_state_dict": token_model.state_dict(),
    }

    return fold_metric, pd.DataFrame(hist_change), pd.DataFrame(hist_token), out_model, out_base, model_bundle


def main():
    df = pd.read_parquet(IN_TOKENS_PQ).copy()
    df["end_ts"] = pd.to_datetime(df["end_ts"], errors="coerce")
    df = df.sort_values(["end_ts", "base_window_id_end"]).reset_index(drop=True)

    x, y, last, ts, is_change, n_vocab, id_to_token = build_samples(df, CFG)
    folds = build_fold_indices(len(y), CFG)

    fold_rows = []
    hist_change_rows = []
    hist_token_rows = []
    pred_model_rows = []
    pred_base_rows = []
    last_bundle = None

    for fold_id, tr_idx, va_idx, te_idx in folds:
        fm, h_ch, h_tk, pmod, pbase, bundle = train_fold(
            fold_id=fold_id,
            x=x,
            y=y,
            last=last,
            ts=ts,
            is_change=is_change,
            n_vocab=n_vocab,
            id_to_token=id_to_token,
            tr_idx=tr_idx,
            va_idx=va_idx,
            te_idx=te_idx,
            cfg=CFG,
        )
        fold_rows.append(fm)
        hist_change_rows.append(h_ch)
        hist_token_rows.append(h_tk)
        pred_model_rows.append(pmod)
        pred_base_rows.append(pbase)
        last_bundle = bundle

    fold_df = pd.DataFrame(fold_rows)
    hist_change_df = pd.concat(hist_change_rows, ignore_index=True) if hist_change_rows else pd.DataFrame()
    hist_token_df = pd.concat(hist_token_rows, ignore_index=True) if hist_token_rows else pd.DataFrame()
    pred_model_df = pd.concat(pred_model_rows, ignore_index=True) if pred_model_rows else pd.DataFrame()
    pred_base_df = pd.concat(pred_base_rows, ignore_index=True) if pred_base_rows else pd.DataFrame()

    fold_df.to_csv(OUT_FOLD_METRICS_CSV, index=False)
    hist_change_df.to_csv(OUT_HIST_CHANGE_CSV, index=False)
    hist_token_df.to_csv(OUT_HIST_TOKEN_CSV, index=False)
    pred_model_df.to_csv(OUT_PRED_MODEL_CSV, index=False)
    pred_base_df.to_csv(OUT_PRED_BASE_CSV, index=False)

    if last_bundle is not None:
        torch.save(
            {
                "change_state_dict": last_bundle["change_state_dict"],
                "token_state_dict": last_bundle["token_state_dict"],
                "config": CFG.__dict__,
                "n_vocab": n_vocab,
                "id_to_token": id_to_token,
            },
            OUT_MODEL_PT,
        )

    metric_cols = [
        "baseline_top1_all",
        "baseline_top3_all",
        "baseline_top5_all",
        "baseline_top1_change",
        "baseline_top3_change",
        "baseline_top5_change",
        "baseline_detector_f1",
        "model_top1_all",
        "model_top3_all",
        "model_top5_all",
        "model_top1_change",
        "model_top3_change",
        "model_top5_change",
        "model_detector_precision",
        "model_detector_recall",
        "model_detector_f1",
        "model_detector_roc_auc",
        "model_detector_pr_auc",
    ]

    summary = {
        "symbol": CFG.symbol,
        "model_tag": "phase8_two_stage_change_detector_plus_next_token",
        "source": "phase6_v6_4_hybrid_balanced_tokens",
        "sequence_len": CFG.seq_len,
        "n_vocab": int(n_vocab),
        "n_samples_total": int(len(y)),
        "n_change_total": int(is_change.sum()),
        "change_ratio_total": float(is_change.mean()),
        "n_folds": int(len(fold_df)),
        "mean_metrics": {
            col: float(fold_df[col].mean())
            for col in metric_cols
            if col in fold_df.columns
        },
        "outputs": {
            "samples_csv": str(OUT_SAMPLES_CSV),
            "fold_metrics_csv": str(OUT_FOLD_METRICS_CSV),
            "history_change_csv": str(OUT_HIST_CHANGE_CSV),
            "history_token_csv": str(OUT_HIST_TOKEN_CSV),
            "predictions_model_csv": str(OUT_PRED_MODEL_CSV),
            "predictions_baseline_csv": str(OUT_PRED_BASE_CSV),
            "model_pt": str(OUT_MODEL_PT),
        },
    }

    OUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

    print("Saved summary:", OUT_SUMMARY_JSON)
    print("Mean metrics:", summary["mean_metrics"])
    return summary, fold_df, hist_change_df, hist_token_df, pred_model_df, pred_base_df


if __name__ == "__main__":
    main()
