#!/usr/bin/env python3
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, top_k_accuracy_score
from torch.utils.data import DataLoader, Dataset

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

ROOT = Path(__file__).resolve().parents[2]
IN_TOKENS_PQ = ROOT / "phases/phase06_context_regimes/output_context_words_GBPUSD_v6_2_balanced_from_v5_1/GBPUSD_context_words_windows_v6_2.parquet"

OUT_DIR = ROOT / "phases/phase07_online_detector/output_next_token_GBPUSD_v3_change"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_SAMPLES_CSV = OUT_DIR / "GBPUSD_next_token_samples_v3.csv"
OUT_FOLD_METRICS_CSV = OUT_DIR / "GBPUSD_next_token_fold_metrics_v3.csv"
OUT_PRED_MODEL_CSV = OUT_DIR / "GBPUSD_next_token_fold_predictions_transformer_v3.csv"
OUT_PRED_BASE_CSV = OUT_DIR / "GBPUSD_next_token_fold_predictions_baseline_v3.csv"
OUT_HISTORY_CSV = OUT_DIR / "GBPUSD_next_token_train_history_v3.csv"
OUT_MODEL_PT = OUT_DIR / "next_token_transformer_v3.pt"
OUT_SUMMARY_JSON = OUT_DIR / "GBPUSD_next_token_summary_v3.json"


@dataclass
class Config:
    symbol: str = "GBPUSD"
    seq_len: int = 64
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 18
    early_stop_patience: int = 4
    min_lr: float = 1e-5
    num_workers: int = 0
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.10
    # change-aware objective
    change_weight_alpha: float = 3.0
    lambda_change_head: float = 0.7
    # walk-forward folds: (train_ratio, val_ratio, test_ratio)
    folds: Tuple[Tuple[float, float, float], ...] = (
        (0.55, 0.10, 0.10),
        (0.65, 0.10, 0.10),
        (0.75, 0.10, 0.10),
    )


CFG = Config()


class SeqDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, c: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)
        self.c = torch.tensor(c, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.c[idx]


class NextTokenChangeTransformer(nn.Module):
    def __init__(self, n_vocab: int, cfg: Config):
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
        self.head_token = nn.Linear(cfg.d_model, n_vocab)
        self.head_change = nn.Linear(cfg.d_model, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, l = x.shape
        pos = torch.arange(l, device=x.device).unsqueeze(0).expand(b, l)
        z = self.tok(x) + self.pos(pos)
        mask = torch.triu(torch.ones(l, l, device=x.device), diagonal=1).bool()
        z = self.enc(z, mask=mask)
        z = self.ln(z[:, -1, :])
        logits_token = self.head_token(z)
        logit_change = self.head_change(z).squeeze(-1)
        return logits_token, logit_change


def build_samples(df: pd.DataFrame, cfg: Config):
    ids = df["word_id"].astype(int).to_numpy()
    tokens = df["word_token"].astype(str).to_numpy()
    ts = pd.to_datetime(df["end_ts"], errors="coerce").to_numpy(dtype="datetime64[ns]")

    x_rows: List[np.ndarray] = []
    y_rows: List[int] = []
    last_rows: List[int] = []
    is_change_rows: List[int] = []
    ts_rows: List[np.datetime64] = []

    for i in range(cfg.seq_len, len(df)):
        x = ids[i - cfg.seq_len : i]
        y = int(ids[i])
        last = int(x[-1])
        x_rows.append(x)
        y_rows.append(y)
        last_rows.append(last)
        is_change_rows.append(1 if y != last else 0)
        ts_rows.append(ts[i])

    x = np.asarray(x_rows, dtype=np.int64)
    y = np.asarray(y_rows, dtype=np.int64)
    last = np.asarray(last_rows, dtype=np.int64)
    is_change = np.asarray(is_change_rows, dtype=np.int64)
    ts_rows = np.asarray(ts_rows)

    n_vocab = int(ids.max()) + 1
    id_to_token = {int(i): t for i, t in zip(df["word_id"], df["word_token"])}
    for i in range(n_vocab):
        if i not in id_to_token:
            id_to_token[i] = f"w{i:02d}"

    out = pd.DataFrame(
        {
            "end_ts": pd.to_datetime(ts_rows),
            "last_token_id": last,
            "target_token_id": y,
            "is_change": is_change,
        }
    )
    out["last_token"] = out["last_token_id"].map(lambda z: id_to_token[int(z)])
    out["target_token"] = out["target_token_id"].map(lambda z: id_to_token[int(z)])
    out.to_csv(OUT_SAMPLES_CSV, index=False)
    return x, y, last, is_change, ts_rows, n_vocab, id_to_token


def build_fold_indices(n: int, cfg: Config):
    folds = []
    idx = np.arange(n)
    for fold_id, (tr_r, va_r, te_r) in enumerate(cfg.folds, start=1):
        a = int(n * tr_r)
        b = int(n * (tr_r + va_r))
        c = int(n * (tr_r + va_r + te_r))
        if a < 100 or b - a < 50 or c - b < 50:
            continue
        tr_idx = idx[:a]
        va_idx = idx[a:b]
        te_idx = idx[b:c]
        folds.append((fold_id, tr_idx, va_idx, te_idx))
    return folds


def class_weights(y_train: np.ndarray, n_vocab: int) -> torch.Tensor:
    cnt = np.bincount(y_train, minlength=n_vocab).astype(float)
    cnt = np.maximum(cnt, 1.0)
    w = cnt.sum() / (n_vocab * cnt)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def run_epoch(model, loader, ce_weight, bce, device, cfg: Config, optimizer=None):
    train = optimizer is not None
    model.train(train)
    losses = []
    losses_token = []
    losses_change = []
    ys = []
    probs = []
    cs = []
    prob_change = []

    for xb, yb, cb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        cb = cb.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            logits_token, logit_change = model(xb)
            # token CE with per-sample change weighting
            ce = nn.functional.cross_entropy(logits_token, yb, weight=ce_weight, reduction="none")
            sample_w = 1.0 + cfg.change_weight_alpha * cb
            loss_token = torch.mean(ce * sample_w)
            loss_change = bce(logit_change, cb)
            loss = loss_token + cfg.lambda_change_head * loss_change

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        losses.append(float(loss.detach().cpu()))
        losses_token.append(float(loss_token.detach().cpu()))
        losses_change.append(float(loss_change.detach().cpu()))
        ys.append(yb.detach().cpu().numpy())
        probs.append(torch.softmax(logits_token, dim=1).detach().cpu().numpy())
        cs.append(cb.detach().cpu().numpy())
        prob_change.append(torch.sigmoid(logit_change).detach().cpu().numpy())

    y_true = np.concatenate(ys) if ys else np.array([], dtype=int)
    p_tok = np.concatenate(probs) if probs else np.zeros((0, 0), dtype=float)
    c_true = np.concatenate(cs) if cs else np.array([], dtype=float)
    p_change = np.concatenate(prob_change) if prob_change else np.array([], dtype=float)

    return {
        "loss": float(np.mean(losses)) if losses else np.nan,
        "loss_token": float(np.mean(losses_token)) if losses_token else np.nan,
        "loss_change": float(np.mean(losses_change)) if losses_change else np.nan,
        "y_true": y_true,
        "p_tok": p_tok,
        "c_true": c_true,
        "p_change": p_change,
    }


def topk_metrics(y_true: np.ndarray, probs: np.ndarray, last_token: np.ndarray, n_vocab: int):
    labels = list(range(n_vocab))
    out = {}
    for k in [1, 3, 5]:
        kk = min(k, n_vocab)
        out[f"top{k}_all"] = float(top_k_accuracy_score(y_true, probs, k=kk, labels=labels))

    m_change = y_true != last_token
    out["n_change"] = int(m_change.sum())
    out["change_ratio"] = float(m_change.mean()) if len(m_change) else np.nan
    if m_change.any():
        for k in [1, 3, 5]:
            kk = min(k, n_vocab)
            out[f"top{k}_change"] = float(top_k_accuracy_score(y_true[m_change], probs[m_change], k=kk, labels=labels))
    else:
        out["top1_change"] = np.nan
        out["top3_change"] = np.nan
        out["top5_change"] = np.nan
    return out


def change_head_metrics(c_true: np.ndarray, p_change: np.ndarray):
    pred = (p_change >= 0.5).astype(int)
    return {
        "change_precision": float(precision_score(c_true, pred, zero_division=0)),
        "change_recall": float(recall_score(c_true, pred, zero_division=0)),
        "change_f1": float(f1_score(c_true, pred, zero_division=0)),
    }


def baseline_markov(last_train: np.ndarray, y_train: np.ndarray, last_test: np.ndarray, n_vocab: int):
    cnt = np.ones((n_vocab, n_vocab), dtype=float)
    for a, b in zip(last_train, y_train):
        cnt[int(a), int(b)] += 1.0
    p = cnt / cnt.sum(axis=1, keepdims=True)
    probs = np.vstack([p[int(a)] for a in last_test])
    return probs


def train_fold(
    fold_id: int,
    x: np.ndarray,
    y: np.ndarray,
    last_token: np.ndarray,
    is_change: np.ndarray,
    ts: np.ndarray,
    n_vocab: int,
    id_to_token: Dict[int, str],
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    te_idx: np.ndarray,
    cfg: Config,
):
    ds_train = SeqDataset(x[tr_idx], y[tr_idx], is_change[tr_idx])
    ds_val = SeqDataset(x[va_idx], y[va_idx], is_change[va_idx])
    ds_test = SeqDataset(x[te_idx], y[te_idx], is_change[te_idx])

    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())
    dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())
    dl_test = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NextTokenChangeTransformer(n_vocab=n_vocab, cfg=cfg).to(device)
    ce_weight = class_weights(y[tr_idx], n_vocab).to(device)
    bce = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=cfg.min_lr)

    history_rows = []
    best_state = None
    best_val = np.inf
    wait = 0

    for epoch in range(1, cfg.max_epochs + 1):
        tr = run_epoch(model, dl_train, ce_weight, bce, device, cfg, optimizer=optimizer)
        va = run_epoch(model, dl_val, ce_weight, bce, device, cfg, optimizer=None)
        scheduler.step(va["loss"])

        mtr = topk_metrics(tr["y_true"], tr["p_tok"], last_token[tr_idx], n_vocab)
        mva = topk_metrics(va["y_true"], va["p_tok"], last_token[va_idx], n_vocab)
        cva = change_head_metrics(va["c_true"], va["p_change"])
        history_rows.append(
            {
                "fold_id": fold_id,
                "epoch": epoch,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "train_loss": tr["loss"],
                "val_loss": va["loss"],
                "train_top3_change": mtr["top3_change"],
                "val_top3_change": mva["top3_change"],
                "val_change_f1": cva["change_f1"],
            }
        )

        if va["loss"] < best_val - 1e-4:
            best_val = va["loss"]
            wait = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.early_stop_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    te = run_epoch(model, dl_test, ce_weight, bce, device, cfg, optimizer=None)
    m_te = topk_metrics(te["y_true"], te["p_tok"], last_token[te_idx], n_vocab)
    c_te = change_head_metrics(te["c_true"], te["p_change"])

    probs_base = baseline_markov(last_token[tr_idx], y[tr_idx], last_token[te_idx], n_vocab)
    m_base = topk_metrics(y[te_idx], probs_base, last_token[te_idx], n_vocab)

    # save per-sample predictions
    topk = min(5, n_vocab)
    ord_model = np.argsort(-te["p_tok"], axis=1)[:, :topk]
    ord_base = np.argsort(-probs_base, axis=1)[:, :topk]

    out_model = pd.DataFrame(
        {
            "fold_id": fold_id,
            "end_ts": pd.to_datetime(ts[te_idx]),
            "last_token_id": last_token[te_idx],
            "target_token_id": y[te_idx],
            "is_change": is_change[te_idx],
            "p_change": te["p_change"],
            "pred_token_id_top1": ord_model[:, 0],
        }
    )
    out_base = pd.DataFrame(
        {
            "fold_id": fold_id,
            "end_ts": pd.to_datetime(ts[te_idx]),
            "last_token_id": last_token[te_idx],
            "target_token_id": y[te_idx],
            "is_change": is_change[te_idx],
            "pred_token_id_top1": ord_base[:, 0],
        }
    )

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
        "baseline_top1_all": m_base["top1_all"],
        "baseline_top3_all": m_base["top3_all"],
        "baseline_top5_all": m_base["top5_all"],
        "baseline_top1_change": m_base["top1_change"],
        "baseline_top3_change": m_base["top3_change"],
        "baseline_top5_change": m_base["top5_change"],
        "model_top1_all": m_te["top1_all"],
        "model_top3_all": m_te["top3_all"],
        "model_top5_all": m_te["top5_all"],
        "model_top1_change": m_te["top1_change"],
        "model_top3_change": m_te["top3_change"],
        "model_top5_change": m_te["top5_change"],
        "model_change_precision": c_te["change_precision"],
        "model_change_recall": c_te["change_recall"],
        "model_change_f1": c_te["change_f1"],
        "model_test_loss": te["loss"],
        "n_change_test": int(m_te["n_change"]),
        "change_ratio_test": float(m_te["change_ratio"]),
    }
    return fold_metric, pd.DataFrame(history_rows), out_model, out_base, model


def main():
    df = pd.read_parquet(IN_TOKENS_PQ).copy()
    df["end_ts"] = pd.to_datetime(df["end_ts"], errors="coerce")
    df = df.sort_values(["end_ts", "base_window_id_end"]).reset_index(drop=True)

    x, y, last_token, is_change, ts, n_vocab, id_to_token = build_samples(df, CFG)
    folds = build_fold_indices(len(y), CFG)

    fold_metrics = []
    history_all = []
    pred_model_all = []
    pred_base_all = []
    last_model = None

    for fold_id, tr_idx, va_idx, te_idx in folds:
        fm, hist, pmod, pbase, last_model = train_fold(
            fold_id=fold_id,
            x=x,
            y=y,
            last_token=last_token,
            is_change=is_change,
            ts=ts,
            n_vocab=n_vocab,
            id_to_token=id_to_token,
            tr_idx=tr_idx,
            va_idx=va_idx,
            te_idx=te_idx,
            cfg=CFG,
        )
        fold_metrics.append(fm)
        history_all.append(hist)
        pred_model_all.append(pmod)
        pred_base_all.append(pbase)

    fold_df = pd.DataFrame(fold_metrics)
    hist_df = pd.concat(history_all, ignore_index=True) if history_all else pd.DataFrame()
    pred_model_df = pd.concat(pred_model_all, ignore_index=True) if pred_model_all else pd.DataFrame()
    pred_base_df = pd.concat(pred_base_all, ignore_index=True) if pred_base_all else pd.DataFrame()

    fold_df.to_csv(OUT_FOLD_METRICS_CSV, index=False)
    hist_df.to_csv(OUT_HISTORY_CSV, index=False)
    pred_model_df.to_csv(OUT_PRED_MODEL_CSV, index=False)
    pred_base_df.to_csv(OUT_PRED_BASE_CSV, index=False)

    if last_model is not None:
        torch.save(
            {
                "state_dict": last_model.state_dict(),
                "config": CFG.__dict__,
                "n_vocab": n_vocab,
                "id_to_token": id_to_token,
            },
            OUT_MODEL_PT,
        )

    summary = {
        "symbol": CFG.symbol,
        "model_tag": "phase7_next_token_transformer_v3_changeaware_walkforward",
        "sequence_len": CFG.seq_len,
        "n_vocab": n_vocab,
        "n_samples_total": int(len(y)),
        "n_folds": int(len(fold_df)),
        "folds": [
            {"train_ratio": tr, "val_ratio": va, "test_ratio": te}
            for tr, va, te in CFG.folds
        ],
        "mean_metrics": {
            col: float(fold_df[col].mean())
            for col in [
                "baseline_top1_all",
                "baseline_top3_all",
                "baseline_top5_all",
                "baseline_top3_change",
                "model_top1_all",
                "model_top3_all",
                "model_top5_all",
                "model_top1_change",
                "model_top3_change",
                "model_top5_change",
                "model_change_f1",
            ]
            if col in fold_df.columns
        },
        "outputs": {
            "samples_csv": str(OUT_SAMPLES_CSV),
            "fold_metrics_csv": str(OUT_FOLD_METRICS_CSV),
            "history_csv": str(OUT_HISTORY_CSV),
            "predictions_model_csv": str(OUT_PRED_MODEL_CSV),
            "predictions_baseline_csv": str(OUT_PRED_BASE_CSV),
            "model_pt": str(OUT_MODEL_PT),
        },
    }
    OUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
    print("Saved summary:", OUT_SUMMARY_JSON)
    print("Mean metrics:", summary["mean_metrics"])
    return summary, fold_df, hist_df, pred_model_df, pred_base_df


if __name__ == "__main__":
    main()
