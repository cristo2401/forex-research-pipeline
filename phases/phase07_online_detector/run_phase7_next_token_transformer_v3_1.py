#!/usr/bin/env python3
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import top_k_accuracy_score
from torch.utils.data import DataLoader, Dataset

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

ROOT = Path(__file__).resolve().parents[2]
IN_TOKENS_PQ = ROOT / "phases/phase06_context_regimes/output_context_words_GBPUSD_v6_2_balanced_from_v5_1/GBPUSD_context_words_windows_v6_2.parquet"

OUT_DIR = ROOT / "phases/phase07_online_detector/output_next_token_GBPUSD_v3_1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_SAMPLES_CSV = OUT_DIR / "GBPUSD_next_token_samples_v3_1.csv"
OUT_FOLD_METRICS_CSV = OUT_DIR / "GBPUSD_next_token_fold_metrics_v3_1.csv"
OUT_HISTORY_CSV = OUT_DIR / "GBPUSD_next_token_train_history_v3_1.csv"
OUT_PRED_MODEL_CSV = OUT_DIR / "GBPUSD_next_token_fold_predictions_transformer_v3_1.csv"
OUT_PRED_BASE_CSV = OUT_DIR / "GBPUSD_next_token_fold_predictions_baseline_v3_1.csv"
OUT_MODEL_PT = OUT_DIR / "next_token_transformer_v3_1.pt"
OUT_SUMMARY_JSON = OUT_DIR / "GBPUSD_next_token_summary_v3_1.json"


@dataclass
class Config:
    symbol: str = "GBPUSD"
    # key improvement from v2/v3 search
    seq_len: int = 16
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 14
    early_stop_patience: int = 4
    min_lr: float = 1e-5
    num_workers: int = 0
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.10
    folds: Tuple[Tuple[float, float, float], ...] = (
        (0.55, 0.10, 0.10),
        (0.65, 0.10, 0.10),
        (0.75, 0.10, 0.10),
    )


CFG = Config()


class SeqDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class NextTokenTransformer(nn.Module):
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
        self.head = nn.Linear(cfg.d_model, n_vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l = x.shape
        pos = torch.arange(l, device=x.device).unsqueeze(0).expand(b, l)
        z = self.tok(x) + self.pos(pos)
        mask = torch.triu(torch.ones(l, l, device=x.device), diagonal=1).bool()
        z = self.enc(z, mask=mask)
        z = self.ln(z[:, -1, :])
        return self.head(z)


def build_samples(df: pd.DataFrame, cfg: Config):
    ids = df["word_id"].astype(int).to_numpy()
    ts = pd.to_datetime(df["end_ts"], errors="coerce").to_numpy(dtype="datetime64[ns]")
    id_to_token = {int(i): str(t) for i, t in zip(df["word_id"], df["word_token"])}

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


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    losses = []
    ys = []
    probs = []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            logits = model(xb)
            loss = criterion(logits, yb)
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
    y_pred = p_tok.argmax(axis=1) if p_tok.size else np.array([], dtype=int)
    return {
        "loss": float(np.mean(losses)) if losses else np.nan,
        "y_true": y_true,
        "y_pred": y_pred,
        "p_tok": p_tok,
    }


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


def baseline_markov(last_train: np.ndarray, y_train: np.ndarray, last_test: np.ndarray, n_vocab: int):
    cnt = np.ones((n_vocab, n_vocab), dtype=float)
    for a, b in zip(last_train, y_train):
        cnt[int(a), int(b)] += 1.0
    p = cnt / cnt.sum(axis=1, keepdims=True)
    return np.vstack([p[int(a)] for a in last_test])


def train_fold(
    fold_id: int,
    x: np.ndarray,
    y: np.ndarray,
    last: np.ndarray,
    ts: np.ndarray,
    n_vocab: int,
    id_to_token: Dict[int, str],
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    te_idx: np.ndarray,
    cfg: Config,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NextTokenTransformer(n_vocab=n_vocab, cfg=cfg).to(device)

    ce_w = class_weights(y[tr_idx], n_vocab).to(device)
    criterion = nn.CrossEntropyLoss(weight=ce_w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=cfg.min_lr)

    dl_train = DataLoader(SeqDataset(x[tr_idx], y[tr_idx]), batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())
    dl_val = DataLoader(SeqDataset(x[va_idx], y[va_idx]), batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())
    dl_test = DataLoader(SeqDataset(x[te_idx], y[te_idx]), batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())

    history_rows = []
    best_state = None
    best_val = np.inf
    wait = 0

    for epoch in range(1, cfg.max_epochs + 1):
        tr = run_epoch(model, dl_train, criterion, device, optimizer=optimizer)
        va = run_epoch(model, dl_val, criterion, device, optimizer=None)
        scheduler.step(va["loss"])

        mtr = topk_metrics(tr["y_true"], tr["p_tok"], last[tr_idx], n_vocab)
        mva = topk_metrics(va["y_true"], va["p_tok"], last[va_idx], n_vocab)
        history_rows.append(
            {
                "fold_id": fold_id,
                "epoch": epoch,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "train_loss": tr["loss"],
                "val_loss": va["loss"],
                "train_top3_all": mtr["top3_all"],
                "train_top3_change": mtr["top3_change"],
                "val_top3_all": mva["top3_all"],
                "val_top3_change": mva["top3_change"],
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

    te = run_epoch(model, dl_test, criterion, device, optimizer=None)
    m_model = topk_metrics(te["y_true"], te["p_tok"], last[te_idx], n_vocab)

    p_base = baseline_markov(last[tr_idx], y[tr_idx], last[te_idx], n_vocab)
    m_base = topk_metrics(y[te_idx], p_base, last[te_idx], n_vocab)

    # Save per-sample test predictions
    topk = min(5, n_vocab)
    ord_model = np.argsort(-te["p_tok"], axis=1)[:, :topk]
    ord_base = np.argsort(-p_base, axis=1)[:, :topk]

    out_model = pd.DataFrame(
        {
            "fold_id": fold_id,
            "end_ts": pd.to_datetime(ts[te_idx]),
            "last_token_id": last[te_idx],
            "target_token_id": y[te_idx],
            "is_change": (y[te_idx] != last[te_idx]).astype(int),
        }
    )
    out_base = out_model.copy()
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
        "model_top1_all": m_model["top1_all"],
        "model_top3_all": m_model["top3_all"],
        "model_top5_all": m_model["top5_all"],
        "model_top1_change": m_model["top1_change"],
        "model_top3_change": m_model["top3_change"],
        "model_top5_change": m_model["top5_change"],
        "model_test_loss": te["loss"],
        "n_change_test": int(m_model["n_change"]),
        "change_ratio_test": float(m_model["change_ratio"]),
    }
    return fold_metric, pd.DataFrame(history_rows), out_model, out_base, model


def main():
    df = pd.read_parquet(IN_TOKENS_PQ).copy()
    df["end_ts"] = pd.to_datetime(df["end_ts"], errors="coerce")
    df = df.sort_values(["end_ts", "base_window_id_end"]).reset_index(drop=True)

    x, y, last, ts, is_change, n_vocab, id_to_token = build_samples(df, CFG)
    folds = build_fold_indices(len(y), CFG)

    fold_rows = []
    hist_rows = []
    pred_model_rows = []
    pred_base_rows = []
    last_model = None

    for fold_id, tr_idx, va_idx, te_idx in folds:
        fm, hist, pmod, pbase, last_model = train_fold(
            fold_id=fold_id,
            x=x,
            y=y,
            last=last,
            ts=ts,
            n_vocab=n_vocab,
            id_to_token=id_to_token,
            tr_idx=tr_idx,
            va_idx=va_idx,
            te_idx=te_idx,
            cfg=CFG,
        )
        fold_rows.append(fm)
        hist_rows.append(hist)
        pred_model_rows.append(pmod)
        pred_base_rows.append(pbase)

    fold_df = pd.DataFrame(fold_rows)
    hist_df = pd.concat(hist_rows, ignore_index=True) if hist_rows else pd.DataFrame()
    pred_model_df = pd.concat(pred_model_rows, ignore_index=True) if pred_model_rows else pd.DataFrame()
    pred_base_df = pd.concat(pred_base_rows, ignore_index=True) if pred_base_rows else pd.DataFrame()

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
        "model_tag": "phase7_next_token_transformer_v3_1_short_context_walkforward",
        "sequence_len": CFG.seq_len,
        "n_vocab": n_vocab,
        "n_samples_total": int(len(y)),
        "n_change_total": int(is_change.sum()),
        "change_ratio_total": float(is_change.mean()),
        "n_folds": int(len(fold_df)),
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
