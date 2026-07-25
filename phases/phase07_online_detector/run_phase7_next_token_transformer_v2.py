#!/usr/bin/env python3
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

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

OUT_DIR = ROOT / "phases/phase07_online_detector/output_next_token_GBPUSD_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_SPLITS_CSV = OUT_DIR / "GBPUSD_next_token_samples_v2.csv"
OUT_HISTORY_CSV = OUT_DIR / "GBPUSD_next_token_train_history_v2.csv"
OUT_MODEL_PT = OUT_DIR / "next_token_transformer_v2.pt"
OUT_BASELINE_PRED_CSV = OUT_DIR / "GBPUSD_next_token_test_predictions_baseline_v2.csv"
OUT_MODEL_PRED_CSV = OUT_DIR / "GBPUSD_next_token_test_predictions_transformer_v2.csv"
OUT_SUMMARY_JSON = OUT_DIR / "GBPUSD_next_token_summary_v2.json"


@dataclass
class Config:
    symbol: str = "GBPUSD"
    seq_len: int = 64
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 24
    early_stop_patience: int = 6
    min_lr: float = 1e-5
    num_workers: int = 0
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.10


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
        # Causal mask for autoregressive conditioning.
        mask = torch.triu(torch.ones(l, l, device=x.device), diagonal=1).bool()
        z = self.enc(z, mask=mask)
        z = self.ln(z[:, -1, :])
        return self.head(z)


def load_sequence_df() -> pd.DataFrame:
    df = pd.read_parquet(IN_TOKENS_PQ).copy()
    df["end_ts"] = pd.to_datetime(df["end_ts"], errors="coerce")
    df = df.sort_values(["end_ts", "base_window_id_end"]).reset_index(drop=True)
    return df


def build_samples(df: pd.DataFrame, cfg: Config):
    ids = df["word_id"].astype(int).to_numpy()
    tokens = df["word_token"].astype(str).to_numpy()
    splits = df["split"].astype(str).to_numpy()
    end_ts = pd.to_datetime(df["end_ts"], errors="coerce").to_numpy(dtype="datetime64[ns]")

    x_rows: List[np.ndarray] = []
    y_rows: List[int] = []
    split_rows: List[str] = []
    end_rows: List[np.datetime64] = []
    last_rows: List[int] = []

    for i in range(cfg.seq_len, len(df)):
        # keep fully split-pure sequences to avoid train/val/test leakage in context
        if len(set(splits[i - cfg.seq_len : i + 1])) != 1:
            continue
        x = ids[i - cfg.seq_len : i]
        y = int(ids[i])
        x_rows.append(x)
        y_rows.append(y)
        split_rows.append(str(splits[i]))
        end_rows.append(end_ts[i])
        last_rows.append(int(x[-1]))

    x = np.asarray(x_rows, dtype=np.int64)
    y = np.asarray(y_rows, dtype=np.int64)
    sp = np.asarray(split_rows)
    ts = np.asarray(end_rows)
    last_token = np.asarray(last_rows, dtype=np.int64)

    n_vocab = int(ids.max()) + 1
    id_to_token = {int(i): t for i, t in zip(df["word_id"], df["word_token"])}
    for i in range(n_vocab):
        if i not in id_to_token:
            id_to_token[i] = f"w{i:02d}"

    samples_df = pd.DataFrame(
        {
            "end_ts": pd.to_datetime(ts),
            "split": sp,
            "last_token_id": last_token,
            "target_token_id": y,
            "is_change": (y != last_token).astype(int),
        }
    )
    samples_df["last_token"] = samples_df["last_token_id"].map(lambda z: id_to_token[int(z)])
    samples_df["target_token"] = samples_df["target_token_id"].map(lambda z: id_to_token[int(z)])
    samples_df.to_csv(OUT_SPLITS_CSV, index=False)

    return x, y, sp, ts, last_token, n_vocab, id_to_token


def make_loaders(x: np.ndarray, y: np.ndarray, sp: np.ndarray, cfg: Config):
    dls = {}
    ys = {}
    for split_name, shuf in [("train", True), ("val", False), ("test", False)]:
        m = sp == split_name
        ys[split_name] = y[m]
        ds = SeqDataset(x[m], y[m])
        dls[split_name] = DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=shuf,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    return dls, ys


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
    y_true = []
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
        y_true.append(yb.detach().cpu().numpy())
        probs.append(torch.softmax(logits, dim=1).detach().cpu().numpy())

    y_true = np.concatenate(y_true) if y_true else np.array([], dtype=int)
    probs = np.concatenate(probs) if probs else np.zeros((0, 0), dtype=float)
    y_pred = probs.argmax(axis=1) if probs.size else np.array([], dtype=int)
    return float(np.mean(losses)) if losses else np.nan, y_true, y_pred, probs


def topk_metrics(y_true: np.ndarray, probs: np.ndarray, last_token: np.ndarray, n_vocab: int) -> Dict[str, float]:
    labels = list(range(n_vocab))
    out = {}
    for k in [1, 3, 5]:
        kk = min(k, n_vocab)
        out[f"top{k}_all"] = float(top_k_accuracy_score(y_true, probs, k=kk, labels=labels))
    is_change = y_true != last_token
    out["n_change"] = int(is_change.sum())
    out["change_ratio"] = float(is_change.mean()) if len(is_change) else np.nan
    if is_change.any():
        for k in [1, 3, 5]:
            kk = min(k, n_vocab)
            out[f"top{k}_change"] = float(
                top_k_accuracy_score(y_true[is_change], probs[is_change], k=kk, labels=labels)
            )
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
    probs = np.vstack([p[int(a)] for a in last_test])
    return probs


def train_and_eval(
    x: np.ndarray,
    y: np.ndarray,
    sp: np.ndarray,
    last_token: np.ndarray,
    ts: np.ndarray,
    n_vocab: int,
    id_to_token: Dict[int, str],
    cfg: Config,
):
    dls, ys = make_loaders(x, y, sp, cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NextTokenTransformer(n_vocab=n_vocab, cfg=cfg).to(device)
    w = class_weights(ys["train"], n_vocab).to(device)
    criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=cfg.min_lr
    )

    history = []
    best_state = None
    best_val = np.inf
    wait = 0

    for epoch in range(1, cfg.max_epochs + 1):
        tr_loss, tr_true, tr_pred, tr_probs = run_epoch(model, dls["train"], criterion, device, optimizer=optimizer)
        va_loss, va_true, va_pred, va_probs = run_epoch(model, dls["val"], criterion, device, optimizer=None)
        scheduler.step(va_loss)

        mtr = topk_metrics(tr_true, tr_probs, last_token[sp == "train"], n_vocab)
        mva = topk_metrics(va_true, va_probs, last_token[sp == "val"], n_vocab)

        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": tr_loss,
            "val_loss": va_loss,
            "train_top1_all": mtr["top1_all"],
            "train_top3_all": mtr["top3_all"],
            "train_top5_all": mtr["top5_all"],
            "train_top3_change": mtr["top3_change"],
            "val_top1_all": mva["top1_all"],
            "val_top3_all": mva["top3_all"],
            "val_top5_all": mva["top5_all"],
            "val_top3_change": mva["top3_change"],
        }
        history.append(row)

        if va_loss < best_val - 1e-4:
            best_val = va_loss
            wait = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.early_stop_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    history_df = pd.DataFrame(history)
    history_df.to_csv(OUT_HISTORY_CSV, index=False)

    # test transformer
    te_loss, te_true, te_pred, te_probs = run_epoch(model, dls["test"], criterion, device, optimizer=None)
    m_te = topk_metrics(te_true, te_probs, last_token[sp == "test"], n_vocab)

    # test baseline
    probs_base = baseline_markov(
        last_train=last_token[sp == "train"],
        y_train=y[sp == "train"],
        last_test=last_token[sp == "test"],
        n_vocab=n_vocab,
    )
    m_base = topk_metrics(y[sp == "test"], probs_base, last_token[sp == "test"], n_vocab)

    # save predictions with top-k
    idx_test = np.where(sp == "test")[0]
    y_test = y[idx_test]
    last_test = last_token[idx_test]
    ts_test = pd.to_datetime(ts[idx_test])

    def pack_predictions(probs_arr: np.ndarray, name: str, out_csv: Path):
        topk = min(5, n_vocab)
        ord_idx = np.argsort(-probs_arr, axis=1)[:, :topk]
        ord_prob = np.take_along_axis(probs_arr, ord_idx, axis=1)

        out = pd.DataFrame(
            {
                "end_ts": ts_test,
                "last_token_id": last_test,
                "last_token": [id_to_token[int(z)] for z in last_test],
                "target_token_id": y_test,
                "target_token": [id_to_token[int(z)] for z in y_test],
                "is_change": (y_test != last_test).astype(int),
                "pred_token_id_top1": ord_idx[:, 0],
                "pred_token_top1": [id_to_token[int(z)] for z in ord_idx[:, 0]],
                "prob_top1": ord_prob[:, 0],
            }
        )
        for j in range(topk):
            out[f"pred_token_id_top{j+1}"] = ord_idx[:, j]
            out[f"pred_token_top{j+1}"] = [id_to_token[int(z)] for z in ord_idx[:, j]]
            out[f"prob_top{j+1}"] = ord_prob[:, j]
        out["model_name"] = name
        out.to_csv(out_csv, index=False)
        return out

    pred_base_df = pack_predictions(probs_base, "baseline_markov1", OUT_BASELINE_PRED_CSV)
    pred_model_df = pack_predictions(te_probs, "transformer_v2", OUT_MODEL_PRED_CSV)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": cfg.__dict__,
            "n_vocab": n_vocab,
            "id_to_token": id_to_token,
        },
        OUT_MODEL_PT,
    )

    summary = {
        "symbol": cfg.symbol,
        "model_tag": "phase7_next_token_transformer_v2",
        "sequence_len": cfg.seq_len,
        "n_vocab": n_vocab,
        "n_samples": {
            "train": int((sp == "train").sum()),
            "val": int((sp == "val").sum()),
            "test": int((sp == "test").sum()),
        },
        "baseline_markov1_test": m_base,
        "transformer_test": {
            "loss": te_loss,
            **m_te,
        },
        "notes": {
            "topk_all_includes_stay_same_token": True,
            "topk_change_focuses_only_when_target_differs_from_last_token": True,
        },
        "outputs": {
            "samples_csv": str(OUT_SPLITS_CSV),
            "history_csv": str(OUT_HISTORY_CSV),
            "model_pt": str(OUT_MODEL_PT),
            "baseline_predictions_csv": str(OUT_BASELINE_PRED_CSV),
            "transformer_predictions_csv": str(OUT_MODEL_PRED_CSV),
        },
    }
    OUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
    return summary, history_df, pred_base_df, pred_model_df


def main():
    df = load_sequence_df()
    x, y, sp, ts, last_token, n_vocab, id_to_token = build_samples(df, CFG)
    summary, history_df, pred_base_df, pred_model_df = train_and_eval(
        x=x,
        y=y,
        sp=sp,
        last_token=last_token,
        ts=ts,
        n_vocab=n_vocab,
        id_to_token=id_to_token,
        cfg=CFG,
    )
    print("Saved summary:", OUT_SUMMARY_JSON)
    print("Baseline test:", summary["baseline_markov1_test"])
    print("Transformer test:", summary["transformer_test"])
    return summary, history_df, pred_base_df, pred_model_df


if __name__ == "__main__":
    main()
