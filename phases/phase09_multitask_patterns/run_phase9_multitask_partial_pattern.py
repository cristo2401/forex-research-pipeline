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
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

ROOT = Path(__file__).resolve().parents[2]
IN_SEG_LONG = ROOT / "phases/phase04_transfer_clustering/output_transfer_GBPUSD_full_v2/GBPUSD_segments_long.parquet"
IN_SEG_ASSIGN = ROOT / "phases/phase04_transfer_clustering/output_transfer_GBPUSD_full_v2/GBPUSD_segment_cluster_assignments.parquet"

OUT_DIR = ROOT / "phases/phase09_multitask_patterns/output_phase9_partial_pattern_GBPUSD"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_SAMPLES_CSV = OUT_DIR / "GBPUSD_phase9_samples.csv"
OUT_HISTORY_CSV = OUT_DIR / "GBPUSD_phase9_train_history.csv"
OUT_TEST_PRED_CSV = OUT_DIR / "GBPUSD_phase9_test_predictions.csv"
OUT_MODEL_PT = OUT_DIR / "phase9_partial_pattern_model.pt"
OUT_SUMMARY_JSON = OUT_DIR / "GBPUSD_phase9_summary.json"


@dataclass
class Config:
    symbol: str = "GBPUSD"
    max_seq_len: int = 64
    in_dim: int = 4
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.10

    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 18
    early_stop_patience: int = 5
    min_lr: float = 1e-5
    num_workers: int = 0

    lambda_reg: float = 0.7

    train_ratio: float = 0.70
    val_ratio: float = 0.15


CFG = Config()


class PatternDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        lengths: np.ndarray,
        y_cls: np.ndarray,
        y_prog: np.ndarray,
        reg_mask: np.ndarray,
    ):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.lengths = torch.tensor(lengths, dtype=torch.long)
        self.y_cls = torch.tensor(y_cls, dtype=torch.long)
        self.y_prog = torch.tensor(y_prog, dtype=torch.float32)
        self.reg_mask = torch.tensor(reg_mask, dtype=torch.bool)

    def __len__(self):
        return len(self.y_cls)

    def __getitem__(self, idx):
        return (
            self.x[idx],
            self.lengths[idx],
            self.y_cls[idx],
            self.y_prog[idx],
            self.reg_mask[idx],
        )


class PartialPatternTransformer(nn.Module):
    def __init__(self, cfg: Config, n_classes: int):
        super().__init__()
        self.in_proj = nn.Linear(cfg.in_dim, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_seq_len, cfg.d_model)
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
        self.cls_head = nn.Linear(cfg.d_model, n_classes)
        self.reg_head = nn.Linear(cfg.d_model, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        b, l, _ = x.shape
        pos_ids = torch.arange(l, device=x.device).unsqueeze(0).expand(b, l)

        z = self.in_proj(x) + self.pos(pos_ids)

        # left padding mask (True means masked/padding)
        left_pad = (l - lengths).unsqueeze(1)
        idx = torch.arange(l, device=x.device).unsqueeze(0)
        pad_mask = idx < left_pad

        h = self.enc(z, src_key_padding_mask=pad_mask)
        pooled = self.ln(h[:, -1, :])

        cls_logits = self.cls_head(pooled)
        prog = torch.sigmoid(self.reg_head(pooled).squeeze(1))
        return cls_logits, prog


def build_splits(seg_assign: pd.DataFrame, cfg: Config):
    seg_sorted = seg_assign[["global_segment_id", "start_ts"]].drop_duplicates().sort_values("start_ts").reset_index(drop=True)
    n = len(seg_sorted)
    n_train = int(n * cfg.train_ratio)
    n_val = int(n * cfg.val_ratio)

    split = np.array(["train"] * n)
    split[n_train : n_train + n_val] = "val"
    split[n_train + n_val :] = "test"

    seg_sorted["split"] = split
    return dict(zip(seg_sorted["global_segment_id"], seg_sorted["split"]))


def build_samples(seg_long: pd.DataFrame, seg_assign: pd.DataFrame, cfg: Config):
    seg_assign = seg_assign.copy()
    seg_assign["start_ts"] = pd.to_datetime(seg_assign["start_ts"], errors="coerce")
    seg_assign["end_ts"] = pd.to_datetime(seg_assign["end_ts"], errors="coerce")

    informative_clusters = sorted(seg_assign.loc[seg_assign["is_informative_cluster"] == True, "cluster_id"].dropna().astype(int).unique().tolist())
    cls_map = {c: i + 1 for i, c in enumerate(informative_clusters)}  # class 0 reserved for noise/unknown

    split_map = build_splits(seg_assign, cfg)

    meta = seg_assign[["global_segment_id", "start_ts", "end_ts", "cluster_id", "is_informative_cluster"]].copy()
    long = seg_long.merge(meta, on="global_segment_id", how="inner")
    long["timestamp"] = pd.to_datetime(long["timestamp"], errors="coerce")
    long = long.sort_values(["global_segment_id", "step"]).reset_index(drop=True)

    x_rows: List[np.ndarray] = []
    len_rows: List[int] = []
    y_cls_rows: List[int] = []
    y_prog_rows: List[float] = []
    reg_mask_rows: List[int] = []
    sample_rows: List[Dict] = []

    for seg_id, g in long.groupby("global_segment_id", sort=False):
        g = g.sort_values("step")
        arr = g[["open", "high", "low", "close"]].to_numpy(dtype=np.float32)

        # normalize by first close to preserve pattern geometry and remove scale
        base = float(arr[0, 3])
        base = base if abs(base) > 1e-8 else 1.0
        arr = (arr / base) - 1.0

        n = len(g)
        cls_raw = int(g["cluster_id"].iloc[0])
        inf = bool(g["is_informative_cluster"].iloc[0])
        y_cls = cls_map[cls_raw] if inf and cls_raw in cls_map else 0

        split = split_map.get(seg_id, "train")

        for s in range(n):
            start = max(0, s - cfg.max_seq_len + 1)
            seq = arr[start : s + 1]
            ln = seq.shape[0]

            x = np.zeros((cfg.max_seq_len, cfg.in_dim), dtype=np.float32)
            x[-ln:] = seq

            prog = float(s / max(n - 1, 1))
            reg_mask = 1 if y_cls > 0 else 0
            y_prog = prog if reg_mask else 0.0

            x_rows.append(x)
            len_rows.append(ln)
            y_cls_rows.append(y_cls)
            y_prog_rows.append(y_prog)
            reg_mask_rows.append(reg_mask)
            sample_rows.append(
                {
                    "global_segment_id": seg_id,
                    "timestamp": g["timestamp"].iloc[s],
                    "step": int(g["step"].iloc[s]),
                    "split": split,
                    "cluster_id_raw": cls_raw,
                    "is_informative_segment": int(inf),
                    "y_class": y_cls,
                    "y_progress": y_prog,
                    "reg_mask": reg_mask,
                }
            )

    x = np.asarray(x_rows, dtype=np.float32)
    lengths = np.asarray(len_rows, dtype=np.int64)
    y_cls = np.asarray(y_cls_rows, dtype=np.int64)
    y_prog = np.asarray(y_prog_rows, dtype=np.float32)
    reg_mask = np.asarray(reg_mask_rows, dtype=np.int64)
    samples_df = pd.DataFrame(sample_rows)

    samples_df.to_csv(OUT_SAMPLES_CSV, index=False)

    return x, lengths, y_cls, y_prog, reg_mask, samples_df, cls_map


def class_weights(y_cls_train: np.ndarray, n_classes: int):
    cnt = np.bincount(y_cls_train, minlength=n_classes).astype(float)
    cnt = np.maximum(cnt, 1.0)
    w = cnt.sum() / (n_classes * cnt)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def compute_metrics(
    y_true_cls: np.ndarray,
    y_pred_cls: np.ndarray,
    y_true_prog: np.ndarray,
    y_pred_prog: np.ndarray,
    reg_mask: np.ndarray,
    n_classes: int,
):
    out = {}
    out["acc"] = float((y_true_cls == y_pred_cls).mean()) if len(y_true_cls) else np.nan

    labels_all = list(range(n_classes))
    try:
        out["f1_macro_all"] = float(f1_score(y_true_cls, y_pred_cls, labels=labels_all, average="macro", zero_division=0))
    except Exception:
        out["f1_macro_all"] = np.nan

    inf_mask = y_true_cls > 0
    if inf_mask.any():
        labels_inf = sorted(list(set(y_true_cls[inf_mask].tolist())))
        out["f1_macro_informative"] = float(f1_score(y_true_cls[inf_mask], y_pred_cls[inf_mask], labels=labels_inf, average="macro", zero_division=0))
    else:
        out["f1_macro_informative"] = np.nan

    m = reg_mask.astype(bool)
    if m.any():
        err = np.abs(y_true_prog[m] - y_pred_prog[m])
        out["mae_progress"] = float(err.mean())
        out["rmse_progress"] = float(np.sqrt(np.mean((y_true_prog[m] - y_pred_prog[m]) ** 2)))

        early = m & (y_true_prog < 0.5)
        out["mae_progress_early"] = float(np.abs(y_true_prog[early] - y_pred_prog[early]).mean()) if early.any() else np.nan
    else:
        out["mae_progress"] = np.nan
        out["rmse_progress"] = np.nan
        out["mae_progress_early"] = np.nan

    out["n_informative"] = int((y_true_cls > 0).sum())
    return out


def run_epoch(model, loader, ce_weight, device, cfg: Config, optimizer=None):
    train = optimizer is not None
    model.train(train)

    losses = []
    ce_losses = []
    reg_losses = []
    y_cls_true = []
    y_cls_pred = []
    y_prog_true = []
    y_prog_pred = []
    reg_masks = []

    for xb, lb, yb_cls, yb_prog, yb_mask in loader:
        xb = xb.to(device, non_blocking=True)
        lb = lb.to(device, non_blocking=True)
        yb_cls = yb_cls.to(device, non_blocking=True)
        yb_prog = yb_prog.to(device, non_blocking=True)
        yb_mask = yb_mask.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            logits, prog = model(xb, lb)

            ce = F.cross_entropy(logits, yb_cls, weight=ce_weight)
            if yb_mask.any():
                reg = F.smooth_l1_loss(prog[yb_mask], yb_prog[yb_mask])
            else:
                reg = torch.tensor(0.0, device=device)

            loss = ce + cfg.lambda_reg * reg

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        losses.append(float(loss.detach().cpu()))
        ce_losses.append(float(ce.detach().cpu()))
        reg_losses.append(float(reg.detach().cpu()))

        y_cls_true.append(yb_cls.detach().cpu().numpy())
        y_cls_pred.append(logits.argmax(dim=1).detach().cpu().numpy())
        y_prog_true.append(yb_prog.detach().cpu().numpy())
        y_prog_pred.append(prog.detach().cpu().numpy())
        reg_masks.append(yb_mask.detach().cpu().numpy().astype(np.int64))

    y_cls_true = np.concatenate(y_cls_true) if y_cls_true else np.array([], dtype=np.int64)
    y_cls_pred = np.concatenate(y_cls_pred) if y_cls_pred else np.array([], dtype=np.int64)
    y_prog_true = np.concatenate(y_prog_true) if y_prog_true else np.array([], dtype=np.float32)
    y_prog_pred = np.concatenate(y_prog_pred) if y_prog_pred else np.array([], dtype=np.float32)
    reg_masks = np.concatenate(reg_masks) if reg_masks else np.array([], dtype=np.int64)

    return {
        "loss": float(np.mean(losses)) if losses else np.nan,
        "ce_loss": float(np.mean(ce_losses)) if ce_losses else np.nan,
        "reg_loss": float(np.mean(reg_losses)) if reg_losses else np.nan,
        "y_cls_true": y_cls_true,
        "y_cls_pred": y_cls_pred,
        "y_prog_true": y_prog_true,
        "y_prog_pred": y_prog_pred,
        "reg_masks": reg_masks,
    }


def main():
    seg_long = pd.read_parquet(IN_SEG_LONG)
    seg_assign = pd.read_parquet(IN_SEG_ASSIGN)

    x, lengths, y_cls, y_prog, reg_mask, samples_df, cls_map = build_samples(seg_long, seg_assign, CFG)

    idx_train = samples_df.index[samples_df["split"] == "train"].to_numpy()
    idx_val = samples_df.index[samples_df["split"] == "val"].to_numpy()
    idx_test = samples_df.index[samples_df["split"] == "test"].to_numpy()

    n_classes = int(max(y_cls.max(), 0)) + 1

    ds_train = PatternDataset(x[idx_train], lengths[idx_train], y_cls[idx_train], y_prog[idx_train], reg_mask[idx_train])
    ds_val = PatternDataset(x[idx_val], lengths[idx_val], y_cls[idx_val], y_prog[idx_val], reg_mask[idx_val])
    ds_test = PatternDataset(x[idx_test], lengths[idx_test], y_cls[idx_test], y_prog[idx_test], reg_mask[idx_test])

    dl_train = DataLoader(ds_train, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers, pin_memory=torch.cuda.is_available())
    dl_val = DataLoader(ds_val, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers, pin_memory=torch.cuda.is_available())
    dl_test = DataLoader(ds_test, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers, pin_memory=torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PartialPatternTransformer(cfg=CFG, n_classes=n_classes).to(device)

    ce_weight = class_weights(y_cls[idx_train], n_classes).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=CFG.min_lr)

    history_rows = []
    best_state = None
    best_val = np.inf
    wait = 0

    for epoch in range(1, CFG.max_epochs + 1):
        tr = run_epoch(model, dl_train, ce_weight, device, CFG, optimizer=optimizer)
        va = run_epoch(model, dl_val, ce_weight, device, CFG, optimizer=None)
        scheduler.step(va["loss"])

        tr_m = compute_metrics(tr["y_cls_true"], tr["y_cls_pred"], tr["y_prog_true"], tr["y_prog_pred"], tr["reg_masks"], n_classes)
        va_m = compute_metrics(va["y_cls_true"], va["y_cls_pred"], va["y_prog_true"], va["y_prog_pred"], va["reg_masks"], n_classes)

        history_rows.append(
            {
                "epoch": epoch,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "train_loss": tr["loss"],
                "train_ce": tr["ce_loss"],
                "train_reg": tr["reg_loss"],
                "train_f1_macro_all": tr_m["f1_macro_all"],
                "train_f1_macro_informative": tr_m["f1_macro_informative"],
                "train_mae_progress": tr_m["mae_progress"],
                "val_loss": va["loss"],
                "val_ce": va["ce_loss"],
                "val_reg": va["reg_loss"],
                "val_f1_macro_all": va_m["f1_macro_all"],
                "val_f1_macro_informative": va_m["f1_macro_informative"],
                "val_mae_progress": va_m["mae_progress"],
                "val_mae_progress_early": va_m["mae_progress_early"],
            }
        )

        if va["loss"] < best_val - 1e-4:
            best_val = va["loss"]
            wait = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= CFG.early_stop_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    te = run_epoch(model, dl_test, ce_weight, device, CFG, optimizer=None)
    te_m = compute_metrics(te["y_cls_true"], te["y_cls_pred"], te["y_prog_true"], te["y_prog_pred"], te["reg_masks"], n_classes)

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(OUT_HISTORY_CSV, index=False)

    test_meta = samples_df.iloc[idx_test].reset_index(drop=True).copy()
    test_meta["pred_class"] = te["y_cls_pred"]
    test_meta["pred_progress"] = te["y_prog_pred"]
    test_meta.to_csv(OUT_TEST_PRED_CSV, index=False)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": CFG.__dict__,
            "n_classes": n_classes,
            "class_map": cls_map,
        },
        OUT_MODEL_PT,
    )

    class_counts = samples_df.groupby(["split", "y_class"]).size().unstack(fill_value=0)

    summary = {
        "symbol": CFG.symbol,
        "model_tag": "phase9_multitask_partial_pattern_cls_plus_progress",
        "input": {
            "segments_long": str(IN_SEG_LONG),
            "segments_assign": str(IN_SEG_ASSIGN),
        },
        "setup": {
            "max_seq_len": CFG.max_seq_len,
            "n_classes": n_classes,
            "class_map": {str(k): int(v) for k, v in cls_map.items()},
            "class_0": "unknown_noise_non_informative",
            "lambda_reg": CFG.lambda_reg,
        },
        "n_samples": {
            "total": int(len(samples_df)),
            "train": int(len(idx_train)),
            "val": int(len(idx_val)),
            "test": int(len(idx_test)),
            "test_informative": int((samples_df.iloc[idx_test]["y_class"] > 0).sum()),
        },
        "class_counts_by_split": class_counts.to_dict(),
        "test_metrics": {
            "acc": te_m["acc"],
            "f1_macro_all": te_m["f1_macro_all"],
            "f1_macro_informative": te_m["f1_macro_informative"],
            "mae_progress": te_m["mae_progress"],
            "rmse_progress": te_m["rmse_progress"],
            "mae_progress_early": te_m["mae_progress_early"],
            "n_informative": te_m["n_informative"],
        },
        "outputs": {
            "samples_csv": str(OUT_SAMPLES_CSV),
            "history_csv": str(OUT_HISTORY_CSV),
            "test_predictions_csv": str(OUT_TEST_PRED_CSV),
            "model_pt": str(OUT_MODEL_PT),
        },
    }

    OUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

    print("Saved summary:", OUT_SUMMARY_JSON)
    print("Test metrics:", summary["test_metrics"])

    return summary, history_df, test_meta


if __name__ == "__main__":
    main()
