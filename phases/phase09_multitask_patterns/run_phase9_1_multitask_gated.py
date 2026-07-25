#!/usr/bin/env python3
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

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

OUT_DIR = ROOT / "phases/phase09_multitask_patterns/output_phase9_1_partial_pattern_GBPUSD"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_SAMPLES_CSV = OUT_DIR / "GBPUSD_phase9_1_samples.csv"
OUT_HISTORY_CSV = OUT_DIR / "GBPUSD_phase9_1_train_history.csv"
OUT_VAL_THRESH_CSV = OUT_DIR / "GBPUSD_phase9_1_val_threshold_scan.csv"
OUT_TEST_PRED_CSV = OUT_DIR / "GBPUSD_phase9_1_test_predictions.csv"
OUT_MODEL_PT = OUT_DIR / "phase9_1_partial_pattern_model.pt"
OUT_SUMMARY_JSON = OUT_DIR / "GBPUSD_phase9_1_summary.json"


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
    max_epochs: int = 20
    early_stop_patience: int = 5
    min_lr: float = 1e-5
    num_workers: int = 0

    w_inf: float = 1.0
    w_sub: float = 1.0
    w_reg: float = 0.7

    train_ratio: float = 0.70
    val_ratio: float = 0.15


CFG = Config()


class PatternDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        lengths: np.ndarray,
        y_class: np.ndarray,
        y_informative: np.ndarray,
        y_sub: np.ndarray,
        y_progress: np.ndarray,
        reg_mask: np.ndarray,
    ):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.lengths = torch.tensor(lengths, dtype=torch.long)
        self.y_class = torch.tensor(y_class, dtype=torch.long)
        self.y_inf = torch.tensor(y_informative, dtype=torch.float32)
        self.y_sub = torch.tensor(y_sub, dtype=torch.long)
        self.y_prog = torch.tensor(y_progress, dtype=torch.float32)
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
            activation="gelu",
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
    class_map = {c: i + 1 for i, c in enumerate(informative_clusters)}
    sub_map = {c: i for i, c in enumerate(informative_clusters)}
    inv_sub_map = {i: c for c, i in sub_map.items()}

    split_map = build_splits(seg_assign, cfg)

    meta = seg_assign[["global_segment_id", "cluster_id", "is_informative_cluster"]].copy()
    long = seg_long.merge(meta, on="global_segment_id", how="inner")
    long["timestamp"] = pd.to_datetime(long["timestamp"], errors="coerce")
    long = long.sort_values(["global_segment_id", "step"]).reset_index(drop=True)

    x_rows: List[np.ndarray] = []
    len_rows: List[int] = []
    y_class_rows: List[int] = []
    y_inf_rows: List[int] = []
    y_sub_rows: List[int] = []
    y_prog_rows: List[float] = []
    reg_mask_rows: List[int] = []
    sample_rows: List[Dict] = []

    for seg_id, g in long.groupby("global_segment_id", sort=False):
        g = g.sort_values("step")
        arr = g[["open", "high", "low", "close"]].to_numpy(dtype=np.float32)

        base = float(arr[0, 3])
        base = base if abs(base) > 1e-8 else 1.0
        arr = (arr / base) - 1.0

        n = len(g)
        cls_raw = int(g["cluster_id"].iloc[0])
        inf = bool(g["is_informative_cluster"].iloc[0])

        y_class = class_map[cls_raw] if inf and cls_raw in class_map else 0
        y_inf = 1 if y_class > 0 else 0
        y_sub = sub_map[cls_raw] if y_inf else 0

        split = split_map.get(seg_id, "train")

        for s in range(n):
            start = max(0, s - cfg.max_seq_len + 1)
            seq = arr[start : s + 1]
            ln = seq.shape[0]

            x = np.zeros((cfg.max_seq_len, cfg.in_dim), dtype=np.float32)
            x[-ln:] = seq

            prog = float(s / max(n - 1, 1))
            reg_mask = 1 if y_inf else 0

            x_rows.append(x)
            len_rows.append(ln)
            y_class_rows.append(y_class)
            y_inf_rows.append(y_inf)
            y_sub_rows.append(y_sub)
            y_prog_rows.append(prog if y_inf else 0.0)
            reg_mask_rows.append(reg_mask)

            sample_rows.append(
                {
                    "global_segment_id": seg_id,
                    "timestamp": g["timestamp"].iloc[s],
                    "step": int(g["step"].iloc[s]),
                    "split": split,
                    "cluster_id_raw": cls_raw,
                    "is_informative_segment": int(inf),
                    "y_class": y_class,
                    "y_inf": y_inf,
                    "y_sub": y_sub,
                    "y_progress": prog if y_inf else 0.0,
                    "reg_mask": reg_mask,
                }
            )

    x = np.asarray(x_rows, dtype=np.float32)
    lengths = np.asarray(len_rows, dtype=np.int64)
    y_class = np.asarray(y_class_rows, dtype=np.int64)
    y_inf = np.asarray(y_inf_rows, dtype=np.int64)
    y_sub = np.asarray(y_sub_rows, dtype=np.int64)
    y_prog = np.asarray(y_prog_rows, dtype=np.float32)
    reg_mask = np.asarray(reg_mask_rows, dtype=np.int64)
    samples_df = pd.DataFrame(sample_rows)

    samples_df.to_csv(OUT_SAMPLES_CSV, index=False)

    return x, lengths, y_class, y_inf, y_sub, y_prog, reg_mask, samples_df, class_map, sub_map, inv_sub_map


def sub_class_weights(y_sub_train: np.ndarray, y_inf_train: np.ndarray, n_subclasses: int):
    ys = y_sub_train[y_inf_train == 1]
    cnt = np.bincount(ys, minlength=n_subclasses).astype(float)
    cnt = np.maximum(cnt, 1.0)
    w = cnt.sum() / (n_subclasses * cnt)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def decode_class_from_outputs(p_inf: np.ndarray, sub_logits: np.ndarray, thr: float):
    sub_pred = sub_logits.argmax(axis=1)
    y_pred = np.where(p_inf >= thr, sub_pred + 1, 0)
    return y_pred.astype(np.int64), sub_pred.astype(np.int64)


def compute_metrics(y_true_class, y_pred_class, y_true_prog, y_pred_prog, reg_mask, n_classes):
    out = {}
    out["acc"] = float((y_true_class == y_pred_class).mean()) if len(y_true_class) else np.nan
    labels_all = list(range(n_classes))
    out["f1_macro_all"] = float(f1_score(y_true_class, y_pred_class, labels=labels_all, average="macro", zero_division=0))

    inf = y_true_class > 0
    if inf.any():
        labels_inf = sorted(list(set(y_true_class[inf].tolist())))
        out["f1_macro_informative"] = float(f1_score(y_true_class[inf], y_pred_class[inf], labels=labels_inf, average="macro", zero_division=0))
    else:
        out["f1_macro_informative"] = np.nan

    out["recall_class0"] = float(((y_pred_class[y_true_class == 0] == 0).mean())) if (y_true_class == 0).any() else np.nan

    m = reg_mask.astype(bool)
    if m.any():
        out["mae_progress"] = float(np.abs(y_true_prog[m] - y_pred_prog[m]).mean())
        out["rmse_progress"] = float(np.sqrt(np.mean((y_true_prog[m] - y_pred_prog[m]) ** 2)))
        early = m & (y_true_prog < 0.5)
        out["mae_progress_early"] = float(np.abs(y_true_prog[early] - y_pred_prog[early]).mean()) if early.any() else np.nan
    else:
        out["mae_progress"] = np.nan
        out["rmse_progress"] = np.nan
        out["mae_progress_early"] = np.nan

    out["n_informative"] = int((y_true_class > 0).sum())
    return out


def run_epoch(model, loader, pos_weight_inf, sub_weight, device, cfg: Config, optimizer=None):
    train = optimizer is not None
    model.train(train)

    losses = []
    inf_losses = []
    sub_losses = []
    reg_losses = []

    y_true_class = []
    p_inf_all = []
    sub_logits_all = []
    y_true_prog = []
    y_pred_prog = []
    reg_masks = []

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
                sub_loss = F.cross_entropy(sub_logits[yb_mask], yb_sub[yb_mask], weight=sub_weight)
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
        inf_losses.append(float(inf_loss.detach().cpu()))
        sub_losses.append(float(sub_loss.detach().cpu()))
        reg_losses.append(float(reg_loss.detach().cpu()))

        y_true_class.append(yb_class.detach().cpu().numpy())
        p_inf_all.append(torch.sigmoid(inf_logit).detach().cpu().numpy())
        sub_logits_all.append(sub_logits.detach().cpu().numpy())
        y_true_prog.append(yb_prog.detach().cpu().numpy())
        y_pred_prog.append(prog.detach().cpu().numpy())
        reg_masks.append(yb_mask.detach().cpu().numpy().astype(np.int64))

    y_true_class = np.concatenate(y_true_class) if y_true_class else np.array([], dtype=np.int64)
    p_inf = np.concatenate(p_inf_all) if p_inf_all else np.array([], dtype=np.float32)
    sub_logits = np.concatenate(sub_logits_all) if sub_logits_all else np.zeros((0, 2), dtype=np.float32)
    y_true_prog = np.concatenate(y_true_prog) if y_true_prog else np.array([], dtype=np.float32)
    y_pred_prog = np.concatenate(y_pred_prog) if y_pred_prog else np.array([], dtype=np.float32)
    reg_masks = np.concatenate(reg_masks) if reg_masks else np.array([], dtype=np.int64)

    return {
        "loss": float(np.mean(losses)) if losses else np.nan,
        "inf_loss": float(np.mean(inf_losses)) if inf_losses else np.nan,
        "sub_loss": float(np.mean(sub_losses)) if sub_losses else np.nan,
        "reg_loss": float(np.mean(reg_losses)) if reg_losses else np.nan,
        "y_true_class": y_true_class,
        "p_inf": p_inf,
        "sub_logits": sub_logits,
        "y_true_prog": y_true_prog,
        "y_pred_prog": y_pred_prog,
        "reg_masks": reg_masks,
    }


def tune_threshold(y_true_class, p_inf, sub_logits, y_true_prog, y_pred_prog, reg_masks, n_classes):
    rows = []
    best = None
    for thr in np.linspace(0.20, 0.80, 25):
        y_pred, _ = decode_class_from_outputs(p_inf, sub_logits, float(thr))
        m = compute_metrics(y_true_class, y_pred, y_true_prog, y_pred_prog, reg_masks, n_classes)
        score = m["f1_macro_all"] + 0.25 * m["f1_macro_informative"] + 0.10 * m["recall_class0"]
        row = {
            "thr": float(thr),
            "score": float(score),
            **m,
        }
        rows.append(row)
        if best is None or row["score"] > best["score"]:
            best = row

    return best, pd.DataFrame(rows)


def main():
    seg_long = pd.read_parquet(IN_SEG_LONG)
    seg_assign = pd.read_parquet(IN_SEG_ASSIGN)

    x, lengths, y_class, y_inf, y_sub, y_prog, reg_mask, samples_df, class_map, sub_map, inv_sub_map = build_samples(seg_long, seg_assign, CFG)

    idx_train = samples_df.index[samples_df["split"] == "train"].to_numpy()
    idx_val = samples_df.index[samples_df["split"] == "val"].to_numpy()
    idx_test = samples_df.index[samples_df["split"] == "test"].to_numpy()

    n_classes = int(max(y_class.max(), 0)) + 1
    n_subclasses = int(max(y_sub.max(), 0)) + 1

    ds_train = PatternDataset(x[idx_train], lengths[idx_train], y_class[idx_train], y_inf[idx_train], y_sub[idx_train], y_prog[idx_train], reg_mask[idx_train])
    ds_val = PatternDataset(x[idx_val], lengths[idx_val], y_class[idx_val], y_inf[idx_val], y_sub[idx_val], y_prog[idx_val], reg_mask[idx_val])
    ds_test = PatternDataset(x[idx_test], lengths[idx_test], y_class[idx_test], y_inf[idx_test], y_sub[idx_test], y_prog[idx_test], reg_mask[idx_test])

    dl_train = DataLoader(ds_train, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers, pin_memory=torch.cuda.is_available())
    dl_val = DataLoader(ds_val, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers, pin_memory=torch.cuda.is_available())
    dl_test = DataLoader(ds_test, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers, pin_memory=torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GatedMultiTaskTransformer(cfg=CFG, n_subclasses=n_subclasses).to(device)

    n_pos = float(y_inf[idx_train].sum())
    n_neg = float(len(idx_train) - n_pos)
    pos_weight_inf = torch.tensor([max(n_neg / max(n_pos, 1.0), 1.0)], dtype=torch.float32, device=device)
    sub_weight = sub_class_weights(y_sub[idx_train], y_inf[idx_train], n_subclasses).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=CFG.min_lr)

    history_rows = []
    best_state = None
    best_val_loss = np.inf
    wait = 0

    for epoch in range(1, CFG.max_epochs + 1):
        tr = run_epoch(model, dl_train, pos_weight_inf, sub_weight, device, CFG, optimizer=optimizer)
        va = run_epoch(model, dl_val, pos_weight_inf, sub_weight, device, CFG, optimizer=None)
        scheduler.step(va["loss"])

        best_thr_val, _ = tune_threshold(
            y_true_class=va["y_true_class"],
            p_inf=va["p_inf"],
            sub_logits=va["sub_logits"],
            y_true_prog=va["y_true_prog"],
            y_pred_prog=va["y_pred_prog"],
            reg_masks=va["reg_masks"],
            n_classes=n_classes,
        )

        y_pred_tr, _ = decode_class_from_outputs(tr["p_inf"], tr["sub_logits"], float(best_thr_val["thr"]))
        tr_m = compute_metrics(tr["y_true_class"], y_pred_tr, tr["y_true_prog"], tr["y_pred_prog"], tr["reg_masks"], n_classes)

        history_rows.append(
            {
                "epoch": epoch,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "train_loss": tr["loss"],
                "train_inf_loss": tr["inf_loss"],
                "train_sub_loss": tr["sub_loss"],
                "train_reg_loss": tr["reg_loss"],
                "train_f1_macro_all": tr_m["f1_macro_all"],
                "train_f1_macro_informative": tr_m["f1_macro_informative"],
                "train_recall_class0": tr_m["recall_class0"],
                "train_mae_progress": tr_m["mae_progress"],
                "val_loss": va["loss"],
                "val_inf_loss": va["inf_loss"],
                "val_sub_loss": va["sub_loss"],
                "val_reg_loss": va["reg_loss"],
                "val_thr": float(best_thr_val["thr"]),
                "val_f1_macro_all": float(best_thr_val["f1_macro_all"]),
                "val_f1_macro_informative": float(best_thr_val["f1_macro_informative"]),
                "val_recall_class0": float(best_thr_val["recall_class0"]),
                "val_mae_progress": float(best_thr_val["mae_progress"]),
            }
        )

        if va["loss"] < best_val_loss - 1e-4:
            best_val_loss = va["loss"]
            wait = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= CFG.early_stop_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # final threshold tune on val
    va = run_epoch(model, dl_val, pos_weight_inf, sub_weight, device, CFG, optimizer=None)
    best_thr_val, scan_df = tune_threshold(
        y_true_class=va["y_true_class"],
        p_inf=va["p_inf"],
        sub_logits=va["sub_logits"],
        y_true_prog=va["y_true_prog"],
        y_pred_prog=va["y_pred_prog"],
        reg_masks=va["reg_masks"],
        n_classes=n_classes,
    )
    scan_df.to_csv(OUT_VAL_THRESH_CSV, index=False)

    # test
    te = run_epoch(model, dl_test, pos_weight_inf, sub_weight, device, CFG, optimizer=None)
    y_pred_test, y_pred_sub = decode_class_from_outputs(te["p_inf"], te["sub_logits"], float(best_thr_val["thr"]))
    te_m = compute_metrics(te["y_true_class"], y_pred_test, te["y_true_prog"], te["y_pred_prog"], te["reg_masks"], n_classes)

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(OUT_HISTORY_CSV, index=False)

    test_meta = samples_df.iloc[idx_test].reset_index(drop=True).copy()
    test_meta["pred_inf_prob"] = te["p_inf"]
    test_meta["pred_subclass"] = y_pred_sub
    test_meta["pred_class"] = y_pred_test
    test_meta["pred_progress"] = te["y_pred_prog"]
    test_meta.to_csv(OUT_TEST_PRED_CSV, index=False)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": CFG.__dict__,
            "n_classes": n_classes,
            "n_subclasses": n_subclasses,
            "class_map": class_map,
            "sub_map": sub_map,
            "inv_sub_map": inv_sub_map,
            "best_thr": float(best_thr_val["thr"]),
        },
        OUT_MODEL_PT,
    )

    class_counts = samples_df.groupby(["split", "y_class"]).size().unstack(fill_value=0)

    summary = {
        "symbol": CFG.symbol,
        "model_tag": "phase9_1_gated_multitask_informative_plus_subclass_plus_progress",
        "input": {
            "segments_long": str(IN_SEG_LONG),
            "segments_assign": str(IN_SEG_ASSIGN),
        },
        "setup": {
            "max_seq_len": CFG.max_seq_len,
            "n_classes": n_classes,
            "n_subclasses": n_subclasses,
            "class_map": {str(k): int(v) for k, v in class_map.items()},
            "class_0": "unknown_noise_non_informative",
            "best_threshold": float(best_thr_val["thr"]),
            "loss_weights": {"inf": CFG.w_inf, "sub": CFG.w_sub, "reg": CFG.w_reg},
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
            "recall_class0": te_m["recall_class0"],
            "mae_progress": te_m["mae_progress"],
            "rmse_progress": te_m["rmse_progress"],
            "mae_progress_early": te_m["mae_progress_early"],
            "n_informative": te_m["n_informative"],
        },
        "outputs": {
            "samples_csv": str(OUT_SAMPLES_CSV),
            "history_csv": str(OUT_HISTORY_CSV),
            "threshold_scan_csv": str(OUT_VAL_THRESH_CSV),
            "test_predictions_csv": str(OUT_TEST_PRED_CSV),
            "model_pt": str(OUT_MODEL_PT),
        },
    }

    OUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

    print("Saved summary:", OUT_SUMMARY_JSON)
    print("Test metrics:", summary["test_metrics"])

    return summary, history_df, test_meta, scan_df


if __name__ == "__main__":
    main()
