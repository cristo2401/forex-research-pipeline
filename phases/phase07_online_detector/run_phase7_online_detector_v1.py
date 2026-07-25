#!/usr/bin/env python3
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

ROOT = Path(__file__).resolve().parents[2]
IN_DIR = ROOT / "phases/phase04_transfer_clustering/output_transfer_GBPUSD_full_v2"
IN_LONG = IN_DIR / "GBPUSD_segments_long.parquet"
IN_ASSIGN = IN_DIR / "GBPUSD_segment_cluster_assignments.parquet"

OUT_DIR = ROOT / "phases/phase07_online_detector/output_online_detector_GBPUSD_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_BARS_PQ = OUT_DIR / "GBPUSD_online_bars_labeled_v1.parquet"
OUT_WINDOWS_NPZ = OUT_DIR / "GBPUSD_online_windows_v1.npz"
OUT_MODEL_PT = OUT_DIR / "online_detector_1d_v1.pt"
OUT_PRED_CSV = OUT_DIR / "GBPUSD_online_test_predictions_v1.csv"
OUT_CONFUSION_CSV = OUT_DIR / "GBPUSD_online_confusion_matrix_v1.csv"
OUT_HISTORY_CSV = OUT_DIR / "GBPUSD_online_train_history_v1.csv"
OUT_SUMMARY_JSON = OUT_DIR / "GBPUSD_online_detector_summary_v1.json"


@dataclass
class Config:
    symbol: str = "GBPUSD"
    window_bars: int = 256
    dist_norm_q_unknown: float = 0.80
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 16
    early_stop_patience: int = 4
    lambda_progress: float = 0.6
    min_lr: float = 1e-5
    num_workers: int = 0


CFG = Config()


class WindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y_cls: np.ndarray, y_prog: np.ndarray, y_mask: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y_cls = torch.tensor(y_cls, dtype=torch.long)
        self.y_prog = torch.tensor(y_prog, dtype=torch.float32)
        self.y_mask = torch.tensor(y_mask, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y_cls)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y_cls[idx], self.y_prog[idx], self.y_mask[idx]


class OnlineDetector1D(nn.Module):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 96, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
            nn.Conv1d(96, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 160, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(160),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.trunk = nn.Sequential(
            nn.Flatten(),
            nn.Linear(160, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.cls_head = nn.Linear(128, n_classes)
        self.prog_head = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # input: [B, L, C] -> [B, C, L]
        z = x.transpose(1, 2)
        z = self.encoder(z)
        z = self.trunk(z)
        logits = self.cls_head(z)
        prog = self.prog_head(z).squeeze(-1)
        return logits, prog


def load_and_label_bars(cfg: Config) -> Tuple[pd.DataFrame, Dict[int, int]]:
    long_df = pd.read_parquet(IN_LONG).copy()
    assign = pd.read_parquet(IN_ASSIGN).copy()

    keep_cols = [
        "global_segment_id",
        "cluster_id",
        "cluster_dist",
        "cluster_dist_norm",
        "is_informative_cluster",
        "selection_score",
        "n_bars",
    ]
    assign = assign[keep_cols].copy()

    bars = long_df.merge(assign, on="global_segment_id", how="left")
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], errors="coerce")

    # Resolve overlaps: keep best candidate per timestamp by informative + score + compact cluster distance
    bars = (
        bars.sort_values(
            ["timestamp", "is_informative_cluster", "selection_score", "cluster_dist"],
            ascending=[True, False, False, True],
        )
        .drop_duplicates("timestamp", keep="first")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # Progress target per segment
    bars["step_max"] = bars.groupby("global_segment_id")["step"].transform("max").astype(float)
    bars["progress"] = np.where(
        bars["step_max"] > 0,
        bars["step"].astype(float) / bars["step_max"],
        0.0,
    )
    bars["progress"] = bars["progress"].clip(0.0, 1.0)

    # Unknown class (0): non-informative or far from cluster center
    informative_mask = bars["is_informative_cluster"].fillna(False).astype(bool)
    dist_thr = float(
        pd.to_numeric(bars.loc[informative_mask, "cluster_dist_norm"], errors="coerce")
        .dropna()
        .quantile(cfg.dist_norm_q_unknown)
    )
    known_mask = informative_mask & (pd.to_numeric(bars["cluster_dist_norm"], errors="coerce") <= dist_thr)

    known_clusters = sorted(pd.to_numeric(bars.loc[known_mask, "cluster_id"], errors="coerce").dropna().astype(int).unique())
    cluster_to_class = {cid: i + 1 for i, cid in enumerate(known_clusters)}

    bars["class_id"] = 0
    bars.loc[known_mask, "class_id"] = (
        pd.to_numeric(bars.loc[known_mask, "cluster_id"], errors="coerce")
        .astype(int)
        .map(cluster_to_class)
        .fillna(0)
        .astype(int)
    )
    bars["is_known_class"] = (bars["class_id"] > 0).astype(int)
    bars["progress_target"] = np.where(bars["class_id"] > 0, bars["progress"], 0.0)
    bars["class_token"] = bars["class_id"].map(lambda x: f"c{x:02d}")

    # Feature engineering
    close_prev = bars["close"].shift(1)
    bars["ret_close"] = (bars["close"] / close_prev - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    bars["ret_open"] = (bars["open"] / close_prev - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    bars["candle_body"] = ((bars["close"] - bars["open"]) / bars["open"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    bars["range_hl"] = ((bars["high"] - bars["low"]) / bars["close"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    bars["hour"] = bars["timestamp"].dt.hour.fillna(0).astype(int)
    bars["hour_sin"] = np.sin(2.0 * np.pi * bars["hour"] / 24.0)
    bars["hour_cos"] = np.cos(2.0 * np.pi * bars["hour"] / 24.0)

    # time gap signal
    gap_h = bars["timestamp"].diff().dt.total_seconds().div(3600.0).fillna(1.0)
    bars["gap_h"] = gap_h
    bars["gap_log"] = np.log1p(np.maximum(gap_h - 1.0, 0.0))

    bars["symbol"] = cfg.symbol
    bars.to_parquet(OUT_BARS_PQ, index=False)
    return bars, cluster_to_class


def make_window_dataset(bars: pd.DataFrame, cfg: Config):
    feat_cols = ["ret_close", "ret_open", "candle_body", "range_hl", "hour_sin", "hour_cos", "gap_log"]
    feats = bars[feat_cols].to_numpy(dtype=np.float32)
    cls = bars["class_id"].to_numpy(dtype=np.int64)
    prog = bars["progress_target"].to_numpy(dtype=np.float32)
    ts = pd.to_datetime(bars["timestamp"], errors="coerce").to_numpy(dtype="datetime64[ns]")

    n = len(bars)
    i_train_end = int(n * cfg.train_ratio)
    i_val_end = int(n * (cfg.train_ratio + cfg.val_ratio))
    w = cfg.window_bars

    xs: List[np.ndarray] = []
    y_cls: List[int] = []
    y_prog: List[float] = []
    y_mask: List[float] = []
    split: List[str] = []
    end_ts: List[np.datetime64] = []

    for end_i in range(w - 1, n):
        start_i = end_i - w + 1
        if end_i < i_train_end:
            sp = "train"
        elif start_i >= i_train_end and end_i < i_val_end:
            sp = "val"
        elif start_i >= i_val_end:
            sp = "test"
        else:
            # skip boundary windows to avoid leakage
            continue

        xs.append(feats[start_i : end_i + 1])
        c = int(cls[end_i])
        y_cls.append(c)
        y_prog.append(float(prog[end_i]))
        y_mask.append(1.0 if c > 0 else 0.0)
        split.append(sp)
        end_ts.append(ts[end_i])

    x = np.stack(xs).astype(np.float32)
    y_cls = np.asarray(y_cls, dtype=np.int64)
    y_prog = np.asarray(y_prog, dtype=np.float32)
    y_mask = np.asarray(y_mask, dtype=np.float32)
    split = np.asarray(split)
    end_ts = np.asarray(end_ts)

    # normalize features using train windows only
    m_train = split == "train"
    feat_mean = x[m_train].mean(axis=(0, 1))
    feat_std = x[m_train].std(axis=(0, 1))
    feat_std = np.where(feat_std < 1e-6, 1.0, feat_std)
    x = (x - feat_mean.reshape(1, 1, -1)) / feat_std.reshape(1, 1, -1)

    np.savez_compressed(
        OUT_WINDOWS_NPZ,
        x=x,
        y_cls=y_cls,
        y_prog=y_prog,
        y_mask=y_mask,
        split=split,
        end_ts=end_ts.astype("datetime64[ns]").astype("int64"),
        feat_cols=np.asarray(feat_cols, dtype=object),
        feat_mean=feat_mean,
        feat_std=feat_std,
    )
    return x, y_cls, y_prog, y_mask, split, end_ts, feat_cols, feat_mean, feat_std


def build_loaders(
    x: np.ndarray,
    y_cls: np.ndarray,
    y_prog: np.ndarray,
    y_mask: np.ndarray,
    split: np.ndarray,
    cfg: Config,
):
    dls = {}
    for sp, shuffle in [("train", True), ("val", False), ("test", False)]:
        m = split == sp
        ds = WindowDataset(x[m], y_cls[m], y_prog[m], y_mask[m])
        dls[sp] = DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    return dls


def compute_class_weights(y_train: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (n_classes * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    ce_loss,
    device: torch.device,
    lambda_progress: float,
    train: bool = True,
):
    model.train(train)
    losses = []
    losses_cls = []
    losses_prog = []
    y_true = []
    y_pred = []
    prog_true = []
    prog_pred = []

    for xb, yb_cls, yb_prog, yb_mask in loader:
        xb = xb.to(device, non_blocking=True)
        yb_cls = yb_cls.to(device, non_blocking=True)
        yb_prog = yb_prog.to(device, non_blocking=True)
        yb_mask = yb_mask.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            logits, pred_prog = model(xb)
            loss_cls = ce_loss(logits, yb_cls)

            known = yb_mask > 0.5
            if known.any():
                loss_prog = torch.mean((pred_prog[known] - yb_prog[known]) ** 2)
            else:
                loss_prog = torch.tensor(0.0, device=device)

            loss = loss_cls + lambda_progress * loss_prog

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        losses.append(float(loss.detach().cpu()))
        losses_cls.append(float(loss_cls.detach().cpu()))
        losses_prog.append(float(loss_prog.detach().cpu()))

        preds = torch.argmax(logits, dim=1)
        y_true.append(yb_cls.detach().cpu().numpy())
        y_pred.append(preds.detach().cpu().numpy())

        prog_true.append(yb_prog.detach().cpu().numpy())
        prog_pred.append(pred_prog.detach().cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    prog_true = np.concatenate(prog_true)
    prog_pred = np.concatenate(prog_pred)

    acc = float(accuracy_score(y_true, y_pred))
    labels_known = sorted(np.unique(y_true[y_true > 0]).tolist())
    f1_macro_known = (
        float(f1_score(y_true, y_pred, labels=labels_known, average="macro", zero_division=0))
        if labels_known
        else np.nan
    )
    recall_unknown = float(np.mean(y_pred[y_true == 0] == 0)) if np.any(y_true == 0) else np.nan
    known_mask = y_true > 0
    mae_prog_known = float(np.mean(np.abs(prog_pred[known_mask] - prog_true[known_mask]))) if np.any(known_mask) else np.nan

    return {
        "loss": float(np.mean(losses)),
        "loss_cls": float(np.mean(losses_cls)),
        "loss_prog": float(np.mean(losses_prog)),
        "acc": acc,
        "f1_macro_known": f1_macro_known,
        "recall_unknown": recall_unknown,
        "mae_prog_known": mae_prog_known,
        "y_true": y_true,
        "y_pred": y_pred,
        "prog_true": prog_true,
        "prog_pred": prog_pred,
    }


def train_detector(
    dls: Dict[str, DataLoader],
    n_features: int,
    n_classes: int,
    y_cls_train: np.ndarray,
    cfg: Config,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OnlineDetector1D(n_features=n_features, n_classes=n_classes).to(device)
    class_weights = compute_class_weights(y_cls_train, n_classes=n_classes).to(device)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=1,
        min_lr=cfg.min_lr,
    )

    best_state = None
    best_val = np.inf
    wait = 0
    history = []

    for epoch in range(1, cfg.max_epochs + 1):
        tr = run_epoch(model, dls["train"], optimizer, ce_loss, device, cfg.lambda_progress, train=True)
        va = run_epoch(model, dls["val"], optimizer, ce_loss, device, cfg.lambda_progress, train=False)

        scheduler.step(va["loss"])
        lr_now = float(optimizer.param_groups[0]["lr"])

        row = {
            "epoch": epoch,
            "lr": lr_now,
            "train_loss": tr["loss"],
            "train_loss_cls": tr["loss_cls"],
            "train_loss_prog": tr["loss_prog"],
            "train_acc": tr["acc"],
            "train_f1_macro_known": tr["f1_macro_known"],
            "train_recall_unknown": tr["recall_unknown"],
            "train_mae_prog_known": tr["mae_prog_known"],
            "val_loss": va["loss"],
            "val_loss_cls": va["loss_cls"],
            "val_loss_prog": va["loss_prog"],
            "val_acc": va["acc"],
            "val_f1_macro_known": va["f1_macro_known"],
            "val_recall_unknown": va["recall_unknown"],
            "val_mae_prog_known": va["mae_prog_known"],
        }
        history.append(row)

        if va["loss"] < best_val - 1e-5:
            best_val = va["loss"]
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
    return model, history_df, device


def evaluate_and_save(
    model: nn.Module,
    dls: Dict[str, DataLoader],
    split: np.ndarray,
    end_ts: np.ndarray,
    y_cls: np.ndarray,
    y_prog: np.ndarray,
    y_mask: np.ndarray,
    feat_cols: List[str],
    feat_mean: np.ndarray,
    feat_std: np.ndarray,
    cluster_to_class: Dict[int, int],
    history_df: pd.DataFrame,
    cfg: Config,
    device: torch.device,
):
    n_classes = int(y_cls.max()) + 1
    ce_loss = nn.CrossEntropyLoss(weight=compute_class_weights(y_cls[split == "train"], n_classes).to(device))

    te = run_epoch(model, dls["test"], optimizer=None, ce_loss=ce_loss, device=device, lambda_progress=cfg.lambda_progress, train=False)
    va = run_epoch(model, dls["val"], optimizer=None, ce_loss=ce_loss, device=device, lambda_progress=cfg.lambda_progress, train=False)

    # test predictions with timestamps
    m_test = split == "test"
    test_df = pd.DataFrame(
        {
            "end_ts": pd.to_datetime(end_ts[m_test]),
            "true_class_id": y_cls[m_test],
            "pred_class_id": te["y_pred"],
            "true_progress": y_prog[m_test],
            "pred_progress": te["prog_pred"],
            "known_mask": y_mask[m_test].astype(int),
        }
    )
    test_df["ok_cls"] = (test_df["true_class_id"] == test_df["pred_class_id"]).astype(int)
    test_df.to_csv(OUT_PRED_CSV, index=False)

    cm = confusion_matrix(te["y_true"], te["y_pred"], labels=list(range(n_classes)))
    cm_df = pd.DataFrame(cm, index=[f"true_{i:02d}" for i in range(n_classes)], columns=[f"pred_{i:02d}" for i in range(n_classes)])
    cm_df.to_csv(OUT_CONFUSION_CSV)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": cfg.__dict__,
            "n_classes": n_classes,
            "n_features": len(feat_cols),
            "feat_cols": feat_cols,
            "feat_mean": feat_mean,
            "feat_std": feat_std,
            "cluster_to_class": cluster_to_class,
        },
        OUT_MODEL_PT,
    )

    summary = {
        "symbol": cfg.symbol,
        "model_tag": "phase7_online_detector_1d_v1",
        "window_bars": cfg.window_bars,
        "unknown_rule": {
            "class_0_non_informative": True,
            "class_0_dist_norm_quantile_threshold": cfg.dist_norm_q_unknown,
        },
        "dataset": {
            "n_bars_labeled": int(pd.read_parquet(OUT_BARS_PQ).shape[0]),
            "n_windows_total": int(len(split)),
            "n_windows_train": int(np.sum(split == "train")),
            "n_windows_val": int(np.sum(split == "val")),
            "n_windows_test": int(np.sum(split == "test")),
            "n_classes_total_including_0": n_classes,
        },
        "metrics_val": {
            "loss": va["loss"],
            "acc": va["acc"],
            "f1_macro_known": va["f1_macro_known"],
            "recall_unknown": va["recall_unknown"],
            "mae_progress_known": va["mae_prog_known"],
        },
        "metrics_test": {
            "loss": te["loss"],
            "acc": te["acc"],
            "f1_macro_known": te["f1_macro_known"],
            "recall_unknown": te["recall_unknown"],
            "mae_progress_known": te["mae_prog_known"],
        },
        "class_mapping_cluster_to_class": {str(k): int(v) for k, v in cluster_to_class.items()},
        "outputs": {
            "bars_labeled": str(OUT_BARS_PQ),
            "windows_npz": str(OUT_WINDOWS_NPZ),
            "model_pt": str(OUT_MODEL_PT),
            "predictions_test_csv": str(OUT_PRED_CSV),
            "confusion_csv": str(OUT_CONFUSION_CSV),
            "history_csv": str(OUT_HISTORY_CSV),
        },
    }
    OUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
    return summary, test_df, cm_df, history_df


def main():
    bars, cluster_to_class = load_and_label_bars(CFG)
    x, y_cls, y_prog, y_mask, split, end_ts, feat_cols, feat_mean, feat_std = make_window_dataset(bars, CFG)
    dls = build_loaders(x, y_cls, y_prog, y_mask, split, CFG)

    n_features = x.shape[2]
    n_classes = int(y_cls.max()) + 1
    model, history_df, device = train_detector(
        dls=dls,
        n_features=n_features,
        n_classes=n_classes,
        y_cls_train=y_cls[split == "train"],
        cfg=CFG,
    )
    summary, test_df, cm_df, history_df = evaluate_and_save(
        model=model,
        dls=dls,
        split=split,
        end_ts=end_ts,
        y_cls=y_cls,
        y_prog=y_prog,
        y_mask=y_mask,
        feat_cols=feat_cols,
        feat_mean=feat_mean,
        feat_std=feat_std,
        cluster_to_class=cluster_to_class,
        history_df=history_df,
        cfg=CFG,
        device=device,
    )

    print("Saved summary:", OUT_SUMMARY_JSON)
    print("Test metrics:", summary["metrics_test"])
    return summary, test_df, cm_df, history_df


if __name__ == "__main__":
    main()
