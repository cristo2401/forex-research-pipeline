#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, kruskal

SEED = 42
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[2]

V62_PQ = ROOT / "phases/phase06_context_regimes/output_context_words_GBPUSD_v6_2_balanced_from_v5_1/GBPUSD_context_words_windows_v6_2.parquet"
V62_SUMMARY = ROOT / "phases/phase06_context_regimes/output_context_words_GBPUSD_v6_2_balanced_from_v5_1/GBPUSD_context_words_summary_v6_2.json"

V63_DIR = ROOT / "phases/phase06_context_regimes/output_context_words_GBPUSD_v6_3_hybrid_from_v5_1"
V63_FUSED_PQ = V63_DIR / "GBPUSD_context_words_fused_v6_3.parquet"
V63_SUMMARY = V63_DIR / "GBPUSD_context_words_hybrid_summary_v6_3.json"

OUT_DIR = ROOT / "phases/phase06_context_regimes/output_context_words_GBPUSD_v6_4_hybrid_balanced_from_v5_1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS_PQ = OUT_DIR / "GBPUSD_context_words_hybrid_balanced_v6_4.parquet"
LABELS_CSV = OUT_DIR / "GBPUSD_context_words_hybrid_balanced_v6_4.csv"
PROFILE_CSV = OUT_DIR / "GBPUSD_context_words_hybrid_balanced_profile_v6_4.csv"
TRANS_CSV = OUT_DIR / "GBPUSD_context_words_hybrid_balanced_transition_matrix_v6_4.csv"
SELECTION_CSV = OUT_DIR / "GBPUSD_context_words_hybrid_balanced_selection_v6_4.csv"
METRICS_CSV = OUT_DIR / "GBPUSD_context_words_hybrid_balanced_metrics_v6_4.csv"
SUMMARY_JSON = OUT_DIR / "GBPUSD_context_words_hybrid_balanced_summary_v6_4.json"

FOLDS = [(0.50, 0.10), (0.60, 0.10), (0.70, 0.10), (0.80, 0.10)]
TOPN_CANDIDATES = [6, 8, 10, 12, 15, 20, 25, 30, 40, 60, 80]
MIN_CLUSTER_N = 20

# Walk-forward feasibility for top_n
MIN_WF_COV = 0.65
MIN_WF_GROUPS = 3.0


def zscore(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    if not np.isfinite(std) or std < 1e-9:
        return np.zeros_like(arr)
    return (arr - mean) / (std + 1e-9)


def evaluate_groups(df: pd.DataFrame, label_col: str, ret_col: str = "ret_fwd_24h_pips") -> dict:
    groups = []
    for _, g in df.groupby(label_col):
        x = pd.to_numeric(g[ret_col], errors="coerce").dropna().to_numpy(dtype=float)
        if len(x) >= MIN_CLUSTER_N:
            groups.append(x)

    if len(groups) >= 2:
        kw_p = float(kruskal(*groups).pvalue)
        anova_p = float(f_oneway(*groups).pvalue)
        allv = np.concatenate(groups)
        gm = np.mean(allv)
        ssb = sum(len(g) * (np.mean(g) - gm) ** 2 for g in groups)
        sst = np.sum((allv - gm) ** 2)
        eta2 = float(ssb / sst) if sst > 0 else np.nan
        coverage = float(sum(len(g) for g in groups) / len(df)) if len(df) else np.nan
    else:
        kw_p, anova_p, eta2, coverage = np.nan, np.nan, np.nan, np.nan

    return {
        "n_groups_active": int(len(groups)),
        "coverage_active_groups": coverage,
        "kw_pvalue": kw_p,
        "anova_pvalue": anova_p,
        "eta2": eta2,
    }


def ensure_v63_fused() -> None:
    if V63_FUSED_PQ.exists():
        return
    script = ROOT / "phases/phase06_context_regimes/run_v6_3_hybrid.py"
    import importlib.util

    spec = importlib.util.spec_from_file_location("run_v6_3_hybrid", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


def build_folds(idx_tv: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    n = len(idx_tv)
    folds = []
    for tr_ratio, va_ratio in FOLDS:
        a = int(n * tr_ratio)
        b = min(n, int(n * (tr_ratio + va_ratio)))
        if b - a >= 30:
            folds.append((idx_tv[:a], idx_tv[a:b]))
    return folds


def main() -> None:
    ensure_v63_fused()

    v62 = pd.read_parquet(V62_PQ).copy()
    v62 = v62[
        [
            "base_window_id_end",
            "start_ts",
            "end_ts",
            "split",
            "ret_fwd_24h_pips",
            "word_id",
            "word_token",
        ]
    ].rename(columns={"word_id": "word_v62_id", "word_token": "word_v62_token"})

    fused = pd.read_parquet(V63_FUSED_PQ).copy()
    fused = fused[["base_window_id_end", "word_fused_id", "word_fused_token"]]

    df = (
        v62.merge(fused, on="base_window_id_end", how="inner")
        .sort_values("base_window_id_end")
        .reset_index(drop=True)
    )
    df["pair_token"] = df["word_v62_id"].map(lambda x: f"w{int(x):02d}") + "_" + df["word_fused_token"].astype(str)

    idx_tv = np.where(df["split"].isin(["train", "val"]).to_numpy())[0]
    folds = build_folds(idx_tv)

    sel_rows = []
    for top_n in TOPN_CANDIDATES:
        eta_f, cov_f, ng_f = [], [], []
        for tr_i, va_i in folds:
            keep = set(df.iloc[tr_i]["pair_token"].value_counts().head(top_n).index)
            va = df.iloc[va_i].copy()
            va["label"] = va["pair_token"].where(va["pair_token"].isin(keep), "other")
            m = evaluate_groups(va, "label")
            eta_f.append(m["eta2"])
            cov_f.append(m["coverage_active_groups"])
            ng_f.append(m["n_groups_active"])

        sel_rows.append(
            {
                "top_n": int(top_n),
                "wf_eta": float(np.nanmean(eta_f)) if eta_f else np.nan,
                "wf_cov": float(np.nanmean(cov_f)) if cov_f else np.nan,
                "wf_n_groups": float(np.nanmean(ng_f)) if ng_f else np.nan,
            }
        )

    sel = pd.DataFrame(sel_rows)
    sel["z_eta"] = zscore(sel["wf_eta"])
    sel["z_cov"] = zscore(sel["wf_cov"])
    sel["z_ng"] = zscore(sel["wf_n_groups"])
    sel["wf_score_balanced"] = 1.3 * sel["z_eta"] + 1.0 * sel["z_cov"] + 0.5 * sel["z_ng"]
    sel["feasible"] = (sel["wf_cov"] >= MIN_WF_COV) & (sel["wf_n_groups"] >= MIN_WF_GROUPS)

    if sel["feasible"].any():
        sel_use = sel[sel["feasible"]].sort_values("wf_score_balanced", ascending=False).reset_index(drop=True)
        selection_mode = "feasible"
    else:
        sel_use = sel.sort_values("wf_score_balanced", ascending=False).reset_index(drop=True)
        selection_mode = "fallback"

    best_top_n = int(sel_use.iloc[0]["top_n"])

    keep_global = set(
        df[df["split"].isin(["train", "val"])]["pair_token"].value_counts().head(best_top_n).index
    )
    df["word_hybrid_balanced_token"] = df["pair_token"].where(df["pair_token"].isin(keep_global), "other")
    df["word_hybrid_balanced_id"] = pd.factorize(df["word_hybrid_balanced_token"], sort=True)[0].astype(int)

    labels = df[
        [
            "base_window_id_end",
            "start_ts",
            "end_ts",
            "split",
            "ret_fwd_24h_pips",
            "word_v62_id",
            "word_v62_token",
            "word_fused_id",
            "word_fused_token",
            "pair_token",
            "word_hybrid_balanced_id",
            "word_hybrid_balanced_token",
        ]
    ].copy()
    labels.to_parquet(LABELS_PQ, index=False)
    labels.to_csv(LABELS_CSV, index=False)

    test_df = labels[labels["split"] == "test"].copy()
    profile = (
        test_df.groupby("word_hybrid_balanced_token")["ret_fwd_24h_pips"]
        .agg(n="count", mean_ret="mean", median_ret="median", std_ret="std")
        .reset_index()
        .sort_values("n", ascending=False)
        .reset_index(drop=True)
    )
    profile["t_like"] = profile["mean_ret"] / profile["std_ret"].replace(0, np.nan)
    profile.to_csv(PROFILE_CSV, index=False)

    trans = pd.crosstab(labels["word_hybrid_balanced_token"], labels["word_hybrid_balanced_token"].shift(-1)).fillna(0)
    trans_prob = trans.div(trans.sum(axis=1), axis=0).fillna(0.0)
    trans_prob.to_csv(TRANS_CSV)

    sel.sort_values("wf_score_balanced", ascending=False).to_csv(SELECTION_CSV, index=False)

    m_v62 = evaluate_groups(test_df.rename(columns={"word_v62_id": "label"}), "label")
    m_pair = evaluate_groups(test_df.rename(columns={"pair_token": "label"}), "label")
    m_bal = evaluate_groups(test_df.rename(columns={"word_hybrid_balanced_token": "label"}), "label")

    metrics = pd.DataFrame(
        [
            {"model": "phase6_v6_2_base", "n_test_rows": int(len(test_df)), **m_v62},
            {"model": "phase6_v6_3_pair_raw", "n_test_rows": int(len(test_df)), **m_pair},
            {"model": "phase6_v6_4_hybrid_balanced", "n_test_rows": int(len(test_df)), **m_bal},
        ]
    )
    metrics.to_csv(METRICS_CSV, index=False)

    v62_summary = json.loads(V62_SUMMARY.read_text()) if V62_SUMMARY.exists() else {}
    v63_summary = json.loads(V63_SUMMARY.read_text()) if V63_SUMMARY.exists() else {}

    summary = {
        "symbol": "GBPUSD",
        "model_tag": "phase6_context_words_v6_4_hybrid_balanced_from_v5_1",
        "source_models": [
            "phase6_v6_2_balanced_from_v5_1",
            "phase6_v6_3_fused_from_v5_1",
        ],
        "selection_mode": selection_mode,
        "selection_constraints": {
            "min_wf_cov": MIN_WF_COV,
            "min_wf_n_groups": MIN_WF_GROUPS,
        },
        "best_top_n": best_top_n,
        "n_rows": int(len(labels)),
        "split_counts": {k: int(v) for k, v in labels["split"].value_counts().to_dict().items()},
        "metrics": {
            row["model"]: {k: row[k] for k in metrics.columns if k != "model"}
            for _, row in metrics.iterrows()
        },
        "baseline_v6_2_eta2_test": v62_summary.get("eta2_test"),
        "baseline_v6_2_kw_pvalue_test": v62_summary.get("kw_pvalue_test"),
        "baseline_v6_3_hybrid_pair_eta2_test": (
            v63_summary.get("metrics", {}).get("phase6_v6_3_hybrid_pair", {}).get("eta2_test")
        ),
        "baseline_v6_3_hybrid_pair_kw_pvalue_test": (
            v63_summary.get("metrics", {}).get("phase6_v6_3_hybrid_pair", {}).get("kw_pvalue_test")
        ),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

    print("Saved:")
    print(" -", LABELS_PQ)
    print(" -", METRICS_CSV)
    print(" -", SUMMARY_JSON)
    print("\nMetrics:")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
