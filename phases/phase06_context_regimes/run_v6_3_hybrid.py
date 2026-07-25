#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, kruskal
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

SEED = 42
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[2]
SEARCH_CSV = ROOT / "phases/phase06_context_regimes/output_context_words_GBPUSD_v6_3_fusion_from_v5_1/search_v6_3_candidates.csv"
BASE_PQ = ROOT / "phases/phase05_context_sequencing/output_superclustering_contextual_GBPUSD_v5_kwalk/GBPUSD_context_window_embeddings_v5_1.parquet"
V62_PQ = ROOT / "phases/phase06_context_regimes/output_context_words_GBPUSD_v6_2_balanced_from_v5_1/GBPUSD_context_words_windows_v6_2.parquet"
V62_SUM = ROOT / "phases/phase06_context_regimes/output_context_words_GBPUSD_v6_2_balanced_from_v5_1/GBPUSD_context_words_summary_v6_2.json"

OUT_DIR = ROOT / "phases/phase06_context_regimes/output_context_words_GBPUSD_v6_3_hybrid_from_v5_1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FUSED_PQ = OUT_DIR / "GBPUSD_context_words_fused_v6_3.parquet"
FUSED_CSV = OUT_DIR / "GBPUSD_context_words_fused_v6_3.csv"
HYB_PQ = OUT_DIR / "GBPUSD_context_words_hybrid_v6_3.parquet"
HYB_CSV = OUT_DIR / "GBPUSD_context_words_hybrid_v6_3.csv"
PROFILE_CSV = OUT_DIR / "GBPUSD_context_words_hybrid_profile_v6_3.csv"
TRANS_CSV = OUT_DIR / "GBPUSD_context_words_hybrid_transition_matrix_v6_3.csv"
METRICS_CSV = OUT_DIR / "GBPUSD_context_words_hybrid_metrics_v6_3.csv"
SUMMARY_JSON = OUT_DIR / "GBPUSD_context_words_hybrid_summary_v6_3.json"

MIN_CLUSTER_N_TEST = 20
EMB_DUMP_DIM = 64


def parse_ngram(value) -> tuple[int, int]:
    if isinstance(value, str):
        txt = value.strip().replace("[", "").replace("]", "").replace(" ", "")
        a, b = txt.split(",")
        return int(a), int(b)
    if isinstance(value, (list, tuple)):
        return int(value[0]), int(value[1])
    raise ValueError(f"Unsupported ngram_range format: {value!r}")


def choose_robust_config(search_df: pd.DataFrame) -> pd.Series:
    df = search_df.copy()
    for c in ["wf_score", "wf_eta", "wf_cov", "wf_bal", "test_eta2", "test_kw_p"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    robust = df[
        (df["selection_mode"].eq("feasible"))
        & (df["wf_score"] >= 3.0)
        & (df["wf_eta"] >= 0.020)
        & (df["wf_cov"] >= 0.90)
        & (df["wf_bal"] >= 0.42)
    ].copy()

    if robust.empty:
        robust = df[df["selection_mode"].eq("feasible")].copy()
    if robust.empty:
        robust = df.copy()

    # Prioriza estabilidad walk-forward y luego separacion.
    robust = robust.sort_values(
        ["wf_cov", "wf_score", "wf_eta", "wf_bal"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return robust.iloc[0]


def evaluate_groups(df: pd.DataFrame, label_col: str, ret_col: str = "ret_fwd_24h_pips") -> dict:
    test_df = df[df["split"] == "test"].copy()
    groups = []
    for _, g in test_df.groupby(label_col):
        x = pd.to_numeric(g[ret_col], errors="coerce").dropna().to_numpy(dtype=float)
        if len(x) >= MIN_CLUSTER_N_TEST:
            groups.append(x)

    if len(groups) >= 2:
        kw_p = float(kruskal(*groups).pvalue)
        anova_p = float(f_oneway(*groups).pvalue)
        allv = np.concatenate(groups)
        gm = np.mean(allv)
        ssb = sum(len(g) * (np.mean(g) - gm) ** 2 for g in groups)
        sst = np.sum((allv - gm) ** 2)
        eta2 = float(ssb / sst) if sst > 0 else np.nan
        coverage = float(sum(len(g) for g in groups) / len(test_df)) if len(test_df) else np.nan
    else:
        kw_p, anova_p, eta2, coverage = np.nan, np.nan, np.nan, np.nan

    return {
        "n_test_rows": int(len(test_df)),
        "n_groups_active_test": int(len(groups)),
        "coverage_test_active_groups": coverage,
        "kw_pvalue_test": kw_p,
        "anova_pvalue_test": anova_p,
        "eta2_test": eta2,
    }


def main() -> None:
    search_df = pd.read_csv(SEARCH_CSV)
    best_cfg = choose_robust_config(search_df)

    context_len = int(best_cfg["context_len"])
    ngram_range = parse_ngram(best_cfg["ngram_range"])
    min_df = int(best_cfg["min_df"])
    max_df = float(best_cfg["max_df"])
    svd_dim = int(best_cfg["svd_dim"])
    extra_name = str(best_cfg["extra_name"])
    extra_weight = float(best_cfg["extra_weight"])
    best_k = int(best_cfg["best_k"])

    base = pd.read_parquet(BASE_PQ).copy()
    base["start_ts"] = pd.to_datetime(base["start_ts"], errors="coerce")
    base["end_ts"] = pd.to_datetime(base["end_ts"], errors="coerce")
    base = base.sort_values(["end_ts", "window_id"]).reset_index(drop=True)

    emb_cols = [c for c in base.columns if c.startswith("ctx_emb_")]
    reg_tokens = base["regime_token"].astype(str).to_numpy()
    reg_ids = base["regime_id"].astype(int).to_numpy()
    split = base["split"].astype(str).to_numpy()
    ret = pd.to_numeric(base["ret_fwd_24h_pips"], errors="coerce").to_numpy(dtype=float)
    starts = base["start_ts"].to_numpy(dtype="datetime64[ns]")
    ends = base["end_ts"].to_numpy(dtype="datetime64[ns]")
    window_ids = base["window_id"].to_numpy(dtype=int)
    emb = base[emb_cols].to_numpy(dtype=float)

    rows = []
    extra_last = []
    extra_mean = []
    extra_delta = []
    for i in range(context_len - 1, len(base)):
        j0 = i - context_len + 1
        seq = reg_tokens[j0 : i + 1]
        e_ctx = emb[j0 : i + 1]
        rows.append(
            {
                "ctx_window_id_fused": int(i - (context_len - 1)),
                "base_window_id_end": int(window_ids[i]),
                "start_ts": pd.Timestamp(starts[j0]),
                "end_ts": pd.Timestamp(ends[i]),
                "split": str(split[i]),
                "token_text": " ".join(seq.tolist()),
                "last_regime_id": int(reg_ids[i]),
                "ret_fwd_24h_pips": float(ret[i]) if np.isfinite(ret[i]) else np.nan,
            }
        )
        extra_last.append(e_ctx[-1])
        extra_mean.append(e_ctx.mean(axis=0))
        extra_delta.append(e_ctx[-1] - e_ctx[0])

    fused = pd.DataFrame(rows)
    extras = {
        "none": None,
        "last": np.vstack(extra_last),
        "mean": np.vstack(extra_mean),
        "delta": np.vstack(extra_delta),
    }

    idx_tv = np.where(fused["split"].isin(["train", "val"]).to_numpy())[0]
    texts = fused["token_text"].astype(str).to_numpy()

    vec = TfidfVectorizer(
        analyzer="word",
        lowercase=False,
        token_pattern=r"[^\s]+",
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
    )
    x_tv = vec.fit_transform(texts[idx_tv])
    x_all = vec.transform(texts)

    if x_tv.shape[1] <= 2:
        z_tv = x_tv.toarray()
        z_all = x_all.toarray()
    else:
        n_comp = int(min(svd_dim, max(2, x_tv.shape[1] - 1)))
        svd = TruncatedSVD(n_components=n_comp, random_state=SEED)
        z_tv = svd.fit_transform(x_tv)
        z_all = svd.transform(x_all)

    sc_t = StandardScaler()
    z_tv = sc_t.fit_transform(z_tv)
    z_all = sc_t.transform(z_all)

    if extra_name != "none":
        m_all = extras[extra_name]
        sc_m = StandardScaler()
        m_tv = sc_m.fit_transform(m_all[idx_tv])
        m_all = sc_m.transform(m_all)
        z_tv = np.hstack([z_tv, extra_weight * m_tv])
        z_all = np.hstack([z_all, extra_weight * m_all])

    km = KMeans(n_clusters=best_k, n_init=40, random_state=SEED)
    km.fit(z_tv)
    fused["word_fused_id"] = km.predict(z_all).astype(int)
    fused["word_fused_token"] = fused["word_fused_id"].map(lambda x: f"wf{x:02d}")
    fused["word_fused_dist"] = km.transform(z_all).min(axis=1).astype(float)
    for j in range(min(EMB_DUMP_DIM, z_all.shape[1])):
        fused[f"word_fused_emb_{j:03d}"] = z_all[:, j]

    fused.to_parquet(FUSED_PQ, index=False)
    fused.to_csv(FUSED_CSV, index=False)

    v62 = pd.read_parquet(V62_PQ).copy()
    v62 = v62[
        [
            "base_window_id_end",
            "split",
            "ret_fwd_24h_pips",
            "word_id",
            "word_token",
        ]
    ].rename(columns={"word_id": "word_v62_id", "word_token": "word_v62_token"})

    hyb = v62.merge(
        fused[
            [
                "base_window_id_end",
                "start_ts",
                "end_ts",
                "split",
                "ret_fwd_24h_pips",
                "word_fused_id",
                "word_fused_token",
                "word_fused_dist",
            ]
        ],
        on="base_window_id_end",
        how="inner",
        suffixes=("_v62", "_fused"),
    )

    hyb["split"] = hyb["split_fused"].fillna(hyb["split_v62"])
    hyb["ret_fwd_24h_pips"] = pd.to_numeric(
        hyb["ret_fwd_24h_pips_fused"].where(hyb["ret_fwd_24h_pips_fused"].notna(), hyb["ret_fwd_24h_pips_v62"]),
        errors="coerce",
    )
    hyb["word_hybrid_token"] = hyb["word_v62_id"].map(lambda x: f"w{int(x):02d}") + "_" + hyb["word_fused_token"].astype(str)
    hyb["word_hybrid_id"] = pd.factorize(hyb["word_hybrid_token"], sort=True)[0].astype(int)

    keep_cols = [
        "base_window_id_end",
        "start_ts",
        "end_ts",
        "split",
        "ret_fwd_24h_pips",
        "word_v62_id",
        "word_v62_token",
        "word_fused_id",
        "word_fused_token",
        "word_fused_dist",
        "word_hybrid_id",
        "word_hybrid_token",
    ]
    hyb = hyb[keep_cols].sort_values("base_window_id_end").reset_index(drop=True)

    m_v62 = evaluate_groups(hyb.rename(columns={"word_v62_id": "label"}), label_col="label")
    m_fused = evaluate_groups(hyb.rename(columns={"word_fused_id": "label"}), label_col="label")
    m_hybrid = evaluate_groups(hyb.rename(columns={"word_hybrid_token": "label"}), label_col="label")

    metrics_rows = [
        {"model": "phase6_v6_2_base", **m_v62},
        {"model": "phase6_v6_3_fused", **m_fused},
        {"model": "phase6_v6_3_hybrid_pair", **m_hybrid},
    ]
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(METRICS_CSV, index=False)

    test_h = hyb[hyb["split"] == "test"].copy()
    prof = (
        test_h.groupby("word_hybrid_token")["ret_fwd_24h_pips"]
        .agg(n="count", mean_ret="mean", median_ret="median", std_ret="std")
        .reset_index()
        .sort_values("n", ascending=False)
        .reset_index(drop=True)
    )
    prof["t_like"] = prof["mean_ret"] / prof["std_ret"].replace(0, np.nan)
    prof.to_csv(PROFILE_CSV, index=False)

    tr = pd.crosstab(hyb["word_hybrid_token"], hyb["word_hybrid_token"].shift(-1)).fillna(0)
    tr_prob = tr.div(tr.sum(axis=1), axis=0).fillna(0.0)
    tr_prob.to_csv(TRANS_CSV)

    v62_sum = json.loads(V62_SUM.read_text()) if V62_SUM.exists() else {}
    summary = {
        "symbol": "GBPUSD",
        "model_tag": "phase6_context_words_v6_3_hybrid_from_v5_1",
        "source_models": [
            "phase6_v6_2_balanced_from_v5_1",
            "phase6_v6_3_fused_from_v5_1",
        ],
        "selection_rule": {
            "type": "robust_walkforward_filter",
            "filters": {
                "selection_mode": "feasible",
                "wf_score_min": 3.0,
                "wf_eta_min": 0.020,
                "wf_cov_min": 0.90,
                "wf_bal_min": 0.42,
            },
            "sort": ["wf_cov desc", "wf_score desc", "wf_eta desc", "wf_bal desc"],
        },
        "selected_fused_config": {
            "context_len": context_len,
            "ngram_range": list(ngram_range),
            "min_df": min_df,
            "max_df": max_df,
            "svd_dim": svd_dim,
            "extra_name": extra_name,
            "extra_weight": extra_weight,
            "best_k": best_k,
            "wf_score": float(best_cfg["wf_score"]),
            "wf_eta": float(best_cfg["wf_eta"]),
            "wf_cov": float(best_cfg["wf_cov"]),
            "wf_bal": float(best_cfg["wf_bal"]),
            "diagnostic_test_eta2_from_search": float(best_cfg["test_eta2"]),
            "diagnostic_test_kw_p_from_search": float(best_cfg["test_kw_p"]),
        },
        "n_rows_fused": int(len(fused)),
        "n_rows_hybrid": int(len(hyb)),
        "split_counts_hybrid": {k: int(v) for k, v in hyb["split"].value_counts().to_dict().items()},
        "metrics": {r["model"]: {k: v for k, v in r.items() if k != "model"} for r in metrics_rows},
        "baseline_v6_2_eta2_test": v62_sum.get("eta2_test"),
        "baseline_v6_2_kw_pvalue_test": v62_sum.get("kw_pvalue_test"),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

    hyb.to_parquet(HYB_PQ, index=False)
    hyb.to_csv(HYB_CSV, index=False)

    print("Saved:")
    print(" -", FUSED_PQ)
    print(" -", HYB_PQ)
    print(" -", METRICS_CSV)
    print(" -", SUMMARY_JSON)
    print("\nMetrics:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
