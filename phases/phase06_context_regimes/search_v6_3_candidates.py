#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

SEED = 42
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[2]
IN_PQ = ROOT / "phases/phase05_context_sequencing/output_superclustering_contextual_GBPUSD_v5_kwalk/GBPUSD_context_window_embeddings_v5_1.parquet"
OUT_DIR = ROOT / "phases/phase06_context_regimes/output_context_words_GBPUSD_v6_3_fusion_from_v5_1"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "search_v6_3_candidates.csv"
OUT_JSON = OUT_DIR / "search_v6_3_best_config.json"

FOLD_SPECS = [(0.50, 0.10), (0.60, 0.10), (0.70, 0.10), (0.80, 0.10)]
K_CANDIDATES = list(range(5, 17))
N_INIT = 40
MIN_CLUSTER_N_FOLD = 15
MIN_CLUSTER_N_TEST = 20
MIN_COV_WF = 0.88
MIN_BAL_WF = 0.35
MIN_ACT_WF = 3.0


def zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    if not np.isfinite(std) or std < 1e-9:
        return np.zeros_like(arr)
    return (arr - mean) / (std + 1e-9)


def build_folds(idx_tv: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    n = len(idx_tv)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for tr_ratio, va_ratio in FOLD_SPECS:
        a = int(n * tr_ratio)
        b = min(n, int(n * (tr_ratio + va_ratio)))
        if b - a >= 30:
            folds.append((idx_tv[:a], idx_tv[a:b]))
    return folds


def build_context(base: pd.DataFrame, emb_cols: list[str], context_len: int) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    reg_tokens = base["regime_token"].astype(str).to_numpy()
    reg_ids = base["regime_id"].astype(int).to_numpy()
    splits = base["split"].astype(str).to_numpy()
    rets = pd.to_numeric(base["ret_fwd_24h_pips"], errors="coerce").to_numpy(dtype=float)
    starts = base["start_ts"].to_numpy(dtype="datetime64[ns]")
    ends = base["end_ts"].to_numpy(dtype="datetime64[ns]")
    window_ids = base["window_id"].to_numpy(dtype=int)
    emb = base[emb_cols].to_numpy(dtype=float)

    rows = []
    mean_rows, std_rows, last_rows, delta_rows = [], [], [], []
    for i in range(context_len - 1, len(base)):
        j0 = i - context_len + 1
        seq = reg_tokens[j0 : i + 1]
        e_ctx = emb[j0 : i + 1]
        rows.append(
            {
                "ctx_window_id": int(i - (context_len - 1)),
                "base_window_id_end": int(window_ids[i]),
                "start_ts": pd.Timestamp(starts[j0]),
                "end_ts": pd.Timestamp(ends[i]),
                "split": str(splits[i]),
                "token_text": " ".join(seq.tolist()),
                "last_regime_id": int(reg_ids[i]),
                "ret_fwd_24h_pips": float(rets[i]) if np.isfinite(rets[i]) else np.nan,
            }
        )
        mean_rows.append(e_ctx.mean(axis=0))
        std_rows.append(e_ctx.std(axis=0))
        last_rows.append(e_ctx[-1])
        delta_rows.append(e_ctx[-1] - e_ctx[0])

    ctx = pd.DataFrame(rows)
    extras = {
        "none": None,
        "mean": np.vstack(mean_rows),
        "std": np.vstack(std_rows),
        "last": np.vstack(last_rows),
        "delta": np.vstack(delta_rows),
    }
    return ctx, extras


def eval_config(
    cfg: dict,
    base: pd.DataFrame,
    emb_cols: list[str],
) -> dict:
    context_len = int(cfg["context_len"])
    ngram_range = tuple(cfg["ngram_range"])
    min_df = int(cfg["min_df"])
    max_df = float(cfg["max_df"])
    svd_dim = int(cfg["svd_dim"])
    extra_name = str(cfg["extra_name"])
    extra_weight = float(cfg["extra_weight"])

    ctx, extras = build_context(base, emb_cols, context_len)
    texts = ctx["token_text"].astype(str).to_numpy()
    ret = ctx["ret_fwd_24h_pips"].to_numpy(dtype=float)
    split = ctx["split"].astype(str).to_numpy()

    idx_tr = np.where(split == "train")[0]
    idx_va = np.where(split == "val")[0]
    idx_tv = np.concatenate([idx_tr, idx_va])
    folds = build_folds(idx_tv)

    rows = []
    for k in K_CANDIDATES:
        eta_f, cov_f, bal_f, act_f = [], [], [], []
        for tr_i, va_i in folds:
            vec = TfidfVectorizer(
                analyzer="word",
                lowercase=False,
                token_pattern=r"[^\s]+",
                ngram_range=ngram_range,
                min_df=min_df,
                max_df=max_df,
            )
            x_tr = vec.fit_transform(texts[tr_i])
            x_va = vec.transform(texts[va_i])

            if x_tr.shape[1] <= 2:
                z_tr = x_tr.toarray()
                z_va = x_va.toarray()
            else:
                n_comp = int(min(svd_dim, max(2, x_tr.shape[1] - 1)))
                svd = TruncatedSVD(n_components=n_comp, random_state=SEED)
                z_tr = svd.fit_transform(x_tr)
                z_va = svd.transform(x_va)

            sc_t = StandardScaler()
            z_tr = sc_t.fit_transform(z_tr)
            z_va = sc_t.transform(z_va)

            if extra_name != "none":
                m_all = extras[extra_name]
                m_tr = m_all[tr_i]
                m_va = m_all[va_i]
                sc_m = StandardScaler()
                m_tr = sc_m.fit_transform(m_tr)
                m_va = sc_m.transform(m_va)
                z_tr = np.hstack([z_tr, extra_weight * m_tr])
                z_va = np.hstack([z_va, extra_weight * m_va])

            lv = KMeans(n_clusters=k, n_init=N_INIT, random_state=SEED).fit(z_tr).predict(z_va)
            cnt = pd.Series(lv).value_counts()
            active = cnt[cnt >= MIN_CLUSTER_N_FOLD]
            cov_f.append(float(active.sum() / len(lv)))
            bal_f.append(float(active.min() / active.max()) if len(active) else 0.0)
            act_f.append(float(len(active)))

            rv = ret[va_i]
            m = np.isfinite(rv)
            l = lv[m]
            r = rv[m]
            groups = []
            for cid in np.unique(l):
                g = r[l == cid]
                if len(g) >= MIN_CLUSTER_N_FOLD:
                    groups.append(g)
            if len(groups) >= 2:
                allv = np.concatenate(groups)
                gm = np.mean(allv)
                ssb = sum(len(g) * (np.mean(g) - gm) ** 2 for g in groups)
                sst = np.sum((allv - gm) ** 2)
                eta_f.append(float(ssb / sst) if sst > 0 else np.nan)

        rows.append(
            {
                "k": int(k),
                "eta_wf": float(np.nanmean(eta_f)) if eta_f else np.nan,
                "cov_wf": float(np.nanmean(cov_f)) if cov_f else np.nan,
                "bal_wf": float(np.nanmean(bal_f)) if bal_f else np.nan,
                "act_wf": float(np.nanmean(act_f)) if act_f else np.nan,
            }
        )

    sel = pd.DataFrame(rows)
    sel["z_eta"] = zscore(sel["eta_wf"].to_numpy())
    sel["z_cov"] = zscore(sel["cov_wf"].to_numpy())
    sel["z_bal"] = zscore(sel["bal_wf"].to_numpy())
    sel["z_act"] = zscore(sel["act_wf"].to_numpy())
    sel["wf_score"] = 1.2 * sel["z_eta"] + 1.0 * sel["z_cov"] + 1.0 * sel["z_bal"] + 0.2 * sel["z_act"]
    sel["feasible"] = (
        (sel["cov_wf"] >= MIN_COV_WF)
        & (sel["bal_wf"] >= MIN_BAL_WF)
        & (sel["act_wf"] >= MIN_ACT_WF)
    )

    if sel["feasible"].any():
        use = sel[sel["feasible"]].sort_values("wf_score", ascending=False).reset_index(drop=True)
        selection_mode = "feasible"
    else:
        use = sel.sort_values("wf_score", ascending=False).reset_index(drop=True)
        selection_mode = "fallback"

    best_k = int(use.iloc[0]["k"])

    # final fit (train+val -> all), test only for diagnostics
    vec_f = TfidfVectorizer(
        analyzer="word",
        lowercase=False,
        token_pattern=r"[^\s]+",
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
    )
    x_tv = vec_f.fit_transform(texts[idx_tv])
    x_all = vec_f.transform(texts)
    if x_tv.shape[1] <= 2:
        z_tv = x_tv.toarray()
        z_all = x_all.toarray()
    else:
        n_comp = int(min(svd_dim, max(2, x_tv.shape[1] - 1)))
        svd_f = TruncatedSVD(n_components=n_comp, random_state=SEED)
        z_tv = svd_f.fit_transform(x_tv)
        z_all = svd_f.transform(x_all)

    sc_tf = StandardScaler()
    z_tv = sc_tf.fit_transform(z_tv)
    z_all = sc_tf.transform(z_all)

    if extra_name != "none":
        m_all = extras[extra_name]
        sc_mf = StandardScaler()
        m_tv = sc_mf.fit_transform(m_all[idx_tv])
        m_all = sc_mf.transform(m_all)
        z_tv = np.hstack([z_tv, extra_weight * m_tv])
        z_all = np.hstack([z_all, extra_weight * m_all])

    km = KMeans(n_clusters=best_k, n_init=N_INIT, random_state=SEED).fit(z_tv)
    y = km.predict(z_all)

    test_mask = split == "test"
    yt = y[test_mask]
    rt = ret[test_mask]

    groups = []
    for cid in np.unique(yt):
        g = rt[(yt == cid) & np.isfinite(rt)]
        if len(g) >= MIN_CLUSTER_N_TEST:
            groups.append(g)

    if len(groups) >= 2:
        kw_p = float(kruskal(*groups).pvalue)
        allv = np.concatenate(groups)
        gm = np.mean(allv)
        ssb = sum(len(g) * (np.mean(g) - gm) ** 2 for g in groups)
        sst = np.sum((allv - gm) ** 2)
        eta2 = float(ssb / sst) if sst > 0 else np.nan
    else:
        kw_p = np.nan
        eta2 = np.nan

    return {
        **cfg,
        "n_context": int(len(ctx)),
        "selection_mode": selection_mode,
        "best_k": int(best_k),
        "wf_score": float(use.iloc[0]["wf_score"]),
        "wf_eta": float(use.iloc[0]["eta_wf"]) if np.isfinite(use.iloc[0]["eta_wf"]) else np.nan,
        "wf_cov": float(use.iloc[0]["cov_wf"]) if np.isfinite(use.iloc[0]["cov_wf"]) else np.nan,
        "wf_bal": float(use.iloc[0]["bal_wf"]) if np.isfinite(use.iloc[0]["bal_wf"]) else np.nan,
        "test_eta2": eta2,
        "test_kw_p": kw_p,
    }


def main() -> None:
    base = pd.read_parquet(IN_PQ).copy()
    base["start_ts"] = pd.to_datetime(base["start_ts"], errors="coerce")
    base["end_ts"] = pd.to_datetime(base["end_ts"], errors="coerce")
    base = base.sort_values(["end_ts", "window_id"]).reset_index(drop=True)
    emb_cols = [c for c in base.columns if c.startswith("ctx_emb_")]

    cfgs = []
    for context_len in [20, 24, 32]:
        for ngram_range in [(1, 2), (1, 3)]:
            for extra_name in ["none", "last", "mean", "delta"]:
                weights = [0.0] if extra_name == "none" else [0.5, 1.0, 1.5]
                for w in weights:
                    cfgs.append(
                        {
                            "context_len": int(context_len),
                            "ngram_range": list(ngram_range),
                            "min_df": 2,
                            "max_df": 0.95,
                            "svd_dim": 48,
                            "extra_name": extra_name,
                            "extra_weight": float(w),
                        }
                    )

    rows = []
    for i, cfg in enumerate(cfgs, 1):
        try:
            out = eval_config(cfg=cfg, base=base, emb_cols=emb_cols)
            rows.append(out)
            print(f"[{i}/{len(cfgs)}] {out}", flush=True)
        except Exception as exc:  # pragma: no cover
            print(f"[{i}/{len(cfgs)}] ERROR {cfg} -> {exc}", flush=True)

    if not rows:
        raise RuntimeError("No se pudo evaluar ninguna configuracion")

    res = pd.DataFrame(rows).sort_values(["wf_score", "test_eta2"], ascending=False).reset_index(drop=True)
    res.to_csv(OUT_CSV, index=False)
    best = res.iloc[0].to_dict()
    OUT_JSON.write_text(json.dumps(best, indent=2, default=str))

    print("\nTop 12 por wf_score", flush=True)
    print(res.head(12).to_string(index=False), flush=True)
    print("\nTop 12 por test_eta2 (solo diagnostico)", flush=True)
    print(res.sort_values("test_eta2", ascending=False).head(12).to_string(index=False), flush=True)
    print(f"\nGuardado: {OUT_CSV}", flush=True)
    print(f"Best cfg: {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
