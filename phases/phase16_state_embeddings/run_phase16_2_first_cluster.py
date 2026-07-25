#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


@dataclass
class Config:
    phase16_run_dir: str
    out_dir: str
    k: int = 0
    k_min: int = 6
    k_max: int = 16
    pca_dim: int = 24
    normalize_embeddings: bool = True
    max_silhouette_samples: int = 15000
    k_objective_val_weight: float = 0.65
    k_objective_purity_weight: float = 0.25
    seed: int = 42


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger('phase16_first_cluster')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    fh = logging.FileHandler(log_path, mode='w')
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n


def split_metrics(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    out = {
        'n': int(len(x)),
        'n_clusters_pred': int(np.unique(y).size),
        'silhouette': float('nan'),
        'calinski_harabasz': float('nan'),
        'davies_bouldin': float('nan'),
    }
    if len(x) < 3 or np.unique(y).size < 2:
        return out
    try:
        out['silhouette'] = float(silhouette_score(x, y))
    except Exception:
        pass
    try:
        out['calinski_harabasz'] = float(calinski_harabasz_score(x, y))
    except Exception:
        pass
    try:
        out['davies_bouldin'] = float(davies_bouldin_score(x, y))
    except Exception:
        pass
    return out


def weighted_purity_from_arrays(cluster_id: np.ndarray, symbols: np.ndarray) -> float:
    d = pd.DataFrame({'cluster_id': cluster_id.astype(int), 'symbol': symbols.astype(str)})
    c = d.groupby(['cluster_id', 'symbol']).size().rename('n').reset_index()
    tot = c.groupby('cluster_id')['n'].sum().rename('tot').reset_index()
    c = c.merge(tot, on='cluster_id', how='left')
    c['share'] = c['n'] / c['tot']
    mx = c.groupby('cluster_id')['share'].max()
    w = tot.set_index('cluster_id')['tot']
    return float((mx * w.loc[mx.index]).sum() / w.loc[mx.index].sum())


def weighted_cluster_purity(df: pd.DataFrame, cluster_col: str = 'cluster_id') -> Tuple[float, Dict[int, float]]:
    c = df.groupby([cluster_col, 'symbol']).size().rename('n').reset_index()
    tot = c.groupby(cluster_col)['n'].sum().rename('tot').reset_index()
    c = c.merge(tot, on=cluster_col, how='left')
    c['share'] = c['n'] / c['tot']

    mx = c.groupby(cluster_col)['share'].max()
    w = tot.set_index(cluster_col)['tot']
    purity = float((mx * w.loc[mx.index]).sum() / w.loc[mx.index].sum())
    return purity, {int(k): float(v) for k, v in mx.to_dict().items()}


def _silhouette_sampled(
    x: np.ndarray,
    y: np.ndarray,
    max_samples: int,
    rng: np.random.Generator,
) -> float:
    if len(x) < 3 or np.unique(y).size < 2:
        return float('nan')
    if len(x) > int(max_samples):
        idx = rng.choice(np.arange(len(x)), size=int(max_samples), replace=False)
        xs = x[idx]
        ys = y[idx]
    else:
        xs = x
        ys = y
    if np.unique(ys).size < 2:
        return float('nan')
    return float(silhouette_score(xs, ys))


def choose_k_robust(
    x_train: np.ndarray,
    x_val: np.ndarray,
    symbols_train: np.ndarray,
    symbols_val: np.ndarray,
    k_min: int,
    k_max: int,
    max_samples: int,
    val_weight: float,
    purity_weight: float,
    seed: int,
    logger: logging.Logger,
) -> Tuple[int, pd.DataFrame]:
    rng = np.random.default_rng(int(seed))

    rows = []
    best_k = None
    best_score = -np.inf
    for k in range(int(k_min), int(k_max) + 1):
        km = KMeans(n_clusters=k, random_state=int(seed), n_init='auto')
        km.fit(x_train)
        y_train = km.predict(x_train)
        y_val = km.predict(x_val) if len(x_val) else np.array([], dtype=np.int32)

        sil_train = _silhouette_sampled(x_train, y_train, max_samples=max_samples, rng=rng)
        sil_val = _silhouette_sampled(x_val, y_val, max_samples=max_samples, rng=rng) if len(x_val) else float('nan')
        purity_train = weighted_purity_from_arrays(y_train, symbols_train)
        purity_val = weighted_purity_from_arrays(y_val, symbols_val) if len(x_val) else float('nan')

        # Robust objective:
        # - prioritize temporal generalization (val)
        # - penalize symbol-collapse
        if np.isnan(sil_val):
            sil_ref = sil_train
            purity_ref = purity_train
        else:
            sil_ref = float(val_weight) * sil_val + (1.0 - float(val_weight)) * sil_train
            purity_ref = purity_val
        objective = sil_ref - float(purity_weight) * purity_ref

        rows.append({
            'k': int(k),
            'objective': float(objective),
            'silhouette_train': float(sil_train),
            'silhouette_val': float(sil_val) if not np.isnan(sil_val) else np.nan,
            'purity_train': float(purity_train),
            'purity_val': float(purity_val) if not np.isnan(purity_val) else np.nan,
        })
        logger.info(
            'k=%s objective=%.6f sil_train=%.6f sil_val=%.6f purity_train=%.6f purity_val=%.6f',
            k,
            objective,
            sil_train,
            sil_val if not np.isnan(sil_val) else float('nan'),
            purity_train,
            purity_val if not np.isnan(purity_val) else float('nan'),
        )
        if objective > best_score:
            best_score = float(objective)
            best_k = k

    if best_k is None:
        raise RuntimeError('Could not choose k')

    return int(best_k), pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description='Phase16.2 - Primer clustering sobre embeddings de estado (fit train, assign val/test).')
    ap.add_argument('--phase16_run_dir', required=True, type=str)
    ap.add_argument('--out_dir', default='', type=str)
    ap.add_argument('--k', type=int, default=0, help='If 0, auto-select with silhouette in [k_min, k_max].')
    ap.add_argument('--k_min', type=int, default=6)
    ap.add_argument('--k_max', type=int, default=16)
    ap.add_argument('--pca_dim', type=int, default=24)
    ap.add_argument('--normalize_embeddings', type=int, choices=[0, 1], default=1)
    ap.add_argument('--max_silhouette_samples', type=int, default=15000)
    ap.add_argument('--k_objective_val_weight', type=float, default=0.65)
    ap.add_argument('--k_objective_purity_weight', type=float, default=0.25)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    cfg = Config(
        phase16_run_dir=str(Path(args.phase16_run_dir).resolve()),
        out_dir=str(Path(args.out_dir).resolve()) if args.out_dir else '',
        k=int(args.k),
        k_min=int(args.k_min),
        k_max=int(args.k_max),
        pca_dim=int(args.pca_dim),
        normalize_embeddings=bool(int(args.normalize_embeddings)),
        max_silhouette_samples=int(args.max_silhouette_samples),
        k_objective_val_weight=float(args.k_objective_val_weight),
        k_objective_purity_weight=float(args.k_objective_purity_weight),
        seed=int(args.seed),
    )

    phase16_dir = Path(cfg.phase16_run_dir)
    if not phase16_dir.exists():
        raise FileNotFoundError(phase16_dir)

    out_dir = Path(cfg.out_dir) if cfg.out_dir else (phase16_dir / 'phase16_2_first_cluster')
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(out_dir / 'phase16_2.log')

    emb_path = phase16_dir / 'phase16_embeddings.parquet'
    if not emb_path.exists():
        raise FileNotFoundError(emb_path)

    d = pd.read_parquet(emb_path)
    emb_cols = [c for c in d.columns if c.startswith('emb_')]
    if not emb_cols:
        raise RuntimeError('No emb_* columns found in phase16_embeddings.parquet')
    if 'split' not in d.columns or 'symbol' not in d.columns:
        raise RuntimeError('Expected split and symbol columns')

    x = d[emb_cols].to_numpy(dtype=np.float32)
    if cfg.normalize_embeddings:
        x = l2_normalize(x)

    split = d['split'].astype(str).to_numpy()
    idx_train = np.where(split == 'train')[0]
    idx_val = np.where(split == 'val')[0]
    idx_test = np.where(split == 'test')[0]

    if len(idx_train) < 100:
        raise RuntimeError('Too few train windows for clustering')

    x_train = x[idx_train]
    x_val = x[idx_val]
    sym_arr = d['symbol'].astype(str).to_numpy()
    sym_train = sym_arr[idx_train]
    sym_val = sym_arr[idx_val]

    pca_dim = int(min(cfg.pca_dim, x.shape[1]))
    pca = PCA(n_components=pca_dim, random_state=int(cfg.seed))
    x_train_p = pca.fit_transform(x_train)
    x_all_p = pca.transform(x)

    # Select k on train only.
    if cfg.k and cfg.k > 1:
        k_use = int(cfg.k)
        k_search_df = pd.DataFrame([{
            'k': k_use,
            'objective': np.nan,
            'silhouette_train': np.nan,
            'silhouette_val': np.nan,
            'purity_train': np.nan,
            'purity_val': np.nan,
        }])
        logger.info('using fixed k=%s', k_use)
    else:
        k_use, k_search_df = choose_k_robust(
            x_train=x_train_p,
            x_val=x_all_p[idx_val],
            symbols_train=sym_train,
            symbols_val=sym_val,
            k_min=cfg.k_min,
            k_max=cfg.k_max,
            max_samples=cfg.max_silhouette_samples,
            val_weight=cfg.k_objective_val_weight,
            purity_weight=cfg.k_objective_purity_weight,
            seed=cfg.seed,
            logger=logger,
        )
        logger.info(
            'selected k=%s by robust objective (val_weight=%.3f purity_weight=%.3f)',
            k_use,
            cfg.k_objective_val_weight,
            cfg.k_objective_purity_weight,
        )

    km = KMeans(n_clusters=int(k_use), random_state=int(cfg.seed), n_init='auto')
    km.fit(x_train_p)
    y_all = km.predict(x_all_p)

    out = d.copy()
    out['cluster_id'] = y_all.astype(np.int32)

    # Metrics by split.
    rows_metrics = []
    for name, idx in [('train', idx_train), ('val', idx_val), ('test', idx_test), ('all', np.arange(len(out)))]:
        if len(idx) == 0:
            continue
        m = split_metrics(x_all_p[idx], y_all[idx])
        m['split'] = name
        rows_metrics.append(m)
    met_df = pd.DataFrame(rows_metrics)

    # Purity / symbol-mix diagnostics.
    purity_all, purity_by_cluster = weighted_cluster_purity(out, cluster_col='cluster_id')
    mix = out.groupby(['cluster_id', 'symbol']).size().rename('n').reset_index()
    mix['cluster_total'] = mix.groupby('cluster_id')['n'].transform('sum')
    mix['symbol_share_in_cluster'] = mix['n'] / mix['cluster_total']

    # Counts diagnostics.
    counts_split_symbol = out.groupby(['split', 'symbol', 'cluster_id']).size().rename('n').reset_index()
    counts_split = out.groupby(['split', 'cluster_id']).size().rename('n').reset_index()

    # Plot: PCA train by cluster.
    vis_train = pd.DataFrame({
        'pc1': x_all_p[idx_train, 0],
        'pc2': x_all_p[idx_train, 1] if x_all_p.shape[1] > 1 else np.zeros(len(idx_train)),
        'cluster_id': y_all[idx_train],
    })
    plt.figure(figsize=(10, 5))
    for cid in sorted(vis_train['cluster_id'].unique()):
        g = vis_train[vis_train['cluster_id'] == cid]
        plt.scatter(g['pc1'], g['pc2'], s=6, alpha=0.55, label=f'c{cid}')
    plt.title('Phase16.2 train windows PCA by cluster')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(loc='best', fontsize=8, ncol=3)
    plt.tight_layout()
    plt.savefig(out_dir / 'phase16_2_pca_train_clusters.png', dpi=140)
    plt.close()

    # Plot: symbol mix heatmap by cluster.
    pivot = mix.pivot(index='cluster_id', columns='symbol', values='symbol_share_in_cluster').fillna(0.0)
    plt.figure(figsize=(8, 5))
    plt.imshow(pivot.values, aspect='auto', cmap='viridis', vmin=0.0, vmax=max(0.25, float(pivot.values.max())))
    plt.colorbar(label='symbol_share_in_cluster')
    plt.title('Phase16.2 cluster-symbol mix')
    plt.xlabel('symbol')
    plt.ylabel('cluster_id')
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha='right')
    plt.yticks(range(len(pivot.index)), [str(x) for x in pivot.index])
    plt.tight_layout()
    plt.savefig(out_dir / 'phase16_2_cluster_symbol_mix.png', dpi=140)
    plt.close()

    # Save outputs.
    out.to_parquet(out_dir / 'phase16_2_cluster_assignments.parquet', index=False)
    mix.to_csv(out_dir / 'phase16_2_cluster_symbol_mix.csv', index=False)
    counts_split_symbol.to_csv(out_dir / 'phase16_2_counts_split_symbol_cluster.csv', index=False)
    counts_split.to_csv(out_dir / 'phase16_2_counts_split_cluster.csv', index=False)
    met_df.to_csv(out_dir / 'phase16_2_metrics_by_split.csv', index=False)
    k_search_df.to_csv(out_dir / 'phase16_2_k_search.csv', index=False)
    np.save(out_dir / 'phase16_2_cluster_centers_pca.npy', km.cluster_centers_)

    summary = {
        'status': 'SUCCESS',
        'phase16_run_dir': str(phase16_dir),
        'out_dir': str(out_dir),
        'n_rows': int(len(out)),
        'n_train': int(len(idx_train)),
        'n_val': int(len(idx_val)),
        'n_test': int(len(idx_test)),
        'embedding_dim_input': int(len(emb_cols)),
        'pca_dim_used': int(pca_dim),
        'normalize_embeddings': bool(cfg.normalize_embeddings),
        'k_selected': int(k_use),
        'cluster_symbol_purity_weighted': float(purity_all),
        'cluster_symbol_purity_by_cluster': purity_by_cluster,
        'metrics_by_split': {str(r['split']): {
            'n': int(r['n']),
            'n_clusters_pred': int(r['n_clusters_pred']),
            'silhouette': None if pd.isna(r['silhouette']) else float(r['silhouette']),
            'calinski_harabasz': None if pd.isna(r['calinski_harabasz']) else float(r['calinski_harabasz']),
            'davies_bouldin': None if pd.isna(r['davies_bouldin']) else float(r['davies_bouldin']),
        } for _, r in met_df.iterrows()},
        'config': asdict(cfg),
        'artifacts': {
            'cluster_assignments_parquet': str(out_dir / 'phase16_2_cluster_assignments.parquet'),
            'cluster_symbol_mix_csv': str(out_dir / 'phase16_2_cluster_symbol_mix.csv'),
            'counts_split_symbol_cluster_csv': str(out_dir / 'phase16_2_counts_split_symbol_cluster.csv'),
            'counts_split_cluster_csv': str(out_dir / 'phase16_2_counts_split_cluster.csv'),
            'metrics_by_split_csv': str(out_dir / 'phase16_2_metrics_by_split.csv'),
            'k_search_csv': str(out_dir / 'phase16_2_k_search.csv'),
            'cluster_centers_pca_npy': str(out_dir / 'phase16_2_cluster_centers_pca.npy'),
            'pca_train_clusters_png': str(out_dir / 'phase16_2_pca_train_clusters.png'),
            'cluster_symbol_mix_png': str(out_dir / 'phase16_2_cluster_symbol_mix.png'),
            'log': str(out_dir / 'phase16_2.log'),
        },
    }

    (out_dir / 'phase16_2_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    logger.info('SUCCESS: %s', out_dir / 'phase16_2_summary.json')


if __name__ == '__main__':
    main()
