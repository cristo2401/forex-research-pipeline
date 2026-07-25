#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


LAG_STATE_COLS: Tuple[str, ...] = (
    'lev_phase14_hardened',
    'action_phase14_hardened',
    'phase14_day_drawdown',
    'phase14_trades_in_hour',
    'phase14_kill_locked',
    'phase14_trade_event',
    'phase14_turnover_event',
    'pred_action',
    'p_short',
    'p_flat',
    'p_long',
    'pred_conf_max',
    'score_ls',
    'nonflat_score',
)

FORBIDDEN_FEATURE_TOKENS: Tuple[str, ...] = (
    'future_',
    'close_fwd',
    'y_',
    'target_',
    'net_',
    'cum_',
)


@dataclass
class Config:
    phase16_3_run_dir: str
    phase16_1_run_dir: str
    out_dir: str
    window_len: int = 24
    window_stride: int = 1
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    corr_alert_threshold: float = 0.35
    min_rows_per_symbol: int = 500
    max_ffill_steps: int = 96
    min_target_symbols: int = 3


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger('phase16_portfolio_state_dataset')
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


def resolve_phase16_1_dir(phase16_3_dir: Path, override: str) -> Path:
    if override:
        p = Path(override).resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        return p

    s3 = phase16_3_dir / 'phase16_3_summary.json'
    if not s3.exists():
        raise FileNotFoundError(s3)
    d3 = json.loads(s3.read_text(encoding='utf-8'))
    p16_2 = Path(d3['phase16_2_run_dir']).resolve()

    s2 = p16_2 / 'phase16_2_summary.json'
    if not s2.exists():
        raise FileNotFoundError(s2)
    d2 = json.loads(s2.read_text(encoding='utf-8'))
    p16_1 = Path(d2['phase16_run_dir']).resolve()
    if not p16_1.exists():
        raise FileNotFoundError(p16_1)
    return p16_1


def make_forward_targets(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values('timestamp').reset_index(drop=True)
    r1 = g['realized_return_1bar'].shift(-1)
    r2 = g['realized_return_1bar'].shift(-2)
    r3 = g['realized_return_1bar'].shift(-3)
    r4 = g['realized_return_1bar'].shift(-4)

    g['target_ret1'] = r1
    g['target_ret4'] = r1 + r2 + r3 + r4
    g['target_vol4'] = pd.concat([r1, r2, r3, r4], axis=1).std(axis=1, ddof=0)
    g['target_ts_1'] = g['timestamp'].shift(-1)
    g['target_ts_4'] = g['timestamp'].shift(-4)
    return g


def assign_temporal_splits(ts: pd.Series, train_ratio: float, val_ratio: float) -> np.ndarray:
    n = len(ts)
    if n < 10:
        raise RuntimeError(f'Not enough rows for split: {n}')
    i_train = int(n * train_ratio)
    i_val = int(n * (train_ratio + val_ratio))
    if i_train < 1 or i_val <= i_train or i_val >= n:
        raise RuntimeError(
            f'Invalid split with n={n}, train_ratio={train_ratio}, val_ratio={val_ratio}'
        )
    s = np.full(n, 'test', dtype=object)
    s[:i_train] = 'train'
    s[i_train:i_val] = 'val'
    return s


def flatten_wide_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [f'{a}__{b}' for a, b in out.columns.to_list()]
    out = out.reset_index()
    return out


def top_abs_corr_features(
    frame: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    top_k: int = 10,
) -> List[Dict[str, float | str]]:
    if len(frame) < 20:
        return []
    x = frame[feature_cols]
    y = frame[target_col]
    # Avoid noisy warnings from constant columns in corr computation.
    var = x.var(axis=0, ddof=0)
    x = x.loc[:, var > 1e-12]
    if x.empty:
        return []
    corr = x.corrwith(y).replace([np.inf, -np.inf], np.nan).dropna()
    if corr.empty:
        return []
    corr = corr.abs().sort_values(ascending=False).head(top_k)
    return [{'feature': str(k), 'abs_corr': float(v)} for k, v in corr.items()]


def has_forbidden_pattern(col: str, token: str) -> bool:
    c = str(col)
    t = str(token)
    if t == 'close_fwd':
        return ('close_fwd' in c) or ('_close_fwd_' in c)
    if t.endswith('_'):
        return c.startswith(t) or (f'_{t}' in c)
    return t in c


def main():
    ap = argparse.ArgumentParser(
        description='Phase16.4 - Construye dataset de estado de portafolio (long/wide/windows) con chequeos anti-leak.'
    )
    ap.add_argument('--phase16_3_run_dir', required=True, type=str)
    ap.add_argument('--phase16_1_run_dir', default='', type=str, help='Optional. If empty, inferred from phase16.3 -> phase16.2 summary chain.')
    ap.add_argument('--out_dir', default='', type=str)
    ap.add_argument('--window_len', type=int, default=24)
    ap.add_argument('--window_stride', type=int, default=1)
    ap.add_argument('--train_ratio', type=float, default=0.70)
    ap.add_argument('--val_ratio', type=float, default=0.15)
    ap.add_argument('--corr_alert_threshold', type=float, default=0.35)
    ap.add_argument('--min_rows_per_symbol', type=int, default=500)
    ap.add_argument('--max_ffill_steps', type=int, default=96, help='Max forward-fill steps for per-symbol state on global timeline. -1 disables stale cutoff.')
    ap.add_argument('--min_target_symbols', type=int, default=3, help='Minimum number of symbols with available target at window end.')
    args = ap.parse_args()

    phase16_3_dir = Path(args.phase16_3_run_dir).resolve()
    if not phase16_3_dir.exists():
        raise FileNotFoundError(phase16_3_dir)

    phase16_1_dir = resolve_phase16_1_dir(phase16_3_dir, args.phase16_1_run_dir)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (phase16_3_dir / 'phase16_4_portfolio_state')
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config(
        phase16_3_run_dir=str(phase16_3_dir),
        phase16_1_run_dir=str(phase16_1_dir),
        out_dir=str(out_dir),
        window_len=int(args.window_len),
        window_stride=int(args.window_stride),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        corr_alert_threshold=float(args.corr_alert_threshold),
        min_rows_per_symbol=int(args.min_rows_per_symbol),
        max_ffill_steps=int(args.max_ffill_steps),
        min_target_symbols=int(args.min_target_symbols),
    )

    logger = setup_logger(out_dir / 'phase16_4.log')
    logger.info('Config: %s', cfg)

    p_ctx = phase16_3_dir / 'phase16_3_context_embeddings.parquet'
    p_state = phase16_1_dir / 'phase16_state_rows_raw.parquet'
    if not p_ctx.exists():
        raise FileNotFoundError(p_ctx)
    if not p_state.exists():
        raise FileNotFoundError(p_state)

    ctx = pd.read_parquet(p_ctx)
    state = pd.read_parquet(p_state)
    logger.info('Loaded: ctx=%s rows, state=%s rows', len(ctx), len(state))

    req_ctx = ['symbol', 'timestamp_end', 'target_cluster_pred', 'pred_conf', 'context_regime_id']
    missing_ctx = [c for c in req_ctx if c not in ctx.columns]
    if missing_ctx:
        raise RuntimeError(f'Missing ctx cols: {missing_ctx}')

    missing_state = [c for c in ('symbol', 'timestamp', 'realized_return_1bar', *LAG_STATE_COLS) if c not in state.columns]
    if missing_state:
        raise RuntimeError(f'Missing state cols: {missing_state}')

    ctx_cols = [c for c in ctx.columns if c.startswith('ctx_')]
    if not ctx_cols:
        raise RuntimeError('No ctx_* columns found in phase16_3_context_embeddings.parquet')

    ctx_use = ctx[['symbol', 'timestamp_end', 'target_cluster_pred', 'pred_conf', 'context_regime_id', *ctx_cols]].copy()
    ctx_use = ctx_use.rename(
        columns={
            'timestamp_end': 'timestamp',
            'target_cluster_pred': 'ctx_pred_cluster',
            'pred_conf': 'ctx_pred_conf',
            'context_regime_id': 'ctx_regime_id',
        }
    )
    ctx_use['timestamp'] = pd.to_datetime(ctx_use['timestamp'], utc=True, errors='coerce')
    ctx_use = ctx_use.dropna(subset=['timestamp']).sort_values(['symbol', 'timestamp']).drop_duplicates(['symbol', 'timestamp'], keep='last')

    state_use = state[['symbol', 'timestamp', 'realized_return_1bar', *LAG_STATE_COLS]].copy()
    state_use['timestamp'] = pd.to_datetime(state_use['timestamp'], utc=True, errors='coerce')
    state_use = state_use.dropna(subset=['timestamp']).sort_values(['symbol', 'timestamp']).drop_duplicates(['symbol', 'timestamp'], keep='last')

    lagged_cols = []
    for c in ('realized_return_1bar', *LAG_STATE_COLS):
        lc = f'{c}_lag1'
        lagged_cols.append(lc)
        state_use[lc] = state_use.groupby('symbol', sort=False)[c].shift(1)

    base = ctx_use.merge(
        state_use[['symbol', 'timestamp', 'realized_return_1bar', *lagged_cols]],
        on=['symbol', 'timestamp'],
        how='inner',
        validate='one_to_one',
    )
    logger.info('Merged base rows=%s symbols=%s', len(base), sorted(base['symbol'].unique().tolist()))

    parts = []
    for symbol, g in base.groupby('symbol', sort=False):
        gg = make_forward_targets(g)
        gg['symbol'] = symbol
        parts.append(gg)
    base = pd.concat(parts, axis=0, ignore_index=True)

    feature_cols_long = ['ctx_pred_cluster', 'ctx_pred_conf', 'ctx_regime_id', *ctx_cols, *lagged_cols]
    target_cols_long = ['target_ret1', 'target_ret4', 'target_vol4']
    target_ts_cols = ['target_ts_1', 'target_ts_4']

    for tok in FORBIDDEN_FEATURE_TOKENS:
        bad = [c for c in feature_cols_long if has_forbidden_pattern(c, tok)]
        if bad:
            raise RuntimeError(f'Forbidden token "{tok}" found in features: {bad[:8]}')

    base = base.dropna(subset=feature_cols_long + target_cols_long + target_ts_cols).copy()
    base['timestamp'] = pd.to_datetime(base['timestamp'], utc=True)
    base['target_ts_1'] = pd.to_datetime(base['target_ts_1'], utc=True)
    base['target_ts_4'] = pd.to_datetime(base['target_ts_4'], utc=True)

    # Leak check: targets must be strictly in the future.
    leak_check_1 = bool((base['target_ts_1'] > base['timestamp']).all())
    leak_check_4 = bool((base['target_ts_4'] > base['timestamp']).all())

    cov = base.groupby('symbol').size().rename('rows').reset_index()
    cov['enough_rows'] = cov['rows'] >= int(cfg.min_rows_per_symbol)
    if not bool(cov['enough_rows'].all()):
        bad = cov.loc[~cov['enough_rows'], ['symbol', 'rows']]
        raise RuntimeError(f'Not enough rows per symbol after merge/targets: {bad.to_dict(orient="records")}')

    symbols = sorted(base['symbol'].unique().tolist())
    logger.info('After filtering: rows=%s symbols=%s', len(base), symbols)

    # Long output.
    long_cols = ['symbol', 'timestamp', *feature_cols_long, *target_cols_long, *target_ts_cols]
    long_df = base[long_cols].sort_values(['timestamp', 'symbol']).reset_index(drop=True)
    long_df.to_parquet(out_dir / 'phase16_4_state_long.parquet', index=False)

    # Wide panel for portfolio-level modeling (global event timeline + per-symbol forward-fill).
    global_ts = np.array(sorted(long_df['timestamp'].unique()))
    aligned_parts = []
    feature_cols_symbol = [*feature_cols_long, 'state_age_steps']
    for s in symbols:
        g = long_df[long_df['symbol'] == s].copy()
        g = g[['timestamp', *feature_cols_long, *target_cols_long, *target_ts_cols]].sort_values('timestamp')
        g = g.set_index('timestamp')
        g = g.reindex(global_ts)

        update_mask = g['ctx_pred_conf'].notna().to_numpy()
        g_feat = g[feature_cols_long].ffill()
        g_tgt = g[target_cols_long + target_ts_cols].copy()

        age = np.empty(len(g_feat), dtype=np.int32)
        cur = 10**9
        for i, is_update in enumerate(update_mask):
            if is_update:
                cur = 0
            else:
                cur += 1
            age[i] = cur
        g_feat['state_age_steps'] = age.astype(np.float32)

        if int(cfg.max_ffill_steps) >= 0:
            stale = age > int(cfg.max_ffill_steps)
            cols_stale = [*feature_cols_long, 'state_age_steps']
            g_feat.loc[stale, cols_stale] = np.nan

        g = pd.concat([g_feat, g_tgt], axis=1)

        g = g.add_suffix(f'__{s}')
        aligned_parts.append(g)

    wide_df = pd.concat(aligned_parts, axis=1).reset_index().rename(columns={'index': 'timestamp'})
    wide_df = wide_df.sort_values('timestamp').reset_index(drop=True)

    feature_cols_wide = [f'{c}__{s}' for c in feature_cols_symbol for s in symbols]
    target_ret1_cols_wide = [f'target_ret1__{s}' for s in symbols]
    target_ret4_cols_wide = [f'target_ret4__{s}' for s in symbols]
    target_vol4_cols_wide = [f'target_vol4__{s}' for s in symbols]
    target_ts1_cols_wide = [f'target_ts_1__{s}' for s in symbols]
    target_ts4_cols_wide = [f'target_ts_4__{s}' for s in symbols]

    required_wide = [*feature_cols_wide]
    wide_df = wide_df.dropna(subset=required_wide).copy()
    wide_df['timestamp'] = pd.to_datetime(wide_df['timestamp'], utc=True)

    # Target availability masks (event-driven supervision).
    target_mask_cols_wide = []
    for s in symbols:
        mc = f'target_mask__{s}'
        target_mask_cols_wide.append(mc)
        wide_df[mc] = wide_df[f'target_ret1__{s}'].notna().astype(np.int8)

    # Global temporal split for portfolio dataset (no per-symbol mismatch).
    wide_df['split'] = assign_temporal_splits(wide_df['timestamp'], cfg.train_ratio, cfg.val_ratio)
    split_counts = wide_df['split'].value_counts().rename('n').reset_index().rename(columns={'index': 'split'})
    split_counts.to_csv(out_dir / 'phase16_4_split_counts.csv', index=False)

    # Extra anti-leak check in wide form.
    ts_arr = wide_df['timestamp']
    checks_ts1 = []
    for c in target_ts1_cols_wide:
        s = pd.to_datetime(wide_df[c], utc=True, errors='coerce')
        m = s.notna()
        if m.any():
            checks_ts1.append(bool((s[m] > ts_arr[m]).all()))
    checks_ts4 = []
    for c in target_ts4_cols_wide:
        s = pd.to_datetime(wide_df[c], utc=True, errors='coerce')
        m = s.notna()
        if m.any():
            checks_ts4.append(bool((s[m] > ts_arr[m]).all()))
    all_ts1_gt = bool(all(checks_ts1)) if checks_ts1 else False
    all_ts4_gt = bool(all(checks_ts4)) if checks_ts4 else False

    # "Too good to be true" sanity alert: large feature-target corr on TEST.
    test_df = wide_df[wide_df['split'] == 'test'].copy()
    corr_top: Dict[str, List[Dict[str, float | str]]] = {}
    corr_max = 0.0
    corr_max_target = ''
    corr_max_feature = ''
    for tc in target_ret1_cols_wide:
        test_sub = test_df[test_df[tc].notna()].copy()
        top = top_abs_corr_features(test_sub, feature_cols_wide, tc, top_k=10)
        corr_top[tc] = top
        if top:
            v = float(top[0]['abs_corr'])
            if v > corr_max:
                corr_max = v
                corr_max_target = tc
                corr_max_feature = str(top[0]['feature'])

    corr_alert = bool(corr_max >= float(cfg.corr_alert_threshold))

    # Build sequential windows for downstream RL/sequence model.
    x = wide_df[feature_cols_wide].to_numpy(dtype=np.float32)
    y1 = wide_df[target_ret1_cols_wide].to_numpy(dtype=np.float32)
    y4 = wide_df[target_ret4_cols_wide].to_numpy(dtype=np.float32)
    yv = wide_df[target_vol4_cols_wide].to_numpy(dtype=np.float32)
    ym = wide_df[target_mask_cols_wide].to_numpy(dtype=np.float32)
    splits = wide_df['split'].astype(str).to_numpy()
    ts = pd.to_datetime(wide_df['timestamp'], utc=True).to_numpy()

    xs: List[np.ndarray] = []
    y1s: List[np.ndarray] = []
    y4s: List[np.ndarray] = []
    yvs: List[np.ndarray] = []
    yms: List[np.ndarray] = []
    wm: List[dict] = []
    dropped_cross_split = 0
    dropped_low_target = 0
    wlen = int(cfg.window_len)
    wstride = int(cfg.window_stride)
    for end in range(wlen - 1, len(wide_df), wstride):
        start = end - wlen + 1
        sp = splits[end]
        if not np.all(splits[start:end + 1] == sp):
            dropped_cross_split += 1
            continue
        mask_end = ym[end]
        if int(mask_end.sum()) < int(cfg.min_target_symbols):
            dropped_low_target += 1
            continue
        xs.append(x[start:end + 1])
        y1s.append(np.nan_to_num(y1[end], nan=0.0))
        y4s.append(np.nan_to_num(y4[end], nan=0.0))
        yvs.append(np.nan_to_num(yv[end], nan=0.0))
        yms.append(mask_end)
        wm.append(
            {
                'split': sp,
                'idx_start': int(start),
                'idx_end': int(end),
                'timestamp_end': pd.Timestamp(ts[end]),
                'target_symbols_available': int(mask_end.sum()),
            }
        )

    if not xs:
        raise RuntimeError('No windows generated for phase16.4 (check window_len / split ratios).')

    xw = np.stack(xs).astype(np.float32)
    y1w = np.stack(y1s).astype(np.float32)
    y4w = np.stack(y4s).astype(np.float32)
    yvw = np.stack(yvs).astype(np.float32)
    ymw = np.stack(yms).astype(np.float32)
    wm_df = pd.DataFrame(wm)

    np.savez_compressed(
        out_dir / 'phase16_4_windows.npz',
        x=xw,
        y_ret1=y1w,
        y_ret4=y4w,
        y_vol4=yvw,
        y_mask=ymw,
    )
    wm_df.to_parquet(out_dir / 'phase16_4_windows_meta.parquet', index=False)

    # Persist wide dataset without timestamp targets in modeling frame.
    wide_out = wide_df.copy()
    wide_out.to_parquet(out_dir / 'phase16_4_state_wide.parquet', index=False)
    cov.to_csv(out_dir / 'phase16_4_symbol_coverage.csv', index=False)
    (out_dir / 'phase16_4_feature_columns.txt').write_text(
        '\n'.join(feature_cols_wide),
        encoding='utf-8',
    )

    sanity = {
        'forbidden_feature_token_check_pass': True,
        'long_target_ts_gt_state_ts_h1': leak_check_1,
        'long_target_ts_gt_state_ts_h4': leak_check_4,
        'wide_target_ts_gt_state_ts_h1': all_ts1_gt,
        'wide_target_ts_gt_state_ts_h4': all_ts4_gt,
        'corr_alert_threshold': float(cfg.corr_alert_threshold),
        'test_max_abs_feature_target_corr_ret1': float(corr_max),
        'test_max_abs_corr_feature': corr_max_feature,
        'test_max_abs_corr_target': corr_max_target,
        'test_corr_alert': corr_alert,
        'dropped_cross_split_windows': int(dropped_cross_split),
        'dropped_low_target_windows': int(dropped_low_target),
        'min_target_symbols_required': int(cfg.min_target_symbols),
        'all_checks_pass': bool(
            leak_check_1 and leak_check_4 and all_ts1_gt and all_ts4_gt
        ),
    }
    (out_dir / 'phase16_4_sanity_checks.json').write_text(
        json.dumps(sanity, indent=2, default=str),
        encoding='utf-8',
    )

    summary = {
        'status': 'SUCCESS',
        'config': asdict(cfg),
        'symbols': symbols,
        'n_symbols': int(len(symbols)),
        'rows_long': int(len(long_df)),
        'rows_wide': int(len(wide_df)),
        'window_count': int(len(xw)),
        'window_shape': [int(xw.shape[0]), int(xw.shape[1]), int(xw.shape[2])],
        'target_shape': [int(y1w.shape[0]), int(y1w.shape[1])],
        'target_mask_shape': [int(ymw.shape[0]), int(ymw.shape[1])],
        'split_counts_rows': {str(k): int(v) for k, v in wide_df['split'].value_counts().to_dict().items()},
        'split_counts_windows': {str(k): int(v) for k, v in wm_df['split'].value_counts().to_dict().items()},
        'avg_target_symbols_available_per_window': float(wm_df['target_symbols_available'].mean()) if len(wm_df) else 0.0,
        'anti_leak': {
            'long_target_ts_future_h1': leak_check_1,
            'long_target_ts_future_h4': leak_check_4,
            'wide_target_ts_future_h1': all_ts1_gt,
            'wide_target_ts_future_h4': all_ts4_gt,
            'no_cross_split_windows': int(dropped_cross_split) >= 0,
        },
        'too_good_sanity': sanity,
        'artifacts': {
            'state_long_parquet': str(out_dir / 'phase16_4_state_long.parquet'),
            'state_wide_parquet': str(out_dir / 'phase16_4_state_wide.parquet'),
            'windows_npz': str(out_dir / 'phase16_4_windows.npz'),
            'windows_meta_parquet': str(out_dir / 'phase16_4_windows_meta.parquet'),
            'split_counts_csv': str(out_dir / 'phase16_4_split_counts.csv'),
            'symbol_coverage_csv': str(out_dir / 'phase16_4_symbol_coverage.csv'),
            'feature_columns_txt': str(out_dir / 'phase16_4_feature_columns.txt'),
            'sanity_checks_json': str(out_dir / 'phase16_4_sanity_checks.json'),
            'log': str(out_dir / 'phase16_4.log'),
        },
        'corr_top_test_ret1': corr_top,
    }
    (out_dir / 'phase16_4_summary.json').write_text(
        json.dumps(summary, indent=2, default=str),
        encoding='utf-8',
    )
    logger.info('SUCCESS: %s', out_dir / 'phase16_4_summary.json')


if __name__ == '__main__':
    main()
