#!/usr/bin/env python3
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


@dataclass
class Config:
    symbol: str = 'GBPUSD'
    max_seq_len: int = 64
    in_dim: int = 4
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.10

    batch_size_infer: int = 1024

    # decision target (within-segment horizon)
    horizon_bars: int = 4
    min_move_pips: float = 4.0
    cost_pips: float = 1.2

    # walk-forward
    train_min_ratio: float = 0.45
    test_ratio: float = 0.10
    max_folds: int = 5
    calib_ratio: float = 0.15
    min_train_rows: int = 2000
    min_test_rows: int = 700


CFG = Config()
ROOT = Path(__file__).resolve().parents[2]
PHASE9_DIR = ROOT / 'phases/phase09_multitask_patterns'
PHASE4_DIR = ROOT / 'phases/phase04_transfer_clustering'
OUT_DIR = ROOT / 'phases/phase10_decision_engine/output_phase10_decision_pre_rl_GBPUSD'
OUT_DIR.mkdir(parents=True, exist_ok=True)

IN_MODEL_PT = PHASE9_DIR / 'output_phase9_3_search_best_GBPUSD/phase9_3_best_model.pt'
IN_SAMPLES_CSV = PHASE9_DIR / 'output_phase9_3_search_best_GBPUSD/GBPUSD_phase9_3_samples.csv'
IN_SEG_LONG = PHASE4_DIR / 'output_transfer_GBPUSD_full_v2/GBPUSD_segments_long.parquet'
IN_THRESH_CSV = PHASE9_DIR / 'output_phase9_3_search_best_GBPUSD/GBPUSD_phase9_3_best_threshold_scan.csv'

OUT_INFER_CSV = OUT_DIR / 'GBPUSD_phase10_phase9_signals_full.csv'
OUT_DATASET_CSV = OUT_DIR / 'GBPUSD_phase10_decision_dataset.csv'
OUT_FOLDS_CSV = OUT_DIR / 'GBPUSD_phase10_walkforward_folds.csv'
OUT_PRED_CSV = OUT_DIR / 'GBPUSD_phase10_fold_predictions.csv'
OUT_METRICS_CSV = OUT_DIR / 'GBPUSD_phase10_walkforward_metrics.csv'
OUT_SUMMARY_JSON = OUT_DIR / 'GBPUSD_phase10_summary.json'
OUT_PNG_CUM = OUT_DIR / 'GBPUSD_phase10_equity_walkforward.png'
OUT_PNG_FOLD = OUT_DIR / 'GBPUSD_phase10_fold_sharpe_and_return.png'
OUT_PNG_ACTION = OUT_DIR / 'GBPUSD_phase10_action_distribution.png'


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
            activation='gelu',
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


def pip_size_for_symbol(symbol: str) -> float:
    return 0.01 if symbol.upper().endswith('JPY') else 0.0001


def load_base_data() -> Tuple[pd.DataFrame, pd.DataFrame, dict, float]:
    model_blob = torch.load(IN_MODEL_PT, map_location='cpu')
    class_map = {int(k): int(v) for k, v in model_blob['class_map'].items()}
    best_threshold = float(model_blob.get('best_threshold', 0.55))

    if IN_THRESH_CSV.exists():
        thr_df = pd.read_csv(IN_THRESH_CSV)
        if 'score' in thr_df.columns and len(thr_df) > 0:
            best_threshold = float(thr_df.sort_values('score', ascending=False).iloc[0]['thr'])

    samples = pd.read_csv(IN_SAMPLES_CSV)
    samples['timestamp'] = pd.to_datetime(samples['timestamp'], errors='coerce')

    seg_long = pd.read_parquet(IN_SEG_LONG)
    seg_long['timestamp'] = pd.to_datetime(seg_long['timestamp'], errors='coerce')

    return samples, seg_long, model_blob, best_threshold


def build_phase9_inputs(samples: pd.DataFrame, seg_long: pd.DataFrame, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    seg_long = seg_long.sort_values(['global_segment_id', 'step']).copy()

    norm_by_segment: Dict[str, np.ndarray] = {}
    for seg_id, g in seg_long.groupby('global_segment_id', sort=False):
        arr = g[['open', 'high', 'low', 'close']].to_numpy(dtype=np.float32)
        base = float(arr[0, 3])
        base = base if abs(base) > 1e-8 else 1.0
        norm_by_segment[seg_id] = (arr / base) - 1.0

    x = np.zeros((len(samples), cfg.max_seq_len, cfg.in_dim), dtype=np.float32)
    lengths = np.ones(len(samples), dtype=np.int64)

    for i, row in samples[['global_segment_id', 'step']].iterrows():
        seg_id = row['global_segment_id']
        step = int(row['step'])
        arr = norm_by_segment[seg_id]

        start = max(0, step - cfg.max_seq_len + 1)
        seq = arr[start : step + 1]
        ln = seq.shape[0]

        x[i, -ln:] = seq
        lengths[i] = ln

    return x, lengths


def infer_phase9_signals(samples: pd.DataFrame, model_blob: dict, x: np.ndarray, lengths: np.ndarray, best_thr: float) -> pd.DataFrame:
    cfg_dict = model_blob.get('config', {})
    cfg = Config(
        symbol=cfg_dict.get('symbol', CFG.symbol),
        max_seq_len=int(cfg_dict.get('max_seq_len', CFG.max_seq_len)),
        in_dim=int(cfg_dict.get('in_dim', CFG.in_dim)),
        d_model=int(cfg_dict.get('d_model', CFG.d_model)),
        n_heads=int(cfg_dict.get('n_heads', CFG.n_heads)),
        n_layers=int(cfg_dict.get('n_layers', CFG.n_layers)),
        dropout=float(cfg_dict.get('dropout', CFG.dropout)),
    )

    n_subclasses = int(model_blob['n_subclasses'])
    model = GatedMultiTaskTransformer(cfg=cfg, n_subclasses=n_subclasses)
    model.load_state_dict(model_blob['state_dict'])
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    pred_inf = []
    pred_class = []
    pred_prog = []
    pred_sub_conf = []
    pred_sub_entropy = []

    bs = CFG.batch_size_infer
    with torch.no_grad():
        for i in range(0, len(samples), bs):
            xb = torch.tensor(x[i : i + bs], dtype=torch.float32, device=device)
            lb = torch.tensor(lengths[i : i + bs], dtype=torch.long, device=device)

            inf_logit, sub_logits, prog = model(xb, lb)
            p_inf = torch.sigmoid(inf_logit)
            p_sub = torch.softmax(sub_logits, dim=1)

            sub_conf, sub_idx = torch.max(p_sub, dim=1)
            entropy = -(p_sub * torch.log(torch.clamp(p_sub, min=1e-8))).sum(dim=1)

            y_pred = torch.where(p_inf >= best_thr, sub_idx + 1, torch.zeros_like(sub_idx))

            pred_inf.append(p_inf.cpu().numpy())
            pred_class.append(y_pred.cpu().numpy())
            pred_prog.append(prog.cpu().numpy())
            pred_sub_conf.append(sub_conf.cpu().numpy())
            pred_sub_entropy.append(entropy.cpu().numpy())

    df = samples.copy()
    df['pred_inf_prob'] = np.concatenate(pred_inf).astype(np.float32)
    df['pred_class'] = np.concatenate(pred_class).astype(np.int64)
    df['pred_progress'] = np.concatenate(pred_prog).astype(np.float32)
    df['pred_sub_conf'] = np.concatenate(pred_sub_conf).astype(np.float32)
    df['pred_sub_entropy'] = np.concatenate(pred_sub_entropy).astype(np.float32)

    return df


def add_market_features(df: pd.DataFrame, seg_long: pd.DataFrame) -> pd.DataFrame:
    g = seg_long[['global_segment_id', 'step', 'open', 'high', 'low', 'close']].copy()
    g = g.sort_values(['global_segment_id', 'step'])

    g['ret_1'] = g['close'].pct_change().fillna(0.0)
    g['ret_2'] = g['close'].pct_change(2).fillna(0.0)
    g['ret_4'] = g['close'].pct_change(4).fillna(0.0)
    g['vol_4'] = g['close'].pct_change().rolling(4, min_periods=1).std().fillna(0.0)

    g['range_1'] = (g['high'] / g['low'] - 1.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    g['body_1'] = (g['close'] - g['open']) / g['open'].replace(0.0, np.nan)
    g['body_1'] = g['body_1'].replace([np.inf, -np.inf], 0.0).fillna(0.0)

    seg_len = g.groupby('global_segment_id')['step'].transform('max') + 1
    g['segment_len'] = seg_len.astype(np.float32)
    g['step_frac'] = (g['step'] / np.maximum(seg_len - 1, 1)).astype(np.float32)

    cols = [
        'global_segment_id', 'step', 'open', 'high', 'low', 'close',
        'ret_1', 'ret_2', 'ret_4', 'vol_4', 'range_1', 'body_1', 'segment_len', 'step_frac'
    ]

    out = df.merge(g[cols], on=['global_segment_id', 'step'], how='left')
    return out


def build_decision_target(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.sort_values(['global_segment_id', 'step']).copy()

    out['close_fwd'] = out.groupby('global_segment_id')['close'].shift(-cfg.horizon_bars)
    out['future_return_h'] = (out['close_fwd'] / out['close']) - 1.0

    pip = pip_size_for_symbol(cfg.symbol)
    out['future_pips_h'] = out['future_return_h'] / pip

    thr = cfg.min_move_pips + cfg.cost_pips
    out['y_action'] = 0
    out.loc[out['future_pips_h'] > thr, 'y_action'] = 1
    out.loc[out['future_pips_h'] < -thr, 'y_action'] = -1

    # rows without horizon target are not usable for supervised learning
    out = out[np.isfinite(out['future_return_h'])].copy()

    # class index for multiclass model
    out['y_action_idx'] = out['y_action'].map({-1: 0, 0: 1, 1: 2}).astype(int)
    return out


def make_walkforward_folds(df: pd.DataFrame, cfg: Config) -> List[Dict]:
    n = len(df)
    train_min = int(n * cfg.train_min_ratio)
    test_size = int(n * cfg.test_ratio)

    folds = []
    for k in range(cfg.max_folds):
        train_end = train_min + k * test_size
        test_start = train_end
        test_end = min(test_start + test_size, n)

        if test_end - test_start < cfg.min_test_rows:
            break

        purge_end = max(cfg.min_train_rows, train_end - cfg.horizon_bars)
        if purge_end < cfg.min_train_rows:
            continue

        folds.append(
            {
                'fold': k + 1,
                'train_start': 0,
                'train_end': purge_end,
                'test_start': test_start,
                'test_end': test_end,
            }
        )

    return folds


def class_weights(y: np.ndarray) -> Dict[int, float]:
    uniq, cnt = np.unique(y, return_counts=True)
    d = {int(u): float(c) for u, c in zip(uniq, cnt)}
    total = float(sum(d.values()))
    n_cls = len(d)
    return {k: total / (n_cls * v) for k, v in d.items()}


def apply_policy(proba: np.ndarray, conf_thr: float, inf_prob: np.ndarray, inf_min: float) -> np.ndarray:
    pred_idx = np.argmax(proba, axis=1)
    conf = np.max(proba, axis=1)
    action = np.where(pred_idx == 0, -1, np.where(pred_idx == 2, 1, 0))

    # abstain to flat if low confidence or weak informative probability
    action = np.where((conf >= conf_thr) & (inf_prob >= inf_min), action, 0)
    return action.astype(int)


def evaluate_trading(actions: np.ndarray, future_ret: np.ndarray, cost_ret: float) -> Dict[str, float]:
    gross = actions * future_ret
    trade = (actions != 0).astype(np.float32)
    net = gross - trade * cost_ret

    eps = 1e-12
    avg = float(np.mean(net))
    std = float(np.std(net))
    sharpe = float((avg / (std + eps)) * np.sqrt(24.0 * 252.0))

    traded = trade > 0
    win_rate = float((net[traded] > 0).mean()) if traded.any() else 0.0

    return {
        'avg_net_return': avg,
        'cum_net_return': float(np.sum(net)),
        'std_net_return': std,
        'sharpe_ann': sharpe,
        'trade_rate': float(trade.mean()),
        'win_rate_trades': win_rate,
    }


def tune_thresholds(calib_df: pd.DataFrame, proba: np.ndarray, cfg: Config) -> Tuple[float, float, Dict[str, float]]:
    pip = pip_size_for_symbol(cfg.symbol)
    cost_ret = cfg.cost_pips * pip

    best = None
    for conf_thr in np.linspace(0.40, 0.75, 8):
        for inf_min in np.linspace(0.30, 0.65, 8):
            a = apply_policy(proba, float(conf_thr), calib_df['pred_inf_prob'].to_numpy(), float(inf_min))
            m = evaluate_trading(a, calib_df['future_return_h'].to_numpy(), cost_ret)
            score = m['avg_net_return'] * 1e4 + 0.12 * m['sharpe_ann']
            row = {'conf_thr': float(conf_thr), 'inf_min': float(inf_min), 'score': float(score), **m}
            if best is None or row['score'] > best['score']:
                best = row

    return float(best['conf_thr']), float(best['inf_min']), best


def run_walkforward(df: pd.DataFrame, cfg: Config):
    feature_cols = [
        'pred_inf_prob', 'pred_progress', 'pred_sub_conf', 'pred_sub_entropy', 'pred_class',
        'ret_1', 'ret_2', 'ret_4', 'vol_4', 'range_1', 'body_1', 'segment_len', 'step_frac',
    ]

    folds = make_walkforward_folds(df, cfg)
    if not folds:
        raise RuntimeError('No valid walk-forward folds were created. Check dataset size and config.')

    fold_rows = []
    pred_rows = []

    pip = pip_size_for_symbol(cfg.symbol)
    cost_ret = cfg.cost_pips * pip

    for fold in folds:
        tr = df.iloc[fold['train_start'] : fold['train_end']].copy()
        te = df.iloc[fold['test_start'] : fold['test_end']].copy()

        calib_n = max(300, int(len(tr) * cfg.calib_ratio))
        if calib_n >= len(tr):
            calib_n = max(100, len(tr) // 5)

        fit = tr.iloc[:-calib_n].copy()
        calib = tr.iloc[-calib_n:].copy()

        x_fit = fit[feature_cols].to_numpy(dtype=np.float32)
        y_fit = fit['y_action_idx'].to_numpy(dtype=np.int64)

        w = class_weights(y_fit)
        sample_w = np.array([w[int(v)] for v in y_fit], dtype=np.float32)

        clf = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=6,
            max_iter=280,
            min_samples_leaf=80,
            l2_regularization=1e-3,
            random_state=SEED,
        )
        clf.fit(x_fit, y_fit, sample_weight=sample_w)

        x_calib = calib[feature_cols].to_numpy(dtype=np.float32)
        p_calib = clf.predict_proba(x_calib)
        conf_thr, inf_min, calib_best = tune_thresholds(calib, p_calib, cfg)

        # retrain on full train portion
        x_tr_full = tr[feature_cols].to_numpy(dtype=np.float32)
        y_tr_full = tr['y_action_idx'].to_numpy(dtype=np.int64)
        w_full = class_weights(y_tr_full)
        sw_full = np.array([w_full[int(v)] for v in y_tr_full], dtype=np.float32)
        clf.fit(x_tr_full, y_tr_full, sample_weight=sw_full)

        x_te = te[feature_cols].to_numpy(dtype=np.float32)
        p_te = clf.predict_proba(x_te)
        a_te = apply_policy(p_te, conf_thr, te['pred_inf_prob'].to_numpy(), inf_min)

        y_true_action = te['y_action'].to_numpy(dtype=np.int64)
        y_pred_idx = np.argmax(p_te, axis=1)
        y_pred_action_raw = np.where(y_pred_idx == 0, -1, np.where(y_pred_idx == 2, 1, 0))

        cls_acc = float((y_true_action == a_te).mean())
        f1_macro = float(
            f1_score(
                y_true_action,
                a_te,
                labels=[-1, 0, 1],
                average='macro',
                zero_division=0,
            )
        )

        trade_m = evaluate_trading(a_te, te['future_return_h'].to_numpy(), cost_ret)

        fold_row = {
            'fold': int(fold['fold']),
            'train_rows': int(len(tr)),
            'test_rows': int(len(te)),
            'timestamp_test_start': str(te['timestamp'].iloc[0]),
            'timestamp_test_end': str(te['timestamp'].iloc[-1]),
            'conf_thr': float(conf_thr),
            'inf_min': float(inf_min),
            'calib_score': float(calib_best['score']),
            'acc_action_3': cls_acc,
            'f1_macro_action_3': f1_macro,
            **trade_m,
        }
        fold_rows.append(fold_row)

        te_out = te[[
            'global_segment_id', 'timestamp', 'step', 'split', 'y_action', 'future_return_h', 'future_pips_h',
            'pred_inf_prob', 'pred_progress', 'pred_sub_conf', 'pred_sub_entropy', 'pred_class'
        ]].copy()
        te_out['fold'] = int(fold['fold'])
        te_out['pred_action_raw'] = y_pred_action_raw.astype(int)
        te_out['pred_action'] = a_te.astype(int)
        te_out['pred_conf_max'] = np.max(p_te, axis=1)
        te_out['p_short'] = p_te[:, 0]
        te_out['p_flat'] = p_te[:, 1]
        te_out['p_long'] = p_te[:, 2]
        te_out['net_return'] = te_out['pred_action'] * te_out['future_return_h'] - (te_out['pred_action'] != 0).astype(float) * cost_ret
        pred_rows.append(te_out)

    folds_df = pd.DataFrame(folds)
    metrics_df = pd.DataFrame(fold_rows)
    preds_df = pd.concat(pred_rows, axis=0, ignore_index=True)

    return folds_df, metrics_df, preds_df


def make_plots(metrics_df: pd.DataFrame, preds_df: pd.DataFrame):
    # Cumulative net return
    p = preds_df.sort_values(['timestamp', 'fold']).copy()
    p['cum_net'] = p['net_return'].cumsum()

    plt.figure(figsize=(11, 4))
    plt.plot(pd.to_datetime(p['timestamp']), p['cum_net'], linewidth=1.5)
    plt.title('Phase10 - Cumulative Net Return (walk-forward test only)')
    plt.xlabel('timestamp')
    plt.ylabel('cumulative return')
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_PNG_CUM, dpi=140)
    plt.close()

    # Fold bar metrics
    fig, ax1 = plt.subplots(figsize=(10, 4))
    x = metrics_df['fold'].astype(int).to_numpy()
    ax1.bar(x - 0.15, metrics_df['cum_net_return'], width=0.3, label='cum_net_return')
    ax1.set_xlabel('fold')
    ax1.set_ylabel('cum net return')

    ax2 = ax1.twinx()
    ax2.plot(x + 0.15, metrics_df['sharpe_ann'], marker='o', linewidth=1.6, label='sharpe_ann', color='tab:red')
    ax2.set_ylabel('sharpe_ann')

    ax1.set_title('Phase10 - Fold Returns and Sharpe')
    ax1.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT_PNG_FOLD, dpi=140)
    plt.close(fig)

    # Action distribution
    vc = preds_df['pred_action'].value_counts().sort_index()
    labels = [str(i) for i in vc.index.tolist()]
    plt.figure(figsize=(6.5, 4))
    plt.bar(labels, vc.values)
    plt.title('Phase10 - Predicted Action Distribution (test folds)')
    plt.xlabel('action (-1 short, 0 flat, 1 long)')
    plt.ylabel('count')
    plt.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_PNG_ACTION, dpi=140)
    plt.close()


def main():
    samples, seg_long, model_blob, best_thr = load_base_data()

    x, lengths = build_phase9_inputs(samples, seg_long, CFG)
    inf_df = infer_phase9_signals(samples, model_blob, x, lengths, best_thr)
    inf_df = add_market_features(inf_df, seg_long)

    data_df = build_decision_target(inf_df, CFG)
    data_df = data_df.sort_values('timestamp').reset_index(drop=True)

    folds_df, metrics_df, preds_df = run_walkforward(data_df, CFG)

    inf_df.to_csv(OUT_INFER_CSV, index=False)
    data_df.to_csv(OUT_DATASET_CSV, index=False)
    folds_df.to_csv(OUT_FOLDS_CSV, index=False)
    metrics_df.to_csv(OUT_METRICS_CSV, index=False)
    preds_df.to_csv(OUT_PRED_CSV, index=False)

    make_plots(metrics_df, preds_df)

    summary = {
        'symbol': CFG.symbol,
        'model_tag': 'phase10_decision_pre_rl',
        'description': 'Pre-RL decision engine with walk-forward validation and costs',
        'config': asdict(CFG),
        'phase9_best_threshold': float(best_thr),
        'dataset': {
            'n_inference_rows': int(len(inf_df)),
            'n_decision_rows': int(len(data_df)),
            'action_distribution': data_df['y_action'].value_counts().sort_index().to_dict(),
            'date_min': str(data_df['timestamp'].min()),
            'date_max': str(data_df['timestamp'].max()),
        },
        'walkforward': {
            'n_folds': int(len(metrics_df)),
            'mean_metrics': metrics_df.mean(numeric_only=True).to_dict(),
            'std_metrics': metrics_df.std(numeric_only=True).to_dict(),
        },
        'outputs': {
            'signals_full_csv': str(OUT_INFER_CSV),
            'decision_dataset_csv': str(OUT_DATASET_CSV),
            'folds_csv': str(OUT_FOLDS_CSV),
            'predictions_csv': str(OUT_PRED_CSV),
            'metrics_csv': str(OUT_METRICS_CSV),
            'plot_cumulative_png': str(OUT_PNG_CUM),
            'plot_fold_png': str(OUT_PNG_FOLD),
            'plot_action_png': str(OUT_PNG_ACTION),
        },
    }

    OUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

    print('Saved summary:', OUT_SUMMARY_JSON)
    print('Walk-forward mean metrics:')
    print(metrics_df.mean(numeric_only=True).to_dict())
    return summary, metrics_df


if __name__ == '__main__':
    main()
