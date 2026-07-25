#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class AuditConfig:
    run_dir: str
    sanity_model: str = 'phase12_2_frozen'
    pvalue_threshold: float = 0.10


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def safe_json_write(path: Path, obj: dict):
    path.write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f'Missing JSON: {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def _shift_stats(shift_block: dict) -> Tuple[bool, float, float, float, float]:
    rows = shift_block.get('results', []) if isinstance(shift_block, dict) else []
    if not rows:
        return False, np.nan, np.nan, np.nan, np.nan
    df = pd.DataFrame(rows)
    if 'shift' not in df.columns:
        return False, np.nan, np.nan, np.nan, np.nan

    if 0 not in set(df['shift'].astype(int).tolist()):
        return False, np.nan, np.nan, np.nan, np.nan

    z = df[df['shift'] == 0].iloc[0]
    z_sh = float(z['sharpe_ann'])
    z_cum = float(z['cum_net_return'])
    others = df[df['shift'] != 0]
    max_other_sh = float(others['sharpe_ann'].max()) if len(others) else -np.inf
    max_other_cum = float(others['cum_net_return'].max()) if len(others) else -np.inf

    # Causal sanity: zero-shift should dominate shifted variants.
    pass_shift = bool((z_sh >= max_other_sh) and (z_cum >= max_other_cum))
    return pass_shift, z_sh, z_cum, max_other_sh, max_other_cum


def audit_symbol(row: pd.Series, cfg: AuditConfig) -> dict:
    out = {
        'symbol': str(row['symbol']),
        'status_input': str(row.get('status', '')),
        'leak_audit_ok': bool(row.get('leak_audit_ok', False)),
        'phase14_metrics_ok': bool(row.get('phase14_metrics_ok', False)),
    }

    if str(row.get('status', '')) != 'SUCCESS':
        out['audit_status'] = 'FAIL'
        out['audit_reason'] = 'symbol status != SUCCESS'
        return out

    sanity_path = Path(str(row.get('phase13_sanity_json', '')))
    p14_dir = Path(str(row.get('phase14_run_dir', '')))

    out['sanity_json_exists'] = bool(sanity_path.exists())
    out['phase14_summary_exists'] = bool((p14_dir / 'phase14_summary.json').exists())

    if not out['sanity_json_exists'] or not out['phase14_summary_exists']:
        out['audit_status'] = 'FAIL'
        out['audit_reason'] = 'missing sanity or phase14 summary'
        return out

    sanity = load_json(sanity_path)
    tests = sanity.get('tests', {})
    mdl = tests.get(cfg.sanity_model)
    if not isinstance(mdl, dict):
        out['audit_status'] = 'FAIL'
        out['audit_reason'] = f'missing sanity model {cfg.sanity_model}'
        return out

    obs = mdl.get('observed_metrics', {})
    perm = mdl.get('permutation_iid', {})
    rnd = mdl.get('random_baseline', {})
    shift = mdl.get('shift_alignment_nonoverlap', {})

    p_cum = float(perm.get('pvalue_cum_ge_obs', 1.0))
    p_sh = float(perm.get('pvalue_sharpe_ge_obs', 1.0))
    out['perm_pvalue_cum'] = p_cum
    out['perm_pvalue_sharpe'] = p_sh
    out['perm_pass'] = bool((p_cum <= cfg.pvalue_threshold) and (p_sh <= cfg.pvalue_threshold))

    obs_cum = float(obs.get('cum_net_return', np.nan))
    obs_sh = float(obs.get('sharpe_ann', np.nan))
    null_cum_p95 = float(rnd.get('null_cum_p95', np.nan))
    null_sh_p95 = float(rnd.get('null_sharpe_p95', np.nan))
    out['obs_cum'] = obs_cum
    out['obs_sharpe'] = obs_sh
    out['rand_null_cum_p95'] = null_cum_p95
    out['rand_null_sharpe_p95'] = null_sh_p95
    out['random_baseline_pass'] = bool((obs_cum > null_cum_p95) and (obs_sh > null_sh_p95))

    shift_pass, z_sh, z_cum, max_other_sh, max_other_cum = _shift_stats(shift)
    out['shift_zero_pass'] = bool(shift_pass)
    out['shift0_sharpe'] = float(z_sh) if np.isfinite(z_sh) else np.nan
    out['shift0_cum'] = float(z_cum) if np.isfinite(z_cum) else np.nan
    out['shift_max_other_sharpe'] = float(max_other_sh) if np.isfinite(max_other_sh) else np.nan
    out['shift_max_other_cum'] = float(max_other_cum) if np.isfinite(max_other_cum) else np.nan

    # Phase14 metrics from symbol results row.
    x1_sh = float(row.get('hardened_test_nooverlap_x1_sharpe_ann', np.nan))
    x1_cum = float(row.get('hardened_test_nooverlap_x1_cum_net_return', np.nan))
    x2_sh = float(row.get('hardened_test_nooverlap_x2_sharpe_ann', np.nan))
    x2_cum = float(row.get('hardened_test_nooverlap_x2_cum_net_return', np.nan))
    x3_sh = float(row.get('hardened_test_nooverlap_x3_sharpe_ann', np.nan))
    x3_cum = float(row.get('hardened_test_nooverlap_x3_cum_net_return', np.nan))

    out['x1_pass'] = bool((x1_sh > 0.0) and (x1_cum > 0.0))
    out['x2_guardrail_pass'] = bool((x2_sh > 0.0) and (x2_cum > 0.0))
    out['x3_guardrail_pass'] = bool((x3_sh > 0.0) and (x3_cum > 0.0))

    # Corrected no-overlap Sharpe diagnostic to avoid stride inflation interpretation.
    p14 = load_json(p14_dir / 'phase14_summary.json')
    stride = int(p14.get('eval_stride', 1))
    stride = max(1, stride)
    out['eval_stride'] = stride
    out['x1_sharpe_stride_adjusted'] = float(x1_sh / np.sqrt(stride)) if np.isfinite(x1_sh) else np.nan

    hard_fail = [
        not out['leak_audit_ok'],
        not out['phase14_metrics_ok'],
        not out['perm_pass'],
        not out['shift_zero_pass'],
        not out['random_baseline_pass'],
        not out['x1_pass'],
    ]

    if any(hard_fail):
        out['audit_status'] = 'FAIL'
    elif out['x2_guardrail_pass']:
        out['audit_status'] = 'PASS'
    else:
        out['audit_status'] = 'WARN'

    reasons: List[str] = []
    if not out['leak_audit_ok']:
        reasons.append('leak gate failed')
    if not out['perm_pass']:
        reasons.append('permutation pvalue not significant')
    if not out['random_baseline_pass']:
        reasons.append('does not beat random baseline p95')
    if not out['shift_zero_pass']:
        reasons.append('shift placebo failed (zero not dominant)')
    if not out['x1_pass']:
        reasons.append('x1 no-overlap not positive')
    if out['audit_status'] == 'WARN' and not out['x2_guardrail_pass']:
        reasons.append('x2 guardrail failed')

    out['audit_reason'] = '; '.join(reasons) if reasons else 'ok'
    return out


def main():
    parser = argparse.ArgumentParser(description='Phase15 anti-fantasy audit (leak, placebo, robustness).')
    parser.add_argument('--run_dir', required=True, type=str, help='Phase15 run directory.')
    parser.add_argument('--sanity_model', default='phase12_2_frozen', choices=['phase12_2_frozen', 'behavior_1x'])
    parser.add_argument('--pvalue_threshold', default=0.10, type=float)
    args = parser.parse_args()

    cfg = AuditConfig(
        run_dir=str(Path(args.run_dir).resolve()),
        sanity_model=str(args.sanity_model),
        pvalue_threshold=float(args.pvalue_threshold),
    )

    run_dir = Path(cfg.run_dir)
    res_csv = run_dir / 'phase15_symbol_results.csv'
    if not res_csv.exists():
        raise FileNotFoundError(f'Missing phase15_symbol_results.csv in {run_dir}')

    df = pd.read_csv(res_csv)
    rows = [audit_symbol(r, cfg) for _, r in df.iterrows()]
    out_df = pd.DataFrame(rows)

    out_csv = run_dir / 'phase15_audit_symbol_checks.csv'
    out_df.to_csv(out_csv, index=False)

    counts = out_df['audit_status'].value_counts().to_dict() if len(out_df) else {}
    summary = {
        'status': 'SUCCESS',
        'created_at_utc': now_utc_iso(),
        'run_dir': str(run_dir),
        'sanity_model': cfg.sanity_model,
        'pvalue_threshold': cfg.pvalue_threshold,
        'n_symbols': int(len(out_df)),
        'n_pass': int(counts.get('PASS', 0)),
        'n_warn': int(counts.get('WARN', 0)),
        'n_fail': int(counts.get('FAIL', 0)),
        'overall': (
            'FAIL' if counts.get('FAIL', 0) > 0
            else ('WARN' if counts.get('WARN', 0) > 0 else 'PASS')
        ),
        'outputs': {
            'symbol_checks_csv': str(out_csv),
        },
    }

    out_json = run_dir / 'phase15_audit_summary.json'
    safe_json_write(out_json, summary)

    print(f'SUCCESS: {out_json}')


if __name__ == '__main__':
    main()
