# Phase 18 - Multi-Agent CTDE

## Script principal

- [`run_phase18_1_multi_agent_ctde.py`](run_phase18_1_multi_agent_ctde.py)
- [`run_phase18_2_ctde_robust.py`](run_phase18_2_ctde_robust.py)
- [`run_phase18_3_blind_holdout_stress.py`](run_phase18_3_blind_holdout_stress.py)

## Objetivo

Entrenar un modelo **multi-agente CTDE** sobre ventanas de estado de `phase16.4`:
- actores descentralizados por divisa (heads por agente),
- critico central global,
- critico local por agente,
- evaluacion walk-forward con costos `x1/x2/x3` y gating contra baselines.

## Ejecucion

```bash
source .venv/bin/activate

python "phases/phase18_multi_agent_ctde/run_phase18_1_multi_agent_ctde.py" \
  --phase16_4_run_dir ./phases/phase16_state_embeddings/output_phase16_4_portfolio_state_full12_tuned_v3 \
  --out_dir "./phases/phase18_multi_agent_ctde/output_phase18_1_multi_agent_ctde_v1" \
  --min_train_windows 1200 \
  --val_windows 128 \
  --test_windows 256 \
  --step_windows 256 \
  --d_model 192 \
  --nhead 6 \
  --num_layers 3 \
  --ff_dim 384 \
  --dropout 0.1 \
  --batch_size 128 \
  --imitation_epochs 8 \
  --rl_epochs 18 \
  --patience 6 \
  --lr 3e-4 \
  --weight_decay 1e-5 \
  --max_gross_leverage 2.0 \
  --target_gross_leverage 0.9 \
  --loss_w_value 0.15 \
  --loss_w_value_local 0.08 \
  --loss_w_agent_excess 0.20 \
  --loss_w_teacher_excess 0.25 \
  --cost_per_turnover 5e-5 \
  --ensemble_seeds 42,43
```

## Salidas

- `phase18_1_summary.json`
- `phase18_1_fold_ranges.csv`
- `phase18_1_train_history.csv`
- `phase18_1_fold_metrics.csv`
- `phase18_1_gate_decisions.csv`
- `phase18_1_pooled_metrics.csv`
- `phase18_1_test_steps.parquet`
- `phase18_1_pooled_test_equity.png`

## Phase 18.2 (robustez x2/x3)

Objetivo:
- seleccionar por fold una politica robusta con validacion en costos duros (`x2/x3`),
- ajustar mezcla `model/teacher`, escala de riesgo y smoothing de turnover antes de test.

Ejecucion:

```bash
source .venv/bin/activate

python "phases/phase18_multi_agent_ctde/run_phase18_2_ctde_robust.py" \
  --phase16_4_run_dir ./phases/phase16_state_embeddings/output_phase16_4_portfolio_state_full12_tuned_v3 \
  --out_dir "./phases/phase18_multi_agent_ctde/output_phase18_2_multi_agent_ctde_robust_v1" \
  --robust_mode 1 \
  --robust_cost_multipliers 2,3 \
  --robust_teacher_mix_candidates 0.00,0.25,0.50,0.75,1.00 \
  --robust_risk_scales 0.70,0.85,1.00 \
  --robust_alpha_candidates 0.55,0.70,0.85
```

Salidas:
- `phase18_2_summary.json`
- `phase18_2_fold_metrics.csv`
- `phase18_2_gate_decisions.csv`
- `phase18_2_pooled_metrics.csv`
- `phase18_2_test_steps.parquet`

## Phase 18.3 (blind holdout + stress + go/no-go)

Objetivo:
- ejecutar una validacion ciega final (1 solo fold holdout, sin retocar hiperparametros),
- correr stress de costos/slippage/latencia sobre el holdout,
- emitir decision automatica `GO_PAPER` o `NO_GO`.

Ejecucion:

```bash
source .venv/bin/activate

python "phases/phase18_multi_agent_ctde/run_phase18_3_blind_holdout_stress.py" \
  --phase16_4_run_dir ./phases/phase16_state_embeddings/output_phase16_4_portfolio_state_full12_tuned_v3 \
  --out_dir "./phases/phase18_multi_agent_ctde/output_phase18_3_blind_holdout_v1"
```

Salidas:
- `phase18_3_blind_summary.json`
- `phase18_3_blind_manifest.json`
- `phase18_3_stress_metrics.csv`
- `phase18_3_blind_command.sh`
- `blind_run/` con todos los artefactos de `phase18.2` (single-fold holdout)
