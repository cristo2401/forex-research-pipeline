# Phase 17 - Meta Portafolio Multi-Divisa

## Script principal

- [`run_phase17_1_meta_portfolio.py`](run_phase17_1_meta_portfolio.py)
- [`run_phase17_2_meta_portfolio_improved.py`](run_phase17_2_meta_portfolio_improved.py)
- [`run_phase17_3_walkforward_validation.py`](run_phase17_3_walkforward_validation.py)
- [`run_phase17_4_walkforward_gated_ensemble.py`](run_phase17_4_walkforward_gated_ensemble.py)
- [`run_phase17_5_hierarchical_gym.py`](run_phase17_5_hierarchical_gym.py)
- [`run_phase17_6_distill_turnover.py`](run_phase17_6_distill_turnover.py)

## Objetivo

Entrenar un meta-modelo de portafolio sobre ventanas de estado de `phase16.4` con targets enmascarados (`y_mask`) y evaluar neto con costos `x1/x2/x3`.

## Ejecucion

```bash
source .venv/bin/activate

python "phases/phase17_meta_portfolio/run_phase17_1_meta_portfolio.py" \
  --phase16_4_run_dir ./phases/phase16_state_embeddings/output_phase16_4_portfolio_state_full12_tuned_v3 \
  --out_dir "./phases/phase17_meta_portfolio/output_phase17_1_meta_portfolio_v1" \
  --epochs 35 \
  --batch_size 128 \
  --d_model 192 \
  --nhead 6 \
  --num_layers 3 \
  --ff_dim 384 \
  --dropout 0.1 \
  --lr 4e-4 \
  --weight_decay 1e-5 \
  --max_gross_leverage 2.0 \
  --weight_temperature 1.0 \
  --cost_per_turnover 5e-5
```

## Salidas

- `phase17_1_summary.json`
- `phase17_1_metrics.csv`
- `phase17_1_train_history.csv`
- `phase17_1_predictions.parquet`
- `phase17_1_backtest_steps.parquet`
- `phase17_1_test_equity.png`
- `phase17_1_model.pt`

## Phase17.2 (mejorada)

Mejoras principales:
- objetivo de exceso vs baseline `equal_weight_target`,
- control explicito de exposicion (`target_gross_leverage`),
- baseline adicional `equal_weight_max` para referencia dura.

Ejecucion:

```bash
source .venv/bin/activate

python "phases/phase17_meta_portfolio/run_phase17_2_meta_portfolio_improved.py" \
  --phase16_4_run_dir ./phases/phase16_state_embeddings/output_phase16_4_portfolio_state_full12_tuned_v3 \
  --out_dir "./phases/phase17_meta_portfolio/output_phase17_2_meta_portfolio_v1" \
  --epochs 50 \
  --batch_size 128 \
  --d_model 192 \
  --nhead 6 \
  --num_layers 3 \
  --ff_dim 384 \
  --dropout 0.1 \
  --lr 3e-4 \
  --weight_decay 1e-5 \
  --max_gross_leverage 2.0 \
  --target_gross_leverage 0.9 \
  --weight_temperature 1.0 \
  --loss_w_ret 1.0 \
  --loss_w_risk 0.3 \
  --loss_w_port 1.0 \
  --loss_w_excess 1.0 \
  --loss_w_exposure 0.15 \
  --loss_w_l1 0.003 \
  --train_shuffle 0 \
  --cost_per_turnover 5e-5
```

Salidas:
- `phase17_2_summary.json`
- `phase17_2_metrics.csv`
- `phase17_2_train_history.csv`
- `phase17_2_predictions.parquet`
- `phase17_2_backtest_steps.parquet`
- `phase17_2_test_equity.png`
- `phase17_2_model.pt`

## Phase17.3 (walk-forward)

Objetivo:
- validar robustez temporal real con reentrenamiento por fold (expanding window),
- comparar en test no solapado vs `equal_weight_target` y `equal_weight_max`.

Ejecucion:

```bash
source .venv/bin/activate

python "phases/phase17_meta_portfolio/run_phase17_3_walkforward_validation.py" \
  --phase16_4_run_dir ./phases/phase16_state_embeddings/output_phase16_4_portfolio_state_full12_tuned_v3 \
  --out_dir "./phases/phase17_meta_portfolio/output_phase17_3_walkforward_v1" \
  --min_train_windows 1200 \
  --val_windows 128 \
  --test_windows 256 \
  --step_windows 256 \
  --epochs 30 \
  --patience 6 \
  --batch_size 128 \
  --d_model 192 \
  --nhead 6 \
  --num_layers 3 \
  --ff_dim 384 \
  --dropout 0.1 \
  --lr 3e-4 \
  --weight_decay 1e-5 \
  --max_gross_leverage 2.0 \
  --target_gross_leverage 0.9 \
  --weight_temperature 1.0 \
  --loss_w_ret 1.0 \
  --loss_w_risk 0.3 \
  --loss_w_port 1.0 \
  --loss_w_excess 1.0 \
  --loss_w_exposure 0.15 \
  --loss_w_l1 0.003 \
  --cost_per_turnover 5e-5
```

Salidas:
- `phase17_3_summary.json`
- `phase17_3_fold_ranges.csv`
- `phase17_3_fold_metrics.csv`
- `phase17_3_pooled_metrics.csv`
- `phase17_3_test_steps.parquet`
- `phase17_3_pooled_test_equity.png`

## Phase17.4 (walk-forward + gating + ensemble)

Objetivo:
- robustecer contra sobreajuste de fold con:
1. ensemble por semillas,
2. reweight temporal (mas peso a datos recientes),
3. penalizacion de turnover en entrenamiento,
4. gating en validacion (si no supera baseline, no opera modelo en ese fold).

Ejecucion:

```bash
source .venv/bin/activate

python "phases/phase17_meta_portfolio/run_phase17_4_walkforward_gated_ensemble.py" \
  --phase16_4_run_dir ./phases/phase16_state_embeddings/output_phase16_4_portfolio_state_full12_tuned_v3 \
  --out_dir "./phases/phase17_meta_portfolio/output_phase17_4_walkforward_gated_v1" \
  --min_train_windows 1200 \
  --val_windows 128 \
  --test_windows 256 \
  --step_windows 256 \
  --epochs 20 \
  --patience 5 \
  --batch_size 128 \
  --d_model 192 \
  --nhead 6 \
  --num_layers 3 \
  --ff_dim 384 \
  --dropout 0.1 \
  --lr 3e-4 \
  --weight_decay 1e-5 \
  --max_gross_leverage 2.0 \
  --target_gross_leverage 0.9 \
  --weight_temperature 1.0 \
  --loss_w_ret 1.0 \
  --loss_w_risk 0.3 \
  --loss_w_port 1.0 \
  --loss_w_excess 1.0 \
  --loss_w_exposure 0.15 \
  --loss_w_l1 0.003 \
  --loss_w_turnover 0.05 \
  --temporal_weight_min 0.6 \
  --temporal_weight_max 1.6 \
  --cost_per_turnover 5e-5 \
  --turnover_smooth_alpha 0.7 \
  --ensemble_seeds 42,43 \
  --gate_min_excess_sharpe 0.10 \
  --gate_min_excess_cum 0.0
```

Salidas:
- `phase17_4_summary.json`
- `phase17_4_fold_ranges.csv`
- `phase17_4_fold_metrics.csv`
- `phase17_4_gate_decisions.csv`
- `phase17_4_pooled_metrics.csv`
- `phase17_4_test_steps.parquet`
- `phase17_4_pooled_test_equity.png`

## Phase17.5 (jerarquico tipo gym + imitacion + RL)

Objetivo:
- agregar politica jerarquica por simbolo (gate + direccion + presupuesto),
- warm-start por imitacion desde señales lag1 del pipeline (`score_ls_lag1`, `pred_action_lag1`, etc.),
- fine-tuning RL-like con reward de exceso vs baseline,
- mantener gating por fold y agregar shield de drawdown en evaluacion.

Ejecucion:

```bash
source .venv/bin/activate

python "phases/phase17_meta_portfolio/run_phase17_5_hierarchical_gym.py" \
  --phase16_4_run_dir ./phases/phase16_state_embeddings/output_phase16_4_portfolio_state_full12_tuned_v3 \
  --out_dir "./phases/phase17_meta_portfolio/output_phase17_5_hierarchical_gym_v1" \
  --min_train_windows 1200 \
  --val_windows 128 \
  --test_windows 256 \
  --step_windows 256 \
  --imitation_epochs 8 \
  --rl_epochs 18 \
  --patience 6 \
  --batch_size 128 \
  --d_model 192 \
  --nhead 6 \
  --num_layers 3 \
  --ff_dim 384 \
  --dropout 0.1 \
  --lr 3e-4 \
  --weight_decay 1e-5 \
  --max_gross_leverage 2.0 \
  --target_gross_leverage 0.9 \
  --min_target_mult 0.35 \
  --max_target_mult 1.45 \
  --weight_temperature 1.0 \
  --cost_per_turnover 5e-5 \
  --turnover_smooth_alpha 0.7 \
  --shield_dd_limit -0.08 \
  --shield_risk_mult 0.55 \
  --ensemble_seeds 42,43 \
  --gate_min_excess_sharpe 0.05 \
  --gate_min_excess_cum 0.0
```

Salidas:
- `phase17_5_summary.json`
- `phase17_5_fold_ranges.csv`
- `phase17_5_train_history.csv`
- `phase17_5_fold_metrics.csv`
- `phase17_5_gate_decisions.csv`
- `phase17_5_pooled_metrics.csv`
- `phase17_5_test_steps.parquet`
- `phase17_5_pooled_test_equity.png`

## Phase17.6 (distill + turnover + gate adaptativo)

Objetivo:
- destilar mejor desde `teacher_lag1` y mezclar en inferencia (`teacher_mix_eval`),
- optimizar exceso vs baseline y vs teacher durante RL,
- gate por fold entre `model`, `teacher` y `equal_weight_target` con control de turnover.

Ejecucion:

```bash
source .venv/bin/activate

python "phases/phase17_meta_portfolio/run_phase17_6_distill_turnover.py" \
  --phase16_4_run_dir ./phases/phase16_state_embeddings/output_phase16_4_portfolio_state_full12_tuned_v3 \
  --out_dir "./phases/phase17_meta_portfolio/output_phase17_6_distill_turnover_v2" \
  --min_train_windows 1200 \
  --val_windows 128 \
  --test_windows 256 \
  --step_windows 256 \
  --imitation_epochs 8 \
  --rl_epochs 18 \
  --patience 6 \
  --batch_size 128 \
  --d_model 192 \
  --nhead 6 \
  --num_layers 3 \
  --ff_dim 384 \
  --dropout 0.1 \
  --lr 3e-4 \
  --weight_decay 1e-5 \
  --max_gross_leverage 2.0 \
  --target_gross_leverage 0.9 \
  --teacher_mix_eval 0.8 \
  --loss_w_teacher_excess 0.25 \
  --gate_min_excess_sharpe 0.05 \
  --gate_min_excess_cum 0.0 \
  --gate_turnover_ratio_max 1.2 \
  --cost_per_turnover 5e-5
```

Salidas:
- `phase17_6_summary.json`
- `phase17_6_fold_ranges.csv`
- `phase17_6_train_history.csv`
- `phase17_6_fold_metrics.csv`
- `phase17_6_gate_decisions.csv`
- `phase17_6_pooled_metrics.csv`
- `phase17_6_test_steps.parquet`
- `phase17_6_pooled_test_equity.png`
