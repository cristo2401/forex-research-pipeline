# Phase 16 - Embeddings de Estado Derivado (Leak-Safe)

Esta fase implementa la primera capa del meta-modelo multi-divisa:

`estado_t -> ventana temporal -> embedding contextual`

No usa precio crudo directo. Usa salidas causales del pipeline (`phase14_policy_strict_predictions.parquet`) y entrena un Transformer auto-supervisado para comprimir cada ventana en un vector fijo.

## Entrada

- `phase15_run_dir` terminado en `SUCCESS` (por ejemplo):
  - `phase15_selected7.csv` (o `phase15_symbol_results.csv`)
  - rutas a `phase14_run_dir` por símbolo
  - en cada `phase14_run_dir`: `phase14_policy_strict_predictions.parquet`

## Script principal

- [`run_phase16_1_state_embedding.py`](run_phase16_1_state_embedding.py)

## Uso

```bash
source .venv/bin/activate

python phase16/run_phase16_1_state_embedding.py \
  --phase15_run_dir ./phases/phase15_multi_currency/output_runs/phase15_full_10pairs_20260322_stride8_v1 \
  --universe selected7 \
  --window_len 48 \
  --window_stride 1 \
  --include_symbol_onehot 0 \
  --scaler_mode per_symbol \
  --cluster_fit_max_per_symbol 3000 \
  --cluster_on_l2_normalized_embeddings 1 \
  --epochs 20 \
  --k_regimes 8
```

## Señales de seguridad anti-leak

- Split temporal estricto por símbolo (`train/val/test`).
- Escalador (`mean/std`) ajustado solo con filas de `train`.
- `estado_t` usa columnas causales (sin `future_*`, `net_*`, `cum_*`, ni targets futuros).
- Embeddings/clustering se entrenan con el pasado y se aplican al resto.
- Anti-colapso por símbolo:
  - por defecto no se incluye one-hot de símbolo (`--include_symbol_onehot 0`)
  - escalado por símbolo (`--scaler_mode per_symbol`)
  - fit de KMeans balanceado por símbolo (`--cluster_fit_max_per_symbol`)

## Salidas

En `out_dir` (por defecto: `<phase15_run_dir>/phase16_state_embeddings`):

- `phase16_summary.json`
- `phase16_state_rows_raw.parquet`
- `phase16_state_rows_std.parquet`
- `phase16_windows_meta.parquet`
- `phase16_embeddings.parquet`
- `phase16_embeddings.npz`
- `phase16_embedding_model.pt`
- `phase16_training_history.csv`
- `phase16_windows_split_count.csv`
- `phase16_embeddings_pca_train.png`
- `phase16_training_loss.png`
- `phase16.log`

## Siguiente paso sugerido

Con estos embeddings ya puedes pasar a:

1. Clustering contextual más fino por secuencia.
2. Construcción del estado de portafolio multi-divisa.
3. Entrenamiento RL con perfiles de riesgo (conservador, balanceado, agresivo).

## Phase16.2 - Primer Cluster

Script:
- [`run_phase16_2_first_cluster.py`](run_phase16_2_first_cluster.py)

Ejemplo:

```bash
source .venv/bin/activate

python phase16/run_phase16_2_first_cluster.py \
  --phase16_run_dir ./phases/phase16_state_embeddings/output_phase16_1_state_embedding_selected7_v2_smoke \
  --out_dir ./phases/phase16_state_embeddings/output_phase16_2_first_cluster_selected7_v2_smoke \
  --k 0 \
  --k_min 6 \
  --k_max 12 \
  --pca_dim 24 \
  --normalize_embeddings 1 \
  --k_objective_val_weight 0.65 \
  --k_objective_purity_weight 0.25
```

## Phase16.3 - Context Transformer + Super-Cluster

Script:
- [`run_phase16_3_context_transformer.py`](run_phase16_3_context_transformer.py)

Ejemplo:

```bash
source .venv/bin/activate

python phase16/run_phase16_3_context_transformer.py \
  --phase16_2_run_dir ./phases/phase16_state_embeddings/output_phase16_2_first_cluster_selected7_v2_full12_tuned \
  --out_dir ./phases/phase16_state_embeddings/output_phase16_3_context_transformer_full12_tuned_v1 \
  --seq_len 48 \
  --d_model 128 \
  --nhead 8 \
  --num_layers 3 \
  --ff_dim 256 \
  --dropout 0.1 \
  --batch_size 512 \
  --epochs 12 \
  --lr 7e-4 \
  --k_context 8
```

Salidas principales:
- `phase16_3_context_embeddings.parquet`
- `phase16_3_context_regime_symbol_mix.csv`
- `phase16_3_next_token_acc_by_split.csv`
- `phase16_3_train_history.csv`
- `phase16_3_context_transformer.pt`
- `phase16_3_summary.json`

## Phase16.4 - Dataset Portafolio (Leak-Safe) para RL

Script:
- [`run_phase16_4_portfolio_state_dataset.py`](run_phase16_4_portfolio_state_dataset.py)

Objetivo:
- Construir estado multi-divisa a nivel portafolio en formato:
  - `long` (por símbolo/tiempo)
  - `wide` (portafolio por timestamp)
  - `windows` (`X` secuencial + targets con máscara)

Puntos importantes de seguridad:
- Features forward-fill controladas por `--max_ffill_steps`.
- Targets **no** se forward-fillean (evita leak temporal).
- Ventanas sin mezcla de splits (`train/val/test`).
- Validación de `target_ts > state_ts` en long y wide.

Ejemplo:

```bash
source .venv/bin/activate

python phase16/run_phase16_4_portfolio_state_dataset.py \
  --phase16_3_run_dir ./phases/phase16_state_embeddings/output_phase16_3_context_transformer_full12_tuned_v1 \
  --out_dir ./phases/phase16_state_embeddings/output_phase16_4_portfolio_state_full12_tuned_v3 \
  --window_len 24 \
  --window_stride 1 \
  --train_ratio 0.70 \
  --val_ratio 0.15 \
  --max_ffill_steps 96 \
  --min_target_symbols 3 \
  --corr_alert_threshold 0.35
```

Artefactos:
- `phase16_4_state_long.parquet`
- `phase16_4_state_wide.parquet`
- `phase16_4_windows.npz` (`x`, `y_ret1`, `y_ret4`, `y_vol4`, `y_mask`)
- `phase16_4_windows_meta.parquet`
- `phase16_4_sanity_checks.json`
- `phase16_4_summary.json`
