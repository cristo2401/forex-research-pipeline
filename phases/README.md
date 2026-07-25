# Índice de fases

El repositorio conserva la evolución experimental completa. Las primeras fases son
principalmente notebooks; desde la fase 6 se incorporan runners de Python y desde la
fase 13 el flujo se consolida en CLIs reproducibles.

| Fase | Directorio | Propósito principal |
|---:|---|---|
| 1 | [`phase01_data_engineering`](phase01_data_engineering/) | Resample M1→H1, gaps e integridad |
| 2 | [`phase02_market_alphabet`](phase02_market_alphabet/) | Segmentación y alfabeto de patrones |
| 3 | [`phase03_segment_analysis`](phase03_segment_analysis/) | Análisis, padding, atención y clustering |
| 4 | [`phase04_transfer_clustering`](phase04_transfer_clustering/) | Transferencia de embeddings entre divisas |
| 5 | [`phase05_context_sequencing`](phase05_context_sequencing/) | Secuenciación y superclustering contextual |
| 6 | [`phase06_context_regimes`](phase06_context_regimes/) | Palabras de contexto y regímenes |
| 7 | [`phase07_online_detector`](phase07_online_detector/) | Detección online y predicción del siguiente token |
| 8 | [`phase08_two_stage_detector`](phase08_two_stage_detector/) | Detector en dos etapas |
| 9 | [`phase09_multitask_patterns`](phase09_multitask_patterns/) | Patrón parcial y aprendizaje multitarea |
| 10 | [`phase10_decision_engine`](phase10_decision_engine/) | Motor de decisión causal pre-RL |
| 11 | [`phase11_offline_rl`](phase11_offline_rl/) | RL offline, meta-políticas y walk-forward |
| 12 | [`phase12_leverage_rl`](phase12_leverage_rl/) | Leverage dinámico en entorno simulado |
| 13 | [`phase13_pipeline_cli`](phase13_pipeline_cli/) | Pipeline individual M1→señal |
| 13.5 | [`phase13_5_individual_paper_live`](phase13_5_individual_paper_live/) | Paper-live individual con auto-lookback |
| 14 | [`phase14_production_hardening`](phase14_production_hardening/) | Costos duros, turnover y guardrails |
| 15 | [`phase15_multi_currency`](phase15_multi_currency/) | Benchmark de 10 pares y selección de 7 |
| 16 | [`phase16_state_embeddings`](phase16_state_embeddings/) | Estado contextual y dataset de portafolio |
| 17 | [`phase17_meta_portfolio`](phase17_meta_portfolio/) | Meta-portafolio y validación walk-forward |
| 18 | [`phase18_multi_agent_ctde`](phase18_multi_agent_ctde/) | Multi-agente CTDE y holdout ciego |
| 18.5 | [`phase18_5_dynamic_leverage`](phase18_5_dynamic_leverage/) | Simulación de margen y leverage cTrader |
| 19 | [`phase19_ctrader_paper_live`](phase19_ctrader_paper_live/) | Integración demo y paper-live H1 |
| 20 | [`phase20_e2e_orchestration`](phase20_e2e_orchestration/) | Live, nightly hardening y estado consolidado |
| 21 | [`phase21_top_engine`](phase21_top_engine/) | Checkpoint, backtest reciente y top engine |

Los directorios `output_*`, datasets, modelos y credenciales no forman parte de la
edición pública. Cuando una fase depende de ellos, su README describe el contrato
esperado.
