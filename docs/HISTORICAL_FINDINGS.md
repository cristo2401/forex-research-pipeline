# Hallazgos históricos auditados

Este documento conserva resultados relevantes de las fases 9–18.5 y, al mismo
tiempo, delimita qué permiten concluir. Son resultados de investigación sobre
backtests y simulaciones; no representan operaciones reales ni una expectativa de
rentabilidad futura.

La evidencia reciente de la fase 21 tiene prioridad sobre estos experimentos
históricos. El estado vigente del proyecto continúa siendo `RESEARCH / PAPER_ONLY`.

## Lectura rápida

La revisión encontró dos cifras cercanas al 5 % que pueden confundirse:

1. **EURGBP produjo 5,34 % neto en el diagnóstico de 365 días de la fase 14**, con
   costos `x2`. Es la coincidencia exacta cuando se busca una divisa individual, pero
   el cálculo usa observaciones solapadas y no debe presentarse como un 5 % anual
   operable.
2. **La cartera de siete pares de la fase 18.5 produjo 15,60 % compuesto** entre los
   extremos temporales de octubre de 2022 y abril de 2025 después de la simulación de
   ejecución cTrader. Su equivalencia geométrica aproximada es 5,96 % anual, pero es
   un resultado de cartera *walk-forward*, no de EURGBP por separado.

## Alcance de la revisión

| Fases | Alcance observado | Papel en el hallazgo |
|---|---|---|
| 9–12 | Desarrollo principalmente sobre GBPUSD; la fase 11.6 añadió una prueba GBPUSD/EURUSD | No aparece una estimación anual comparable y defendible cercana al 5 % |
| 13–15 | Pipeline causal, hardening y benchmark final de 10 pares | Aquí aparece el diagnóstico de EURGBP |
| 16–18.5 | Estado multi-divisa, meta-portafolio, CTDE y ejecución cTrader | Los resultados pasan a ser de cartera; no hay atribución final de PnL por par |

## EURGBP: resultado individual que coincide con el 5 %

El run `phase15_full_10pairs_20260322_stride8_v1` ejecutó el hardening de fase 14
sobre diez pares. En EURGBP, el reporte `phase14_recent_period_metrics.csv` produjo
las siguientes métricas para el modelo `phase14_hardened` en la ventana nominal de
365 días:

| Escenario de costo | Retorno neto acumulado | Sharpe anualizado | Max drawdown |
|---|---:|---:|---:|
| `x1` — 1 pip | 6,66 % | 6,92 | -3,18 % |
| `x2` — 2 pips | **5,34 %** | 5,57 | -3,58 % |
| `x3` — 3 pips | 4,02 % | 4,21 | -4,29 % |

La muestra contenía 403 observaciones y el PnL simulado era positivo después del
costo supuesto. Sin embargo, esta tabla es un diagnóstico exploratorio, no la prueba
principal de ejecución.

### Por qué el 5,34 % necesita una advertencia

La implementación pública permite auditar la diferencia:

- [`evaluate_levels`](../phases/phase14_production_hardening/run_phase14_1_hardening.py#L64)
  define `cum_net_return` como la suma aritmética de retornos netos;
- [`maybe_nonoverlap`](../phases/phase14_production_hardening/run_phase14_1_hardening.py#L338)
  selecciona observaciones no solapadas mediante `eval_stride`;
- [`compute_recent_metrics`](../phases/phase14_production_hardening/run_phase14_1_hardening.py#L428)
  evalúa todas las filas recientes y no aplica esa selección;
- la llamada que genera el reporte reciente recibe directamente `d_hardened`, como
  muestra el [flujo principal](../phases/phase14_production_hardening/run_phase14_1_hardening.py#L605).

Los datos están organizados en segmentos de varios pasos y los retornos futuros de
filas próximas se solapan. Por ello, sumar las 403 observaciones puede contar
exposición temporal repetida y elevar el retorno y el Sharpe.

La prueba cronológica estricta, con `eval_stride=8` y sin solapamiento, es más
conservadora:

| Escenario de costo | Retorno neto acumulado | Sharpe anualizado | Max drawdown |
|---|---:|---:|---:|
| `x1` | 2,36 % | 1,16 | -2,08 % |
| `x2` | **1,54 %** | 0,76 | -2,10 % |
| `x3` | 0,71 % | 0,35 | -2,18 % |

Esta prueba contiene 251 observaciones en la cola cronológica comprendida entre abril
de 2022 y noviembre de 2025. El 1,54 % es acumulado para esa muestra; tampoco debe
reinterpretarse como retorno anual.

**Conclusión sobre EURGBP:** existió una señal histórica positiva y suficientemente
interesante para justificar investigación adicional. La evidencia disponible no
permite afirmar que EURGBP generase de forma robusta un 5 % anual.

## Resultados posteriores de cartera

Desde la fase 16 se utilizaron siete pares seleccionados:

`AUDUSD`, `EURGBP`, `EURUSD`, `GBPUSD`, `NZDUSD`, `USDCAD` y `USDCHF`.

### Fase 18.3: holdout ciego

El run `phase18_3_blind_holdout_v2` evaluó 378 pasos desde el 11 de julio de 2024
hasta el 7 de septiembre de 2025. La política `model_ctde_robust` registró:

| Escenario | Retorno neto compuesto | Sharpe anualizado | Max drawdown |
|---|---:|---:|---:|
| `x1` | 5,00 % | 6,74 | -1,23 % |
| `x2` | 4,38 % | 5,93 | -1,36 % |
| `x3` | 3,77 % | 5,12 | -1,50 % |
| `x4` | 3,16 % | 4,32 | -1,63 % |
| `x5` | 2,56 % | 3,51 | -1,77 % |

Este protocolo emitió `GO_PAPER`. Esa decisión significaba avanzar a observación
simulada, no habilitar capital real.

### Fase 18.5: simulación de ejecución cTrader

El run `output_phase18_5_final_main_v3c` seleccionó `model_ctde_robust` y procesó
1.024 pasos de prueba entre el 5 de octubre de 2022 y el 7 de abril de 2025.

| Capa | Retorno neto compuesto | Sharpe anualizado | Max drawdown |
|---|---:|---:|---:|
| Modelo base | 17,35 % | 7,24 | -1,53 % |
| Ejecución cTrader simulada | **15,60 %** | 6,56 | -1,62 % |

La simulación de ejecución incluye margen, costo adicional y *slippage*. El retorno
compuesto se calcula mediante `cumprod`, tanto en la
[capa CTDE](../phases/phase18_multi_agent_ctde/run_phase18_2_ctde_robust.py#L923)
como en la
[capa de ejecución](../phases/phase18_5_dynamic_leverage/run_phase18_5_dynamic_leverage_ctrader.py#L178).

Dividir o anualizar el 15,60 % por el tiempo transcurrido da una referencia aproximada
de 5,96 % anual compuesto. Esa cifra es derivada, no una métrica original del run, y
el muestreo *walk-forward* no equivale a una cuenta operando continuamente durante
todo el intervalo.

Además, los artefactos finales guardan métricas agregadas de cartera. No conservan
una atribución auditable que permita asignar ese 15,60 % a una sola divisa.

## Qué se puede afirmar

- EURGBP tuvo PnL simulado positivo en el experimento histórico revisado.
- El diagnóstico reciente de fase 14 registró 5,34 % con costos `x2`.
- La prueba sin solapamiento de EURGBP también fue positiva, pero mucho más modesta.
- Las fases 18.3 y 18.5 encontraron desempeño histórico positivo para una cartera de
  siete pares bajo sus respectivos supuestos de costos y ejecución.
- Estos resultados justificaron avanzar a paper trading y nuevas pruebas.

## Qué no se debe afirmar

- Que EURGBP produjo un 5 % anual robusto.
- Que los resultados son PnL real de broker.
- Que el 15,60 % de fase 18.5 pertenece a EURGBP.
- Que un Sharpe anualizado alto sobre muestras solapadas demuestra capacidad
  predictiva estable.
- Que `GO_PAPER` equivale a `GO_LIVE`.
- Que los resultados históricos prevalecen sobre el fallo de los guardrails recientes.

## Estado posterior

Las evaluaciones de fase 21 sobre ventanas recientes de 90 y 120 días fueron
negativas y el guardrail terminó en `FAILED`. Esa evidencia posterior impide usar los
hallazgos de este documento para declarar el sistema listo para producción.

Consulta [Resultados y roadmap](RESULTS_AND_ROADMAP.md) para el estado vigente y los
criterios necesarios antes de volver a `GO_PAPER`.
