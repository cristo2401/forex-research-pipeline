# Estado y reproducibilidad

## Qué puede comprobar un lector

| Nivel | Evidencia disponible | Límite |
|---|---|---|
| Preparación de datos | `examples/synthetic_data_demo.py` ejecuta fase 1 con un fixture determinista y contrasta OHLCV, conteos, completitud y gaps. | No cubre todas las features, todos los formatos ni duplicados entre row groups. |
| Código experimental | Notebooks y runners organizados en `phases/`. | La existencia del código no prueba integración ni validez estadística. |
| Resultados históricos | [Resultados](RESULTS_AND_ROADMAP.md), [hallazgos](HISTORICAL_FINDINGS.md) y figuras. | Reportes de corridas previas; no se han regenerado desde esta edición pública sin sus datos/modelos. |
| Inferencia de fase 13 | Runner con modo `infer_frozen`. | Requiere bundles locales; `retrain_full` ejecuta la misma ruta congelada en v1. |
| Portafolio y ejecución simulada | Prototipos de fases 16–21 y documentación de costos/guardrails. | No hay demostración pública autocontenida del recorrido completo. La evaluación reciente registrada falla los guardrails. |

La etiqueta *leak-safe* no se usa como garantía global. Algunos experimentos tienen
particiones temporales, auditorías y separación de conjuntos; falta demostrar que
todos los ajustes, decisiones y transformaciones respetan esas restricciones al
ejecutarse juntos.

## Próximos hitos verificables

1. Ampliar los casos de datos: duplicados entre row groups, desorden global,
   timestamps inválidos y consistencia de features al cambiar la partición del Parquet.
2. Fijar versiones y describir origen, versión y esquema de cada artefacto requerido.
3. Construir una ruta pequeña con datos distribuibles que entrene, infiera y evalúe
   desde cero, sin dependencias de directorios locales.
4. Validar temporalmente todo ajuste: escalado, clustering, selección de modelos y
   umbrales; incluir baselines simples, ventanas sin solapamiento y sensibilidad a costos.
5. Separar pruebas de software de criterios estadísticos. Pasar CI o completar una
   corrida no demuestra rentabilidad ni habilita operación real.

## Prueba pública

```bash
python -m pip install -r requirements-demo.txt
python examples/synthetic_data_demo.py
```

La demo usa solo NumPy, pandas y PyArrow. El conjunto completo de dependencias de
investigación está en `requirements.txt`; instalarlo no proporciona los modelos ni
datos omitidos.
