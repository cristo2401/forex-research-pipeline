# Contribuir

Las contribuciones deben preservar los principios de causalidad, reproducibilidad y
seguridad del proyecto.

Antes de proponer cambios:

1. documenta el contrato de datos y el periodo evaluado;
2. evita ajustar parámetros sobre el holdout que después reportas;
3. incluye costos y slippage en las comparaciones;
4. conserva `paper_only=true` y la ejecución real deshabilitada;
5. no agregues datasets, modelos pesados ni credenciales;
6. ejecuta `python -m compileall -q phases`.

Los resultados negativos también son resultados válidos y deben quedar documentados.
