# Seguridad

## Alcance

Este proyecto es de investigación y *paper trading*. La ejecución de órdenes reales
no está soportada ni autorizada por el diseño publicado.

## Credenciales

- Nunca confirmes archivos `.env`, tokens OAuth, secretos de cliente o IDs de cuenta.
- Usa únicamente `phases/phase19_ctrader_paper_live/.env.example` como plantilla.
- Mantén las credenciales reales fuera del repositorio y usa una cuenta demo.
- Si una credencial aparece en un commit, revócala antes de limpiar el historial.

## Reporte de vulnerabilidades

No publiques credenciales ni detalles explotables en un issue abierto. Informa el
problema al propietario del repositorio mediante un canal privado de GitHub.

## Riesgo financiero

Una validación técnica exitosa no implica rentabilidad. Los guardrails, límites de
drawdown y kill switches no eliminan el riesgo de pérdida.
