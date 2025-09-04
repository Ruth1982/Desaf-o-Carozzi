# Guión de Presentación - Predicción de Ventas Carozzi

## 1. INTRODUCCIÓN (2 min)

"Buenos días/tardes. Hoy les presentaré los resultados del proyecto de predicción de ventas que desarrollamos para Carozzi.

Nuestro objetivo principal fue crear un modelo predictivo capaz de estimar las ventas diarias por tienda con un horizonte de 6 meses. Este proyecto es fundamental para la optimización de inventarios y la planificación estratégica de la compañía.

## 2. ANÁLISIS Y METODOLOGÍA (5 min)

Para abordar este desafío, trabajamos con tres fuentes principales de datos:
- Histórico de ventas por tienda
- Información detallada de cada tienda
- Datos de precios del petróleo como variable macroeconómica

Después de un análisis exhaustivo, seleccionamos XGBoost como nuestro algoritmo principal por tres razones fundamentales:
1. Su capacidad para manejar relaciones no lineales complejas
2. Excelente rendimiento con datos temporales
3. Robustez ante valores atípicos y datos faltantes

## 3. RESULTADOS PRINCIPALES (7 min)

Me complace compartir que nuestro modelo alcanzó métricas muy sólidas:
- Un R² de 0.901 en el conjunto de prueba, lo que significa que explicamos el 90.1% de la variabilidad en las ventas
- Un error medio absoluto (MAE) de 205.84, que representa una desviación promedio muy aceptable para los volúmenes que manejamos

[MOSTRAR GRÁFICO DE PREDICCIONES VS REALES]

Como pueden ver en esta gráfica, nuestras predicciones (en azul) siguen muy de cerca los valores reales (en naranja), capturando incluso los patrones estacionales más sutiles.

## 4. INSIGHTS DE NEGOCIO (4 min)

Nuestro análisis reveló varios hallazgos importantes:

Primero, identificamos patrones claros en las ventas:
- Los fines de semana muestran un incremento del 15% en las ventas
- Diciembre presenta un pico estacional con un aumento del 25%

[MOSTRAR GRÁFICO DE PATRONES ESTACIONALES]

Un descubrimiento interesante fue la relación con el precio del petróleo:
- Las tiendas grandes muestran una correlación positiva (0.45)
- Las tiendas pequeñas presentan una correlación negativa (-0.28)

Esto sugiere diferentes estrategias según el tipo de tienda.

## 5. RECOMENDACIONES Y PRÓXIMOS PASOS (2 min)

Basándonos en estos hallazgos, recomendamos:

1. Gestión de Inventario:
   - Ajustar stocks considerando los patrones estacionales identificados
   - Implementar políticas diferenciadas por tipo de tienda

2. Optimización Operativa:
   - Reforzar personal en fines de semana y períodos pico
   - Adaptar la logística según el tipo de tienda

[CONCLUSIÓN]

Para concluir, este modelo nos permite:
- Predecir ventas con un 90.1% de precisión
- Anticipar patrones estacionales
- Optimizar operaciones por tipo de tienda

El siguiente paso es implementar estas recomendaciones y monitorear su impacto en el negocio.

¿Tienen alguna pregunta?

## RESPUESTAS A POSIBLES PREGUNTAS

1. Si preguntan sobre la confiabilidad del modelo:
"Realizamos una validación exhaustiva utilizando datos históricos y pruebas en diferentes períodos. El modelo mantiene su precisión incluso en condiciones variadas."

2. Si preguntan sobre la implementación:
"El modelo está diseñado para actualizarse automáticamente con nuevos datos, permitiendo ajustes y mejoras continuas."

3. Si preguntan sobre el impacto en el negocio:
"Estimamos que la implementación de estas recomendaciones puede optimizar los niveles de inventario en un 15-20% y mejorar la eficiencia operativa en períodos pico."

## FRASES CLAVE PARA ENFATIZAR

- "Nuestro modelo captura tanto patrones evidentes como sutiles en el comportamiento de ventas"
- "La precisión del 90.1% nos da una base sólida para la toma de decisiones"
- "Cada recomendación está respaldada por datos concretos"
- "El modelo es adaptable y escalable para futuros requerimientos"
