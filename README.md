# Modelo de Predicción de Ventas - Desafío Carozzi

## Descripción del Proyecto
Este proyecto implementa un modelo predictivo de ventas diarias por tienda utilizando XGBoost. El modelo está diseñado para predecir ventas con un horizonte de 6 meses (26 semanas) para múltiples tiendas en Ecuador.

## Métricas del Modelo

| Conjunto     | RMSE     | MAE      | R²      |
|-------------|----------|----------|---------|
| Train       | 372.81   | 170.46   | 0.889   |
| Validation  | 382.15   | 178.92   | 0.878   |
| Test        | 379.63   | 175.31   | 0.882   |

## Estructura del Proyecto
```
├── data/
│   ├── data.csv      # Datos de ventas
│   ├── stores.csv    # Información de tiendas
│   └── oil.csv       # Precios del petróleo
├── Desafío_Carozzi.ipynb  # Notebook principal
└── README.md
```

## Justificación del uso de XGBoost

XGBoost fue seleccionado como el algoritmo principal para este caso de predicción de ventas por las siguientes razones:

1. **Complejidad del Problema**
   - Las ventas minoristas tienen patrones no lineales y múltiples dependencias
   - Existen interacciones complejas entre variables (ej: temporada, tipo de tienda, precio del petróleo)
   - Los patrones de venta varían significativamente entre diferentes tipos de tiendas

2. **Manejo de Características Temporales**
   - Excelente capacidad para manejar features de series temporales
   - Procesa eficientemente lags y medias móviles
   - Captura patrones estacionales a través de features cíclicas

3. **Rendimiento y Escalabilidad**
   - Entrenamiento eficiente con grandes volúmenes de datos
   - Paralelización automática que acelera el procesamiento
   - Manejo eficiente de memoria para datasets grandes

4. **Control de Overfitting**
   - Regularización incorporada (L1 y L2)
   - Early stopping para prevenir sobreajuste
   - Validación cruzada para optimización de hiperparámetros

5. **Robustez**
   - Manejo efectivo de valores faltantes
   - Resistente a outliers en los datos de ventas
   - Capacidad para capturar patrones a diferentes escalas temporales

## Características del Modelo

### Features Principales
1. **Features Base**
   - Número de tienda (store_nbr)
   - Precio del petróleo (dcoilwtico)
   - Indicador de fin de semana (is_weekend)

2. **Features Temporales**
   - Componentes estacionales mensuales (mes_sin, mes_cos)
   - Componentes diarios (dia_sin, dia_cos)
   - Tendencia normalizada (trend_norm)

3. **Features de Lag y Medias Móviles**
   - Lags semanales (1-4 semanas)
   - Medias móviles (7, 14, 30 días)

4. **Encodings Categóricos**
   - Tipos de tienda (one-hot encoding)

### Configuración del Modelo XGBoost
```python
XGBRegressor(
    max_depth=7,
    n_estimators=500,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective='reg:squarederror'
)
```

## Resultados y Métricas

### Métricas de Rendimiento
| Conjunto    | RMSE    | MAE     | R²      |
|-------------|---------|---------|---------|
| Train       | 372.81  | 170.46  | 0.889   |
| Validation  | 382.15  | 178.92  | 0.878   |
| Test        | 379.63  | 175.31  | 0.882   |



### Features Más Importantes
1. sales_ma_30d (Media móvil 30 días)
2. sales_ma_14d (Media móvil 14 días)
3. sales_lag_1w (Ventas semana anterior)
4. store_nbr (Número de tienda)
5. trend_norm (Tendencia normalizada)

### Predicciones a 6 Meses
- Crecimiento proyectado: 2.8%
- Patrones estacionales identificados con picos en diciembre y julio
- Correlación moderada con precio del petróleo (0.32)

## Insights Principales

### Patrones Temporales
- Mayores ventas en fines de semana (+15% vs días entre semana)
- Pico estacional en diciembre (incremento del 25%)
- Tendencia de crecimiento sostenido por tipo de tienda

### Impacto del Precio del Petróleo
- Correlación positiva en tiendas grandes (0.45)
- Correlación negativa en tiendas pequeñas (-0.28)
- Impacto en costos operativos y comportamiento del consumidor

## Recomendaciones
1. **Gestión de Inventario**
   - Ajustar stocks según patrones estacionales
   - Considerar impacto del precio del petróleo

2. **Estrategias por Tipo de Tienda**
   - Personalización según características específicas
   - Adaptación a patrones de demanda

3. **Optimización Operativa**
   - Ajuste de personal según patrones semanales
   - Consideración de factores externos

## Requisitos de Instalación
```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
```

## Uso
1. Clonar el repositorio
2. Instalar dependencias
3. Ejecutar el notebook Desafío_Carozzi.ipynb

## Autor
Ruth1982

## Licencia
Este proyecto es parte del Desafío Carozzi y está sujeto a sus términos y condiciones.
