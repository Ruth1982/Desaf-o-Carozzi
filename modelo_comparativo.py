import os
import json
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

def create_time_features(df):
    """Crea características temporales más detalladas"""
    df = df.copy()
    df['day_of_week'] = pd.to_datetime(df['week']).dt.dayofweek
    df['month'] = pd.to_datetime(df['week']).dt.month
    df['year'] = pd.to_datetime(df['week']).dt.year
    df['week_of_year'] = pd.to_datetime(df['week']).dt.isocalendar().week
    
    # Características cíclicas para capturar estacionalidad
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['week_sin'] = np.sin(2 * np.pi * df['week_of_year']/52)
    df['week_cos'] = np.cos(2 * np.pi * df['week_of_year']/52)
    
    return df

def create_lag_features(df, lags=[1, 2, 3, 4, 8, 12, 26, 52]):
    """Crea características de rezago y ventanas móviles"""
    df = df.copy()
    
    # Ordenar por tienda y fecha
    df = df.sort_values(['store', 'week'])
    
    # Crear rezagos de ventas
    for lag in lags:
        df[f'sales_lag_{lag}'] = df.groupby('store')['sales'].shift(lag)
    
    # Medias móviles
    windows = [4, 8, 12, 26]
    for window in windows:
        df[f'sales_rolling_mean_{window}'] = df.groupby('store')['sales'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )
        df[f'sales_rolling_std_{window}'] = df.groupby('store')['sales'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
        )
    
    return df

def create_store_features(df):
    """Crea características específicas de la tienda"""
    df = df.copy()
    
    # Estadísticas históricas por tienda
    store_stats = df.groupby('store').agg({
        'sales': ['mean', 'std', 'median']
    }).reset_index()
    
    store_stats.columns = ['store', 'store_mean_sales', 'store_std_sales', 'store_median_sales']
    df = df.merge(store_stats, on='store', how='left')
    
    # Ratio de ventas respecto a la media de la tienda
    df['sales_to_mean_ratio'] = df['sales'] / df['store_mean_sales']
    
    return df

def prepare_features(df):
    """Prepara todas las características para el modelo"""
    df = create_time_features(df)
    df = create_lag_features(df)
    df = create_store_features(df)
    
    return df

def train_store_model(train_data, valid_data, features):
    """Entrena un modelo XGBoost para una tienda específica"""
    # Preparar los datos
    X_train = train_data[features]
    y_train = train_data['sales']
    X_valid = valid_data[features]
    y_valid = valid_data['sales']
    
    # Escalar características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    
    # Configurar modelo XGBoost
    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    # Entrenar modelo
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_valid_scaled, y_valid)],
        early_stopping_rounds=20,
        verbose=False
    )
    
    # Predicciones
    y_pred = model.predict(X_valid_scaled)
    
    # Calcular métricas
    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    r2 = r2_score(y_valid, y_pred)
    
    return {
        'model': model,
        'scaler': scaler,
        'rmse': rmse,
        'r2': r2
    }

def load_data():
    """Carga y prepara los datos iniciales"""
    print("Cargando archivos originales...")
    
    try:
        # Cargar los archivos originales usando chunks para manejar archivos grandes
        print("Intentando cargar data.csv...")
        chunks = []
        for chunk in pd.read_csv('data/data.csv', chunksize=10000):
            chunks.append(chunk)
        df_data = pd.concat(chunks)
    df_stores = pd.read_csv('data/stores.csv')
    df_oil = pd.read_csv('data/oil.csv')
    
    # Normalizar nombres de columnas
    df_data.columns = df_data.columns.str.strip().str.lower()
    df_stores.columns = df_stores.columns.str.strip().str.lower()
    df_oil.columns = df_oil.columns.str.strip().str.lower()
    
    # Convertir fechas
    df_data['date'] = pd.to_datetime(df_data['date'])
    if 'date' in df_oil.columns:
        df_oil['date'] = pd.to_datetime(df_oil['date'])
    elif 'fecha' in df_oil.columns:
        df_oil['date'] = pd.to_datetime(df_oil['fecha'])
        df_oil = df_oil.drop('fecha', axis=1)
    
    # Asegurar que tenemos precio del petróleo
    if 'dcoilwtico' in df_oil.columns:
        df_oil['oil_price'] = pd.to_numeric(df_oil['dcoilwtico'], errors='coerce')
    elif 'oil_price' in df_oil.columns:
        df_oil['oil_price'] = pd.to_numeric(df_oil['oil_price'], errors='coerce')
    
    # Interpolar valores faltantes de petróleo
    df_oil['oil_price'] = df_oil['oil_price'].interpolate(method='linear')
    
    # Merge de los datos
    weekly_panel = df_data.merge(
        df_oil[['date', 'oil_price']],
        on='date',
        how='left'
    )
    
    print(f"Datos cargados y preparados: {len(weekly_panel)} filas")
    return weekly_panel

def main():
    print("Iniciando proceso...")
    
    # Cargar y preparar datos
    weekly_panel = load_data()
    weekly_panel['week'] = pd.to_datetime(weekly_panel['date'])
    print(f"Datos procesados: {len(weekly_panel)} filas")
    
    # Preparar características
    print("Preparando características...")
    data = prepare_features(weekly_panel)
    print(f"Características preparadas. Shape: {data.shape}")
    
    # Definir características para el modelo
    features = [
        'promo_rate', 'oil',
        'month_sin', 'month_cos', 'week_sin', 'week_cos',
        'sales_lag_1', 'sales_lag_2', 'sales_lag_3', 'sales_lag_4',
        'sales_lag_8', 'sales_lag_12', 'sales_lag_26', 'sales_lag_52',
        'sales_rolling_mean_4', 'sales_rolling_mean_8',
        'sales_rolling_mean_12', 'sales_rolling_mean_26',
        'sales_rolling_std_4', 'sales_rolling_std_8',
        'sales_rolling_std_12', 'sales_rolling_std_26',
        'store_mean_sales', 'store_std_sales', 'store_median_sales',
        'sales_to_mean_ratio'
    ]
    
    # Definir punto de corte para validación (últimas 12 semanas)
    valid_start = data['week'].max() - pd.Timedelta(weeks=12)
    
    # Resultados por tienda
    store_results = []
    
    for store in data['store'].unique():
        # Filtrar datos de la tienda
        store_data = data[data['store'] == store].copy()
        
        # Dividir en entrenamiento y validación
        train_data = store_data[store_data['week'] < valid_start]
        valid_data = store_data[store_data['week'] >= valid_start]
        
        # Entrenar modelo y obtener resultados
        if len(train_data) > 0 and len(valid_data) > 0:
            results = train_store_model(train_data, valid_data, features)
            store_results.append({
                'store_id': int(store),
                'rmse': float(results['rmse']),
                'r2': float(results['r2'])
            })
    
    # Calcular métricas globales
    rmse_values = [r['rmse'] for r in store_results]
    r2_values = [r['r2'] for r in store_results]
    
    metrics = {
        'metricas_globales': {
            'rmse_promedio': float(np.mean(rmse_values)),
            'rmse_mediana': float(np.median(rmse_values)),
            'rmse_min': float(np.min(rmse_values)),
            'rmse_max': float(np.max(rmse_values)),
            'r2_promedio': float(np.mean(r2_values)),
            'r2_mediana': float(np.median(r2_values)),
            'r2_min': float(np.min(r2_values)),
            'r2_max': float(np.max(r2_values))
        },
        'mejores_tiendas': sorted(store_results, key=lambda x: x['rmse'])[:5],
        'peores_tiendas': sorted(store_results, key=lambda x: x['rmse'], reverse=True)[:5]
    }
    
    # Guardar métricas
    with open('outputs/metricas_modelo_comparativo.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print("\nMétricas globales del modelo comparativo:")
    print(f"R² promedio: {metrics['metricas_globales']['r2_promedio']:.4f}")
    print(f"R² mediana: {metrics['metricas_globales']['r2_mediana']:.4f}")
    print(f"RMSE promedio: {metrics['metricas_globales']['rmse_promedio']:.4f}")
    print(f"RMSE mediana: {metrics['metricas_globales']['rmse_mediana']:.4f}")

if __name__ == "__main__":
    main()
