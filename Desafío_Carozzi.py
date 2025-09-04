import os
import json
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings('ignore')



# Crear carpeta de salida 

os.makedirs("outputs", exist_ok=True)

def ensure_outputs():
    os.makedirs('outputs', exist_ok=True)


def log(msg: str):
    print(f'[INFO] {msg}')


def parse_date_series(s: pd.Series) -> pd.Series:
    d1 = pd.to_datetime(s, errors='coerce', infer_datetime_format=True)
    mask = d1.isna()
    if mask.any():
        d2 = pd.to_datetime(s[mask], errors='coerce', dayfirst=True)
        d1[mask] = d2
    return d1


def week_floor(dt_series: pd.Series) -> pd.Series:
    # Inicio de semana en lunes
    return dt_series.dt.to_period('W-MON').dt.start_time


# 1) Carga de datos

def load_data():
    # Rutas (puedes cambiar a minúsculas si tus archivos están como data.csv)
    data_path   = 'data/Data.csv'   if os.path.exists('data/Data.csv')   else 'data/data.csv'
    stores_path = 'data/Stores.csv' if os.path.exists('data/Stores.csv') else 'data/stores.csv'
    oil_path    = 'data/Oil.csv'    if os.path.exists('data/Oil.csv')    else 'data/oil.csv'

    log(f'Cargando archivos:\n - {data_path}\n - {stores_path}\n - {oil_path}')

    # Carga base
    df_data   = pd.read_csv(data_path)
    df_stores = pd.read_csv(stores_path)

    # Detectar header en Oil (como en tu código)
    tmp = pd.read_csv(oil_path, header=None, dtype=str, engine='python')
    header_row = next(
        (i for i, r in tmp.iterrows()
         if ('date' in r.astype(str).str.lower().tolist())
         or ('fecha' in r.astype(str).str.lower().tolist())),
        None
    )
    df_oil = pd.read_csv(oil_path, header=header_row, engine='python') if header_row is not None else pd.read_csv(oil_path)

    # Normalización de nombres
    df_data.columns   = df_data.columns.str.strip().str.lower()
    df_stores.columns = df_stores.columns.str.strip().str.lower()
    df_oil.columns    = df_oil.columns.str.strip().str.lower().str.replace('fecha', 'date', regex=False)

    # Fechas
    df_data['date'] = parse_date_series(df_data['date'])
    df_oil['date']  = parse_date_series(df_oil['date'])
    df_oil = df_oil.dropna(subset=['date']).sort_values('date').reset_index(drop=True)

    # Interpolación de petróleo si existe dcoilwtico
    if 'dcoilwtico' in df_oil.columns:
        df_oil['dcoilwtico'] = pd.to_numeric(df_oil['dcoilwtico'], errors='coerce')
        df_oil['oil_price_interp'] = df_oil['dcoilwtico'].interpolate(method='linear', limit_direction='both')
    elif 'oil_price' in df_oil.columns:
        df_oil['oil_price'] = pd.to_numeric(df_oil['oil_price'], errors='coerce')
        df_oil['oil_price_interp'] = df_oil['oil_price'].interpolate(method='linear', limit_direction='both')

    return df_data, df_stores, df_oil

## EDA de datos

# Resumen de exploración


# Resumen simple
def resumen_simple(df, nombre):
    with open(f"outputs/{nombre}_resumen.txt", "w", encoding="utf-8") as f:
        f.write(f"=== {nombre} ===\n")
        df.info(buf=f)
        f.write("\nNulos por columna:\n")
        f.write(str(df.isnull().sum()))
        f.write("\n\nÚnicos por columna:\n")
        f.write(str(df.nunique()))

# Gráfico promedio anual
def plot_promedio_ventas_anual(df_data):
    df_data['sales'] = pd.to_numeric(df_data['sales'], errors='coerce')
    df_data['year'] = df_data['date'].dt.year
    ventas_anuales = df_data.groupby('year')['sales'].mean().reset_index()

    plt.figure(figsize=(10, 5))
    plt.bar(ventas_anuales['year'].astype(str), ventas_anuales['sales'], color='skyblue')
    plt.title('Promedio de Ventas por Año')
    plt.xlabel('Año')
    plt.ylabel('Promedio de Ventas')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('outputs/promedio_ventas_anual.png')
    plt.show()

    print(ventas_anuales)

# === EJECUCIÓN ===
df_data, df_stores, df_oil = load_data()

resumen_simple(df_data, "DATA")
resumen_simple(df_stores, "STORES")
resumen_simple(df_oil, "OIL")

plot_promedio_ventas_anual(df_data)
