
"""
Exploración y limpieza inicial - Desafío Data Scientist (Carozzi)
Autor: <tu_nombre>
Requisitos: pandas, numpy, matplotlib
Entradas esperadas en data: data.csv, stores.csv, oil.csv
Salidas: data/eda/* (resúmenes CSV y gráficos PNG)
"""

import os
import json
import math
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# -------------------------
# Configuración y rutas
# -------------------------
DATA_DIR = "data"
OUT_DIR = os.path.join(DATA_DIR, "eda")
os.makedirs(OUT_DIR, exist_ok=True)

DATA_PATH = os.path.join(DATA_DIR, "data.csv")
STORES_PATH = os.path.join(DATA_DIR, "stores.csv")
OIL_PATH = os.path.join(DATA_DIR, "oil.csv")

# -------------------------
# 1) Carga de datos
# -------------------------
df = pd.read_csv(DATA_PATH)
stores = pd.read_csv(STORES_PATH)
oil = pd.read_csv(OIL_PATH)

# Guardar tamaños
meta = {
    "data_shape": df.shape,
    "stores_shape": stores.shape,
    "oil_shape": oil.shape
}

with open(os.path.join(OUT_DIR, "00_shapes.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

# -------------------------
# 2) Tipos & conversión de fechas
# -------------------------
for col in ["date"]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    if col in oil.columns:
        oil[col] = pd.to_datetime(oil[col], errors="coerce")

# -------------------------
# 3) Resumen rápido de columnas
# -------------------------
def quick_summary(df_in: pd.DataFrame) -> pd.DataFrame:
    summary = []
    for c in df_in.columns:
        s = df_in[c]
        summary.append({
            "col": c,
            "dtype": str(s.dtype),
            "n_null": int(s.isna().sum()),
            "pct_null": float((s.isna().mean()*100)),
            "n_unique": int(s.nunique(dropna=True)),
            "sample_values": ", ".join(map(str, s.dropna().astype(str).head(5).tolist()))
        })
    return pd.DataFrame(summary).sort_values("pct_null", ascending=False)

quick_summary(df).to_csv(os.path.join(OUT_DIR, "01_summary_data.csv"), index=False)
quick_summary(stores).to_csv(os.path.join(OUT_DIR, "01_summary_stores.csv"), index=False)
quick_summary(oil).to_csv(os.path.join(OUT_DIR, "01_summary_oil.csv"), index=False)

# -------------------------
# 4) Rango de fechas y checks básicos
# -------------------------
def date_range_info(s: pd.Series) -> dict:
    s = pd.to_datetime(s, errors="coerce")
    return {
        "min_date": str(s.min()),
        "max_date": str(s.max()),
        "n_days": int((s.max() - s.min()).days) if s.notna().any() else None
    }

date_meta = {
    "data_date_range": date_range_info(df["date"]) if "date" in df.columns else None,
    "oil_date_range": date_range_info(oil["date"]) if "date" in oil.columns else None
}
with open(os.path.join(OUT_DIR, "02_date_ranges.json"), "w", encoding="utf-8") as f:
    json.dump(date_meta, f, ensure_ascii=False, indent=2)

# Duplicados
dup_id = int(df.duplicated(subset=["id"]).sum()) if "id" in df.columns else None
dup_all = int(df.duplicated().sum())
dup_info = {"dup_by_id": dup_id, "dup_full_rows": dup_all}
with open(os.path.join(OUT_DIR, "03_duplicates.json"), "w", encoding="utf-8") as f:
    json.dump(dup_info, f, ensure_ascii=False, indent=2)

# -------------------------
# 5) Limpieza mínima
# -------------------------
# Imputación simple del petróleo: interpolación por tiempo
if "dcoilwtico" in oil.columns:
    oil = oil.sort_values("date").reset_index(drop=True)
    # Interpolar valores faltantes linealmente
    oil["dcoilwtico"] = oil["dcoilwtico"].astype(float)
    oil["dcoilwtico_interp"] = oil["dcoilwtico"].interpolate(method="linear", limit_direction="both")

# -------------------------
# 6) Integración de datasets
# -------------------------
# Merge con stores (características estáticas)
df_merged = df.merge(stores, on="store_nbr", how="left")

# Merge con oil por fecha
if "date" in df_merged.columns and "date" in oil.columns:
    df_merged = df_merged.merge(oil[["date", "dcoilwtico_interp"]], on="date", how="left")

# -------------------------
# 7) Ingeniería de features básicas de tiempo
# -------------------------
if "date" in df_merged.columns:
    df_merged["year"] = df_merged["date"].dt.year
    df_merged["month"] = df_merged["date"].dt.month
    df_merged["day"] = df_merged["date"].dt.day
    df_merged["dow"] = df_merged["date"].dt.dayofweek  # 0=lunes
    df_merged["week"] = df_merged["date"].dt.isocalendar().week.astype(int)
    df_merged["year_week"] = df_merged["date"].dt.strftime("%G-W%V")

# Señales de promoción
if "onpromotion" in df_merged.columns:
    # Algunas tiendas/categorías pueden tener onpromotion=0/NaN; asegurar numérico
    df_merged["onpromotion"] = pd.to_numeric(df_merged["onpromotion"], errors="coerce").fillna(0).astype(int)

# Asegurar columna de ventas numérica
if "sales" in df_merged.columns:
    df_merged["sales"] = pd.to_numeric(df_merged["sales"], errors="coerce")

# Guardar versión ligera (sin texto largo) para análisis
cols_keep = [c for c in df_merged.columns if c not in ["city", "state"]]
df_merged[cols_keep].to_csv(os.path.join(OUT_DIR, "04_df_merged_lite.csv"), index=False)

# -------------------------
# 8) Resúmenes EDA
# -------------------------
# Estadísticas de ventas
sales_desc = df_merged["sales"].describe(percentiles=[0.01, 0.05, 0.95, 0.99]).to_frame(name="sales_stats")
sales_desc.to_csv(os.path.join(OUT_DIR, "05_sales_describe.csv"))

# Ventas por tienda
sales_by_store = df_merged.groupby("store_nbr")["sales"].sum().reset_index().sort_values("sales", ascending=False)
sales_by_store.to_csv(os.path.join(OUT_DIR, "06_sales_by_store.csv"), index=False)

# Ventas por familia
if "family" in df_merged.columns:
    sales_by_family = df_merged.groupby("family")["sales"].sum().reset_index().sort_values("sales", ascending=False)
    sales_by_family.to_csv(os.path.join(OUT_DIR, "07_sales_by_family.csv"), index=False)

# Promedio promociones por tienda
prom_by_store = df_merged.groupby("store_nbr")["onpromotion"].mean().reset_index().rename(columns={"onpromotion":"mean_onpromotion"})
prom_by_store.to_csv(os.path.join(OUT_DIR, "08_promotions_by_store.csv"), index=False)

# Agregación semanal (para modelado futuro)
weekly = (
    df_merged
    .groupby(["store_nbr", "year", "week"], as_index=False)
    .agg(
        weekly_sales=("sales", "sum"),
        weekly_promo=("onpromotion", "sum"),
        oil_mean=("dcoilwtico_interp", "mean")
    )
)
# Año-semana legible
weekly["year_week"] = weekly["year"].astype(str) + "-W" + weekly["week"].astype(str).str.zfill(2)
weekly.to_csv(os.path.join(OUT_DIR, "09_weekly_store.csv"), index=False)

# Correlación simple a nivel global (ventas totales vs petróleo semanal)
weekly_global = weekly.groupby(["year", "week"], as_index=False).agg(
    total_sales=("weekly_sales", "sum"),
    oil_mean=("oil_mean", "mean")
)
corr_val = weekly_global[["total_sales", "oil_mean"]].corr().iloc[0,1]
with open(os.path.join(OUT_DIR, "10_corr_sales_oil.json"), "w", encoding="utf-8") as f:
    json.dump({"pearson_corr_total_sales_oil": float(corr_val)}, f, ensure_ascii=False, indent=2)

# Outliers por IQR (a nivel diario)
q1, q3 = df_merged["sales"].quantile(0.25), df_merged["sales"].quantile(0.75)
iqr = q3 - q1
low_bound = q1 - 1.5*iqr
high_bound = q3 + 1.5*iqr
outlier_share = float(((df_merged["sales"] < low_bound) | (df_merged["sales"] > high_bound)).mean())
with open(os.path.join(OUT_DIR, "11_outliers_sales.json"), "w", encoding="utf-8") as f:
    json.dump({
        "q1": float(q1), "q3": float(q3), "iqr": float(iqr),
        "low_bound": float(low_bound), "high_bound": float(high_bound),
        "outlier_share": outlier_share
    }, f, ensure_ascii=False, indent=2)

# -------------------------
# 9) Gráficos principales (Matplotlib, sin estilos ni colores específicos)
# -------------------------
def save_lineplot(x, y, title, xlabel, ylabel, path):
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def save_scatter(x, y, title, xlabel, ylabel, path):
    plt.figure()
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# 9.1 Serie total de ventas por día
daily_total = df_merged.groupby("date", as_index=False)["sales"].sum()
if not daily_total.empty:
    save_lineplot(
        daily_total["date"], daily_total["sales"],
        "Ventas totales diarias",
        "Fecha", "Ventas",
        os.path.join(OUT_DIR, "plot_01_total_daily_sales.png")
    )

# 9.2 Precio del petróleo (interpolado)
oil_plot = oil[["date", "dcoilwtico_interp"]].dropna()
if not oil_plot.empty:
    save_lineplot(
        oil_plot["date"], oil_plot["dcoilwtico_interp"],
        "Precio del petróleo (interpolado)",
        "Fecha", "WTI",
        os.path.join(OUT_DIR, "plot_02_oil_wti.png")
    )

# 9.3 Ventas semanales de una tienda ejemplo (store 1 o la top)
store_top = int(sales_by_store.iloc[0]["store_nbr"]) if not sales_by_store.empty else 1
wk_store = weekly[weekly["store_nbr"] == store_top].sort_values(["year", "week"])
if not wk_store.empty:
    save_lineplot(
        range(len(wk_store)),
        wk_store["weekly_sales"],
        f"Ventas semanales - Tienda {store_top}",
        "Semana (índice)", "Ventas semanales",
        os.path.join(OUT_DIR, "plot_03_weekly_sales_top_store.png")
    )

# 9.4 Dispersión ventas totales semanales vs petróleo
if len(weekly_global) > 5:
    save_scatter(
        weekly_global["oil_mean"], weekly_global["total_sales"],
        "Ventas semanales totales vs. precio petróleo",
        "WTI semanal promedio", "Ventas semanales totales",
        os.path.join(OUT_DIR, "plot_04_scatter_sales_vs_oil.png")
    )

# 9.5 Promociones vs ventas (global semanal)
weekly_global2 = weekly.groupby(["year", "week"], as_index=False).agg(
    total_sales=("weekly_sales", "sum"),
    total_promo=("weekly_promo", "sum")
)
if len(weekly_global2) > 5:
    save_scatter(
        weekly_global2["total_promo"], weekly_global2["total_sales"],
        "Promociones vs. ventas (agregado semanal)",
        "Total promos semana", "Ventas semanales totales",
        os.path.join(OUT_DIR, "plot_05_scatter_promo_vs_sales.png")
    )

# -------------------------
# 10) Guardar una muestra pequeña del dataset integrado
# -------------------------
df_merged.sample(min(50000, len(df_merged)), random_state=7).to_csv(
    os.path.join(OUT_DIR, "12_sample_50k_df_merged.csv"),
    index=False
)

print("EDA terminada. Archivos guardados en:", OUT_DIR)
