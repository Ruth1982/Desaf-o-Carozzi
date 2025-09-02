import pandas as pd
import matplotlib.pyplot as plt

 #Rutas 
data_path   = "data/Data.csv"
stores_path = "data/Stores.csv"
oil_path    = "data/Oil.csv"

#  Carga 
df_data   = pd.read_csv(data_path)
df_stores = pd.read_csv(stores_path)

# Detectar header en Oil 
tmp = pd.read_csv(oil_path, header=None, dtype=str, engine="python")
header_row = next(
    (i for i, r in tmp.iterrows()
     if ("date" in r.astype(str).str.lower().tolist())
     or ("fecha" in r.astype(str).str.lower().tolist())),
    None
)
df_oil = pd.read_csv(oil_path, header=header_row, engine="python") if header_row is not None else pd.read_csv(oil_path)

# Normalizar nombres
df_data.columns   = df_data.columns.str.strip().str.lower()
df_stores.columns = df_stores.columns.str.strip().str.lower()
df_oil.columns    = df_oil.columns.str.strip().str.lower().str.replace("fecha", "date", regex=False)

# Fechas
def parse_date_series(s):
    d1 = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
    mask = d1.isna()
    if mask.any():
        d2 = pd.to_datetime(s[mask], errors="coerce", dayfirst=True)
        d1[mask] = d2
    return d1

df_data["date"] = parse_date_series(df_data["date"])
df_oil["date"]  = parse_date_series(df_oil["date"])
df_oil = df_oil.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


# Interpolación petróleo si existe
if "dcoilwtico" in df_oil.columns:
    df_oil["dcoilwtico"] = pd.to_numeric(df_oil["dcoilwtico"], errors="coerce")
    df_oil["dcoilwtico_interp"] = df_oil["dcoilwtico"].interpolate(method="linear", limit_direction="both")

# Resumen simple
def resumen_simple(df, nombre):
    print(f"\n=== {nombre} ===")
    df.info()
    print("\nNulos por columna:\n", df.isnull().sum())
    print("\nÚnicos por columna:\n", df.nunique())

resumen_simple(df_data, "DATA")
resumen_simple(df_stores, "STORES")
resumen_simple(df_oil, "OIL")




