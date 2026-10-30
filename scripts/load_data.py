"""Carga el dataset de churn, valida sus columnas y muestra su tamaño.

Por defecto lee `datos.ruta` de configs/config.yaml. Las rutas se resuelven desde la raíz
del repositorio, así que ejecuta el script desde allí.

Ejecución:
    poetry run python scripts/load_data.py

Para usarla desde un flow:
    from scripts.load_data import cargar_datos
"""
import pandas as pd
import yaml

RUTA_CONFIG = "configs/config.yaml"

# Columnas del dataset de churn: las variables predictivas y el objetivo `Exited`.
# Los identificadores (RowNumber, CustomerId, Surname) no se cargan.
COLUMNAS = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Complain",
    "Satisfaction Score",
    "Point Earned",
    "Geography",
    "Gender",
    "Card Type",
    "Exited",
]


def cargar_datos(ruta=None):
    """Devuelve el DataFrame de churn con las columnas esperadas. Falla si falta alguna o hay nulos."""
    if ruta is None:
        with open(RUTA_CONFIG, encoding="utf-8") as f:
            ruta = yaml.safe_load(f)["datos"]["ruta"]

    df = pd.read_csv(ruta)
    faltan = [c for c in COLUMNAS if c not in df.columns]
    if faltan:
        raise ValueError(f"Al CSV le faltan columnas: {faltan}")
    df = df[COLUMNAS].copy()
    if df.isna().any().any():
        raise ValueError("El CSV tiene valores nulos en las columnas del modelo")
    return df


def main():
    df = cargar_datos()
    print(f"Filas: {len(df)}  columnas: {df.shape[1]}  nulos: {int(df.isna().sum().sum())}")
    print(f"Proporción de Exited = 1: {df['Exited'].mean():.3f}")


if __name__ == "__main__":
    main()
