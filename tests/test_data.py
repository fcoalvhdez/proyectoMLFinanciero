import pandas as pd
import pytest

from scripts.load_data import COLUMNAS, cargar_datos

# Las pruebas se ejecutan desde la raíz del repositorio (ver README).
CSV = "data/Customer-Churn-Records.csv"
IDENTIFICADORES = ["RowNumber", "CustomerId", "Surname"]
OBJETIVO = "Exited"


def test_carga_el_csv_con_las_columnas_esperadas():
    df = cargar_datos(CSV)
    assert list(df.columns) == COLUMNAS
    assert len(df) == 10000


def test_sin_nulos_y_objetivo_binario():
    df = cargar_datos(CSV)
    assert df.isna().sum().sum() == 0
    assert set(df[OBJETIVO].unique()) <= {0, 1}


def test_no_incluye_identificadores():
    df = cargar_datos(CSV)
    assert not set(IDENTIFICADORES) & set(df.columns)


def test_sin_ruta_lee_el_csv_de_la_configuracion():
    df = cargar_datos()
    assert len(df) == 10000


def test_columna_faltante_falla(tmp_path):
    ruta = tmp_path / "incompleto.csv"
    pd.DataFrame({"Age": [1], "Exited": [0]}).to_csv(ruta, index=False)
    with pytest.raises(ValueError):
        cargar_datos(ruta)
