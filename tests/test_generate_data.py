import pandas as pd
import pytest

from scripts.generate_data import generar_datos
from scripts.load_data import COLUMNAS, cargar_datos


def test_tiene_todas_las_columnas_del_dataset_real():
    df = generar_datos(filas=100, proporcion_objetivo=0.2, semilla=1)
    assert set(COLUMNAS) <= set(df.columns)
    assert df.shape[1] == 18


def test_numero_de_filas():
    df = generar_datos(filas=500, proporcion_objetivo=0.2, semilla=1)
    assert len(df) == 500


def test_proporcion_de_la_clase_objetivo_es_exacta():
    df = generar_datos(filas=1000, proporcion_objetivo=0.3, semilla=1)
    assert df["Exited"].sum() == 300


def test_sin_nulos_y_valores_dentro_de_rango():
    df = generar_datos(filas=1000, proporcion_objetivo=0.2, semilla=1)
    assert df.isna().sum().sum() == 0
    assert df["Age"].between(18, 92).all()
    assert set(df["Exited"].unique()) <= {0, 1}


def test_misma_semilla_da_el_mismo_resultado():
    a = generar_datos(filas=300, proporcion_objetivo=0.2, semilla=7)
    b = generar_datos(filas=300, proporcion_objetivo=0.2, semilla=7)
    pd.testing.assert_frame_equal(a, b)


def test_otra_semilla_da_otro_resultado():
    a = generar_datos(filas=300, proporcion_objetivo=0.2, semilla=7)
    b = generar_datos(filas=300, proporcion_objetivo=0.2, semilla=8)
    assert not a.equals(b)


@pytest.mark.parametrize("filas", [0, 1])
def test_pocas_filas_falla(filas):
    with pytest.raises(ValueError):
        generar_datos(filas=filas, proporcion_objetivo=0.2, semilla=1)


@pytest.mark.parametrize("proporcion", [0, 1, -0.1, 1.5])
def test_proporcion_fuera_de_rango_falla(proporcion):
    with pytest.raises(ValueError):
        generar_datos(filas=100, proporcion_objetivo=proporcion, semilla=1)


def test_el_csv_generado_pasa_el_cargador(tmp_path):
    ruta = tmp_path / "sintetico.csv"
    generar_datos(filas=200, proporcion_objetivo=0.2, semilla=1).to_csv(ruta, index=False)
    df = cargar_datos(ruta)
    assert list(df.columns) == COLUMNAS
    assert len(df) == 200
