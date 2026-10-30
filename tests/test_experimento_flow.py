import mlflow
import pytest
import yaml

from flows.experimento_flow import churn_training, entrenar_y_registrar
from scripts.generate_data import generar_datos

CONFIG = "configs/config.yaml"


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """Configuración del proyecto con MLflow en una base temporal, para no escribir en mlflow.db."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    with open(CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def datos():
    return generar_datos(filas=1000, proporcion_objetivo=0.2, semilla=42)


def test_entrenar_y_registrar_crea_un_run(cfg, datos):
    run_id, metricas = entrenar_y_registrar(datos, cfg)

    assert set(metricas) == {"auc", "accuracy", "precision", "recall", "f1"}
    assert all(0 <= valor <= 1 for valor in metricas.values())

    run = mlflow.get_run(run_id)
    assert run.data.params["modelo"] == cfg["modelo"]["tipo"]
    assert run.data.params["semilla"] == str(cfg["division"]["random_state"])
    assert run.data.metrics["auc"] == pytest.approx(metricas["auc"])


def test_modelo_no_soportado_falla(cfg, datos):
    cfg["modelo"]["tipo"] = "xgboost"
    with pytest.raises(ValueError, match="no soportado"):
        entrenar_y_registrar(datos, cfg)


def test_el_flow_completo_termina_y_registra_el_run(cfg):
    run_id = churn_training()

    assert mlflow.get_run(run_id).info.status == "FINISHED"
