# Proyecto MLOps: predicción de churn bancario

Proyecto del curso *MLOps en Python – De la práctica a la producción* (Universidad Santo Tomás). El grupo construye, a lo largo del curso, un modelo que predice si un cliente de un banco abandonará la entidad (`Exited = 1`), con trazabilidad completa en **MLflow** y orquestación con **Prefect**.

- **Actividad 1** (este estado del repositorio): estructura, entorno reproducible con Poetry, carga y validación de datos, servicios validados y primer experimento registrado.
- Memoria de la actividad: [docs/memoria-actividad-1.md](docs/memoria-actividad-1.md).

## Caso de uso y datos

Dataset público de Kaggle *Bank Customer Churn* (`data/Customer-Churn-Records.csv`, 10 000 clientes, 18 columnas, sin nulos, ~20 % de churn). Se versiona en Git porque es pequeño (≈ 820 KB). Fuente: https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn (copiado del repositorio del curso `actividades/data/`).

Como alternativa sin el CSV real, `scripts/generate_data.py` genera un dataset sintético con el mismo esquema y semilla fija.

## Estructura

```
proyectoMLFinanciero/
├── configs/config.yaml        # datos, variables, división, modelo, MLflow y umbral
├── data/                      # dataset de churn (y el sintético, que no se versiona)
├── flows/
│   ├── prueba_flow.py         # flow mínimo: comprueba que Prefect y MLflow responden
│   └── experimento_flow.py    # cargar datos → entrenar y registrar en MLflow → resumen
├── scripts/
│   ├── load_data.py           # carga el CSV y valida columnas y nulos
│   └── generate_data.py       # dataset sintético reproducible (semilla fija)
├── tests/                     # pruebas con pytest (datos, generador y flow)
├── notebooks/                 # exploración (próximas actividades)
├── services/                  # MLflow y Prefect: Docker Compose (opción A) o sin Docker (opción B)
├── docs/                      # memorias de actividad y evidencias
├── pyproject.toml             # dependencias declaradas con versión fijada (==)
└── poetry.lock                # versiones exactas resueltas (no editar a mano)
```

Cada carpeta lleva un `__init__.py` para que Git la versione aunque esté vacía.

## Requisitos

| Herramienta | Versión | Comprobación |
|---|---|---|
| Python | 3.12 a 3.14 | `python --version` |
| Poetry | 2.x | `poetry --version` |
| Git | reciente | `git --version` |
| Docker Desktop (opcional) | reciente | `docker --version` |

## 1. Preparar el entorno

```powershell
git clone https://github.com/fcoalvhdez/proyectoMLFinanciero.git
cd proyectoMLFinanciero
poetry env use 3.12
poetry install
poetry run python -c "import prefect, mlflow, sklearn; print('ok')"
poetry run pytest -q
```

`pyproject.toml` declara lo que se necesita y `poetry.lock` registra lo que se instaló; ambos se versionan. Si el lock no coincide con `pyproject.toml`, solo quien cambió las dependencias ejecuta `poetry lock`.

## 2. Levantar MLflow y Prefect

Ver [services/README.md](services/README.md). Usa una sola opción:

- **Opción A (Docker):** `cd services; docker compose up -d`
- **Opción B (sin Docker)**, en dos terminales desde `services/`:

  ```powershell
  poetry run mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow_store/mlflow.db --artifacts-destination mlflow_store/artifacts --serve-artifacts
  ```

  ```powershell
  poetry run prefect server start --host 127.0.0.1 --port 4200
  ```

UIs: MLflow en http://localhost:5000 y Prefect en http://localhost:4200.

## 3. Ejecutar los flows

Desde la raíz del repositorio, con las variables definidas en la misma terminal:

```powershell
$env:PREFECT_API_URL = "http://localhost:4200/api"
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
poetry run python flows/prueba_flow.py        # run "prueba" + experimento "prueba-validacion"
poetry run python scripts/load_data.py        # valida el dataset y muestra su tamaño
poetry run python flows/experimento_flow.py   # run "experimento-a1" + experimento "churn-training-mlops"
```

Sin `PREFECT_API_URL`, Prefect crea un servidor temporal y el run no aparece en la UI.

## Configuración

Todo lo que puede cambiar vive en `configs/config.yaml` (ruta de datos, variables y columnas excluidas, `test_size` y semilla, tipo de modelo y parámetros, experimento de MLflow y umbral de decisión). `MLFLOW_TRACKING_URI`, si existe, tiene prioridad sobre `mlflow.tracking_uri`.

La columna `Complain` se excluye del modelo: coincide con `Exited` en ~99,7 % de las filas (fuga de datos) y da un AUC irreal de 0,999.
