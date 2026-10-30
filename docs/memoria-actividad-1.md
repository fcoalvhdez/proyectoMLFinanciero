# Memoria – Actividad 1: Fundamentos, versionamiento y tracking

**Curso:** MLOps en Python – De la práctica a la producción (20262_01_11117_18025_1B)
**Integrantes:** _[Nombre 1 – usuario GitHub]_, _[Nombre 2 – usuario GitHub]_, _[Nombre 3 – usuario GitHub]_
**Repositorio:** https://github.com/fcoalvhdez/proyectoMLFinanciero
**Fecha:** 29 de octubre de 2026

## 1. Objetivo

Construir la base reproducible del proyecto del grupo: un repositorio Git con estructura profesional, un entorno de Python controlado con Poetry (`pyproject.toml` + `poetry.lock`), la carga y validación de datos versionables y un primer experimento registrado con trazabilidad en MLflow y orquestado con Prefect.

Es la primera actividad del curso, así que no depende de actividades anteriores; sí sienta la base de las siguientes: la actividad 2 (pipeline y empaquetado) partirá de esta estructura, del `config.yaml` y de las tareas de Prefect ya separadas (cargar datos → entrenar/registrar → resumen).

**Caso de uso elegido:** predicción de abandono (*churn*) de clientes bancarios, una clasificación binaria con el dataset público *Bank Customer Churn* de Kaggle (10 000 clientes, objetivo `Exited`). Para un banco, identificar a tiempo a los clientes con riesgo de irse permite lanzar acciones de retención, que suelen ser más baratas que captar clientes nuevos.

## 2. Descripción del trabajo

1. **Herramientas.** Se instaló Python 3.12.10 (el equipo tenía 3.10, fuera del rango 3.12–3.14 que exige Prefect 3.8.8) y Poetry 2.5.1.
2. **Repositorio y estructura.** Se creó el repositorio con la estructura de abajo. Cada carpeta tiene un `__init__.py`, porque Git no versiona carpetas vacías. Al `.gitignore` se añadieron `mlruns/`, `mlflow.db`, `mlflow_store/` y `.venv/`.
3. **Entorno.** `poetry init --no-interaction --name mi-proyecto-mlops --python ">=3.12,<3.15"`, `poetry env use` con Python 3.12 y `poetry add` de las dependencias con versión fijada (`prefect==3.8.8`, `mlflow==3.17.0`, `scikit-learn==1.9.1`, `pandas==3.0.6`, `numpy==2.5.4`, `pyyaml==6.0.3`, `fastapi==0.143.0`; dev: `pytest==9.1.1`). Se usó `package-mode = false` porque el proyecto no es una librería.
4. **Datos (paso 2).** `scripts/load_data.py` lee el CSV indicado en `configs/config.yaml`, comprueba que existan las 15 columnas esperadas y que no haya nulos, y descarta los identificadores (`RowNumber`, `CustomerId`, `Surname`). `scripts/generate_data.py` genera un dataset sintético con el mismo esquema y semilla fija, como alternativa sin el CSV real.
5. **Servicios (paso 3).** MLflow (puerto 5000) y Prefect (puerto 4200) se levantaron con la opción B (sin Docker) y se validaron con `flows/prueba_flow.py`.
6. **Pipeline de entrenamiento (paso 4, opcional).** `flows/experimento_flow.py` (flow `experimento-a1`), con tres tareas:
   - `cargar_datos`: usa `scripts/load_data.py` y registra en el log el tamaño del dataset.
   - `entrenar_y_registrar`: división estratificada 80/20 con semilla 42; `Pipeline` de sklearn (StandardScaler + OneHotEncoder + estimador); métricas AUC, accuracy, precision, recall y F1; registro en MLflow de parámetros, métricas, el `config.yaml` como artefacto y el modelo con firma y ejemplo de entrada.
   - Resumen: artefacto Markdown **Experimento A1** en Prefect con el modelo, las filas, las métricas y el ID del run de MLflow.
7. **Pruebas.** 21 pruebas con pytest (carga y validación de datos, generador sintético, entrenamiento, registro en MLflow con una base temporal y ejecución completa del flow).

Estructura del repositorio:

```
proyectoMLFinanciero/
├── configs/config.yaml        # parámetros: datos, variables, división, modelo, MLflow, umbral
├── data/Customer-Churn-Records.csv
├── flows/prueba_flow.py       # validación de Prefect + MLflow
├── flows/experimento_flow.py  # flow experimento-a1
├── scripts/load_data.py, generate_data.py
├── tests/                     # test_data, test_generate_data, test_experimento_flow
├── notebooks/                 # (próximas actividades)
├── services/                  # docker-compose.yml (opción A), requirements.txt (opción B)
├── docs/                      # esta memoria y evidencias
├── pyproject.toml, poetry.lock, README.md, .gitignore
```

## 3. Decisiones técnicas

| Decisión | Alternativas consideradas | Justificación |
|---|---|---|
| **Poetry** para el entorno | `requirements.txt` + venv; Conda | Poetry separa lo declarado (`pyproject.toml`) de lo resuelto (`poetry.lock`, con hashes y dependencias transitivas). Con `requirements.txt` habría que congelar a mano las transitivas; Conda es más pesado y el curso estandariza Poetry. |
| **Versiones fijadas con `==`** | Rangos (`^`, `>=`) | Cualquier integrante obtiene exactamente el mismo entorno; evita roturas por versiones nuevas (ver problema 2: FastAPI). |
| **Dataset real de Kaggle** (churn bancario) | Datos sintéticos; problema propio | Disponible de inmediato, sin datos sensibles, pequeño (820 KB) para versionarlo en Git y alineado con las actividades 2–4 del curso. El generador sintético queda como respaldo. |
| **Versionar el CSV en Git** | DVC; descarga documentada | Con 820 KB no compensa añadir DVC todavía; se reevaluará si el dataset crece. |
| **Excluir `Complain`** del modelo | Mantenerla (como en el ejemplo del curso) | `Complain` coincide con `Exited` en ~99,9 % (9 986 de 10 000) de las filas: es una fuga de datos (la queja suele registrarse junto con la baja). Con ella el AUC es 0,999, un valor irreal. Se excluye desde `config.yaml` (`variables.excluir`) y se registra como parámetro `excluidas` en MLflow. |
| **`class_weight: balanced`** | Sin ponderación; sobremuestreo (SMOTE) | La clase positiva es ~20 %. La ponderación mejora el recall (detectar a quienes se van) sin añadir dependencias. |
| **Configuración en YAML** | Constantes en el código; argumentos de línea de comandos | Cambiar datos, modelo o umbral no exige tocar código, y el YAML se guarda como artefacto en cada run (auditoría). |
| **Servicios sin Docker (opción B)** | Docker Compose (opción A) | Docker Desktop no estaba instalado y requiere WSL2 y reinicios. La opción B usa las mismas versiones (MLflow 3.17.0, Prefect 3.8.8) y la misma base `mlflow_store/mlflow.db`, así que es intercambiable con la A. Se ejecutó con el entorno de Poetry del proyecto, que ya trae ambas versiones. |
| **`package-mode = false`** + raíz del repo en `sys.path` | Empaquetar `scripts/` como paquete | El proyecto no es una librería; así se evita el error de instalación que menciona la guía. |

## 4. Evidencias

Capturas (en `docs/evidencias/`):

- [ ] `prefect_ui.png`: UI de Prefect en http://localhost:4200.
- [ ] `prefect_run_prueba.png`: run del flow `prueba` en **Completed**, con el log `Suma de 1 a 5 = 15`.
- [ ] `mlflow_prueba_validacion.png`: experimento `prueba-validacion` con el run `prueba-mlflow` (parámetro `n = 5`, métrica `suma = 15`).
- [ ] `prefect_run_experimento.png`: run del flow `experimento-a1` en **Completed**, con sus tareas, logs y el artefacto **Experimento A1**.
- [ ] `mlflow_churn_training.png`: experimento `churn-training-mlops` con los runs, parámetros, métricas y el modelo registrado.
- [ ] `github_colaboradores.png`: *Settings → Collaborators* con los tres integrantes.

Salidas de terminal guardadas:

- `docs/evidencias/salida_prueba_flow.txt`: flow run `impetuous-chicken` → `Suma de 1 a 5 = 15` → `Finished in state Completed()`; run de MLflow `43fade8059dc4dd4b00041e88652c9f4` en el experimento `prueba-validacion`.
- `docs/evidencias/salida_load_data.txt`: `Filas: 10000  columnas: 15  nulos: 0`, proporción de `Exited = 1` = 0,204.
- `docs/evidencias/salida_experimento_flow.txt`: flow run `alluring-dachshund` (`experimento-a1`) en Completed; run de MLflow `0d641acb5b684b7c8653e08bd3181672`.
- `docs/evidencias/resumen_runs_mlflow.txt`: tabla de los runs del experimento `churn-training-mlops`.
- `poetry run pytest -q` → **21 passed**.

Commits: _[enlace al commit "chore: estructura inicial y entorno con Poetry" y siguientes]_.

## 5. Resultados

Experimento `churn-training-mlops`. Conjunto de prueba: 2 000 clientes (20 %, estratificado, semilla 42), umbral 0,5.

| Run | Modelo | Columnas excluidas | AUC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| `churn-training` (flow) | Regresión logística (C=1, balanced) | `Complain` | 0,7798 | 0,7115 | 0,3887 | 0,7230 | 0,5056 |
| `gb-sin-complain` | Gradient Boosting (200 árboles, prof. 3) | `Complain` | **0,8763** | 0,8670 | 0,7610 | 0,5074 | 0,6088 |
| `lr-con-complain` | Regresión logística (C=1, balanced) | ninguna | 0,9988 | 0,9985 | 0,9975 | 0,9951 | 0,9963 |

Lectura de los resultados:

- **La fuga de datos es real:** solo con añadir `Complain`, la regresión logística pasa de AUC 0,78 a 0,999. Un modelo así no serviría en producción, porque en el momento de predecir todavía no existe la queja asociada a la baja.
- Sin la fuga, **Gradient Boosting** separa mejor las clases (AUC 0,876 frente a 0,780), porque capta relaciones no lineales (por ejemplo, edad × número de productos).
- La regresión logística ponderada tiene más **recall** (0,72 frente a 0,51): detecta más clientes que se van, a costa de más falsos positivos. El umbral y la ponderación se ajustarán en las próximas actividades según el costo de cada error.
- Todo lo anterior es reproducible: misma semilla, mismo `config.yaml` (guardado como artefacto de cada run) y mismo `poetry.lock`.

## 6. Problemas encontrados

1. **Versión de Python incompatible.** El equipo tenía Python 3.10 y Prefect 3.8.8 exige 3.12–3.14. *Solución:* se instaló Python 3.12.10 con `winget install Python.Python.3.12 --scope user` y se fijó con `poetry env use`.
2. **Error del servidor temporal de Prefect** (`TypeError: Body() got an unexpected keyword argument 'min_items'`) al ejecutar las pruebas del flow. *Causa:* Poetry resolvió FastAPI 0.162.0 (transitiva de MLflow/Prefect), que no es compatible con Prefect 3.8.8. *Solución:* fijar `fastapi==0.143.0`, la misma versión del lock del curso. Las 21 pruebas pasaron después. Este es un buen ejemplo de por qué conviene fijar también las dependencias críticas que entran como transitivas.
3. **Docker no disponible.** *Solución:* opción B de `services/README.md` (MLflow y Prefect como procesos locales).
4. **Runs adicionales en la UI de Prefect.** Las pruebas de pytest usan el modo temporal de Prefect, que comparte la base `~/.prefect` con `prefect server start`, así que sus runs (`gracious-buffalo`, `spotted-civet`) también aparecen en la UI. No afecta a MLflow, porque las pruebas usan una base SQLite temporal. *Mejora pendiente:* aislar `PREFECT_HOME` en las pruebas.
5. **Avisos de MLflow** sobre columnas enteras en la firma del modelo (pueden fallar si llegan nulos en inferencia). Se dejan documentados para la actividad de despliegue.

## 7. Conclusiones

- `pyproject.toml` declara qué se necesita y `poetry.lock` registra exactamente qué se instaló; con ambos versionados, otra persona clona el repositorio, ejecuta `poetry install` y reproduce el entorno sin volver a añadir dependencias. `poetry run` ejecuta comandos dentro del entorno sin activarlo.
- MLflow registra para cada run sus parámetros, métricas, configuración y modelo, así que comparar experimentos (y detectar la fuga de `Complain`) fue directo y auditable.
- Prefect separa el proceso en tareas observables, con estado, logs y artefactos, lo que facilita depurar y, más adelante, programar y desplegar.
- **Con más tiempo o recursos:** usar Docker Compose para igualar los servicios entre integrantes; validación del esquema con `pandera`; búsqueda de hiperparámetros y calibración del umbral según el costo de negocio; registrar el mejor modelo en el Model Registry; CI con GitHub Actions que ejecute `poetry install` y `pytest` en cada push; DVC si el dataset crece.

## 8. Uso de herramientas de IA

- **Herramientas utilizadas:** Claude Code (Anthropic), un asistente de programación en el editor/terminal, para:
  - leer la guía de la actividad y el repositorio del curso y proponer un plan de trabajo;
  - instalar Python 3.12 y Poetry, generar la estructura del proyecto, el `config.yaml`, la adaptación de `load_data.py`/`experimento_flow.py` (variables y exclusiones desde la configuración, validación de nulos, más parámetros registrados) y las pruebas, a partir de los ejemplos del curso;
  - diagnosticar el error de compatibilidad FastAPI/Prefect;
  - redactar el borrador del README y de esta memoria.
- **Qué partes se hicieron sin IA:** _[completar por el grupo: p. ej., creación del repositorio en GitHub e invitación de colaboradores, clonación y `poetry install` de cada integrante, capturas de pantalla, elección del caso de uso, revisión y discusión de los resultados]_.
- **Qué salidas de la IA se revisaron, corrigieron o descartaron:**
  - Se revisó la decisión de excluir `Complain` comparando los runs con y sin ella en MLflow, antes de adoptarla.
  - Se descartó instalar Docker Desktop por el tiempo disponible y se usó la opción B, que la guía permite.
  - El código copiado del curso se adaptó a la raíz del repositorio (rutas) y se comprobó con pytest (21 pruebas en verde) y con ejecuciones reales contra los servicios.
  - _[completar con la revisión hecha por cada integrante]_.
- **Contribución individual:** _[completar: qué hizo cada integrante]_.
