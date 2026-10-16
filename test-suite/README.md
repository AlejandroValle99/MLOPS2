# Demo en vivo: Stroke Risk Service

Paso a paso para levantar el sistema y mostrarlo funcionando: REST, gRPC, streaming, data
lake, el ciclo de vida del modelo y qué pasa cuando algo se cae. Los comandos se corren
desde la raíz del repositorio.

Se usan dos terminales, **T1** (comandos) y **T2** (la bandeja de un hospital), y
**Docker Desktop → Containers → mlops2**, donde cada servicio tiene su pestaña **Logs** y
sus botones **Stop / Start**.

## Mapa del sistema

| Servicio | Qué hace |
|---|---|
| `api` | REST para el clínico: valida el pedido y pide el score al model server |
| `model-server` | gRPC: tiene el modelo `champion` en memoria y lo recarga en caliente |
| `redpanda` | Broker compatible con Kafka: eventos de los hospitales y resultados |
| `stream-worker` | Valida cada evento, lo puntúa por lotes, publica el score y vigila el drift |
| `lake-sink` | Archiva predicciones y reportes de ventana en MinIO, en Parquet |
| `mlflow` | Experimentos y registro de modelos |
| `airflow` | Entrenamiento de candidatos y gate de promoción |
| `minio` | Almacenamiento S3: datos crudos, modelos y el lake |
| `postgres` | Metadatos de Airflow y MLflow |

## 0. Levantar todo

En **T1**:

```bash
docker compose --env-file .env.local --profile streaming up -d --build
```

Listo cuando Docker Desktop muestra todos los servicios en verde (los `*-init` terminan y
quedan detenidos, es normal). La primera vez tarda unos dos minutos: `model-init` entrena
el primer modelo.

| Interfaz | URL | Acceso |
|---|---|---|
| API (Swagger) | http://localhost:8000/docs | una clave de `API_KEYS` |
| MLflow | http://localhost:5001 | — |
| Airflow | http://localhost:8080 | `AIRFLOW_ADMIN_USER` / `AIRFLOW_ADMIN_PASSWORD` |
| MinIO | http://localhost:9001 | `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD` |

Todos los valores están en `.env.local`.

## 1. REST

En **Swagger**: **Authorize** → pegar la clave → `POST /v1/predict` → **Try it out** →
**Execute** con el paciente de ejemplo.

- **200:** probabilidad, `high_risk`, el umbral del modelo y su versión.
- Cambiar `age` a `-5` → **422**, nombrando el campo.
- `GET /v1/model`: el modelo en uso (versión, familia, umbral).

## 2. gRPC

La API no tiene el modelo: se lo pide al model server por gRPC. Hay un solo modelo en
memoria, así que REST y streaming puntúan siempre con la misma versión.

En **T1**: `grpcui -plaintext localhost:50051` (se abre el navegador). Elegir `Predict`,
pestaña **Raw Request (JSON)**, pegar el mismo paciente y **Invoke**:

```json
{"request_id": "demo-1", "patient": {"gender": "GENDER_MALE", "age": 67, "hypertension": 0,
 "heart_disease": 1, "ever_married": "EVER_MARRIED_YES", "work_type": "WORK_TYPE_PRIVATE",
 "residence_type": "RESIDENCE_TYPE_URBAN", "avg_glucose_level": 228.69, "bmi": 36.6,
 "smoking_status": "SMOKING_STATUS_FORMERLY_SMOKED"}}
```

**Qué se ve:** la misma probabilidad que devolvió REST. Con `"hypertension": 2` responde
`InvalidArgument`: el servidor valida igual que la API. Cerrar con Ctrl-C.

## 3. Streaming

Los sistemas de los hospitales publican un evento por cada paciente que ingresa. El worker
lo valida con el mismo contrato que REST, lo puntúa y publica el score; cada hospital lee
los suyos. Hay cuatro hospitales simulados (zona urbana/rural × mayores o menores de 60).

En **T2**, la bandeja de un hospital (sus pacientes de alto riesgo):

```bash
uv run --group stream python scripts/hospital_inbox.py rural-60plus
```

En **T1**, los hospitales empiezan a enviar eventos, en segundo plano, y siguen hasta el
final de la demo. Al minuto, `rural-60plus` cambia su población (glucosa × 1,2), y uno de
cada 500 eventos llega inválido:

```bash
uv run --group ml --group stream python -m services.producer.main --rate 100 \
  --drift-hospital rural-60plus --drift-from 6000 --factor 1.2 --bad-every 500 > /dev/null 2>&1 &
```

**Qué se ve:**

- **T2:** `follow up p-…: p=0.42 (threshold 0.10, model v6)`, a medida que llegan.
- **Logs de `stream-worker`:**
  - `window {…}` cada 300 eventos de un hospital: throughput, latencia, tasa de positivos
    y el PSI de cada variable contra la línea base de ese hospital.
  - `dead-lettered 1 of N events`: el evento inválido va a un topic aparte con el motivo,
    y el resto del lote se puntúa igual.
  - Cerca de los 2 minutos: `drift alert` para `rural-60plus` en `avg_glucose_level`
    (buscarlo con el buscador de los logs). La alerta exige PSI > 0,25 en dos ventanas
    seguidas, para que una ventana ruidosa sola no alcance.

## 4. Data lake

| Zona | Dónde | Lo escribe |
|---|---|---|
| Crudo | `datasets/`: el CSV y su `_MANIFEST.json` (checksum) | `minio-init` |
| Modelos | `mlflow-artifacts/` | MLflow, en cada entrenamiento |
| Salidas | `lake/outputs/predictions/` y `lake/outputs/window-reports/`, Parquet, una carpeta por día | `lake-sink` |

El split de entrenamiento no se guarda: se reconstruye del archivo crudo verificado, con
una semilla fija. El sink usa código boto3 que funciona igual contra AWS S3: solo cambian
el endpoint y las credenciales, que son variables de entorno.

En la **consola de MinIO**: `lake` → `outputs` → `predictions` → `date=…`. Cada archivo
cubre 1.000 mensajes y su nombre dice cuáles (`p0-<desde>-<hasta>.parquet`). El último se
reescribe cada 30 s mientras se llena.

Para ver qué hay adentro, en **T1**:

```bash
docker compose --env-file .env.local exec -T lake-sink python - < test-suite/ver_lake.py
```

**Qué se ve:** filas por día, las columnas de cada tabla y `columnas con datos del
paciente: ninguna`. El lake guarda scores y métricas, nunca los datos clínicos del
paciente.

El sink es un servicio aparte para que una falla de MinIO no frene los scores (paso 6) y
para que el worker no necesite credenciales de almacenamiento.

## 5. Ciclo de vida del modelo

Un modelo nuevo llega a producción solo si pasa el gate: se compara con el champion sobre
el mismo test congelado, con 2.000 remuestreos bootstrap, y se promueve solo si su F2 es
mejor en al menos el 90% de ellos.

**MLflow, lo que ya hay:**

- **Experiments → `stroke-risk`:** una corrida por candidato (`age_only`, `logreg`,
  `random_forest`, `xgboost`) con sus hiperparámetros (`hp_*`) y métricas. El finalista
  se elige por `oof_f2` (fuera de muestra, solo con datos de entrenamiento); `test_f2` es
  solo para el registro.
- **Models → `stroke-model`:** las versiones y sus alias. La que tiene `champion` es la
  que se sirve. **Version 10** → tags: `gate_decision = rejected`, `gate_p_better =
  0.0085`: no le ganaba al champion de forma consistente, así que no lo reemplazó.

**Airflow, forzar una promoción:** `etl_train_models_process_taskflow` → **▶ Trigger
DAG** → `n_trials = 5`, `gate_min_p_better = 0` → **Trigger**. En **Graph**: `verify_raw`
→ los cuatro `train_<familia>` en paralelo → `register_and_gate` (1–2 min). Con el umbral
en 0 el gate promueve igual, y queda registrado como decisión del gate.

**Qué se ve, sin reiniciar nada:**

- **Logs de `model-server`**, en menos de 10 s: `now serving stroke-model version N`.
- **T2:** los pacientes pasan a `model vN`.
- **MLflow → Models:** la versión nueva tiene `champion` y sus tags `gate_*`; la anterior
  quedó como `previous_champion`.

**Rollback, en MLflow:** abrir la versión con `previous_champion` → **Aliases** → editar:
quitar `previous_champion`, agregar `champion` → **Save**. En menos de 10 s el model server
vuelve a esa versión, con su propio umbral, y T2 lo muestra. Es lo mismo que hace
`scripts/rollback.py`.

## 6. Fallas

Con los eventos todavía corriendo, en Docker Desktop: **Stop** de un servicio, mirar, y
**Start**.

| Servicio | Mientras está caído | Al volver |
|---|---|---|
| `model-server` | T2 se detiene. Swagger `GET /ready` → 503. Logs de `stream-worker`: `model server call failed (UNAVAILABLE); retrying the same events` | Carga el modelo (~30 s) y el worker puntúa los eventos que esperaban en el broker: no se pierde ninguno |
| `minio` | T2 sigue recibiendo scores. Logs de `lake-sink`: `write failed` | El sink se pone al día en segundos; `ver_lake.py` muestra todas las filas, sin huecos ni duplicados |

## Al terminar

T2: Ctrl-C. En **T1**:

```bash
kill %1                                                   # corta los eventos
docker compose --env-file .env.local --profile streaming down
```

`down` conserva los datos. Nunca agregar `-v`: borra los volúmenes de MLflow, PostgreSQL
y MinIO.

Los tests automáticos no necesitan el stack (entrenan modelos chicos en memoria y simulan
el broker y el almacenamiento):

```bash
uv run --group api --group ml --group train --group serve --group stream --group lake --group dev pytest
```
