"""Simula un ataque a la cadena de suministro: modifica los archivos de una versión del
modelo en MinIO, para ver que el model server se niega a cargarla. Y la deja como estaba.

    docker compose --env-file .env.local exec -T model-server \
        python - romper 2 < test-suite/manipular_modelo.py
    docker compose --env-file .env.local exec -T model-server \
        python - reparar 2 < test-suite/manipular_modelo.py

`romper` agrega una línea al final del MLmodel de esa versión (el archivo que dice cómo
cargarla); `reparar` la saca, y comprueba que el checksum vuelve a coincidir con el tag
`artifact_sha256` de la versión. Se niega a romper la versión que hoy es `champion`.

Corre dentro del contenedor model-server, que ya tiene MLflow, boto3 y las credenciales
de MinIO: no hace falta nada instalado ni configurado en la laptop.
"""

import os
import sys
from urllib.parse import urlparse

import boto3
from mlflow.tracking import MlflowClient

from stroke.registry import DIGEST_TAG, digest_of, get_alias

NAME = os.getenv("MODEL_NAME", "stroke-model")
MARCA = b"\n# modificado por la demo de seguridad\n"

accion, version = sys.argv[1], sys.argv[2]
client = MlflowClient()
mv = client.get_model_version(NAME, version)
champion = get_alias(client, NAME)
if accion == "romper" and champion is not None and str(champion.version) == version:
    sys.exit(f"v{version} es el champion: elegí otra versión, o el model server no arranca")

# Dónde están de verdad los archivos de la versión, en el bucket de MLflow
destino = urlparse(client.get_logged_model(mv.source.rsplit("/", 1)[-1]).artifact_location)
bucket, key = destino.netloc, destino.path.lstrip("/") + "/MLmodel"
s3 = boto3.client("s3", endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"])
actual = s3.get_object(Bucket=bucket, Key=key)["Body"].read()

if accion == "romper" and not actual.endswith(MARCA):
    s3.put_object(Bucket=bucket, Key=key, Body=actual + MARCA)
elif accion == "reparar" and actual.endswith(MARCA):
    s3.put_object(Bucket=bucket, Key=key, Body=actual[: -len(MARCA)])
elif accion not in ("romper", "reparar"):
    sys.exit("uso: romper <versión> | reparar <versión>")

ahora, esperado = digest_of(f"models:/{NAME}/{version}"), mv.tags.get(DIGEST_TAG, "")
print(f"s3://{bucket}/{key}")
print(f"v{version}: checksum de los archivos {ahora[:12]}, tag artifact_sha256 {esperado[:12]}")
if ahora == esperado:
    print("coinciden: el model server la puede cargar")
else:
    print("NO coinciden: el model server se niega a cargarla")
