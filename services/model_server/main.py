"""The gRPC model server: python -m services.model_server.main

Serves stroke_rpc.v1.Scoring, the standard grpc.health.v1 health service (NOT_SERVING
until the first model loads) and server reflection (so grpcurl can list the services).
A background thread keeps the served model in step with the registry's champion alias.

Every call except health needs one of MODEL_SERVER_KEYS (services/model_server/auth.py);
the server refuses to start without one. With TLS_CERT_FILE and TLS_KEY_FILE it serves TLS
only, and with TLS_CLIENT_CA_FILE as well it also requires a client certificate signed by
that CA (mutual TLS). Without them it serves plaintext and says so in its log.
"""

import logging
import os
import signal
import threading
from collections.abc import Callable, Collection
from concurrent import futures
from pathlib import Path

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
from grpc_reflection.v1alpha import reflection

from services.model_server.auth import AuthInterceptor
from services.model_server.holder import ModelHolder, ModelState
from services.model_server.reloader import Reloader, mlflow_loader, mlflow_lookup
from services.model_server.servicer import ScoringServicer
from stroke_rpc.v1 import scoring_pb2 as pb
from stroke_rpc.v1 import scoring_pb2_grpc as pb_grpc

log = logging.getLogger(__name__)

SERVICE_NAME = pb.DESCRIPTOR.services_by_name["Scoring"].full_name  # "stroke_rpc.v1.Scoring"

# One valid patient, scored once by every candidate model before it is swapped in.
WARMUP_ROW = {
    "gender": "Male",
    "age": 67.0,
    "hypertension": 0,
    "heart_disease": 1,
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
    "smoking_status": "formerly smoked",
}


# A PredictBatch of a few hundred patients is a few tens of KB; anything this large is not
# a request the service makes, and the server refuses it before reading it into memory.
MAX_MESSAGE_BYTES = 1024 * 1024


def server_credentials(
    cert_file: str, key_file: str, client_ca_file: str | None = None
) -> grpc.ServerCredentials:
    """TLS with this certificate and key. With a client CA, mutual TLS: the handshake fails
    unless the client presents a certificate that CA signed."""
    client_ca = Path(client_ca_file).read_bytes() if client_ca_file else None
    return grpc.ssl_server_credentials(
        [(Path(key_file).read_bytes(), Path(cert_file).read_bytes())],
        root_certificates=client_ca,
        require_client_auth=client_ca is not None,
    )


def create_server(
    holder: ModelHolder,
    address: str,
    keys: Collection[str],
    credentials: grpc.ServerCredentials | None = None,
    max_message_bytes: int = MAX_MESSAGE_BYTES,
) -> tuple[grpc.Server, health.HealthServicer, int]:
    """Build (not start) the server. Returns it, its health servicer and the bound port.
    Every call but health must carry one of `keys`; with `credentials` the port is TLS only."""
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=int(os.getenv("GRPC_MAX_WORKERS", "8"))),
        interceptors=[AuthInterceptor(keys)],
        # Over this many RPCs in flight, new calls get RESOURCE_EXHAUSTED instead of queueing.
        maximum_concurrent_rpcs=int(os.getenv("GRPC_MAX_CONCURRENT_RPCS", "32")),
        options=[("grpc.max_receive_message_length", max_message_bytes)],
    )
    pb_grpc.add_ScoringServicer_to_server(ScoringServicer(holder), server)

    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    for service in ("", SERVICE_NAME):  # "" is the whole server
        health_servicer.set(service, health_pb2.HealthCheckResponse.NOT_SERVING)

    reflection.enable_server_reflection(
        (SERVICE_NAME, health.SERVICE_NAME, reflection.SERVICE_NAME), server
    )
    if credentials is None:
        port = server.add_insecure_port(address)
    else:
        port = server.add_secure_port(address, credentials)
    return server, health_servicer, port


def mark_serving(health_servicer: health.HealthServicer) -> Callable[[ModelState], None]:
    """The reloader's on_loaded callback: report SERVING once a model is in the holder."""

    def on_loaded(_state: ModelState) -> None:
        for service in ("", SERVICE_NAME):
            health_servicer.set(service, health_pb2.HealthCheckResponse.SERVING)

    return on_loaded


def serve() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    name = os.getenv("MODEL_NAME", "stroke-model")
    alias = os.getenv("MODEL_ALIAS", "champion")

    keys = [key.strip() for key in os.getenv("MODEL_SERVER_KEYS", "").split(",") if key.strip()]
    if not keys:
        raise SystemExit("MODEL_SERVER_KEYS is empty: refusing to serve without authentication")
    cert, key = os.getenv("TLS_CERT_FILE"), os.getenv("TLS_KEY_FILE")
    credentials = None
    if cert and key:
        credentials = server_credentials(cert, key, os.getenv("TLS_CLIENT_CA_FILE") or None)
    else:
        log.warning("TLS_CERT_FILE/TLS_KEY_FILE unset: serving PLAINTEXT, for local use only")

    holder = ModelHolder()
    server, health_servicer, port = create_server(
        holder,
        f"[::]:{os.getenv('MODEL_SERVER_PORT', '50051')}",
        keys,
        credentials,
        int(os.getenv("GRPC_MAX_MESSAGE_BYTES", str(MAX_MESSAGE_BYTES))),
    )
    reloader = Reloader(
        holder,
        name,
        lookup=mlflow_lookup(tracking_uri, name, alias),
        loader=mlflow_loader(tracking_uri, name),
        warmup_row=WARMUP_ROW,
        interval_s=float(os.getenv("RELOAD_INTERVAL_S", "10")),
        on_loaded=mark_serving(health_servicer),
    )
    stop = threading.Event()
    thread = threading.Thread(target=reloader.run_forever, args=(stop,), daemon=True)
    thread.start()

    server.start()
    log.info(
        "model server listening on port %d (%s), following %s@%s",
        port,
        "plaintext" if credentials is None else "TLS",
        name,
        alias,
    )

    def shutdown(*_: object) -> None:
        log.info("shutting down: draining in-flight calls")
        stop.set()
        health_servicer.enter_graceful_shutdown()  # health answers NOT_SERVING while draining
        server.stop(grace=10)  # docker stop sends SIGTERM; finish running calls first

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
