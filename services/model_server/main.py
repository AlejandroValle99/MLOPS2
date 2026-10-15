"""The gRPC model server: python -m services.model_server.main

Serves stroke_rpc.v1.Scoring, the standard grpc.health.v1 health service (NOT_SERVING
until the first model loads) and server reflection (so grpcurl can list the services).
A background thread keeps the served model in step with the registry's champion alias.
"""

import logging
import os
import signal
import threading
from collections.abc import Callable
from concurrent import futures

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
from grpc_reflection.v1alpha import reflection

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


def create_server(
    holder: ModelHolder, address: str
) -> tuple[grpc.Server, health.HealthServicer, int]:
    """Build (not start) the server. Returns it, its health servicer and the bound port."""
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=int(os.getenv("GRPC_MAX_WORKERS", "8"))),
        # Over this many RPCs in flight, new calls get RESOURCE_EXHAUSTED instead of queueing.
        maximum_concurrent_rpcs=int(os.getenv("GRPC_MAX_CONCURRENT_RPCS", "32")),
    )
    pb_grpc.add_ScoringServicer_to_server(ScoringServicer(holder), server)

    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    for service in ("", SERVICE_NAME):  # "" is the whole server
        health_servicer.set(service, health_pb2.HealthCheckResponse.NOT_SERVING)

    reflection.enable_server_reflection(
        (SERVICE_NAME, health.SERVICE_NAME, reflection.SERVICE_NAME), server
    )
    port = server.add_insecure_port(address)
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

    holder = ModelHolder()
    server, health_servicer, port = create_server(
        holder, f"[::]:{os.getenv('MODEL_SERVER_PORT', '50051')}"
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
    log.info("model server listening on port %d, following %s@%s", port, name, alias)

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
