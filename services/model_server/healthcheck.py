"""Docker's healthcheck for the model server: exit 0 only when health reports SERVING.

python -m services.model_server.healthcheck (no grpc_health_probe binary needed).
"""

import os
import sys

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc


def main() -> int:
    target = f"localhost:{os.getenv('MODEL_SERVER_PORT', '50051')}"
    try:
        with grpc.insecure_channel(target) as channel:
            response = health_pb2_grpc.HealthStub(channel).Check(
                health_pb2.HealthCheckRequest(), timeout=2
            )
    except grpc.RpcError:
        return 1
    return 0 if response.status == health_pb2.HealthCheckResponse.SERVING else 1


if __name__ == "__main__":
    sys.exit(main())
