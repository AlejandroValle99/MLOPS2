"""services.model_server.healthcheck: Docker's healthcheck exits 0 only while SERVING."""

from grpc_health.v1 import health_pb2

from services.model_server.healthcheck import main


def test_exit_code_follows_the_health_status(model_server, monkeypatch):
    monkeypatch.setenv("MODEL_SERVER_PORT", model_server.target.rsplit(":", 1)[1])
    assert main() == 1  # no model loaded yet: NOT_SERVING
    model_server.health.set("", health_pb2.HealthCheckResponse.SERVING)
    assert main() == 0


def test_exit_code_is_1_when_nothing_is_listening(monkeypatch):
    monkeypatch.setenv("MODEL_SERVER_PORT", "1")  # a port no server listens on
    assert main() == 1
