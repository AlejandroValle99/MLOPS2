"""services.model_server.healthcheck: Docker's healthcheck exits 0 only while SERVING."""

from grpc_health.v1 import health_pb2

from services.model_server.healthcheck import main
from services.model_server.holder import ModelHolder
from services.model_server.main import create_server, server_credentials


def test_exit_code_follows_the_health_status(model_server, monkeypatch):
    monkeypatch.setenv("MODEL_SERVER_PORT", model_server.target.rsplit(":", 1)[1])
    assert main() == 1  # no model loaded yet: NOT_SERVING
    model_server.health.set("", health_pb2.HealthCheckResponse.SERVING)
    assert main() == 0


def test_exit_code_is_1_when_nothing_is_listening(monkeypatch):
    monkeypatch.setenv("MODEL_SERVER_PORT", "1")  # a port no server listens on
    assert main() == 1


def test_over_tls_it_checks_with_the_configured_ca(certs, monkeypatch):
    credentials = server_credentials(str(certs / "server.crt"), str(certs / "server.key"))
    server, health_servicer, port = create_server(ModelHolder(), "localhost:0", {"k"}, credentials)
    server.start()
    try:
        monkeypatch.setenv("MODEL_SERVER_PORT", str(port))
        monkeypatch.delenv("TLS_CA_FILE", raising=False)
        health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
        assert main() == 1  # plaintext against a TLS server: no answer at all
        monkeypatch.setenv("TLS_CA_FILE", str(certs / "ca.crt"))
        assert main() == 0
    finally:
        server.stop(None)
