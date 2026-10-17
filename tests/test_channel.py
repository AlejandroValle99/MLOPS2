"""stroke_rpc.channel: TLS and mutual TLS to the model server, with certificates made by
scripts/make_certs.sh (the `certs` fixture). A client that can't prove what the server
expects never gets a call through: the handshake fails, and the call is UNAVAILABLE."""

import subprocess
from pathlib import Path

import grpc
import pytest

from services.model_server.holder import ModelHolder
from services.model_server.main import create_server, server_credentials
from stroke_rpc.channel import ClientTLS, open_channel
from stroke_rpc.v1 import scoring_pb2 as pb
from stroke_rpc.v1 import scoring_pb2_grpc as pb_grpc

KEY = "tls-test-key"
MAKE_CERTS = Path(__file__).parent.parent / "scripts/make_certs.sh"


def tls_server(certs: Path, make_state, mutual: bool):
    holder = ModelHolder()
    holder.set(make_state(version="3"))
    credentials = server_credentials(
        str(certs / "server.crt"),
        str(certs / "server.key"),
        str(certs / "ca.crt") if mutual else None,
    )
    server, _, port = create_server(holder, "localhost:0", {KEY}, credentials)
    server.start()
    return server, f"localhost:{port}"  # localhost: a name the certificate lists


@pytest.fixture
def tls_target(certs, make_state):
    server, target = tls_server(certs, make_state, mutual=False)
    yield target
    server.stop(None)


@pytest.fixture
def mtls_target(certs, make_state):
    server, target = tls_server(certs, make_state, mutual=True)
    yield target
    server.stop(None)


def model_version(target: str, tls: ClientTLS | None) -> str:
    with open_channel(target, key=KEY, tls=tls) as channel:
        return pb_grpc.ScoringStub(channel).GetModelInfo(pb.ModelInfoRequest(), timeout=3).version


def refused_with(target: str, tls: ClientTLS | None) -> grpc.StatusCode:
    with pytest.raises(grpc.RpcError) as err:
        model_version(target, tls)
    return err.value.code()


def test_a_tls_client_reaches_a_tls_server(certs, tls_target):
    assert model_version(tls_target, ClientTLS(ca=(certs / "ca.crt").read_bytes())) == "3"


def test_a_plaintext_client_cannot_reach_a_tls_server(tls_target):
    assert refused_with(tls_target, None) == grpc.StatusCode.UNAVAILABLE


def test_a_server_whose_certificate_another_ca_signed_is_not_trusted(tls_target, tmp_path):
    """What a client sees when something else answers at the model server's address."""
    other = tmp_path / "other"
    subprocess.run(["bash", str(MAKE_CERTS), str(other)], check=True, capture_output=True)
    tls = ClientTLS(ca=(other / "ca.crt").read_bytes())
    assert refused_with(tls_target, tls) == grpc.StatusCode.UNAVAILABLE


def test_mutual_tls_needs_a_client_certificate(certs, mtls_target):
    ca = (certs / "ca.crt").read_bytes()
    assert refused_with(mtls_target, ClientTLS(ca=ca)) == grpc.StatusCode.UNAVAILABLE
    with_cert = ClientTLS(
        ca=ca, cert=(certs / "client.crt").read_bytes(), key=(certs / "client.key").read_bytes()
    )
    assert model_version(mtls_target, with_cert) == "3"


def test_client_tls_comes_from_the_environment(certs, monkeypatch):
    monkeypatch.delenv("TLS_CA_FILE", raising=False)
    assert ClientTLS.from_env() is None  # no CA configured: a plaintext channel
    monkeypatch.setenv("TLS_CA_FILE", str(certs / "ca.crt"))
    monkeypatch.setenv("TLS_CLIENT_CERT_FILE", str(certs / "client.crt"))
    monkeypatch.setenv("TLS_CLIENT_KEY_FILE", str(certs / "client.key"))
    tls = ClientTLS.from_env()
    assert tls is not None
    assert tls.ca == (certs / "ca.crt").read_bytes()
    assert tls.cert == (certs / "client.crt").read_bytes()
    assert tls.key == (certs / "client.key").read_bytes()
