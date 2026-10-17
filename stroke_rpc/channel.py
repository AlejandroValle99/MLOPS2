"""How every client opens its channel to the model server: TLS when a CA is configured, and
the client's key attached to every call.

Imports only grpc and the standard library: the stream worker's image has nothing else.
"""

import os
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import grpc


@dataclass(frozen=True)
class ClientTLS:
    """The CA that signed the model server's certificate, and for mutual TLS this client's
    own certificate and key (PEM bytes)."""

    ca: bytes
    cert: bytes | None = None
    key: bytes | None = None

    @classmethod
    def from_env(cls) -> "ClientTLS | None":
        """TLS_CA_FILE, plus TLS_CLIENT_CERT_FILE and TLS_CLIENT_KEY_FILE for mutual TLS.
        None when TLS_CA_FILE is unset: the channel is then plaintext."""
        ca = os.getenv("TLS_CA_FILE")
        if not ca:
            return None
        cert, key = os.getenv("TLS_CLIENT_CERT_FILE"), os.getenv("TLS_CLIENT_KEY_FILE")
        return cls(
            ca=Path(ca).read_bytes(),
            cert=Path(cert).read_bytes() if cert else None,
            key=Path(key).read_bytes() if key else None,
        )


class _CallDetails(
    namedtuple(
        "_CallDetails",
        ("method", "timeout", "metadata", "credentials", "wait_for_ready", "compression"),
    ),
    grpc.ClientCallDetails,
):
    pass


class _BearerKey(
    grpc.UnaryUnaryClientInterceptor,
    grpc.UnaryStreamClientInterceptor,
    grpc.StreamUnaryClientInterceptor,
    grpc.StreamStreamClientInterceptor,
):
    """Adds `authorization: Bearer <key>` to every call, keeping the call's own metadata
    (the request id, for one)."""

    def __init__(self, key: str) -> None:
        self._header = ("authorization", f"Bearer {key}")

    def _with_key(self, details: Any) -> _CallDetails:
        return _CallDetails(
            details.method,
            details.timeout,
            [*(details.metadata or ()), self._header],
            details.credentials,
            details.wait_for_ready,
            details.compression,
        )

    def intercept_unary_unary(self, continuation, client_call_details, request):
        return continuation(self._with_key(client_call_details), request)

    def intercept_unary_stream(self, continuation, client_call_details, request):
        return continuation(self._with_key(client_call_details), request)

    def intercept_stream_unary(self, continuation, client_call_details, request_iterator):
        return continuation(self._with_key(client_call_details), request_iterator)

    def intercept_stream_stream(self, continuation, client_call_details, request_iterator):
        return continuation(self._with_key(client_call_details), request_iterator)


def open_channel(target: str, key: str | None = None, tls: ClientTLS | None = None) -> grpc.Channel:
    """One long-lived channel to `target`. With `tls`, the server's certificate must be
    signed by `tls.ca` and name the host in `target`; a plaintext server then refuses the
    connection (UNAVAILABLE), as does an mTLS server when `tls` holds no client certificate.
    With `key`, every call carries it; without one, the model server answers UNAUTHENTICATED."""
    if tls is None:
        channel = grpc.insecure_channel(target)
    else:
        credentials = grpc.ssl_channel_credentials(
            root_certificates=tls.ca, private_key=tls.key, certificate_chain=tls.cert
        )
        channel = grpc.secure_channel(target, credentials)
    if key:
        channel = grpc.intercept_channel(channel, _BearerKey(key))
    return channel
