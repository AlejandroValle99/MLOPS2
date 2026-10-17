"""Every call to the model server must carry one of its keys, except health checks.

Clients send `authorization: Bearer <key>` in the call's metadata (stroke_rpc.channel does
it for them). Health stays public, because Docker's healthcheck and load balancers ask it
without credentials and it reveals nothing. Reflection is not public: it lists the whole
service, so grpcurl and grpcui need the key as well (`-H "authorization: Bearer ..."`).
"""

import logging
import secrets
from collections.abc import Collection

import grpc

log = logging.getLogger(__name__)

PUBLIC_PREFIXES = ("/grpc.health.v1.Health/",)


def _deny(handler: grpc.RpcMethodHandler) -> grpc.RpcMethodHandler:
    """A handler that refuses the call, of the same shape (unary or streaming, each way) as
    the one it replaces, so the refused call is read the way the client sends it."""

    def abort(request_or_iterator, context: grpc.ServicerContext):
        context.abort(grpc.StatusCode.UNAUTHENTICATED, "missing or invalid credentials")

    if handler.request_streaming and handler.response_streaming:
        return grpc.stream_stream_rpc_method_handler(abort)
    if handler.request_streaming:
        return grpc.stream_unary_rpc_method_handler(abort)
    if handler.response_streaming:
        return grpc.unary_stream_rpc_method_handler(abort)
    return grpc.unary_unary_rpc_method_handler(abort)


class AuthInterceptor(grpc.ServerInterceptor):
    def __init__(self, keys: Collection[str]) -> None:
        self.keys = [key for key in keys if key]
        if not self.keys:
            raise ValueError("the model server needs at least one key; refusing to run open")

    def _authorized(self, metadata: dict) -> bool:
        scheme, _, presented = str(metadata.get("authorization", "")).partition(" ")
        if scheme != "Bearer" or not presented:
            return False
        # Every key is compared, without stopping at a match, each in constant time
        given = presented.encode()
        return any([secrets.compare_digest(given, key.encode()) for key in self.keys])

    def intercept_service(self, continuation, handler_call_details):
        method = handler_call_details.method
        if method.startswith(PUBLIC_PREFIXES):
            return continuation(handler_call_details)
        if self._authorized(dict(handler_call_details.invocation_metadata or ())):
            return continuation(handler_call_details)
        handler = continuation(handler_call_details)
        if handler is None:  # an unknown method: let gRPC answer UNIMPLEMENTED
            return None
        log.warning("refused %s: missing or invalid key", method)
        return _deny(handler)
