"""Who may call the REST API, how often, and with how large a request.

Every route but /health and /ready needs a JWT (`Authorization: Bearer <token>`), signed
with JWT_SECRET (HS256), naming the caller (`sub`) and one role:

- `predict`: clinicians, and the hospital systems that call REST. May score patients.
- `admin`: the company's operators. May also read the served model's details.

401 means "who are you?" (no token, a bad signature, expired); 403 means "I know who you
are, and you may not". Each caller gets RATE_LIMIT_PER_MINUTE predictions per minute (429
past that), so no single caller can scrape the model or swamp it. Refusals go to the audit
log. Tokens are issued by whoever holds the secret, with this module's command line:

    python -m api.security <subject> <role> [--minutes 15]
"""

import argparse
import math
import os
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.types import ASGIApp, Receive, Scope, Send

from stroke.audit import audit_event

ALGORITHM = "HS256"
ISSUER = "stroke-risk"
AUDIENCE = "stroke-api"
ROLES = ("predict", "admin")
# HS256 needs a key at least as long as its 256-bit output (RFC 7518, section 3.2)
MIN_SECRET_BYTES = 32

_bearer = HTTPBearer(auto_error=False)


def _secret(name: str) -> str:
    value = os.getenv(name, "")
    if len(value.encode()) < MIN_SECRET_BYTES:
        raise RuntimeError(
            f"{name} must be set to at least {MIN_SECRET_BYTES} bytes; generate one with "
            "python -c 'import secrets; print(secrets.token_urlsafe(32))'"
        )
    return value


def jwt_secret() -> str:
    return _secret("JWT_SECRET")


def audit_key() -> str:
    """The key of the audit log's input fingerprints. It must stay the same across restarts,
    or old fingerprints can no longer be checked."""
    return _secret("AUDIT_KEY")


def issue_token(subject: str, role: str, minutes: int = 15, secret: str | None = None) -> str:
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}; expected one of {ROLES}")
    now = datetime.now(UTC)
    claims = {
        "sub": subject,
        "role": role,
        "iss": ISSUER,
        "aud": AUDIENCE,
        "iat": now,
        "exp": now + timedelta(minutes=minutes),
    }
    return jwt.encode(claims, secret or jwt_secret(), algorithm=ALGORITHM)


def _unauthorized(request: Request, reason: str) -> HTTPException:
    audit_event("anonymous", "auth-denied", reason=reason, path=request.url.path)
    return HTTPException(status_code=401, detail=reason, headers={"WWW-Authenticate": "Bearer"})


def require_role(*allowed: str) -> Callable[..., dict]:
    """A dependency: Depends(require_role("predict", "admin")) returns the token's claims, or
    answers 401 or 403."""

    def dependency(
        request: Request,
        creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    ) -> dict:
        if creds is None:
            raise _unauthorized(request, "missing bearer token")
        try:
            # algorithms is fixed, never read from the token: a token can't pick "none"
            claims = jwt.decode(
                creds.credentials,
                jwt_secret(),
                algorithms=[ALGORITHM],
                audience=AUDIENCE,
                issuer=ISSUER,
                options={"require": ["exp", "iat", "sub", "role"]},
            )
        except jwt.ExpiredSignatureError:
            raise _unauthorized(request, "token expired") from None
        except jwt.InvalidTokenError:
            raise _unauthorized(request, "invalid token") from None
        if claims["role"] not in allowed:
            audit_event(claims["sub"], "forbidden", role=claims["role"], path=request.url.path)
            raise HTTPException(status_code=403, detail=f"role {claims['role']!r} may not do this")
        return claims

    return dependency


class RateLimiter:
    """A sliding window per caller: at most `per_minute` calls in any 60 seconds.

    It counts calls to this process only. One API process is what compose runs; several
    replicas would each count on their own and need a shared counter (Redis, for one)."""

    def __init__(self, per_minute: int, clock: Callable[[], float] = time.monotonic) -> None:
        self.per_minute, self.clock = per_minute, clock
        self._calls: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()  # sync routes run in a thread pool

    def check(self, caller: str) -> float | None:
        """Count one call. None if allowed, else the seconds until the caller may call again."""
        with self._lock:
            now = self.clock()
            calls = self._calls[caller]
            while calls and now - calls[0] >= 60:
                calls.popleft()
            if len(calls) >= self.per_minute:
                return 60 - (now - calls[0]) if calls else 60.0  # a limit of 0 admits no one
            calls.append(now)
            return None


def rate_limited(
    request: Request, claims: Annotated[dict, Depends(require_role("predict", "admin"))]
) -> dict:
    """require_role("predict", "admin"), then the caller's rate limit (429 when over it)."""
    retry_after = request.app.state.rate_limiter.check(claims["sub"])
    if retry_after is not None:
        audit_event(claims["sub"], "rate-limited", path=request.url.path)
        raise HTTPException(
            status_code=429,
            detail="rate limit exceeded",
            headers={"Retry-After": str(math.ceil(retry_after))},
        )
    return claims


class BodySizeLimit:
    """Refuse a request body larger than `max_bytes` (413) before reading it, and a body of
    unknown length (411): the server would otherwise buffer whatever arrives in memory.
    The server never reads past the declared Content-Length, so checking it is enough."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["method"] in ("POST", "PUT", "PATCH"):
            headers = dict(scope["headers"])
            length = headers.get(b"content-length")
            refusal = None
            if length is None:
                refusal = JSONResponse({"detail": "Content-Length required"}, status_code=411)
            elif not length.isdigit() or int(length) > self.max_bytes:
                refusal = JSONResponse({"detail": "request body too large"}, status_code=413)
            if refusal is not None:
                await refusal(scope, receive, send)
                return
        await self.app(scope, receive, send)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Issue a token for the REST API (JWT_SECRET).")
    parser.add_argument("subject", help="who the token is for, e.g. dr-house or hospital-a")
    parser.add_argument("role", choices=ROLES)
    parser.add_argument("--minutes", type=int, default=15, help="lifetime (default 15)")
    args = parser.parse_args(argv)
    print(issue_token(args.subject, args.role, args.minutes))


if __name__ == "__main__":
    main()
