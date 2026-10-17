"""api.security on its own: the rate limiter's window, token issuing and its command line.
How the routes use them (401, 403, 429, 413) is in test_api.py."""

import jwt
import pytest

from api.security import AUDIENCE, ISSUER, RateLimiter, issue_token, main

SECRET = "test-jwt-secret-that-is-long-enough-for-hs256"


def test_the_rate_limit_window_slides():
    now = [0.0]
    limiter = RateLimiter(per_minute=2, clock=lambda: now[0])
    assert limiter.check("a") is None and limiter.check("a") is None
    assert limiter.check("a") == pytest.approx(60)  # full: retry when the first call expires
    assert limiter.check("b") is None  # every caller has a window of their own
    now[0] = 59.9
    assert limiter.check("a") == pytest.approx(0.1)
    now[0] = 60.0  # the first call has left the window
    assert limiter.check("a") is None


def test_a_refused_call_is_not_counted():
    now = [0.0]
    limiter = RateLimiter(per_minute=1, clock=lambda: now[0])
    assert limiter.check("a") is None
    for t in (10.0, 20.0, 30.0):  # refused calls must not push the window forward
        now[0] = t
        assert limiter.check("a") is not None
    now[0] = 60.0
    assert limiter.check("a") is None


def test_a_token_says_who_what_role_and_until_when():
    claims = jwt.decode(
        issue_token("dr-house", "predict", minutes=5, secret=SECRET),
        SECRET,
        algorithms=["HS256"],
        audience=AUDIENCE,
        issuer=ISSUER,
    )
    assert (claims["sub"], claims["role"]) == ("dr-house", "predict")
    assert claims["exp"] - claims["iat"] == 300


def test_issue_token_refuses_an_unknown_role():
    with pytest.raises(ValueError):
        issue_token("x", "superuser", secret=SECRET)


def test_issue_token_refuses_a_short_secret(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "change_me_to_32_bytes_or_more")  # 29 bytes
    with pytest.raises(RuntimeError, match="at least 32 bytes"):
        issue_token("x", "admin")


def test_the_command_line_prints_a_token(monkeypatch, capsys):
    monkeypatch.setenv("JWT_SECRET", SECRET)
    main(["ops-1", "admin", "--minutes", "60"])
    token = capsys.readouterr().out.strip()
    claims = jwt.decode(token, SECRET, algorithms=["HS256"], audience=AUDIENCE, issuer=ISSUER)
    assert (claims["sub"], claims["role"]) == ("ops-1", "admin")
