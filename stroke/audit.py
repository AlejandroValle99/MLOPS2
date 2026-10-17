"""The audit log: one JSON line per action that matters, saying who did what, and when.

Predictions, gate decisions, rollbacks and refused calls each leave one line on the `audit`
logger, which the services print with the rest of their logs. A line never holds patient
features. A prediction's input is recorded as a keyed fingerprint instead: whoever holds the
key can check that a given input was the one scored, but the line alone doesn't reveal it.
The probability isn't recorded either: the input and the model version are enough to compute
it again. Standard library only, so every image that imports it stays as slim as it is.
"""

import hashlib
import hmac
import json
import logging
from datetime import UTC, datetime

audit_log = logging.getLogger("audit")


def input_fingerprint(row: dict, key: str) -> str:
    """A keyed hash (HMAC-SHA256) of the input. It has to be keyed: the ten features take
    few enough values that a plain hash of them could be reversed by trying every combination."""
    message = json.dumps(row, sort_keys=True, default=str).encode()
    return hmac.new(key.encode(), message, hashlib.sha256).hexdigest()[:32]


def audit_event(actor: str, action: str, **details: object) -> dict:
    """Write one audit line and return it. `details` must never contain features: pass
    input_fingerprint(row, key) instead of the row."""
    entry = {
        "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
        "actor": actor,
        "action": action,
        **details,
    }
    audit_log.info(json.dumps(entry, default=str))
    return entry
