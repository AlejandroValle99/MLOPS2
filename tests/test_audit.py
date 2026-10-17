"""stroke.audit: one JSON line per action, and an input fingerprint that proves what was
scored without revealing it."""

import json
import logging

from scripts.mini_tp1_client import VALID
from stroke.audit import audit_event, input_fingerprint

KEY = "audit-test-key-that-is-long-enough!!"


def test_the_fingerprint_is_stable_and_keyed():
    row = dict(VALID)
    assert input_fingerprint(row, KEY) == input_fingerprint(dict(reversed(row.items())), KEY)
    assert input_fingerprint(row, KEY) != input_fingerprint({**row, "age": 68}, KEY)
    # Without the key, nobody can recompute it, so nobody can test guesses against the log
    assert input_fingerprint(row, KEY) != input_fingerprint(row, KEY + "x")
    assert len(input_fingerprint(row, KEY)) == 32


def test_an_audit_line_is_one_json_object_with_who_what_and_when(caplog):
    caplog.set_level(logging.INFO, logger="audit")
    entry = audit_event("ops-1", "rollback", model="stroke-model", to_version="4")
    (record,) = [r for r in caplog.records if r.name == "audit"]
    assert json.loads(record.getMessage()) == entry
    assert (entry["actor"], entry["action"], entry["to_version"]) == ("ops-1", "rollback", "4")
    assert entry["ts"].endswith("+00:00")  # UTC
