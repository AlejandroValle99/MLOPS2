"""Shared constants for stroke/ (the new lifecycle modules). Env-driven so the same code
runs unchanged on the laptop and inside containers.

Deliberately minimal for now (Phase 0 scope): just what stroke/data.py needs for the
frozen split. Phase 5 adds the gate/registry/evaluation constants on top of this.
"""

RANDOM_SEED = 42
TEST_SIZE = 0.20
