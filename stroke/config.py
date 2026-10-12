"""Shared constants for stroke/ (the new lifecycle modules). Env-driven so the same code
runs unchanged on the laptop and inside containers.

Holds only what stroke/data.py needs for the frozen split so far; constants for model
evaluation and the registry belong here too as those modules are added.
"""

RANDOM_SEED = 42
TEST_SIZE = 0.20
