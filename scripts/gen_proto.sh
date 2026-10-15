#!/usr/bin/env bash
set -euo pipefail
uv run python -m grpc_tools.protoc -I proto \
  --python_out=. --pyi_out=. --grpc_python_out=. \
  proto/stroke_rpc/v1/scoring.proto
touch stroke_rpc/__init__.py stroke_rpc/v1/__init__.py