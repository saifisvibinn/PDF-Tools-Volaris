#!/usr/bin/env bash
set -euo pipefail

# Simple entrypoint to allow switching modes later.
# Modes: streamlit | fastapi
MODE=${MODE:-streamlit}
PORT=${PORT:-7860}

echo "[entrypoint] Starting in $MODE mode on port $PORT"

if [ "$MODE" = "fastapi" ]; then
  # Run FastAPI via uvicorn (expects an api_server.py you can add later)
  uvicorn app:app --host 0.0.0.0 --port "$PORT" --workers 1
else
  streamlit run pdf_extractor_gui.py --server.address=0.0.0.0 --server.port="$PORT"
fi
