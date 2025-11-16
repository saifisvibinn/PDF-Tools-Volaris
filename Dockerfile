# syntax=docker/dockerfile:1
#
# Multi-purpose Dockerfile for Flask or FastAPI apps, GPU-ready
# - Base: Python 3.10-slim (small, production-friendly)
# - Installs build deps and common native libs (for Pillow, OpenCV, etc.)
# - Installs project dependencies from pyproject.toml (preferred) or requirements.txt
# - CMD switches between Gunicorn (Flask) and Uvicorn (FastAPI) based on APP_TYPE
# - No reload mode (production)
# - GPU access is provided by NVIDIA Container Toolkit on the host; no CUDA install here
#

ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim AS base

# Allow switching app type at build/run time: "flask" (default) or "fastapi"
ARG APP_TYPE=flask
ENV APP_TYPE=${APP_TYPE}

# Set workdir
WORKDIR /app

# System deps: build tools and runtime libs used by common Python packages
# - gcc, g++: build wheels if needed
# - libgl1: OpenCV image backend
# - libffi, libjpeg, zlib: Pillow and friends
# - netcat-openbsd: for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc g++ \
    libgl1 \
    libffi-dev \
    libjpeg62-turbo-dev \
    zlib1g-dev \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip/setuptools/wheel for reliable builds
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy dependency files first for better Docker layer caching
COPY pyproject.toml requirements.txt* ./

# Install server runners and helpers (small footprint)
# - gunicorn: production WSGI server for Flask
# - uvicorn: ASGI server for FastAPI
RUN python -m pip install --no-cache-dir gunicorn uvicorn

# Install project dependencies from pyproject.toml (preferred), else requirements.txt
# Note: If using CUDA-enabled PyTorch, ensure your pyproject/requirements pins a +cuXX wheel.
# Since we're using Python 3.12 (matching pyproject.toml's requires-python), pip install . works fine.
RUN if [ -f "pyproject.toml" ]; then \
        python -m pip install --no-cache-dir . ; \
    elif [ -f "requirements.txt" ]; then \
        python -m pip install --no-cache-dir -r requirements.txt ; \
    else \
        echo "No dependency file found. Skipping dependency install."; \
    fi

# Copy the rest of the application
COPY . .

# Expose the appropriate port depending on APP_TYPE
# (This is informational; docker-compose will publish the right port.)
EXPOSE 5000
EXPOSE 8000

# Healthcheck (simple TCP check)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD sh -c "nc -z localhost ${PORT:-5000} || exit 1"

# Runtime user (optional hardening). Comment if you need write access everywhere.
# RUN useradd --create-home appuser && chown -R appuser:appuser /app
# USER appuser

# Start the correct server based on APP_TYPE (no reload, production-ready)
# - Flask: gunicorn -b 0.0.0.0:5000 app:app
# - FastAPI: uvicorn app:app --host 0.0.0.0 --port 8000
#
# To switch at runtime: docker run -e APP_TYPE=fastapi ...
CMD ["/bin/sh", "-lc", "\
if [ \"$APP_TYPE\" = \"fastapi\" ]; then \
    export PORT=${PORT:-8000}; \
    echo \"Starting FastAPI on port ${PORT}\"; \
    exec uvicorn app:app --host 0.0.0.0 --port ${PORT}; \
else \
    export PORT=${PORT:-5000}; \
    echo \"Starting Flask on port ${PORT}\"; \
    exec gunicorn -b 0.0.0.0:${PORT} app:app; \
fi"]


