FROM python:3.13-slim

# git is required only to `pip install market-structure-engine @ git+https://...`
# (see requirements.txt) -- removed after install to keep the image lean.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/models/market_structure
# This container's own package root needs models/__init__.py + this
# folder's __init__.py for `models.market_structure.xxx`-style imports to
# work exactly like the multi-model inference-service layout.
RUN mkdir -p /app/models && touch /app/models/__init__.py

EXPOSE 8000

# Shell form + exec so ${PORT:-8000} expands (Railway injects PORT; local
# Compose doesn't, falls back to 8000) and uvicorn becomes PID 1 (receives
# SIGTERM directly for clean shutdown instead of a shell child eating it).
CMD exec uvicorn models.market_structure.main:app --host 0.0.0.0 --port ${PORT:-8000}
