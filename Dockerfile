# Knowledge Graph-RAG System - Offline Deployment Dockerfile
# Multi-stage build for secure, optimized offline deployment

# =============================================================================
# Stage 1: Base Dependencies
# =============================================================================
FROM ubuntu:22.04 as base

# Install system dependencies for offline environment
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    build-essential \
    curl \
    wget \
    git \
    cmake \
    pkg-config \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    libjpeg-dev \
    libpng-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash appuser && \
    mkdir -p /app /app/models /app/data /app/config /app/logs && \
    chown -R appuser:appuser /app

# Set up Python environment
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install basic Python packages
RUN pip install --upgrade pip setuptools wheel

# =============================================================================
# Stage 2: Python Dependencies
# =============================================================================
FROM base as python-deps

# Copy requirements for dependency installation
COPY pyproject.toml /tmp/
WORKDIR /tmp

# Install Python dependencies
RUN pip install --no-cache-dir -e .[prod,gpu] && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Download and cache models for offline use
RUN python -c "
import sentence_transformers
from transformers import AutoModel, AutoTokenizer
import torch

# Download BGE embedding model
model = sentence_transformers.SentenceTransformer('BAAI/bge-large-en-v1.5')
model.save('/opt/models/bge-large-en-v1.5')

# Download additional models for offline use
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-en-v1.5')
tokenizer.save_pretrained('/opt/models/bge-large-en-v1.5-tokenizer')

print('Models downloaded successfully')
"

# =============================================================================
# Stage 3: Production Runtime
# =============================================================================
FROM base as runtime

# Copy Python environment from deps stage
COPY --from=python-deps /opt/venv /opt/venv
COPY --from=python-deps /opt/models /app/models

# Ensure virtual environment is activated
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app/src:$PYTHONPATH"

# Copy application code
COPY src/ /app/src/
COPY config/ /app/config/
COPY scripts/ /app/scripts/
COPY pyproject.toml README.md /app/

# Copy offline model configurations
COPY docker/models/ /app/models/
COPY docker/offline-packages/ /tmp/offline-packages/

# Install any additional offline packages
RUN if [ -d "/tmp/offline-packages" ]; then \
    pip install --no-index --find-links /tmp/offline-packages /tmp/offline-packages/*.whl; \
fi

# Set up configuration
COPY .env.example /app/.env
RUN chmod 600 /app/.env

# Create necessary directories with proper permissions
RUN mkdir -p \
    /app/data/documents \
    /app/data/vectors \
    /app/data/neo4j \
    /app/logs/audit \
    /app/logs/application \
    /app/config/security \
    /app/config/personas \
    /app/config/mcp \
    && chown -R appuser:appuser /app

# Security hardening
RUN chmod 755 /app/scripts/*.sh 2>/dev/null || true && \
    chmod 600 /app/config/security/* 2>/dev/null || true && \
    find /app -type f -name "*.py" -exec chmod 644 {} \; && \
    find /app -type d -exec chmod 755 {} \;

# Switch to non-root user
USER appuser
WORKDIR /app

# Environment variables for offline mode
ENV OFFLINE_MODE=true
ENV MODEL_CACHE_DIR=/app/models
ENV DATA_DIR=/app/data
ENV LOG_DIR=/app/logs
ENV CONFIG_DIR=/app/config

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Expose ports
EXPOSE 8000 8001 8002 8003 8004 8005 9090

# Default command
CMD ["python", "-m", "kg_rag.api.main"]