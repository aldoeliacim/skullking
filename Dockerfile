# Python Dockerfile for Skull King
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy dependency files and README (required by poetry-core metadata)
COPY pyproject.toml uv.lock README.md ./

# Install dependencies only (without dev dependencies)
RUN uv sync --no-dev --frozen --no-install-project

# Copy application code
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY static/ ./static/

# Install the project itself
RUN uv sync --no-dev --frozen

# Copy trained models if available
COPY models/ ./models/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
