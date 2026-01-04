FROM mcr.microsoft.com/devcontainers/python:3.11-bullseye AS base
WORKDIR /app

FROM base AS builder

# Copy uv binaries from astral-sh image
COPY --from=ghcr.io/astral-sh/uv:0.6.3 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Copy project files first for caching
COPY pyproject.toml requirements.txt /app/

# Generate lock file and install dependencies (no dev) using uv
RUN uv lock && uv sync --no-install-project --no-dev

# Copy application code and install project dependencies
COPY . /app
RUN uv sync --no-dev

# Final stage
FROM base

WORKDIR /app
COPY --from=builder /app /app
COPY --from=builder /bin/uv /bin/uv

# Set PATH to use venv created by uv
ENV PATH="/app/.venv/bin:$PATH"

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app

# Run the RFP Radar pipeline (batch job, not a server)
CMD ["uv", "run", "python", "-m", "rfp_radar.main"]
