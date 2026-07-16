# Dockerfile for the MolScribe image-to-SMILES server (CPU, linux/amd64).
# torch/torchvision come from the PyTorch CPU-only index (pinned in pyproject.toml), whose wheels
# are published for x86_64 — hence an amd64 image (runs on Apple Silicon via emulation). This keeps
# the image small; the GPU counterpart is Dockerfile.cuda, which uses the CUDA torch build.
#
## -------- Builder Stage ----------##
FROM python:3.11-slim-bullseye AS builder

RUN apt-get update && apt-get install -y \
    build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    pip install --upgrade pip

# Keeps Python from buffering stdout/stderr so logs aren't lost on a crash.
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY . /app

# Install into /opt/venv via uv. UV_PYTHON_PREFERENCE=only-system forces the container's
# Python (so venv symlinks stay valid in the production stage). --no-editable copies files
# into the venv so the source tree can be dropped.
RUN pip install uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV UV_PYTHON_PREFERENCE=only-system
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN uv sync --no-dev --no-editable

# MolScribe is installed WITHOUT its dependencies: its metadata pins torch<2.0, but the
# resolved environment above already provides a modern torch and MolScribe's real imports
# work fine with it. MolScribe itself is a pure-python wheel, so this is a plain download.
RUN uv pip install --python /opt/venv/bin/python --no-deps molscribe

# Verify the installation
RUN uv pip list

## -------- Production Stage ----------##
FROM python:3.11-slim-bullseye AS production
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# HuggingFace cache lives under the mounted volume so the ~400 MB checkpoint downloads once.
ENV HF_HOME=/home/appuser/.data/molscribe
RUN mkdir -p /home/appuser/.data/molscribe

# Copy the application and the pre-built venv from the builder stage
COPY --from=builder /app /app
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Run as a non-root user
RUN adduser --disabled-password --gecos '' appuser \
    && chown -R appuser:appuser /app \
    && chown -R appuser:appuser /home/appuser/.data
USER appuser

# Port the server listens on
EXPOSE 8999

ENV PYTHONPATH=/app

CMD ["python", "molscribe_server.py"]
