# ============================================================
# Oblisk – single-container build
#   Stage 1 (node): compile the Vite/TypeScript frontend
#   Stage 2 (python): run FastAPI + serve the compiled frontend
# ============================================================

# ---- Stage 1: build frontend -----------------------------------
FROM node:20-alpine AS frontend-builder

WORKDIR /frontend
COPY web/frontend/package.json web/frontend/package-lock.json* ./
RUN npm install --frozen-lockfile

COPY web/frontend/ ./
RUN npm run build        # → /frontend/dist


# ---- Stage 2: Python runtime -----------------------------------
FROM python:3.11-slim

# System deps needed by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python deps (minimal web set; local GPU dev uses requirements.txt)
COPY web/requirements-web.txt /tmp/requirements-web.txt
RUN pip install --no-cache-dir \
        -r /tmp/requirements-web.txt \
        fastapi \
        "uvicorn[standard]" \
        python-multipart

# Copy oblisk source (all of it – same context as docker-compose)
WORKDIR /app/oblisk
COPY . /app/oblisk

# Drop compiled frontend into the expected location
COPY --from=frontend-builder /frontend/dist /app/oblisk/web/frontend/dist

# Data volume mount point
RUN mkdir -p /app/data/uploads /app/data/results

ENV PYTHONPATH=/app/oblisk
# Persistent uploads/results (Coolify: mount storage at this path)
ENV DATA_DIR=/app/data
ENV CUDA_VISIBLE_DEVICES=
ENV NVIDIA_VISIBLE_DEVICES=void

EXPOSE 32212

CMD ["uvicorn", "web.backend.main:app", \
     "--host", "0.0.0.0", "--port", "32212", \
     "--workers", "2"]
