# Baseball Pitch Simulator API. Build from the repo root:
#   docker build -f api/Dockerfile -t pitch-sim-api .
FROM python:3.13-slim

# libgomp1 is required by LightGBM at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code and the ML package it imports (ml.features, ml.explain).
COPY api/ ./api/
COPY ml/ ./ml/

# Serialized models + precomputed lookup tables (gitignored, from build context).
COPY ml/artifacts/ ./ml/artifacts/
COPY data/processed/ ./data/processed/

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
