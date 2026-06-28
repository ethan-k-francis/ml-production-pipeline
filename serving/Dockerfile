# ============================================================
# FastAPI Serving — Dockerfile (Phase 04: MLflow Integration)
# Updated to include training code so models can be trained inside
# the container, and to support MLflow model loading.
# ============================================================

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Create a non-root user — containers should never run as root
RUN useradd --create-home --uid 65532 appuser

# Install serving dependencies
COPY serving/requirements.txt ./serving-requirements.txt
RUN pip install --no-cache-dir -r serving-requirements.txt

# Install training dependencies (needed for in-container training)
COPY training/requirements.txt ./training-requirements.txt
RUN pip install --no-cache-dir -r training-requirements.txt

# Copy serving application
COPY serving/app.py .

# Copy training code for in-container model training
COPY training/ /app/training/

# Model directory — mounted as volume or populated by training
RUN mkdir -p /models /app/data && chown -R appuser:appuser /models /app/data

# Run as non-root for security — least-privilege principle
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
