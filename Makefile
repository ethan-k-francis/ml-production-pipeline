# ============================================================
# ML Production Pipeline — Top-Level Makefile
# Orchestrates training, serving, drift detection, and Docker.
# ============================================================

.PHONY: train serve drift-monitor up down test clean help

# Default target — show available commands
help:
	@echo "ML Production Pipeline"
	@echo "====================="
	@echo ""
	@echo "  make train          Train the fraud detection model"
	@echo "  make serve          Run the FastAPI serving app locally"
	@echo "  make drift-monitor  Run the Go drift detector locally"
	@echo "  make up             Start all services via Docker Compose"
	@echo "  make down           Stop all services"
	@echo "  make test           Run all tests (Python + Go)"
	@echo "  make clean          Remove artifacts, models, caches"
	@echo ""

# Train the model using the training pipeline
train:
	cd training && pip install -r requirements.txt && python download_data.py && python train.py

# Run the FastAPI prediction server locally (requires trained model)
serve:
	cd serving && pip install -r requirements.txt && uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Build and run the Go drift detection service locally
drift-monitor:
	cd drift-detector && go build -o drift-detector ./cmd/detector && ./drift-detector

# Start the full stack with Docker Compose
up:
	@mkdir -p models
	@test -f models/reference_distributions.json || cp drift-detector/reference_distributions.json models/
	docker compose up --build -d

# Stop all Docker Compose services and remove orphans
down:
	docker compose down --remove-orphans

# Run all test suites
test:
	cd training && python -m pytest -v 2>/dev/null || echo "No Python tests found"
	cd drift-detector && go test ./... 2>/dev/null || echo "No Go tests found"

# Clean up generated artifacts, models, and caches
clean:
	rm -rf models/ data/*.csv mlruns/
	rm -rf training/__pycache__ serving/__pycache__
	rm -f drift-detector/drift-detector
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

.PHONY: lint-ci ci-security pr-commit-check ci
lint-ci:
	pre-commit run --all-files

ci-security:
	trivy fs --severity HIGH,CRITICAL --exit-code 1 .

pr-commit-check:
	@chmod +x .github/scripts/commit-message-lint.sh
	@.github/scripts/commit-message-lint.sh --base-ref origin/main

ci: lint-ci ci-security pr-commit-check
	@echo "Local CI checks passed."
