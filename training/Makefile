# ============================================================
# Training Pipeline — Makefile
# Targets for data generation, model training, and evaluation.
# ============================================================

.PHONY: train evaluate generate-data clean help

# Default: show available targets
help:
	@echo "Training Pipeline"
	@echo "================="
	@echo ""
	@echo "  make generate-data  Generate synthetic fraud dataset"
	@echo "  make train          Train the fraud detection model"
	@echo "  make evaluate       Train and show detailed evaluation metrics"
	@echo "  make clean          Remove generated data and model artifacts"
	@echo ""

# Generate synthetic credit card fraud dataset
generate-data:
	python download_data.py
	@echo "Dataset generated in ../data/"

# Full training pipeline: generate data (if needed) → train → save model
train: generate-data
	python train.py
	@echo "Model saved in ../models/"

# Evaluate mode: same as train but with verbose output for inspection
evaluate: generate-data
	python train.py
	@echo ""
	@echo "Check MLflow UI at http://localhost:5000 for experiment details"

# Remove generated artifacts
clean:
	rm -rf ../data/*.csv ../models/*.joblib
	rm -rf __pycache__
	@echo "Cleaned training artifacts"
