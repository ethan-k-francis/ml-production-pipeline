# ML Production Pipeline

**End-to-end ML serving with automated drift detection**

A production-grade machine learning pipeline that trains a fraud detection model, serves predictions via FastAPI, and monitors for data drift using a custom Go service — with automated alerting when model performance may be degrading.

---

## Design Document

### Problem

ML models silently degrade in production as real-world data distributions shift from training data — without monitoring, prediction quality drops undetected. A fraud detection model trained on historical patterns may miss emerging fraud techniques simply because the incoming data no longer resembles what the model learned. This project solves that by building automated drift detection directly into the serving pipeline.

### Trade-offs

| Decision | Why |
|---|---|
| **Go for drift detection** | The drift monitor needs fast statistical calculations over streaming data. Go's low-latency, compiled performance handles high-throughput PSI/KS computations without the GIL bottleneck — Python would struggle at high request volumes. |
| **FastAPI for serving** | Industry standard for ML serving with native async support, automatic OpenAPI docs, and Pydantic validation. Integrates cleanly with scikit-learn and MLflow. |
| **Redis for prediction logging** | Predictions flow from serving → Redis stream → drift detector. Decouples the services and provides a reliable buffer. Falls back to in-memory when Redis is unavailable. |
| **PSI + KS for drift detection** | Population Stability Index catches gradual distribution shifts; Kolmogorov-Smirnov catches abrupt changes. Together they cover the full drift spectrum. |

### Outcome

- Drift detected and alerted within **5 minutes** of distribution shift
- Automated retraining trigger via configurable webhook
- Zero-downtime model serving with health checks and metrics

---

## Architecture

```
┌─────────────┐     ┌──────────┐     ┌─────────────────┐
│  Training    │────▶│  MLflow  │────▶│ Model Registry  │
│  (Python)    │     │  Server  │     │  (Artifacts)    │
└─────────────┘     └──────────┘     └────────┬────────┘
                                              │
                                              ▼
┌─────────────┐     ┌──────────────────────────────────┐
│   Client     │────▶│  FastAPI Serving (Python)        │
│   Request    │◀────│  POST /predict  GET /health      │
└─────────────┘     └──────────┬───────────────────────┘
                               │ logs predictions
                               ▼
                    ┌──────────────────────┐
                    │   Redis Stream       │
                    └──────────┬───────────┘
                               │ reads predictions
                               ▼
                    ┌──────────────────────┐     ┌──────────────┐
                    │  Go Drift Monitor    │────▶│  Webhook     │
                    │  (PSI + KS stats)    │     │  Alerts      │
                    └──────────────────────┘     └──────────────┘
```

---

## Quick Start

```bash
# Start all services (MLflow, Postgres, Redis, Serving, Drift Monitor)
make up

# Train the fraud detection model
make train

# Run predictions against the serving endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"amount": 150.0, "time": 3600, "v1": -1.35, "v2": 1.19, "v3": 0.27, "v4": 0.16, "v5": 0.45, "v6": 0.06, "v7": -0.68, "v8": 0.09, "v9": -0.25, "v10": -0.17}'

# Tear everything down
make down
```

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Model Training | Python, scikit-learn | RandomForest fraud classifier |
| Experiment Tracking | MLflow | Parameter/metric/artifact logging |
| Model Serving | FastAPI, Uvicorn | REST API for predictions |
| Drift Detection | Go (stdlib only) | PSI, KS, KL divergence statistics |
| Prediction Buffer | Redis Streams | Decouple serving from monitoring |
| Orchestration | Docker Compose | Multi-service local deployment |
| Metadata Store | PostgreSQL | MLflow backend storage |
| CI/CD | GitHub Actions | Lint, test, build on every push |

---

## Structure

```
ml-production-pipeline/
├── training/               # Model training pipeline
│   ├── train.py            # Training script with MLflow tracking
│   ├── download_data.py    # Synthetic data generation
│   └── requirements.txt
├── serving/                # FastAPI prediction service
│   ├── app.py              # REST API with /predict, /health, /metrics
│   ├── Dockerfile
│   └── requirements.txt
├── drift-detector/         # Go drift monitoring service
│   ├── cmd/detector/       # Entry point
│   ├── internal/
│   │   ├── config/         # Environment-based configuration
│   │   ├── detector/       # PSI, KS, monitoring loop
│   │   └── alerting/       # Webhook notifications
│   ├── reference_distributions.json
│   ├── Dockerfile
│   └── go.mod
├── docker-compose.yaml     # Full stack orchestration
├── .github/workflows/      # CI pipelines
├── Makefile                # Developer commands
└── README.md
```

---

## Future Enhancements

- **Model A/B testing** — serve multiple model versions and compare live performance
- **Prometheus + Grafana** — dashboards for prediction latency, drift scores, model accuracy
- **Automated retraining pipeline** — trigger retrain on drift detection via webhook → CI
- **Feature store integration** — centralized feature management with Feast
- **Kubernetes deployment** — Helm charts for production-scale serving
- **Shadow mode** — run new models alongside production without affecting users

---

## License

MIT — see [LICENSE](LICENSE) for details.
