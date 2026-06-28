# ML Production Pipeline

**Train a model, serve predictions, and get warned when real-world data drifts from what the model learned**

This is an end-to-end **machine learning (ML) in production** demo. You train a fraud-detection model, expose it as a web Application Programming Interface (API), and run a separate monitor that watches incoming data and alerts you when patterns change enough that predictions may become unreliable.

Built for learning: every service runs locally with Docker Compose.

---

## The problem in plain English

A model trained on last year's data assumes the world looks a certain way. When reality shifts — new fraud patterns, seasonal spending changes, different customer behavior — the model keeps making predictions, but accuracy quietly drops. Nobody notices until money is lost or users complain.

This project adds an automated **"is the data still familiar?"** check and sends alerts when it isn't.

---

## What you'll learn

| Concept | Plain English |
|---|---|
| **Training** | Teach a model from example data |
| **Serving** | Run the model behind an Application Programming Interface (API) so apps can ask for predictions |
| **Experiment tracking** | Log what you tried (settings, scores, model files) so you can compare runs |
| **Drift** | Incoming data looks different from training data |
| **Population Stability Index (PSI)** | A score for gradual drift — how much the data distribution has shifted over time |
| **Kolmogorov–Smirnov test (KS test)** | A score for sudden distribution shifts |
| **Webhook alert** | Hypertext Transfer Protocol (HTTP) POST to Slack, Discord, or any URL when drift is detected |

---

## How it works

```
┌─────────────┐     ┌──────────┐     ┌─────────────────┐
│  Training    │────▶│  MLflow  │────▶│ Model Registry  │
│  (Python)    │     │  Server  │     │  (saved models) │
└─────────────┘     └──────────┘     └────────┬────────┘
                                              │
                                              ▼
┌─────────────┐     ┌──────────────────────────────────┐
│   Client     │────▶│  FastAPI Serving (Python)        │
│   Request    │◀────│  POST /predict  GET /health      │
└─────────────┘     └──────────┬───────────────────────┘
                               │ logs each prediction
                               ▼
                    ┌──────────────────────┐
                    │   Redis Stream       │
                    │   (message buffer)   │
                    └──────────┬───────────┘
                               │ reads predictions
                               ▼
                    ┌──────────────────────┐     ┌──────────────┐
                    │  Go Drift Monitor    │────▶│  Webhook     │
                    │  compares to baseline│     │  Alerts      │
                    └──────────────────────┘     └──────────────┘
```

**Step by step:**

1. **Train** a fraud classifier and save it via MLflow.
2. **Serve** predictions through a FastAPI endpoint (`POST /predict`).
3. **Log** each prediction to Redis so serving and monitoring stay independent.
4. **Monitor** with a Go service that compares live feature values to a saved baseline using Population Stability Index (PSI) and Kolmogorov–Smirnov (KS) statistics.
5. **Alert** via webhook when drift scores cross thresholds (typically within ~5 minutes).

---

## Quick start

```bash
# Start all services (MLflow, Postgres, Redis, Application Programming Interface (API), drift monitor)
make up

# Train the fraud detection model
make train

# Ask for a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"amount": 150.0, "time": 3600, "v1": -1.35, "v2": 1.19, "v3": 0.27, "v4": 0.16, "v5": 0.45, "v6": 0.06, "v7": -0.68, "v8": 0.09, "v9": -0.25, "v10": -0.17}'

# Stop everything
make down
```

---

## What's inside

| Piece | Technology | What it does |
|---|---|---|
| Training | Python, scikit-learn | Builds a RandomForest fraud classifier |
| Experiment tracking | MLflow | Records runs, metrics, and model artifacts |
| Application Programming Interface (API) | FastAPI | Serves predictions over Hypertext Transfer Protocol (HTTP) |
| Drift monitor | Go (stdlib only) | Computes Population Stability Index (PSI) and Kolmogorov–Smirnov (KS) statistics on live data |
| Buffer | Redis Streams | Holds predictions between the API and monitor |
| Orchestration | Docker Compose | Runs the full stack locally |
| Database | PostgreSQL | Stores MLflow metadata |

---

## Project layout

```
ml-production-pipeline/
├── training/               # Train script + data download
├── serving/                # FastAPI app (/predict, /health, /metrics)
├── drift-detector/         # Go drift monitor + alerting
├── docker-compose.yaml
├── Makefile
└── .github/workflows/      # Continuous Integration (CI): lint, test, build
```

---

## Design choices (for the curious)

| Decision | Why |
|---|---|
| Go for drift detection | Fast number-crunching on a stream of predictions |
| FastAPI for serving | Simple async Application Programming Interface (API) with automatic docs at `/docs` |
| Redis between services | API stays fast even if the monitor is briefly slow |
| Population Stability Index (PSI) + Kolmogorov–Smirnov (KS) together | PSI catches slow drift; KS catches sudden shifts |

---

## Ideas for extending this

- A/B test two model versions side by side
- Grafana dashboards for latency and drift scores
- Auto-retrain when drift alert fires
- Deploy on Kubernetes (K8s) for production scale

---

## License

MIT — see [LICENSE](LICENSE).
