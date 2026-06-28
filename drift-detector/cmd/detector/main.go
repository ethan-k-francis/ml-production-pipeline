// ML Production Pipeline — Drift Detector Entry Point (Phase 06: with alerting)
//
// Updated to pass the webhook URL to the monitor for drift alerting.
// When WEBHOOK_URL is set, drift alerts are POSTed to that endpoint.
// When empty, drift is still detected and logged — just not pushed externally.
//
// Usage:
//
//	SERVING_URL=http://serving:8000 \
//	DRIFT_THRESHOLD=0.2 \
//	CHECK_INTERVAL=300 \
//	WEBHOOK_URL=https://hooks.slack.com/services/... \
//	./drift-detector
package main

import (
	"log/slog"
	"os"

	"github.com/ethan-k-francis/ml-production-pipeline/drift-detector/internal/config"
	"github.com/ethan-k-francis/ml-production-pipeline/drift-detector/internal/detector"
)

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))

	// Load configuration from environment variables.
	// Fails fast if any required value is malformed.
	cfg, err := config.Load()
	if err != nil {
		logger.Error("failed to load configuration", "error", err.Error())
		os.Exit(1)
	}

	logger.Info("configuration loaded",
		"serving_url", cfg.ServingURL,
		"check_interval", cfg.CheckInterval.String(),
		"drift_threshold", cfg.DriftThreshold,
		"window_size", cfg.WindowSize,
		"webhook_url", cfg.WebhookURL,
		"reference_path", cfg.ReferenceDistPath,
	)

	// Load reference distributions from the training pipeline output.
	ref, err := detector.LoadReference(cfg.ReferenceDistPath)
	if err != nil {
		logger.Error("failed to load reference distributions", "error", err.Error())
		os.Exit(1)
	}

	logger.Info("reference distributions loaded", "features", len(ref))

	// Create the drift monitor with alerting support.
	// The webhook URL is passed to enable alert delivery when drift is detected.
	monitor := detector.NewMonitor(
		cfg.ServingURL,
		cfg.CheckInterval,
		cfg.DriftThreshold,
		cfg.WindowSize,
		ref,
		cfg.WebhookURL,
	)

	monitor.Run()
}
