// Package config provides environment-based configuration for the drift detector.
//
// All settings come from environment variables with sensible defaults.
// This pattern (env vars → struct) follows the 12-Factor App methodology:
// config lives in the environment, not in code — making the service
// portable across local dev, Docker, and Kubernetes without code changes.
package config

import (
	"fmt"
	"os"
	"strconv"
	"time"
)

// Config holds all runtime configuration for the drift detection service.
// Each field maps to an environment variable with a default value.
type Config struct {
	// ServingURL is the base URL of the FastAPI serving endpoint.
	// Used to fetch recent predictions for distribution analysis.
	ServingURL string

	// CheckInterval controls how often the detector polls for new predictions.
	// Shorter intervals catch drift faster but increase load on the serving API.
	CheckInterval time.Duration

	// DriftThreshold is the PSI value above which drift is flagged.
	// PSI < 0.1 = no drift, 0.1-0.2 = moderate, > 0.2 = significant drift.
	// Industry standard thresholds from credit risk modeling literature.
	DriftThreshold float64

	// WebhookURL is the endpoint to POST alerts when drift is detected.
	// Supports generic webhooks (Slack, Discord, PagerDuty, custom).
	WebhookURL string

	// WindowSize is the number of recent predictions to analyze per check.
	// Larger windows give more stable statistics but delay drift detection.
	WindowSize int

	// ReferenceDistPath is the path to the JSON file containing training
	// data distributions used as the baseline for drift comparison.
	ReferenceDistPath string
}

// Load reads configuration from environment variables, falling back to defaults.
// Returns an error only if a value exists but can't be parsed (e.g., non-numeric interval).
func Load() (*Config, error) {
	cfg := &Config{
		ServingURL:        getEnv("SERVING_URL", "http://localhost:8000"),
		WebhookURL:        getEnv("WEBHOOK_URL", ""),
		ReferenceDistPath: getEnv("REFERENCE_DIST_PATH", "reference_distributions.json"),
	}

	// Parse check interval from seconds string → time.Duration
	intervalSec, err := strconv.Atoi(getEnv("CHECK_INTERVAL", "300"))
	if err != nil {
		return nil, fmt.Errorf("invalid CHECK_INTERVAL: %w", err)
	}
	cfg.CheckInterval = time.Duration(intervalSec) * time.Second

	// Parse drift threshold (float)
	threshold, err := strconv.ParseFloat(getEnv("DRIFT_THRESHOLD", "0.2"), 64)
	if err != nil {
		return nil, fmt.Errorf("invalid DRIFT_THRESHOLD: %w", err)
	}
	cfg.DriftThreshold = threshold

	// Parse window size (int)
	windowSize, err := strconv.Atoi(getEnv("WINDOW_SIZE", "500"))
	if err != nil {
		return nil, fmt.Errorf("invalid WINDOW_SIZE: %w", err)
	}
	cfg.WindowSize = windowSize

	return cfg, nil
}

// getEnv returns the environment variable value or a default if unset/empty.
func getEnv(key, defaultVal string) string {
	if val := os.Getenv(key); val != "" {
		return val
	}
	return defaultVal
}
