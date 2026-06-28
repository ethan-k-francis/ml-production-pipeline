// Package alerting — webhook.go
//
// Sends drift alerts to a configurable webhook URL when drift is detected.
// Designed to be generic: works with Slack incoming webhooks, Discord webhooks,
// PagerDuty events API, or any custom HTTP endpoint that accepts JSON POST.
//
// Why webhooks instead of a specific integration?
// - Vendor-agnostic: users choose their own notification platform
// - Simple: just HTTP POST with a JSON body — no SDKs or auth flows
// - Composable: webhook receivers can fan out to multiple destinations
package alerting

import (
	"bytes"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"time"
)

// WebhookClient sends drift alert notifications via HTTP POST.
type WebhookClient struct {
	// url is the webhook endpoint to POST alerts to
	url string

	// client is a shared HTTP client with timeouts configured.
	// Reusing a single client enables connection pooling.
	client *http.Client

	logger *slog.Logger
}

// NewWebhookClient creates a webhook client for the given URL.
// Returns nil if the URL is empty (alerting disabled).
func NewWebhookClient(url string) *WebhookClient {
	if url == "" {
		return nil
	}

	return &WebhookClient{
		url: url,
		client: &http.Client{
			// 10-second timeout prevents hanging if the webhook endpoint is slow.
			// Drift alerts aren't latency-critical, so this is generous.
			Timeout: 10 * time.Second,
		},
		logger: slog.New(slog.NewJSONHandler(os.Stdout, nil)),
	}
}

// SendAlert posts a drift alert summary to the webhook endpoint.
// The payload is the JSON-serialized AlertSummary.
//
// Errors are logged but not returned — alert delivery failure should not
// crash the drift detector or block the monitoring loop.
func (w *WebhookClient) SendAlert(summary AlertSummary) {
	if w == nil {
		return
	}

	// Serialize the alert summary to JSON
	payload, err := summary.ToJSON()
	if err != nil {
		w.logger.Error("failed to serialize alert", "error", err.Error())
		return
	}

	w.logger.Info("sending drift alert",
		"url", w.url,
		"drifted_features", summary.DriftedCount,
		"max_psi", fmt.Sprintf("%.4f", summary.MaxPSI),
		"severity", summary.MaxSeverity,
	)

	// POST the JSON payload to the webhook endpoint
	req, err := http.NewRequest(http.MethodPost, w.url, bytes.NewReader(payload))
	if err != nil {
		w.logger.Error("failed to create webhook request", "error", err.Error())
		return
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "ml-drift-detector/1.0")

	resp, err := w.client.Do(req)
	if err != nil {
		w.logger.Error("webhook request failed", "error", err.Error())
		return
	}
	defer resp.Body.Close()

	// Read and discard the response body to allow connection reuse.
	// HTTP/1.1 connection pooling requires the body to be fully consumed.
	_, _ = io.Copy(io.Discard, resp.Body)

	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		w.logger.Info("drift alert sent successfully", "status", resp.StatusCode)
	} else {
		w.logger.Warn("webhook returned non-2xx status",
			"status", resp.StatusCode,
			"url", w.url,
		)
	}
}

// SendText posts a plain-text formatted alert. Useful for platforms
// that prefer text over structured JSON (e.g., some Slack webhook formats).
func (w *WebhookClient) SendText(summary AlertSummary) {
	if w == nil {
		return
	}

	text := summary.ToText()
	payload := []byte(fmt.Sprintf(`{"text": %q}`, text))

	req, err := http.NewRequest(http.MethodPost, w.url, bytes.NewReader(payload))
	if err != nil {
		w.logger.Error("failed to create text webhook request", "error", err.Error())
		return
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "ml-drift-detector/1.0")

	resp, err := w.client.Do(req)
	if err != nil {
		w.logger.Error("text webhook request failed", "error", err.Error())
		return
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, resp.Body)
}
