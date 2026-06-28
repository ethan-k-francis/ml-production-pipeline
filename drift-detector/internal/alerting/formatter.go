// Package alerting provides drift alert formatting and delivery.
//
// When the drift detector identifies a significant distribution shift,
// this package formats the alert into both structured JSON (for machine
// consumption) and human-readable text (for Slack/Discord/email).
package alerting

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

// DriftAlert contains all information about a detected drift event.
// This struct is serialized to JSON for webhook payloads and formatted
// to text for human-readable notifications.
type DriftAlert struct {
	// Timestamp of when drift was detected (ISO 8601)
	Timestamp string `json:"timestamp"`

	// Feature is the name of the drifted feature (e.g., "v1", "amount")
	Feature string `json:"feature"`

	// PSI is the Population Stability Index value that triggered the alert
	PSI float64 `json:"psi"`

	// KS is the Kolmogorov-Smirnov test statistic
	KS float64 `json:"ks"`

	// KL is the Kullback-Leibler divergence
	KL float64 `json:"kl"`

	// Threshold is the PSI threshold that was exceeded
	Threshold float64 `json:"threshold"`

	// Severity classifies the drift: "moderate", "significant", "critical"
	Severity string `json:"severity"`

	// MeanShift is the difference between current and reference means
	MeanShift float64 `json:"mean_shift"`

	// RefMean is the reference (training) distribution mean
	RefMean float64 `json:"ref_mean"`

	// CurrMean is the current (production) distribution mean
	CurrMean float64 `json:"curr_mean"`
}

// AlertSummary aggregates multiple feature drift alerts into a single notification.
// Sent when one or more features exceed the drift threshold in a single check.
type AlertSummary struct {
	Timestamp      string       `json:"timestamp"`
	TotalFeatures  int          `json:"total_features"`
	DriftedCount   int          `json:"drifted_count"`
	MaxPSI         float64      `json:"max_psi"`
	MaxSeverity    string       `json:"max_severity"`
	DriftedAlerts  []DriftAlert `json:"drifted_alerts"`
}

// NewDriftAlert creates an alert from drift detection results.
func NewDriftAlert(
	feature string,
	psi, ks, kl, threshold, meanShift, refMean, currMean float64,
	severity string,
) DriftAlert {
	return DriftAlert{
		Timestamp: time.Now().UTC().Format(time.RFC3339),
		Feature:   feature,
		PSI:       psi,
		KS:        ks,
		KL:        kl,
		Threshold: threshold,
		Severity:  severity,
		MeanShift: meanShift,
		RefMean:   refMean,
		CurrMean:  currMean,
	}
}

// NewAlertSummary aggregates multiple drift alerts into a summary.
func NewAlertSummary(alerts []DriftAlert, totalFeatures int) AlertSummary {
	summary := AlertSummary{
		Timestamp:     time.Now().UTC().Format(time.RFC3339),
		TotalFeatures: totalFeatures,
		DriftedCount:  len(alerts),
		DriftedAlerts: alerts,
	}

	// Find the maximum PSI and severity across all drifted features.
	// The worst feature determines the overall alert severity.
	severityRank := map[string]int{"none": 0, "moderate": 1, "significant": 2, "critical": 3}
	maxRank := 0

	for _, a := range alerts {
		if a.PSI > summary.MaxPSI {
			summary.MaxPSI = a.PSI
		}
		if rank, ok := severityRank[a.Severity]; ok && rank > maxRank {
			maxRank = rank
			summary.MaxSeverity = a.Severity
		}
	}

	return summary
}

// ToJSON serializes an alert summary to indented JSON.
// Used as the webhook POST body for machine-readable consumers.
func (s AlertSummary) ToJSON() ([]byte, error) {
	return json.MarshalIndent(s, "", "  ")
}

// ToText formats an alert summary as human-readable text.
// Designed for readability in Slack, Discord, or email notifications.
func (s AlertSummary) ToText() string {
	var b strings.Builder

	// Header with severity emoji mapping (universal, not platform-specific)
	severityIcon := map[string]string{
		"moderate":    "[!]",
		"significant": "[!!]",
		"critical":    "[!!!]",
	}
	icon := severityIcon[s.MaxSeverity]
	if icon == "" {
		icon = "[i]"
	}

	b.WriteString(fmt.Sprintf("%s DATA DRIFT DETECTED — %s\n", icon, strings.ToUpper(s.MaxSeverity)))
	b.WriteString(fmt.Sprintf("Time: %s\n", s.Timestamp))
	b.WriteString(fmt.Sprintf("Drifted: %d / %d features\n", s.DriftedCount, s.TotalFeatures))
	b.WriteString(fmt.Sprintf("Max PSI: %.4f\n\n", s.MaxPSI))

	// Per-feature breakdown
	for _, a := range s.DriftedAlerts {
		b.WriteString(fmt.Sprintf("  Feature: %s\n", a.Feature))
		b.WriteString(fmt.Sprintf("    PSI: %.4f (threshold: %.4f)\n", a.PSI, a.Threshold))
		b.WriteString(fmt.Sprintf("    KS:  %.4f\n", a.KS))
		b.WriteString(fmt.Sprintf("    Mean shift: %.4f (ref: %.4f → curr: %.4f)\n", a.MeanShift, a.RefMean, a.CurrMean))
		b.WriteString(fmt.Sprintf("    Severity: %s\n\n", a.Severity))
	}

	b.WriteString("Action: Investigate feature distributions and consider model retraining.\n")

	return b.String()
}
