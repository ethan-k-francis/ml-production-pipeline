// Package detector implements statistical methods for detecting data drift.
//
// Data drift occurs when the distribution of incoming production data diverges
// from the training data distribution — causing model predictions to silently
// degrade. We use three complementary statistical tests:
//
//   - PSI (Population Stability Index): measures overall distribution shift
//   - KS (Kolmogorov-Smirnov): detects the maximum pointwise divergence
//   - KL (Kullback-Leibler divergence): information-theoretic distance
//
// Together, PSI catches gradual drift, KS catches abrupt shifts, and KL
// quantifies how much information is "lost" if you model current data
// with the training distribution.
package detector

import (
	"math"
	"sort"
)

// FeatureDistribution represents a feature's reference distribution from training.
// These are loaded from JSON (saved by the training pipeline) and used as the
// baseline for all drift comparisons.
type FeatureDistribution struct {
	Mean      float64   `json:"mean"`
	Std       float64   `json:"std"`
	Min       float64   `json:"min"`
	Max       float64   `json:"max"`
	Histogram Histogram `json:"histogram"`
}

// Histogram stores a density histogram with bin edges.
// Counts are probability densities (sum × bin_width ≈ 1), not raw counts.
type Histogram struct {
	Counts   []float64 `json:"counts"`
	BinEdges []float64 `json:"bin_edges"`
}

// DriftResult holds the output of a drift check for a single feature.
type DriftResult struct {
	Feature    string  `json:"feature"`
	PSI        float64 `json:"psi"`
	KS         float64 `json:"ks"`
	KL         float64 `json:"kl"`
	IsDrifted  bool    `json:"is_drifted"`
	Severity   string  `json:"severity"`
	RefMean    float64 `json:"ref_mean"`
	CurrMean   float64 `json:"curr_mean"`
	MeanShift  float64 `json:"mean_shift"`
}

// PSI computes the Population Stability Index between two distributions.
//
// PSI = Σ (actual_i - expected_i) × ln(actual_i / expected_i)
//
// Interpretation (industry standard from credit risk modeling):
//   - PSI < 0.1:  no significant drift
//   - 0.1 ≤ PSI < 0.2:  moderate drift — investigate
//   - PSI ≥ 0.2:  significant drift — action required
//
// The epsilon prevents log(0) when a bin is empty. Using 1e-6 is standard;
// it adds negligible noise but avoids NaN/Inf in the calculation.
func PSI(expected, actual []float64) float64 {
	const epsilon = 1e-6

	if len(expected) != len(actual) {
		return 0.0
	}

	// Normalize to proportions (so they sum to 1)
	expSum := sum(expected)
	actSum := sum(actual)

	if expSum == 0 || actSum == 0 {
		return 0.0
	}

	psi := 0.0
	for i := range expected {
		// Add epsilon to both numerator and denominator to avoid division by
		// zero or log(0). This is a standard smoothing technique.
		e := (expected[i] / expSum) + epsilon
		a := (actual[i] / actSum) + epsilon
		psi += (a - e) * math.Log(a/e)
	}

	return psi
}

// KSStatistic computes the Kolmogorov-Smirnov test statistic.
//
// KS = max |F_ref(x) - F_curr(x)| over all x
//
// This measures the maximum vertical distance between two empirical CDFs.
// Unlike PSI which summarizes overall shift, KS pinpoints the single largest
// divergence — good for catching sudden, localized distribution changes.
//
// Inputs are raw sample values (not histograms), sorted internally.
func KSStatistic(reference, current []float64) float64 {
	if len(reference) == 0 || len(current) == 0 {
		return 0.0
	}

	// Sort both samples — required to build empirical CDFs
	refSorted := make([]float64, len(reference))
	curSorted := make([]float64, len(current))
	copy(refSorted, reference)
	copy(curSorted, current)
	sort.Float64s(refSorted)
	sort.Float64s(curSorted)

	// Merge both sorted arrays and walk through, tracking CDF positions.
	// At each point, the CDF step is 1/n for its respective sample.
	nRef := float64(len(refSorted))
	nCur := float64(len(curSorted))

	var i, j int
	var cdfRef, cdfCur float64
	maxDiff := 0.0

	for i < len(refSorted) && j < len(curSorted) {
		if refSorted[i] <= curSorted[j] {
			cdfRef = float64(i+1) / nRef
			i++
		} else {
			cdfCur = float64(j+1) / nCur
			j++
		}

		diff := math.Abs(cdfRef - cdfCur)
		if diff > maxDiff {
			maxDiff = diff
		}
	}

	// Process remaining elements in whichever array wasn't exhausted
	for i < len(refSorted) {
		cdfRef = float64(i+1) / nRef
		diff := math.Abs(cdfRef - cdfCur)
		if diff > maxDiff {
			maxDiff = diff
		}
		i++
	}
	for j < len(curSorted) {
		cdfCur = float64(j+1) / nCur
		diff := math.Abs(cdfRef - cdfCur)
		if diff > maxDiff {
			maxDiff = diff
		}
		j++
	}

	return maxDiff
}

// KLDivergence computes the Kullback-Leibler divergence: KL(P || Q).
//
// KL = Σ P(i) × ln(P(i) / Q(i))
//
// Information-theoretic interpretation: KL measures the "extra bits" needed
// if you encode data from distribution P using a code optimized for Q.
// KL = 0 means the distributions are identical.
//
// Note: KL is asymmetric (KL(P||Q) ≠ KL(Q||P)). We compute KL(current || reference)
// which answers: "how surprised would the training distribution be by current data?"
func KLDivergence(p, q []float64) float64 {
	const epsilon = 1e-6

	if len(p) != len(q) {
		return 0.0
	}

	pSum := sum(p)
	qSum := sum(q)

	if pSum == 0 || qSum == 0 {
		return 0.0
	}

	kl := 0.0
	for i := range p {
		pi := (p[i] / pSum) + epsilon
		qi := (q[i] / qSum) + epsilon
		kl += pi * math.Log(pi/qi)
	}

	return kl
}

// ClassifySeverity maps a PSI value to a human-readable severity level.
// Thresholds from industry practice in credit risk and financial modeling.
func ClassifySeverity(psi float64) string {
	switch {
	case psi < 0.1:
		return "none"
	case psi < 0.2:
		return "moderate"
	case psi < 0.5:
		return "significant"
	default:
		return "critical"
	}
}

// Histogramize bins raw values into a histogram matching reference bin edges.
// This lets us compare current data directly against the reference histogram
// using the same bins — essential for accurate PSI calculation.
func Histogramize(values []float64, binEdges []float64) []float64 {
	nBins := len(binEdges) - 1
	if nBins <= 0 {
		return nil
	}

	counts := make([]float64, nBins)
	for _, v := range values {
		// Find which bin this value falls into via binary search.
		// Values outside the range go into the first or last bin.
		idx := sort.SearchFloat64s(binEdges, v) - 1
		if idx < 0 {
			idx = 0
		}
		if idx >= nBins {
			idx = nBins - 1
		}
		counts[idx]++
	}

	// Normalize to density (proportions) so PSI comparison is scale-independent
	total := sum(counts)
	if total > 0 {
		for i := range counts {
			counts[i] /= total
		}
	}

	return counts
}

// sum returns the total of a float64 slice. Utility for normalization.
func sum(vals []float64) float64 {
	total := 0.0
	for _, v := range vals {
		total += v
	}
	return total
}

// mean returns the arithmetic mean of a float64 slice.
func mean(vals []float64) float64 {
	if len(vals) == 0 {
		return 0.0
	}
	return sum(vals) / float64(len(vals))
}
