package audio

import "math"

// normalizePerFeature applies per-mel-bin mean/variance normalization in-place.
// melSpec is row-major [nBins × nFrames].
// Matches NeMo's "per_feature" normalize_type.
func normalizePerFeature(melSpec []float64, nBins, nFrames int) {
	for b := 0; b < nBins; b++ {
		row := melSpec[b*nFrames : (b+1)*nFrames]
		mean, std := meanStd(row)
		if std < 1e-10 {
			std = 1e-10
		}
		for i := range row {
			row[i] = (row[i] - mean) / std
		}
	}
}

// normalizeAllFeatures applies global mean/variance normalization in-place.
// Matches NeMo's "all_features" normalize_type.
func normalizeAllFeatures(melSpec []float64) {
	mean, std := meanStd(melSpec)
	if std < 1e-10 {
		std = 1e-10
	}
	for i := range melSpec {
		melSpec[i] = (melSpec[i] - mean) / std
	}
}

func meanStd(v []float64) (mean, std float64) {
	if len(v) == 0 {
		return 0, 0
	}
	n := float64(len(v))
	for _, x := range v {
		mean += x
	}
	mean /= n
	for _, x := range v {
		d := x - mean
		std += d * d
	}
	std = math.Sqrt(std / n)
	return
}
