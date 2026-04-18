package audio

import (
	"fmt"
	"math"
)

// buildMelFilterbank constructs a [nMelBins][nFreqs] triangular mel filterbank
// matrix, where nFreqs = nfft/2 + 1.
//
// melNorm == "" or "htk" → HTK mel scale (NeMo default).
// melNorm == "slaney"    → Slaney/librosa piecewise mel scale.
//
// The filterbank matches librosa.filters.mel with the corresponding norm setting.
func buildMelFilterbank(nMelBins, nfft int, sampleRate, fMin, fMax float64, melNorm string) ([][]float64, error) {
	if nMelBins <= 0 {
		return nil, fmt.Errorf("nMelBins must be > 0")
	}
	nFreqs := nfft/2 + 1

	useSlaney := melNorm == "slaney"

	// Build nMelBins+2 mel-spaced center frequencies in Hz.
	melMin := hzToMel(fMin, useSlaney)
	melMax := hzToMel(fMax, useSlaney)

	// Linear spacing in mel domain.
	centers := make([]float64, nMelBins+2)
	for i := range centers {
		m := melMin + float64(i)*(melMax-melMin)/float64(nMelBins+1)
		centers[i] = melToHz(m, useSlaney)
	}

	// FFT frequency bins (linear Hz).
	fftFreqs := make([]float64, nFreqs)
	for f := 0; f < nFreqs; f++ {
		fftFreqs[f] = float64(f) * sampleRate / float64(nfft)
	}

	fb := make([][]float64, nMelBins)
	for b := 0; b < nMelBins; b++ {
		fb[b] = make([]float64, nFreqs)
		lo := centers[b]
		mid := centers[b+1]
		hi := centers[b+2]

		for f, freq := range fftFreqs {
			var w float64
			if freq >= lo && freq <= mid {
				w = (freq - lo) / (mid - lo)
			} else if freq > mid && freq <= hi {
				w = (hi - freq) / (hi - mid)
			}

			// Slaney norm: scale each filter to unit area.
			if useSlaney && w > 0 {
				w *= 2.0 / (hi - lo)
			}

			fb[b][f] = w
		}
	}
	return fb, nil
}

// hzToMel converts Hz to mel scale.
func hzToMel(hz float64, slaney bool) float64 {
	if slaney {
		return hzToMelSlaney(hz)
	}
	// HTK formula.
	return 2595.0 * math.Log10(1.0+hz/700.0)
}

// melToHz converts mel to Hz.
func melToHz(mel float64, slaney bool) float64 {
	if slaney {
		return melToHzSlaney(mel)
	}
	// HTK formula.
	return 700.0 * (math.Pow(10.0, mel/2595.0) - 1.0)
}

// Slaney (librosa default) piecewise mel scale.
// Linear below 1000 Hz (f_sp = 200/3 Hz/mel), log above.
const (
	slaneySp     = 200.0 / 3.0 // Hz per mel in linear region
	slaneyMinLog = 1000.0       // Hz where log region starts
	// slaneyLogStep = 27.0 / ln(6.4) — computed at init time below
)

// slaneyLogStep = 27 / log(6.4), used in the log portion of Slaney mel scale.
// Computed as a var because math.Log is not a constant expression.
var slaneyLogStep = 27.0 / math.Log(6.4)

func hzToMelSlaney(hz float64) float64 {
	minLogMel := slaneyMinLog / slaneySp
	if hz < slaneyMinLog {
		return hz / slaneySp
	}
	return minLogMel + math.Log(hz/slaneyMinLog)*slaneyLogStep
}

func melToHzSlaney(mel float64) float64 {
	minLogMel := slaneyMinLog / slaneySp
	if mel < minLogMel {
		return mel * slaneySp
	}
	return slaneyMinLog * math.Exp((mel-minLogMel)/slaneyLogStep)
}
