package audio

import (
	"fmt"
	"math"
)

// NormType controls how log-mel features are normalized.
type NormType string

const (
	NormPerFeature  NormType = "per_feature"   // normalize each mel bin independently
	NormAllFeatures NormType = "all_features"  // normalize over all bins jointly
	NormNone        NormType = "none"
)

// Config mirrors NeMo's AudioToMelSpectrogramPreprocessor parameters.
type Config struct {
	SampleRate    int      // e.g. 16000
	WindowSizeSec float64  // e.g. 0.02  → 320 samples at 16 kHz
	WindowStride  float64  // e.g. 0.01  → 160 samples at 16 kHz
	NFft          int      // 0 = auto (next power-of-2 ≥ window samples)
	NMelBins      int      // e.g. 80 or 128
	FMin          float64  // e.g. 0.0
	FMax          float64  // 0.0 = SampleRate/2
	// MelNorm: "slaney" (NeMo default) uses Slaney Hz scale + area normalization.
	// "" or "htk" uses HTK mel scale, no area normalization.
	MelNorm       string
	LogOffset     float64  // added before log to avoid log(0), e.g. 1e-6
	NormType      NormType // normalization applied after log
	Dither        float64  // additive noise amplitude (0 = off)
	Preemph       float64  // pre-emphasis coefficient (0 = off, NeMo default 0.97)
	// Center: if true (NeMo default, exact_pad=false), pad n_fft/2 zeros on each
	// side before STFT, matching torch.stft center=True behavior.
	Center        bool
}

// DefaultConfig returns parameters that match NeMo's parakeet/canary defaults.
func DefaultConfig() Config {
	return Config{
		SampleRate:    16000,
		WindowSizeSec: 0.02,
		WindowStride:  0.01,
		NMelBins:      80,
		FMin:          0.0,
		FMax:          0.0,       // → 8000
		MelNorm:       "slaney", // NeMo default
		LogOffset:     1e-6,
		NormType:      NormPerFeature,
		Preemph:       0.97,
		Center:        true, // NeMo default (exact_pad=False → center=True)
	}
}

// Preprocessor extracts log-mel spectrograms from raw PCM audio.
type Preprocessor struct {
	cfg       Config
	winSamples int
	hopSamples int
	nfft       int
	fmax       float64
	filterbank [][]float64 // [nMelBins][nfft/2+1]
}

// New constructs a Preprocessor and pre-computes the mel filterbank.
func New(cfg Config) (*Preprocessor, error) {
	if cfg.SampleRate <= 0 {
		return nil, fmt.Errorf("SampleRate must be > 0")
	}
	if cfg.NMelBins <= 0 {
		return nil, fmt.Errorf("NMelBins must be > 0")
	}

	winSamples := int(math.Round(cfg.WindowSizeSec * float64(cfg.SampleRate)))
	hopSamples := int(math.Round(cfg.WindowStride * float64(cfg.SampleRate)))

	nfft := cfg.NFft
	if nfft == 0 {
		nfft = nextPow2(winSamples)
	}

	fmax := cfg.FMax
	if fmax <= 0 {
		fmax = float64(cfg.SampleRate) / 2.0
	}

	fb, err := buildMelFilterbank(cfg.NMelBins, nfft, float64(cfg.SampleRate), cfg.FMin, fmax, cfg.MelNorm)
	if err != nil {
		return nil, err
	}

	return &Preprocessor{
		cfg:        cfg,
		winSamples: winSamples,
		hopSamples: hopSamples,
		nfft:       nfft,
		fmax:       fmax,
		filterbank: fb,
	}, nil
}

// LogMel converts mono 16 kHz PCM to a log-mel spectrogram.
// Returns a flat []float32 in row-major order [nMelBins × nFrames],
// and nFrames.
func (p *Preprocessor) LogMel(pcm []float32) ([]float32, int, error) {
	if len(pcm) == 0 {
		return nil, 0, fmt.Errorf("empty PCM input")
	}

	sig := make([]float64, len(pcm))
	for i, v := range pcm {
		sig[i] = float64(v)
	}

	// Pre-emphasis: y[n] = x[n] - coeff*x[n-1]
	if p.cfg.Preemph > 0 {
		applyPreemphasis(sig, p.cfg.Preemph)
	}

	// center=True: pad n_fft/2 zeros on each side, matching torch.stft center=True.
	if p.cfg.Center {
		pad := p.nfft / 2
		padded := make([]float64, pad+len(sig)+pad)
		copy(padded[pad:], sig)
		sig = padded
	}

	// STFT → power spectrum [nFrames][nfft/2+1]
	powerSpec, nFrames := stftPower(sig, p.winSamples, p.hopSamples, p.nfft)

	// Mel filterbank: [nMelBins][nFrames]
	nBins := p.cfg.NMelBins
	melSpec := make([]float64, nBins*nFrames)
	for b := 0; b < nBins; b++ {
		for t := 0; t < nFrames; t++ {
			var s float64
			for f, w := range p.filterbank[b] {
				s += w * powerSpec[t][f]
			}
			melSpec[b*nFrames+t] = math.Log(s + p.cfg.LogOffset)
		}
	}

	// Normalization
	switch p.cfg.NormType {
	case NormPerFeature:
		normalizePerFeature(melSpec, nBins, nFrames)
	case NormAllFeatures:
		normalizeAllFeatures(melSpec)
	}

	out := make([]float32, len(melSpec))
	for i, v := range melSpec {
		out[i] = float32(v)
	}
	return out, nFrames, nil
}

// NMelBins returns the number of mel frequency bins.
func (p *Preprocessor) NMelBins() int { return p.cfg.NMelBins }

// NFrames returns the number of output frames for a given number of PCM samples,
// without allocating or computing the full spectrogram.
func (p *Preprocessor) NFrames(nSamples int) int {
	if nSamples <= 0 {
		return 0
	}
	effective := nSamples
	if p.cfg.Center {
		// center=True pads n_fft/2 on each side, adding n_fft total.
		effective += p.nfft
	}
	if effective < p.nfft {
		return 0
	}
	return 1 + (effective-p.nfft)/p.hopSamples
}

func nextPow2(n int) int {
	p := 1
	for p < n {
		p <<= 1
	}
	return p
}
