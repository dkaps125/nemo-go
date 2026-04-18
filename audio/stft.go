package audio

import (
	"math"
)

// stftPower computes a short-time Fourier transform and returns the power
// spectrum (magnitude²) for each frame.
//
// Returns powerSpec[frame][freqBin] where freqBin is in [0, nfft/2].
// Uses a real-valued Hann window and the Cooley-Tukey radix-2 FFT.
func stftPower(sig []float64, winSamples, hopSamples, nfft int) ([][]float64, int) {
	win := hannWindow(winSamples)

	// Frame count uses nfft (not winSamples), matching torch.stft / librosa behavior:
	// a frame is valid only when a full nfft-sized block is available.
	nFrames := 0
	if len(sig) >= nfft {
		nFrames = 1 + (len(sig)-nfft)/hopSamples
	}

	nFreqs := nfft/2 + 1
	power := make([][]float64, nFrames)

	// torch.stft (and librosa) center the win_length window within the n_fft buffer:
	// buf[0..winOffset-1] = 0
	// buf[winOffset..winOffset+winSamples-1] = signal * hann
	// buf[winOffset+winSamples..nfft-1] = 0
	// where winOffset = (nfft - winSamples) / 2
	winOffset := (nfft - winSamples) / 2

	buf := make([]complex128, nfft)

	for t := 0; t < nFrames; t++ {
		start := t * hopSamples

		// Zero the whole buffer, then fill the windowed region.
		// Frame t occupies sig[start .. start+nfft-1].
		// The window sits at buf[winOffset..winOffset+winSamples-1], so the
		// signal read is sig[start+winOffset .. start+winOffset+winSamples-1].
		for i := range buf {
			buf[i] = 0
		}
		sigBase := start + winOffset
		for i := 0; i < winSamples; i++ {
			buf[winOffset+i] = complex(sig[sigBase+i]*win[i], 0)
		}

		fft(buf)

		frame := make([]float64, nFreqs)
		for f := 0; f < nFreqs; f++ {
			re, im := real(buf[f]), imag(buf[f])
			frame[f] = re*re + im*im
		}
		power[t] = frame
	}

	return power, nFrames
}

// hannWindow returns a periodic (not symmetric) Hann window of length n,
// matching torch.hann_window(n, periodic=True) used by NeMo.
func hannWindow(n int) []float64 {
	w := make([]float64, n)
	for i := range w {
		w[i] = 0.5 * (1.0 - math.Cos(2*math.Pi*float64(i)/float64(n)))
	}
	return w
}

// fft performs an in-place Cooley-Tukey radix-2 DIT FFT.
// len(x) must be a power of 2.
func fft(x []complex128) {
	n := len(x)
	if n <= 1 {
		return
	}

	// Bit-reversal permutation.
	j := 0
	for i := 1; i < n; i++ {
		bit := n >> 1
		for ; j&bit != 0; bit >>= 1 {
			j ^= bit
		}
		j ^= bit
		if i < j {
			x[i], x[j] = x[j], x[i]
		}
	}

	// Butterfly stages.
	for length := 2; length <= n; length <<= 1 {
		angle := -2 * math.Pi / float64(length)
		wlen := complex(math.Cos(angle), math.Sin(angle))
		for i := 0; i < n; i += length {
			w := complex(1, 0)
			for k := 0; k < length/2; k++ {
				u := x[i+k]
				v := x[i+k+length/2] * w
				x[i+k] = u + v
				x[i+k+length/2] = u - v
				w *= wlen
			}
		}
	}
}

// applyPreemphasis applies a first-order high-pass filter in-place:
// y[0] = x[0] (unchanged), y[n] = x[n] - coeff*x[n-1] for n > 0.
// Matches NeMo: torch.cat((x[:,0:1], x[:,1:] - preemph*x[:,:-1]), dim=1).
// Must iterate backward to avoid using already-modified values.
func applyPreemphasis(sig []float64, coeff float64) {
	for i := len(sig) - 1; i >= 1; i-- {
		sig[i] -= coeff * sig[i-1]
	}
	// sig[0] is left unchanged (NeMo behavior).
}
