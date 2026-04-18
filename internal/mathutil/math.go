package mathutil

import "math"

// Argmax returns the index of the maximum value in v.
// Returns -1 if v is empty.
func Argmax(v []float32) int {
	if len(v) == 0 {
		return -1
	}
	best := 0
	for i := 1; i < len(v); i++ {
		if v[i] > v[best] {
			best = i
		}
	}
	return best
}

// LogSoftmax applies log-softmax in-place over v.
func LogSoftmax(v []float32) {
	if len(v) == 0 {
		return
	}
	// Numerically stable: subtract max first.
	max := v[0]
	for _, x := range v[1:] {
		if x > max {
			max = x
		}
	}
	var sumExp float64
	for _, x := range v {
		sumExp += math.Exp(float64(x - max))
	}
	logSum := float32(math.Log(sumExp)) + max
	for i := range v {
		v[i] -= logSum
	}
}

// Softmax applies softmax in-place over v.
func Softmax(v []float32) {
	if len(v) == 0 {
		return
	}
	max := v[0]
	for _, x := range v[1:] {
		if x > max {
			max = x
		}
	}
	var sum float32
	for i, x := range v {
		v[i] = float32(math.Exp(float64(x - max)))
		sum += v[i]
	}
	for i := range v {
		v[i] /= sum
	}
}
