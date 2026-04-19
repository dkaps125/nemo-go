package parakeet

import (
	"github.com/dkaps125/nemo-go/internal/mathutil"
	"github.com/dkaps125/nemo-go/tokenizer"
)

// greedyCTCDecode decodes a CTC logit matrix using greedy (best-path) decoding.
//
// logits is a flat [vocabSize × nFrames] slice (NeMo output layout).
// nFrames is the actual (unpadded) frame count for this sequence.
// blankID is the CTC blank token index.
//
// Returns the decoded token IDs (blank-collapsed, without blanks).
func greedyCTCDecode(logits []float32, vocabSize, nFrames, blankID int) []int32 {
	if nFrames == 0 || vocabSize == 0 {
		return nil
	}

	// NeMo ONNX logits shape: [batch, nFrames, vocabSize] — caller provides
	// the single-batch slice already as [nFrames × vocabSize].
	// We argmax over the vocab dimension at each time step.
	var ids []int32
	prev := -1
	for t := 0; t < nFrames; t++ {
		frame := logits[t*vocabSize : (t+1)*vocabSize]
		best := mathutil.Argmax(frame)
		if best == blankID || best == prev {
			prev = best
			continue
		}
		prev = best
		if best != blankID {
			ids = append(ids, int32(best))
		}
	}
	return ids
}

// ctcDecode is the high-level decode function: logits → text.
func ctcDecode(logits []float32, vocabSize, nFrames, blankID int, tok tokenizer.Tokenizer) (string, error) {
	ids := greedyCTCDecode(logits, vocabSize, nFrames, blankID)
	if len(ids) == 0 {
		return "", nil
	}
	return tok.Decode(ids)
}
