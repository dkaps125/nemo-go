package canary

import (
	"context"
	"fmt"

	"github.com/danielkapit/nemo-go/internal/mathutil"
	"github.com/danielkapit/nemo-go/onnx"
	"github.com/danielkapit/nemo-go/tokenizer"
)

// ONNX tensor names for the Canary decoder.
// Verify these match your exported model:
//   python -c "import onnx; m=onnx.load('decoder.onnx'); print([i.name for i in m.graph.input])"
const (
	decInputEncoderOut     = "encoder_output"   // [B, T, D]
	decInputEncoderLens    = "encoder_lengths"   // [B]
	decInputTargets        = "targets"           // [B, U] int64 prompt tokens
	decInputTargetLens     = "target_lengths"    // [B]
	decOutputLogProbs      = "log_probs"         // [B, U, V]
)

// autoregressiveDecode runs greedy autoregressive decoding for a single
// audio segment. encoderOut is the [1, T, D] float32 tensor output by the
// encoder. promptIDs seeds the decoder (Canary-specific prompt tokens).
//
// Returns the generated token IDs (excluding the prompt and EOS).
func autoregressiveDecode(
	ctx context.Context,
	decSession *onnx.Session,
	encoderOut onnx.Float32Output,
	encoderLen int64,
	promptIDs []int32,
	tok *tokenizer.AggregateTokenizer,
	maxLen int,
) ([]int32, error) {
	eosID := int32(tok.EOSID())
	if eosID < 0 {
		return nil, fmt.Errorf("canary: tokenizer has no EOS token")
	}

	// Current decoder input: starts as the prompt, grows by one token per step.
	tokens := make([]int32, len(promptIDs))
	copy(tokens, promptIDs)

	generated := make([]int32, 0, maxLen)

	for step := 0; step < maxLen; step++ {
		if err := ctx.Err(); err != nil {
			return generated, err
		}

		U := int64(len(tokens))
		T := encoderOut.Shape[1]
		D := encoderOut.Shape[2]

		// Build int64 token array for ONNX.
		tokenData := make([]int64, len(tokens))
		for i, id := range tokens {
			tokenData[i] = int64(id)
		}

		inputs := map[string]onnx.AnyInput{
			decInputEncoderOut:  onnx.Float32Input{Data: encoderOut.Data, Shape: []int64{1, T, D}},
			decInputEncoderLens: onnx.Int64Input{Data: []int64{encoderLen}, Shape: []int64{1}},
			decInputTargets:     onnx.Int64Input{Data: tokenData, Shape: []int64{1, U}},
			decInputTargetLens:  onnx.Int64Input{Data: []int64{U}, Shape: []int64{1}},
		}

		outputs, err := decSession.RunMixed(inputs)
		if err != nil {
			return nil, fmt.Errorf("canary: decoder step %d: %w", step, err)
		}

		logProbs, ok := outputs[decOutputLogProbs]
		if !ok {
			return nil, fmt.Errorf("canary: missing decoder output %q", decOutputLogProbs)
		}

		// logProbs: [1, U, V] — take the last time step.
		V := int(logProbs.Shape[2])
		lastLogits := logProbs.Data[(len(tokens)-1)*V : len(tokens)*V]

		// Greedy: pick the most likely token.
		nextID := int32(mathutil.Argmax(lastLogits))
		if nextID == eosID {
			break
		}
		generated = append(generated, nextID)
		tokens = append(tokens, nextID)
	}

	return generated, nil
}
