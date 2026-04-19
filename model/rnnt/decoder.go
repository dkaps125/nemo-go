package rnnt

import (
	"context"
	"fmt"

	"github.com/dkaps125/nemo-go/internal/mathutil"
	"github.com/dkaps125/nemo-go/onnx"
)

const (
	// ONNX tensor names produced by scripts/export_onnx.py for RNNT models.
	djInputEncoderOutputs = "encoder_outputs"
	djInputTargets        = "targets"
	djInputTargetLength   = "target_length"
	djInputStates1        = "input_states_1"
	djInputStates2        = "input_states_2"
	djOutputLogits  = "outputs"
	djOutputStates1 = "output_states_1"
	djOutputStates2 = "output_states_2"
)

const maxSymbolsPerStep = 10

// greedyTDTDecode runs TDT greedy decoding for a single audio segment.
//
// encoderOut is the encoder output flat [1, D, T]. encodedLen is the number
// of valid encoder frames. durations is the TDT skip list (e.g. [0,1,2,3,4]).
func greedyTDTDecode(
	ctx context.Context,
	djSession *onnx.Session,
	encoderOut []float32,
	D, T int,
	encodedLen int,
	durations []int,
	state1Shape, state2Shape []int64,
) ([]int32, error) {
	// Initialise LSTM states to zero.
	state1 := make([]float32, shapeProduct(state1Shape))
	state2 := make([]float32, shapeProduct(state2Shape))

	lastToken := int32(0) // SOS token: prediction network embedding starts at 0
	generated := make([]int32, 0, 64)

	timeIdx := 0
	for timeIdx < encodedLen {
		if err := ctx.Err(); err != nil {
			return generated, err
		}

		// Single encoder frame: slice encoderOut[0, :, timeIdx] → [1, D, 1].
		frame := make([]float32, D)
		for d := 0; d < D; d++ {
			frame[d] = encoderOut[d*T+timeIdx]
		}

		symbolsAdded := 0
		needLoop := true
		skip := 1

		for needLoop && symbolsAdded < maxSymbolsPerStep {
			inputs := map[string]onnx.AnyInput{
				djInputEncoderOutputs: onnx.Float32Input{
					Data:  frame,
					Shape: []int64{1, int64(D), 1},
				},
				djInputTargets: onnx.Int32Input{
					Data:  []int32{lastToken},
					Shape: []int64{1, 1},
				},
				djInputTargetLength: onnx.Int32Input{
					Data:  []int32{1},
					Shape: []int64{1},
				},
				djInputStates1: onnx.Float32Input{Data: state1, Shape: state1Shape},
				djInputStates2: onnx.Float32Input{Data: state2, Shape: state2Shape},
			}

			outputs, err := djSession.RunMixed(inputs)
			if err != nil {
				return nil, fmt.Errorf("rnnt: decoder+joint step: %w", err)
			}

			logitsOut, ok := outputs[djOutputLogits]
			if !ok {
				return nil, fmt.Errorf("rnnt: missing output %q", djOutputLogits)
			}
			st1Out, ok := outputs[djOutputStates1]
			if !ok {
				return nil, fmt.Errorf("rnnt: missing output %q", djOutputStates1)
			}
			st2Out, ok := outputs[djOutputStates2]
			if !ok {
				return nil, fmt.Errorf("rnnt: missing output %q", djOutputStates2)
			}

			// logitsOut shape: [1, enc_time, tgt_time, logit_dim]
			// The model prepends SOS internally (tgt_time = input_tgt_time + 1).
			// We want the last tgt position (after processing lastToken).
			numDur := len(durations)
			logitDim := int(logitsOut.Shape[3])
			tgtTime := int(logitsOut.Shape[2])
			stepOffset := (tgtTime - 1) * logitDim
			logits := logitsOut.Data[stepOffset : stepOffset+logitDim]

			// Joint output layout: [vocab..., blank, dur0, dur1, ...]
			// jointBlankID = logitDim - numDur - 1
			jointBlankID := logitDim - numDur - 1
			tokenLogits := logits[:jointBlankID+1]
			durLogits := append([]float32{}, logits[jointBlankID+1:]...)
			mathutil.LogSoftmax(durLogits)

			tokenIdx := mathutil.Argmax(tokenLogits)
			durIdx := mathutil.Argmax(durLogits)
			skip = durations[durIdx]

			// Always update states (prediction network ran regardless of token type).
			state1 = st1Out.Data
			state2 = st2Out.Data
			if tokenIdx != jointBlankID {
				generated = append(generated, int32(tokenIdx))
				lastToken = int32(tokenIdx)
			}

			symbolsAdded++
			timeIdx += skip
			needLoop = skip == 0
		}

		// Safety: if blank was emitted with duration=0, force advance one frame.
		if skip == 0 {
			timeIdx++
		}
	}

	return generated, nil
}

func shapeProduct(shape []int64) int {
	n := 1
	for _, s := range shape {
		n *= int(s)
	}
	return n
}
