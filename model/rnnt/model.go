// Package rnnt implements inference for NeMo RNNT/TDT ASR models.
//
// Pipeline:
//
//	AudioSegment
//	  → audio.Preprocessor  → log-mel spectrogram
//	  → ONNX encoder         → encoded_output [1, D, T]
//	  → greedyTDTDecode      → token IDs
//	  → SentencePiece        → transcript string
package rnnt

import (
	"context"
	"fmt"
	"strings"

	"github.com/danielkapit/nemo-go/audio"
	"github.com/danielkapit/nemo-go/checkpoint"
	"github.com/danielkapit/nemo-go/model"
	"github.com/danielkapit/nemo-go/onnx"
	"github.com/danielkapit/nemo-go/tokenizer"
)

// ONNX tensor names for the RNNT encoder (same as Canary encoder but output
// name is "outputs" not "encoder_output", and shape is [B,D,T] not [B,T,D]).
const (
	encInputAudio   = "audio_signal"
	encInputLength  = "length"
	encOutputFrames = "outputs"          // [B, D, T]
	encOutputLen    = "encoded_lengths"  // [B]
)

// Model is an RNNT/TDT inference model.
type Model struct {
	preprocessor *audio.Preprocessor
	encSession   *onnx.Session
	djSession    *onnx.Session
	tok          tokenizer.Tokenizer
	durations    []int
	blankID      int
	state1Shape  []int64
	state2Shape  []int64
}

// Load instantiates an RNNT model from a resolved checkpoint.
func Load(ck *checkpoint.RNNTCheckpoint, opts onnx.SessionOptions) (*Model, error) {
	pp, err := audio.New(ck.AudioConfig)
	if err != nil {
		return nil, fmt.Errorf("rnnt: create preprocessor: %w", err)
	}

	encSess, err := onnx.NewSession(
		ck.EncoderONNXPath,
		[]string{encInputAudio, encInputLength},
		[]string{encOutputFrames, encOutputLen},
		opts,
	)
	if err != nil {
		return nil, fmt.Errorf("rnnt: load encoder ONNX: %w", err)
	}

	djSess, err := onnx.NewSession(
		ck.DecoderJointONNXPath,
		[]string{djInputEncoderOutputs, djInputTargets, djInputTargetLength,
			djInputStates1, djInputStates2},
		[]string{djOutputLogits, djOutputStates1, djOutputStates2},
		opts,
	)
	if err != nil {
		encSess.Close()
		return nil, fmt.Errorf("rnnt: load decoder+joint ONNX: %w", err)
	}

	tok, err := tokenizer.LoadSentencePiece(ck.TokenizerPath, -1)
	if err != nil {
		encSess.Close()
		djSess.Close()
		return nil, fmt.Errorf("rnnt: load tokenizer: %w", err)
	}

	return &Model{
		preprocessor: pp,
		encSession:   encSess,
		djSession:    djSess,
		tok:          tok,
		durations:    ck.Durations,
		blankID:      ck.BlankID,
		state1Shape:  ck.State1Shape,
		state2Shape:  ck.State2Shape,
	}, nil
}

// Transcribe runs ASR on a batch of audio segments.
func (m *Model) Transcribe(ctx context.Context, segments []model.AudioSegment, _ model.TranscribeOptions) ([]string, error) {
	if len(segments) == 0 {
		return nil, nil
	}

	nMels := int64(m.preprocessor.NMelBins())
	transcripts := make([]string, len(segments))

	for i, seg := range segments {
		mel, nFrames, err := m.preprocessor.LogMel(seg.PCM)
		if err != nil {
			return nil, fmt.Errorf("rnnt: preprocess segment %d: %w", i, err)
		}

		T := int64(nFrames)
		encInputs := map[string]onnx.AnyInput{
			encInputAudio: onnx.Float32Input{
				Data:  mel,
				Shape: []int64{1, nMels, T},
			},
			encInputLength: onnx.Int64Input{
				Data:  []int64{T},
				Shape: []int64{1},
			},
		}

		encOutputs, err := m.encSession.RunMixed(encInputs)
		if err != nil {
			return nil, fmt.Errorf("rnnt: encode segment %d: %w", i, err)
		}

		encFrames, ok := encOutputs[encOutputFrames]
		if !ok {
			return nil, fmt.Errorf("rnnt: encoder output %q not found", encOutputFrames)
		}
		encLens, ok := encOutputs[encOutputLen]
		if !ok {
			return nil, fmt.Errorf("rnnt: encoder output %q not found", encOutputLen)
		}

		// encFrames shape: [1, D, T']
		D := int(encFrames.Shape[1])
		Tenc := int(encFrames.Shape[2])
		encodedLen := int(encLens.Data[0])

		ids, err := greedyTDTDecode(
			ctx,
			m.djSession,
			encFrames.Data,
			D, Tenc,
			encodedLen,
			m.durations,
			m.state1Shape,
			m.state2Shape,
		)
		if err != nil {
			return nil, fmt.Errorf("rnnt: decode segment %d: %w", i, err)
		}

		text, err := m.tok.Decode(ids)
		if err != nil {
			return nil, fmt.Errorf("rnnt: tokenizer decode segment %d: %w", i, err)
		}
		transcripts[i] = strings.TrimSpace(text)
	}
	return transcripts, nil
}

// Close releases all resources.
func (m *Model) Close() error {
	err1 := m.encSession.Close()
	err2 := m.djSession.Close()
	if err1 != nil {
		return err1
	}
	return err2
}
