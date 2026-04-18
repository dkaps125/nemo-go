// Package parakeet implements inference for NeMo Parakeet CTC ASR models.
//
// Pipeline:
//
//	[]AudioSegment
//	  → audio.Preprocessor  → log-mel spectrograms
//	  → ONNX encoder+CTC     → logits [B, T, V]
//	  → greedyCTCDecode       → []string transcripts
package parakeet

import (
	"context"
	"fmt"

	"github.com/danielkapit/nemo-go/audio"
	"github.com/danielkapit/nemo-go/checkpoint"
	"github.com/danielkapit/nemo-go/model"
	"github.com/danielkapit/nemo-go/onnx"
	"github.com/danielkapit/nemo-go/tokenizer"
)

// ONNX input/output names for NeMo Parakeet CTC export.
// Verify these against your exported model with:
//
//	python -c "import onnx; m=onnx.load('encoder_ctc.onnx'); print([i.name for i in m.graph.input])"
const (
	inputAudioSignal = "audio_signal"
	inputLength      = "length"
	outputLogprobs   = "logprobs"
)

// Model is a Parakeet CTC inference model.
type Model struct {
	preprocessor *audio.Preprocessor
	session      *onnx.Session
	tok          tokenizer.Tokenizer
	vocabSize    int
	blankID      int
}

// Load instantiates a Parakeet model from a resolved checkpoint.
func Load(ck *checkpoint.ParakeetCheckpoint, opts onnx.SessionOptions) (*Model, error) {
	pp, err := audio.New(ck.AudioConfig)
	if err != nil {
		return nil, fmt.Errorf("parakeet: create preprocessor: %w", err)
	}

	sess, err := onnx.NewSession(
		ck.EncoderCTCONNXPath,
		[]string{inputAudioSignal, inputLength},
		[]string{outputLogprobs},
		opts,
	)
	if err != nil {
		return nil, fmt.Errorf("parakeet: load ONNX: %w", err)
	}

	tok, err := tokenizer.LoadSentencePiece(ck.TokenizerPath, -1)
	if err != nil {
		sess.Close()
		return nil, fmt.Errorf("parakeet: load tokenizer: %w", err)
	}

	return &Model{
		preprocessor: pp,
		session:      sess,
		tok:          tok,
		vocabSize:    ck.VocabSize,
		blankID:      ck.BlankID,
	}, nil
}

// Transcribe runs ASR on a batch of audio segments.
// Implements model.ASRModel.
func (m *Model) Transcribe(_ context.Context, segments []model.AudioSegment, _ model.TranscribeOptions) ([]string, error) {
	if len(segments) == 0 {
		return nil, nil
	}

	nMels := int64(m.preprocessor.NMelBins())
	transcripts := make([]string, len(segments))

	for i, seg := range segments {
		mel, nFrames, err := m.preprocessor.LogMel(seg.PCM)
		if err != nil {
			return nil, fmt.Errorf("parakeet: preprocess segment %d: %w", i, err)
		}

		T := int64(nFrames)
		inputs := map[string]onnx.AnyInput{
			inputAudioSignal: onnx.Float32Input{
				Data:  mel,
				Shape: []int64{1, nMels, T},
			},
			inputLength: onnx.Int64Input{
				Data:  []int64{T},
				Shape: []int64{1},
			},
		}

		outputs, err := m.session.RunMixed(inputs)
		if err != nil {
			return nil, fmt.Errorf("parakeet: run encoder segment %d: %w", i, err)
		}

		logprobs, ok := outputs[outputLogprobs]
		if !ok {
			return nil, fmt.Errorf("parakeet: output %q not found", outputLogprobs)
		}

		// logprobs shape: [1, T', V]
		outT := int(logprobs.Shape[1])
		V := int(logprobs.Shape[2])
		text, err := ctcDecode(logprobs.Data[:outT*V], V, outT, m.blankID, m.tok)
		if err != nil {
			return nil, fmt.Errorf("parakeet: decode segment %d: %w", i, err)
		}
		transcripts[i] = text
	}
	return transcripts, nil
}

// Close releases all resources held by the model.
func (m *Model) Close() error {
	return m.session.Close()
}
