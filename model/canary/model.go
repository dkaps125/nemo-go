// Package canary implements inference for NeMo Canary AED (encoder-decoder) ASR models.
//
// Pipeline:
//
//	[]AudioSegment
//	  → audio.Preprocessor  → log-mel spectrograms
//	  → ONNX encoder         → encoder_output [B, T, D]
//	  → autoregressiveDecode → token IDs per segment
//	  → AggregateTokenizer   → []string transcripts
package canary

import (
	"context"
	"fmt"
	"strings"

	"github.com/dkaps125/nemo-go/audio"
	"github.com/dkaps125/nemo-go/checkpoint"
	"github.com/dkaps125/nemo-go/model"
	"github.com/dkaps125/nemo-go/onnx"
	"github.com/dkaps125/nemo-go/tokenizer"
)

// ONNX tensor names for the Canary encoder.
const (
	encInputAudioSignal = "audio_signal"
	encInputLength      = "length"
	encOutputEncoderOut = "encoder_output"
	encOutputEncLen     = "encoded_lengths"
)

// Model is a Canary AED inference model.
type Model struct {
	preprocessor *audio.Preprocessor
	encSession   *onnx.Session
	decSession   *onnx.Session
	tok          *tokenizer.AggregateTokenizer
	maxDecodeLen int
}

// Load instantiates a Canary model from a resolved checkpoint.
func Load(ck *checkpoint.CanaryCheckpoint, opts onnx.SessionOptions) (*Model, error) {
	pp, err := audio.New(ck.AudioConfig)
	if err != nil {
		return nil, fmt.Errorf("canary: create preprocessor: %w", err)
	}

	encSess, err := onnx.NewSession(
		ck.EncoderONNXPath,
		[]string{encInputAudioSignal, encInputLength},
		[]string{encOutputEncoderOut, encOutputEncLen},
		opts,
	)
	if err != nil {
		return nil, fmt.Errorf("canary: load encoder ONNX: %w", err)
	}

	decSess, err := onnx.NewSession(
		ck.DecoderONNXPath,
		[]string{decInputEncoderOut, decInputEncoderLens, decInputTargets, decInputTargetLens},
		[]string{decOutputLogProbs},
		opts,
	)
	if err != nil {
		encSess.Close()
		return nil, fmt.Errorf("canary: load decoder ONNX: %w", err)
	}

	speTok, err := tokenizer.LoadSentencePiece(ck.TokenizerPath, -1)
	if err != nil {
		encSess.Close()
		decSess.Close()
		return nil, fmt.Errorf("canary: load SPE tokenizer: %w", err)
	}

	aggTok, err := tokenizer.NewAggregateTokenizer(speTok, ck.SpecialTokens)
	if err != nil {
		encSess.Close()
		decSess.Close()
		return nil, fmt.Errorf("canary: build aggregate tokenizer: %w", err)
	}

	return &Model{
		preprocessor: pp,
		encSession:   encSess,
		decSession:   decSess,
		tok:          aggTok,
		maxDecodeLen: ck.MaxDecodeLen,
	}, nil
}

// Transcribe runs AED inference on a batch of audio segments.
// Implements model.ASRModel.
func (m *Model) Transcribe(ctx context.Context, segments []model.AudioSegment, opts model.TranscribeOptions) ([]string, error) {
	if len(segments) == 0 {
		return nil, nil
	}

	lang := opts.Language
	if lang == "" {
		lang = "en"
	}
	task := opts.Task
	if task == "" {
		task = "transcribe"
	}
	tgtLang := opts.TargetLanguage
	if tgtLang == "" {
		tgtLang = lang
	}

	promptIDs, err := m.tok.BuildPromptIDs(lang, task, tgtLang, opts.PnC)
	if err != nil {
		return nil, fmt.Errorf("canary: build prompt: %w", err)
	}

	transcripts := make([]string, len(segments))

	// Run each segment individually (batched autoregressive decoding is much
	// more complex; single-item is correct and sufficient for most use cases).
	for i, seg := range segments {
		mel, nFrames, err := m.preprocessor.LogMel(seg.PCM)
		if err != nil {
			return nil, fmt.Errorf("canary: preprocess segment %d: %w", i, err)
		}

		nMels := int64(m.preprocessor.NMelBins())
		T := int64(nFrames)

		encInputs := map[string]onnx.AnyInput{
			encInputAudioSignal: onnx.Float32Input{
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
			return nil, fmt.Errorf("canary: encode segment %d: %w", i, err)
		}

		encoderOut, ok := encOutputs[encOutputEncoderOut]
		if !ok {
			return nil, fmt.Errorf("canary: encoder output %q not found", encOutputEncoderOut)
		}
		encLens, ok := encOutputs[encOutputEncLen]
		if !ok {
			return nil, fmt.Errorf("canary: encoder output %q not found", encOutputEncLen)
		}
		encoderLen := encLens.Data[0]

		genIDs, err := autoregressiveDecode(
			ctx,
			m.decSession,
			encoderOut,
			int64(encoderLen),
			promptIDs,
			m.tok,
			m.maxDecodeLen,
		)
		if err != nil {
			return nil, fmt.Errorf("canary: decode segment %d: %w", i, err)
		}

		text, err := m.tok.Decode(genIDs)
		if err != nil {
			return nil, fmt.Errorf("canary: tokenizer decode segment %d: %w", i, err)
		}
		transcripts[i] = strings.TrimSpace(text)
	}
	return transcripts, nil
}

// Close releases all resources.
func (m *Model) Close() error {
	err1 := m.encSession.Close()
	err2 := m.decSession.Close()
	if err1 != nil {
		return err1
	}
	return err2
}
