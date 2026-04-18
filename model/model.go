// Package model defines the shared ASR model interface.
package model

import "context"

// AudioSegment is mono PCM audio at the model's expected sample rate (16 kHz).
type AudioSegment struct {
	PCM        []float32
	SampleRate int // should be 16000
}

// TranscribeOptions controls inference behavior.
type TranscribeOptions struct {
	// Language is the source language code, e.g. "en", "de". Canary only.
	Language string
	// Task is "transcribe" or "translate". Canary only.
	Task string
	// TargetLanguage is the output language for translation. Canary only.
	TargetLanguage string
	// BeamSize controls decoding. 1 = greedy (default, fastest).
	BeamSize int
	// PnC enables punctuation and capitalisation. Canary only.
	PnC bool
}

// DefaultOptions returns sensible defaults for transcription.
func DefaultOptions() TranscribeOptions {
	return TranscribeOptions{
		Language: "en",
		Task:     "transcribe",
		BeamSize: 1,
		PnC:      true,
	}
}

// ASRModel is the common interface for Parakeet and Canary inference.
type ASRModel interface {
	// Transcribe converts a batch of audio segments to transcripts.
	Transcribe(ctx context.Context, segments []AudioSegment, opts TranscribeOptions) ([]string, error)
	// Close releases all native resources (ONNX sessions, etc.).
	Close() error
}
