// nemo-transcribe runs ASR inference on WAV files using Parakeet or Canary models.
//
// Usage:
//
//	nemo-transcribe --model parakeet --checkpoint /path/to/model.nemo audio1.wav audio2.wav
//	nemo-transcribe --model canary   --checkpoint /path/to/model.nemo --lang en --task translate --tgt-lang de audio.wav
//
// Prerequisites:
//  1. Export ONNX files: python3 scripts/export_onnx.py --model <parakeet|canary> --checkpoint <path>
//  2. Set ORT_LIB_PATH env var (or pass --ort-lib) to the ONNX Runtime shared library.
//     macOS: typically /usr/local/lib/libonnxruntime.dylib
//     Linux: typically /usr/lib/libonnxruntime.so.1
package main

import (
	"context"
	"encoding/binary"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/dkaps125/nemo-go/checkpoint"
	"github.com/dkaps125/nemo-go/model"
	"github.com/dkaps125/nemo-go/model/canary"
	"github.com/dkaps125/nemo-go/model/parakeet"
	modelrnnt "github.com/dkaps125/nemo-go/model/rnnt"
	"github.com/dkaps125/nemo-go/onnx"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	modelType := flag.String("model", "", "Model type: parakeet, rnnt, or canary (required)")
	ckPath := flag.String("checkpoint", "", "Path to .nemo file or checkpoint directory (required)")
	ortLib := flag.String("ort-lib", "", "Path to ONNX Runtime shared library (or set ORT_LIB_PATH env)")
	lang := flag.String("lang", "en", "Source language (Canary only)")
	task := flag.String("task", "transcribe", "Task: transcribe or translate (Canary only)")
	tgtLang := flag.String("tgt-lang", "", "Target language for translation (Canary only)")
	pnc := flag.Bool("pnc", true, "Enable punctuation and capitalisation (Canary only)")
	cudaDevice := flag.Int("cuda", -1, "CUDA device ID (-1 = CPU)")
	flag.Parse()

	if *modelType == "" || *ckPath == "" {
		flag.Usage()
		return fmt.Errorf("--model and --checkpoint are required")
	}
	audioFiles := flag.Args()
	if len(audioFiles) == 0 {
		return fmt.Errorf("no audio files provided")
	}

	// Initialise ORT.
	lib := *ortLib
	if lib == "" {
		lib = os.Getenv("ORT_LIB_PATH")
	}
	if lib == "" {
		return fmt.Errorf("ORT library path not set: use --ort-lib or ORT_LIB_PATH env var")
	}
	if err := checkpoint.SetORTLibraryPath(lib); err != nil {
		return fmt.Errorf("init ORT: %w", err)
	}

	opts := onnx.SessionOptions{CUDADeviceID: *cudaDevice}

	// Load audio files.
	segments := make([]model.AudioSegment, len(audioFiles))
	for i, path := range audioFiles {
		seg, err := loadAudio(path)
		if err != nil {
			return fmt.Errorf("load %s: %w", path, err)
		}
		segments[i] = seg
	}

	transcribeOpts := model.TranscribeOptions{
		Language:       *lang,
		Task:           *task,
		TargetLanguage: *tgtLang,
		BeamSize:       1,
		PnC:            *pnc,
	}
	if transcribeOpts.TargetLanguage == "" {
		transcribeOpts.TargetLanguage = *lang
	}

	var transcripts []string

	ctx := context.Background()
	start := time.Now()

	var transcribeErr error
	switch *modelType {
	case "parakeet":
		ck, err := checkpoint.LoadParakeetCheckpoint(*ckPath)
		if err != nil {
			return err
		}
		m, err := parakeet.Load(ck, opts)
		if err != nil {
			return err
		}
		defer m.Close()
		transcripts, transcribeErr = m.Transcribe(ctx, segments, transcribeOpts)

	case "canary":
		ck, err := checkpoint.LoadCanaryCheckpoint(*ckPath)
		if err != nil {
			return err
		}
		m, err := canary.Load(ck, opts)
		if err != nil {
			return err
		}
		defer m.Close()
		transcripts, transcribeErr = m.Transcribe(ctx, segments, transcribeOpts)

	case "rnnt":
		ck, err := checkpoint.LoadRNNTCheckpoint(*ckPath)
		if err != nil {
			return err
		}
		m, err := modelrnnt.Load(ck, opts)
		if err != nil {
			return err
		}
		defer m.Close()
		transcripts, transcribeErr = m.Transcribe(ctx, segments, transcribeOpts)

	default:
		return fmt.Errorf("unknown model type %q (must be 'parakeet', 'rnnt', or 'canary')", *modelType)
	}

	if transcribeErr != nil {
		return fmt.Errorf("transcribe: %w", transcribeErr)
	}

	elapsed := time.Since(start)

	var totalSamples int
	for _, seg := range segments {
		totalSamples += len(seg.PCM)
	}
	audioDur := float64(totalSamples) / 16000.0
	rtf := elapsed.Seconds() / audioDur

	var totalWords int
	for _, t := range transcripts {
		totalWords += len(strings.Fields(t))
	}

	fmt.Fprintf(os.Stderr, "audio: %.2fs  transcription: %.2fs  RTF: %.3f  words: %d\n",
		audioDur, elapsed.Seconds(), rtf, totalWords)

	for i, t := range transcripts {
		fmt.Printf("[%s]\n%s\n\n", audioFiles[i], t)
		if t == "" {
			fmt.Fprintf(os.Stderr, "warning: empty transcript for %s\n", audioFiles[i])
		}
	}
	return nil
}

// loadAudio decodes any audio file to 16 kHz mono float32 PCM using ffmpeg.
// ffmpeg must be available on PATH.
func loadAudio(path string) (model.AudioSegment, error) {
	// Ask ffmpeg to decode to raw s16le at 16 kHz mono on stdout.
	cmd := exec.Command("ffmpeg",
		"-hide_banner", "-loglevel", "error",
		"-i", path,
		"-ar", "16000",
		"-ac", "1",
		"-f", "s16le",
		"-",
	)
	raw, err := cmd.Output()
	if err != nil {
		var ee *exec.ExitError
		if errors.As(err, &ee) {
			return model.AudioSegment{}, fmt.Errorf("ffmpeg: %s", ee.Stderr)
		}
		return model.AudioSegment{}, fmt.Errorf("ffmpeg: %w", err)
	}

	if len(raw)%2 != 0 {
		raw = raw[:len(raw)-1]
	}
	pcm := make([]float32, len(raw)/2)
	for i := range pcm {
		s := int16(binary.LittleEndian.Uint16(raw[i*2 : i*2+2]))
		pcm[i] = float32(s) / 32768.0
	}
	return model.AudioSegment{PCM: pcm, SampleRate: 16000}, nil
}
