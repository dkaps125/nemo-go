package checkpoint

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/danielkapit/nemo-go/audio"
	"github.com/danielkapit/nemo-go/tokenizer"
)

// ORTLibraryPath sets the path to the ONNX Runtime shared library.
// Must be called before loading any model.
// On macOS: typically /usr/local/lib/libonnxruntime.dylib
// On Linux:  typically /usr/lib/libonnxruntime.so
func SetORTLibraryPath(path string) error {
	ort.SetSharedLibraryPath(path)
	return ort.InitializeEnvironment()
}

// ParakeetCheckpoint holds resolved paths for a Parakeet CTC model.
type ParakeetCheckpoint struct {
	// EncoderCTCONNXPath is the path to the exported encoder+CTC head ONNX file.
	// Export with: scripts/export_onnx.py --model parakeet
	EncoderCTCONNXPath string
	TokenizerPath      string
	AudioConfig        audio.Config
	VocabSize          int
	BlankID            int
}

// CanaryCheckpoint holds resolved paths for a Canary AED model.
type CanaryCheckpoint struct {
	EncoderONNXPath string
	DecoderONNXPath string
	TokenizerPath   string
	SpecialTokens   []string
	AudioConfig     audio.Config
	MaxDecodeLen    int
}

// LoadParakeetCheckpoint resolves paths and config for a Parakeet model.
//
// nemoOrDir may be:
//   - path to a .nemo file  (tokenizer extracted; ONNX file looked up alongside)
//   - path to a directory   (expects encoder_ctc.onnx and tokenizer.model)
func LoadParakeetCheckpoint(nemoOrDir string) (*ParakeetCheckpoint, error) {
	var (
		onnxPath  string
		tokPath   string
		audioCfg  audio.Config
		vocabSize int
		blankID   int
	)

	if strings.HasSuffix(nemoOrDir, ".nemo") {
		meta, err := OpenNemo(nemoOrDir)
		if err != nil {
			return nil, err
		}
		defer meta.CleanupTokenizerModel()

		tokPath = meta.TokenizerModelPath
		audioCfg, err = audioConfigFromMeta(meta)
		if err != nil {
			return nil, err
		}

		// Look for ONNX next to the .nemo file.
		base := strings.TrimSuffix(nemoOrDir, ".nemo")
		onnxPath = base + "_encoder_ctc.onnx"
	} else {
		// Directory layout.
		tokPath = filepath.Join(nemoOrDir, "tokenizer.model")
		onnxPath = filepath.Join(nemoOrDir, "encoder_ctc.onnx")
		audioCfg = audio.DefaultConfig()
	}

	if err := requireFile(onnxPath, "encoder+CTC ONNX (run scripts/export_onnx.py first)"); err != nil {
		return nil, err
	}
	if err := requireFile(tokPath, "SentencePiece tokenizer model"); err != nil {
		return nil, err
	}

	// Load tokenizer to determine vocab size.
	tok, err := tokenizer.LoadSentencePiece(tokPath, -1)
	if err != nil {
		return nil, fmt.Errorf("checkpoint: load tokenizer: %w", err)
	}
	vocabSize = tok.VocabSize()
	blankID = tok.BlankID() // NeMo convention: vocab_size (one past the last real token)

	return &ParakeetCheckpoint{
		EncoderCTCONNXPath: onnxPath,
		TokenizerPath:      tokPath,
		AudioConfig:        audioCfg,
		VocabSize:          vocabSize,
		BlankID:            blankID,
	}, nil
}

// RNNTCheckpoint holds resolved paths for a pure RNNT/TDT model.
type RNNTCheckpoint struct {
	EncoderONNXPath     string
	DecoderJointONNXPath string
	TokenizerPath       string
	AudioConfig         audio.Config
	// TDT duration values: skip[i] is how many encoder frames to advance when
	// duration index i is predicted. Typically [0,1,2,3,4,5,6,7,8].
	Durations   []int
	VocabSize   int
	BlankID     int
	State1Shape []int64 // LSTM h state: [num_layers, 1, hidden]
	State2Shape []int64 // LSTM c state: [num_layers, 1, hidden]
}

// rnntMeta mirrors the JSON written by scripts/export_onnx.py for RNNT models.
type rnntMeta struct {
	Durations   []int   `json:"durations"`
	VocabSize   int     `json:"vocab_size"`
	BlankID     int     `json:"blank_id"`
	State1Shape []int64 `json:"state1_shape"`
	State2Shape []int64 `json:"state2_shape"`
	NMels       int     `json:"n_mels"`
}

// LoadRNNTCheckpoint resolves paths and config for an RNNT/TDT model.
//
// nemoOrDir may be:
//   - path to a .nemo file  (ONNX files looked up alongside)
//   - path to a directory   (expects encoder.onnx, decoder_joint.onnx, tokenizer.model, *_rnnt_meta.json)
func LoadRNNTCheckpoint(nemoOrDir string) (*RNNTCheckpoint, error) {
	var (
		encONNX  string
		djONNX   string
		metaPath string
		tokPath  string
		audioCfg audio.Config
	)

	if strings.HasSuffix(nemoOrDir, ".nemo") {
		meta, err := OpenNemo(nemoOrDir)
		if err != nil {
			return nil, err
		}
		defer meta.CleanupTokenizerModel()

		tokPath = meta.TokenizerModelPath
		audioCfg, err = audioConfigFromMeta(meta)
		if err != nil {
			return nil, err
		}

		base := strings.TrimSuffix(nemoOrDir, ".nemo")
		encONNX = base + "_encoder.onnx"
		djONNX = base + "_decoder_joint.onnx"
		metaPath = base + "_rnnt_meta.json"
	} else {
		tokPath = filepath.Join(nemoOrDir, "tokenizer.model")
		audioCfg = audio.DefaultConfig()
		entries, _ := os.ReadDir(nemoOrDir)
		for _, e := range entries {
			name := e.Name()
			switch {
			case strings.HasSuffix(name, "_encoder.onnx"):
				encONNX = filepath.Join(nemoOrDir, name)
			case strings.HasSuffix(name, "_decoder_joint.onnx"):
				djONNX = filepath.Join(nemoOrDir, name)
			case strings.HasSuffix(name, "_rnnt_meta.json"):
				metaPath = filepath.Join(nemoOrDir, name)
			}
		}
	}

	for _, p := range []struct{ path, desc string }{
		{encONNX, "RNNT encoder ONNX (run scripts/export_onnx.py --model rnnt first)"},
		{djONNX, "RNNT decoder+joint ONNX (run scripts/export_onnx.py --model rnnt first)"},
		{tokPath, "SentencePiece tokenizer model"},
		{metaPath, "RNNT metadata JSON (run scripts/export_onnx.py --model rnnt first)"},
	} {
		if err := requireFile(p.path, p.desc); err != nil {
			return nil, err
		}
	}

	raw, err := os.ReadFile(metaPath)
	if err != nil {
		return nil, fmt.Errorf("checkpoint: read RNNT meta: %w", err)
	}
	var m rnntMeta
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, fmt.Errorf("checkpoint: parse RNNT meta: %w", err)
	}

	if m.NMels > 0 {
		audioCfg.NMelBins = m.NMels
	}

	return &RNNTCheckpoint{
		EncoderONNXPath:      encONNX,
		DecoderJointONNXPath: djONNX,
		TokenizerPath:        tokPath,
		AudioConfig:          audioCfg,
		Durations:            m.Durations,
		VocabSize:            m.VocabSize,
		BlankID:              m.BlankID,
		State1Shape:          m.State1Shape,
		State2Shape:          m.State2Shape,
	}, nil
}

// LoadCanaryCheckpoint resolves paths and config for a Canary model.
func LoadCanaryCheckpoint(nemoOrDir string) (*CanaryCheckpoint, error) {
	var (
		encONNX      string
		decONNX      string
		tokPath      string
		specialToks  []string
		audioCfg     audio.Config
		maxDecodeLen = 400
	)

	if strings.HasSuffix(nemoOrDir, ".nemo") {
		meta, err := OpenNemo(nemoOrDir)
		if err != nil {
			return nil, err
		}
		defer meta.CleanupTokenizerModel()

		tokPath = meta.TokenizerModelPath
		audioCfg, err = audioConfigFromMeta(meta)
		if err != nil {
			return nil, err
		}
		specialToks, err = meta.ParseCanaryTokenizerConfig()
		if err != nil {
			return nil, err
		}

		if v, ok := nestedGet(meta.Config, "decoding", "max_generation_length").(int); ok {
			maxDecodeLen = v
		}

		base := strings.TrimSuffix(nemoOrDir, ".nemo")
		encONNX = base + "_encoder.onnx"
		decONNX = base + "_decoder.onnx"
	} else {
		tokPath = filepath.Join(nemoOrDir, "tokenizer.model")
		encONNX = filepath.Join(nemoOrDir, "encoder.onnx")
		decONNX = filepath.Join(nemoOrDir, "decoder.onnx")
		audioCfg = audio.DefaultConfig()
		specialToks = tokenizer.CanarySpecialTokens([]string{"en"})
	}

	for _, p := range []struct{ path, desc string }{
		{encONNX, "Canary encoder ONNX (run scripts/export_onnx.py first)"},
		{decONNX, "Canary decoder ONNX (run scripts/export_onnx.py first)"},
		{tokPath, "SentencePiece tokenizer model"},
	} {
		if err := requireFile(p.path, p.desc); err != nil {
			return nil, err
		}
	}

	return &CanaryCheckpoint{
		EncoderONNXPath: encONNX,
		DecoderONNXPath: decONNX,
		TokenizerPath:   tokPath,
		SpecialTokens:   specialToks,
		AudioConfig:     audioCfg,
		MaxDecodeLen:    maxDecodeLen,
	}, nil
}

// audioConfigFromMeta builds an audio.Config from a model_config.yaml.
func audioConfigFromMeta(meta *NemoMeta) (audio.Config, error) {
	cfg := audio.DefaultConfig()
	pp, err := meta.PreprocessorConfig()
	if err != nil {
		// Use defaults if preprocessor section missing.
		return cfg, nil
	}

	if v, ok := floatVal(pp, "sample_rate"); ok {
		cfg.SampleRate = int(v)
	}
	if v, ok := floatVal(pp, "window_size"); ok {
		cfg.WindowSizeSec = v
	}
	if v, ok := floatVal(pp, "window_stride"); ok {
		cfg.WindowStride = v
	}
	if v, ok := floatVal(pp, "n_fft"); ok {
		cfg.NFft = int(v)
	}
	if v, ok := floatVal(pp, "n_mels"); ok {
		cfg.NMelBins = int(v)
	} else if v, ok := floatVal(pp, "features"); ok {
		cfg.NMelBins = int(v)
	}
	if v, ok := floatVal(pp, "highfreq"); ok {
		cfg.FMax = v
	}
	if v, ok := pp["mel_norm"].(string); ok {
		cfg.MelNorm = v
	}
	if v, ok := floatVal(pp, "preemph"); ok {
		cfg.Preemph = v
	}
	if v, ok := pp["normalize"].(string); ok {
		switch v {
		case "per_feature":
			cfg.NormType = audio.NormPerFeature
		case "all_features":
			cfg.NormType = audio.NormAllFeatures
		default:
			cfg.NormType = audio.NormNone
		}
	}
	if ep, ok := pp["exact_pad"].(bool); ok && ep {
		cfg.Center = false // exact_pad=True → center=False in NeMo
	}

	return cfg, nil
}

func floatVal(m map[string]any, key string) (float64, bool) {
	v, ok := m[key]
	if !ok {
		return 0, false
	}
	switch val := v.(type) {
	case float64:
		return val, true
	case int:
		return float64(val), true
	case float32:
		return float64(val), true
	}
	return 0, false
}

func requireFile(path, desc string) error {
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return fmt.Errorf("checkpoint: %s not found at %s", desc, path)
	}
	return nil
}

func nestedGet(m map[string]any, keys ...string) any {
	var cur any = m
	for _, k := range keys {
		mm, ok := cur.(map[string]any)
		if !ok {
			return nil
		}
		cur = mm[k]
	}
	return cur
}
