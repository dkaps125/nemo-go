// Package checkpoint handles loading NeMo checkpoints.
//
// A .nemo file is a ZIP archive containing:
//   - model_config.yaml  — Hydra/OmegaConf configuration
//   - model_weights.ckpt — PyTorch checkpoint (not used by Go inference)
//   - tokenizer.model    — SentencePiece .model file
//
// Go inference requires pre-exported ONNX files alongside the .nemo file.
// Use scripts/export_onnx.py to generate them first.
package checkpoint

import (
	"archive/zip"
	"fmt"
	"os"
	"path/filepath"

	"gopkg.in/yaml.v3"
)

// NemoMeta holds the parsed essentials from a .nemo archive.
type NemoMeta struct {
	// ModelType is the NeMo _target_ class, e.g.
	// "nemo.collections.asr.models.EncDecCTCModelBPE" or
	// "nemo.collections.asr.models.EncDecMultiTaskModel".
	ModelType string

	// Config is the raw model_config.yaml unmarshalled into a map.
	Config map[string]any

	// TokenizerModelBytes is the raw bytes of tokenizer.model (SentencePiece).
	TokenizerModelBytes []byte

	// TokenizerModelPath is a temp file path where TokenizerModelBytes was written.
	// Callers must call CleanupTokenizerModel() when done.
	TokenizerModelPath string
}

// OpenNemo opens a .nemo ZIP archive and extracts the config and tokenizer.
func OpenNemo(nemoPath string) (*NemoMeta, error) {
	r, err := zip.OpenReader(nemoPath)
	if err != nil {
		return nil, fmt.Errorf("checkpoint: open %s: %w", nemoPath, err)
	}
	defer r.Close()

	meta := &NemoMeta{Config: make(map[string]any)}

	for _, f := range r.File {
		switch filepath.Base(f.Name) {
		case "model_config.yaml":
			rc, err := f.Open()
			if err != nil {
				return nil, fmt.Errorf("checkpoint: open model_config.yaml: %w", err)
			}
			var raw map[string]any
			if err := yaml.NewDecoder(rc).Decode(&raw); err != nil {
				rc.Close()
				return nil, fmt.Errorf("checkpoint: parse model_config.yaml: %w", err)
			}
			rc.Close()
			meta.Config = raw
			if t, ok := raw["_target_"].(string); ok {
				meta.ModelType = t
			}

		case "tokenizer.model":
			rc, err := f.Open()
			if err != nil {
				return nil, fmt.Errorf("checkpoint: open tokenizer.model: %w", err)
			}
			buf := make([]byte, f.UncompressedSize64)
			if _, err := rc.Read(buf); err != nil && err.Error() != "EOF" {
				rc.Close()
				return nil, fmt.Errorf("checkpoint: read tokenizer.model: %w", err)
			}
			rc.Close()
			meta.TokenizerModelBytes = buf
		}
	}

	if len(meta.TokenizerModelBytes) == 0 {
		return nil, fmt.Errorf("checkpoint: tokenizer.model not found in %s", nemoPath)
	}

	// Write tokenizer to a temp file (SentencePiece loader requires a path).
	tmp, err := os.CreateTemp("", "nemo-spe-*.model")
	if err != nil {
		return nil, fmt.Errorf("checkpoint: create temp tokenizer file: %w", err)
	}
	if _, err := tmp.Write(meta.TokenizerModelBytes); err != nil {
		tmp.Close()
		os.Remove(tmp.Name())
		return nil, fmt.Errorf("checkpoint: write temp tokenizer: %w", err)
	}
	tmp.Close()
	meta.TokenizerModelPath = tmp.Name()

	return meta, nil
}

// CleanupTokenizerModel removes the temp tokenizer file written by OpenNemo.
func (m *NemoMeta) CleanupTokenizerModel() {
	if m.TokenizerModelPath != "" {
		os.Remove(m.TokenizerModelPath)
	}
}

// IsParakeet reports whether the checkpoint is a Parakeet CTC model.
func (m *NemoMeta) IsParakeet() bool {
	return containsAny(m.ModelType,
		"EncDecCTCModel",
		"EncDecCTCModelBPE",
		"EncDecHybridRNNTCTCModel",
	)
}

// IsRNNT reports whether the checkpoint is a pure RNNT/TDT model.
func (m *NemoMeta) IsRNNT() bool {
	return containsAny(m.ModelType, "EncDecRNNT")
}

// IsCanary reports whether the checkpoint is a Canary AED model.
func (m *NemoMeta) IsCanary() bool {
	return containsAny(m.ModelType, "EncDecMultiTaskModel")
}

// PreprocessorConfig extracts AudioToMelSpectrogramPreprocessor config fields.
func (m *NemoMeta) PreprocessorConfig() (map[string]any, error) {
	pp, ok := m.Config["preprocessor"].(map[string]any)
	if !ok {
		return nil, fmt.Errorf("checkpoint: preprocessor key not found in config")
	}
	return pp, nil
}

// ParseCanaryTokenizerConfig reads the special token list from the Canary
// model_config.yaml in the order NeMo uses for ID assignment.
//
// Returns the ordered list of special token strings.
func (m *NemoMeta) ParseCanaryTokenizerConfig() ([]string, error) {
	// Canary config structure (simplified):
	// model_defaults:
	//   asr_enc_hidden: 1024
	// decoder:
	//   tokenizer:
	//     langs: [en, de, es, fr, ...]
	//     special_tokens: [<|transcribe|>, <|translate|>, ...]
	//
	// Actual path varies; try common locations.
	langs, err := m.extractLangs()
	if err != nil {
		return nil, err
	}

	// NeMo's Canary tokenizer special_tokens order is hardcoded in the class.
	// See: nemo/collections/common/tokenizers/canary_tokenizer.py
	// Order: <pad>, <|startoftranscript|>, <|endoftext|>, <|transcribe|>,
	//        <|translate|>, <|nopnc|>, <|pnc|>, <|en|>, ...other langs
	tokens := []string{
		"<pad>",
		"<|startoftranscript|>",
		"<|endoftext|>",
		"<|transcribe|>",
		"<|translate|>",
		"<|nopnc|>",
		"<|pnc|>",
	}
	for _, lang := range langs {
		tokens = append(tokens, fmt.Sprintf("<|%s|>", lang))
	}
	return tokens, nil
}

func (m *NemoMeta) extractLangs() ([]string, error) {
	// Try decoder.tokenizer.langs
	if dec, ok := m.Config["decoder"].(map[string]any); ok {
		if tok, ok := dec["tokenizer"].(map[string]any); ok {
			if langs, ok := tok["langs"].([]any); ok {
				return anySliceToStrings(langs)
			}
		}
	}
	// Try model_defaults.languages
	if md, ok := m.Config["model_defaults"].(map[string]any); ok {
		if langs, ok := md["languages"].([]any); ok {
			return anySliceToStrings(langs)
		}
	}
	// Default: just English.
	return []string{"en"}, nil
}

func anySliceToStrings(s []any) ([]string, error) {
	out := make([]string, len(s))
	for i, v := range s {
		str, ok := v.(string)
		if !ok {
			return nil, fmt.Errorf("checkpoint: expected string language tag, got %T", v)
		}
		out[i] = str
	}
	return out, nil
}

func containsAny(s string, subs ...string) bool {
	for _, sub := range subs {
		if len(s) >= len(sub) {
			for i := 0; i <= len(s)-len(sub); i++ {
				if s[i:i+len(sub)] == sub {
					return true
				}
			}
		}
	}
	return false
}
