package tokenizer

import (
	"fmt"
	"strings"
)

// AggregateTokenizer combines a SentencePiece base tokenizer with a map of
// special tokens, exactly mirroring NeMo's AggregateTokenizer used by Canary.
//
// Special token IDs start at SPE vocab size and are assigned in the order
// they appear in SpecialTokens. This ordering MUST match the order in
// model_config.yaml's tokenizer.langs + special_tokens fields.
type AggregateTokenizer struct {
	spe          Tokenizer         // base tokenizer (usually *SentencePieceTokenizer)
	specialNames []string          // ordered list of special token strings
	specialToID  map[string]int32
	idToSpecial  map[int32]string
	totalVocab   int
	bosID        int
	eosID        int
}

// NewAggregateTokenizer builds an AggregateTokenizer.
//
//   - spe: the base tokenizer (typically *SentencePieceTokenizer)
//   - specialTokens: ordered list of special token strings matching NeMo's
//     tokenizer config order. The first token gets ID = spe.VocabSize(),
//     the second gets spe.VocabSize()+1, etc.
func NewAggregateTokenizer(spe Tokenizer, specialTokens []string) (*AggregateTokenizer, error) {
	if len(specialTokens) == 0 {
		return nil, fmt.Errorf("aggregate: specialTokens must not be empty")
	}
	base := spe.VocabSize()
	t := &AggregateTokenizer{
		spe:          spe,
		specialNames: specialTokens,
		specialToID:  make(map[string]int32, len(specialTokens)),
		idToSpecial:  make(map[int32]string, len(specialTokens)),
		totalVocab:   base + len(specialTokens),
		bosID:        -1,
		eosID:        -1,
	}
	for i, name := range specialTokens {
		id := int32(base + i)
		t.specialToID[name] = id
		t.idToSpecial[id] = name
		switch name {
		case "<|startoftranscript|>":
			t.bosID = int(id)
		case "<|endoftext|>":
			t.eosID = int(id)
		}
	}
	return t, nil
}

// Encode tokenizes text. If text is a known special token it returns that ID.
// Otherwise falls through to the SentencePiece tokenizer.
func (t *AggregateTokenizer) Encode(text string) ([]int32, error) {
	if id, ok := t.specialToID[text]; ok {
		return []int32{id}, nil
	}
	return t.spe.Encode(text)
}

// Decode converts token IDs to text. Special token IDs are skipped (they are
// control tokens that should not appear in the output transcript).
func (t *AggregateTokenizer) Decode(ids []int32) (string, error) {
	var baseIDs []int32
	for _, id := range ids {
		if _, ok := t.idToSpecial[id]; ok {
			// Skip special tokens in output.
			continue
		}
		baseIDs = append(baseIDs, id)
	}
	return t.spe.Decode(baseIDs)
}

// VocabSize returns the total vocabulary size (SPE + special).
func (t *AggregateTokenizer) VocabSize() int { return t.totalVocab }

// BlankID delegates to the underlying SPE tokenizer (not used by Canary).
func (t *AggregateTokenizer) BlankID() int { return t.spe.BlankID() }

// BOSID returns the <|startoftranscript|> token ID.
func (t *AggregateTokenizer) BOSID() int { return t.bosID }

// EOSID returns the <|endoftext|> token ID.
func (t *AggregateTokenizer) EOSID() int { return t.eosID }

// TokenID returns the ID of a token string (checks special tokens first).
func (t *AggregateTokenizer) TokenID(token string) (int32, bool) {
	if id, ok := t.specialToID[token]; ok {
		return id, true
	}
	return t.spe.TokenID(token)
}

// CanarySpecialTokens returns the default special token list for Canary models
// in the exact order NeMo assigns IDs.
//
// The order matches NeMo's CanaryTokenizer definition:
// ["<pad>", "<|startoftranscript|>", "<|endoftext|>", "<|transcribe|>",
//  "<|translate|>", "<|nopnc|>", "<|pnc|>", "<|en|>", "<|de|>", "<|es|>",
//  "<|fr|>", "<|hr|>", ...other languages in NeMo config order]
//
// WARNING: This list is model-specific. For production use, parse it from
// model_config.yaml via checkpoint.ParseCanaryTokenizerConfig().
func CanarySpecialTokens(langs []string) []string {
	base := []string{
		"<pad>",
		"<|startoftranscript|>",
		"<|endoftext|>",
		"<|transcribe|>",
		"<|translate|>",
		"<|nopnc|>",
		"<|pnc|>",
	}
	// Language tokens in the order provided by the model config.
	for _, lang := range langs {
		base = append(base, fmt.Sprintf("<|%s|>", lang))
	}
	return base
}

// BuildPromptIDs constructs the Canary decoder prompt token sequence.
// Returns [<|startoftranscript|>, <|srcLang|>, <|task|>, <|tgtLang|>, <|pnc|>/<|nopnc|>].
func (t *AggregateTokenizer) BuildPromptIDs(srcLang, task, tgtLang string, pnc bool) ([]int32, error) {
	var tokens []string

	tokens = append(tokens, "<|startoftranscript|>")
	tokens = append(tokens, fmt.Sprintf("<|%s|>", srcLang))

	taskToken := fmt.Sprintf("<|%s|>", strings.ToLower(task))
	tokens = append(tokens, taskToken)

	if strings.ToLower(task) == "translate" {
		tokens = append(tokens, fmt.Sprintf("<|%s|>", tgtLang))
	}

	if pnc {
		tokens = append(tokens, "<|pnc|>")
	} else {
		tokens = append(tokens, "<|nopnc|>")
	}

	ids := make([]int32, len(tokens))
	for i, tok := range tokens {
		id, ok := t.TokenID(tok)
		if !ok {
			return nil, fmt.Errorf("aggregate: unknown prompt token %q", tok)
		}
		ids[i] = id
	}
	return ids, nil
}
