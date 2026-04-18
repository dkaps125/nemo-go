// Package tokenizer provides tokenization for NeMo ASR models.
// Parakeet uses a plain SentencePiece BPE tokenizer.
// Canary uses an AggregateTokenizer: SentencePiece + special tokens.
package tokenizer

// Tokenizer is the common interface for all NeMo tokenizers.
type Tokenizer interface {
	// Encode converts text to token IDs.
	Encode(text string) ([]int32, error)
	// Decode converts token IDs to text, skipping special tokens.
	Decode(ids []int32) (string, error)
	// VocabSize returns the total number of tokens (including special).
	VocabSize() int
	// BlankID returns the CTC blank token ID (used by Parakeet).
	BlankID() int
	// BOSID returns the beginning-of-sequence token ID (used by Canary).
	BOSID() int
	// EOSID returns the end-of-sequence token ID (used by Canary).
	EOSID() int
	// TokenID returns the ID of a specific token string, and whether it was found.
	TokenID(token string) (int32, bool)
}
