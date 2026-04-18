package tokenizer_test

import (
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/danielkapit/nemo-go/tokenizer"
)

// TestAggregateTokenizerSpecialTokens verifies that the AggregateTokenizer
// assigns special token IDs starting at spe.VocabSize() in the correct order.
func TestAggregateTokenizerSpecialTokens(t *testing.T) {
	// We can't load a real .model file in unit tests, so we use a stub.
	stub := &stubSPE{vocabSize: 1024}
	specials := []string{
		"<pad>",
		"<|startoftranscript|>",
		"<|endoftext|>",
		"<|transcribe|>",
		"<|translate|>",
		"<|nopnc|>",
		"<|pnc|>",
		"<|en|>",
		"<|de|>",
	}
	agg, err := tokenizer.NewAggregateTokenizer(stub, specials)
	require.NoError(t, err)

	assert.Equal(t, 1024+len(specials), agg.VocabSize())

	for i, tok := range specials {
		id, ok := agg.TokenID(tok)
		assert.True(t, ok, "token %q not found", tok)
		assert.Equal(t, int32(1024+i), id, "wrong ID for %q", tok)
	}

	assert.Equal(t, 1024+1, agg.BOSID(), "<|startoftranscript|> should be BOS")
	assert.Equal(t, 1024+2, agg.EOSID(), "<|endoftext|> should be EOS")
}

// TestAggregateTokenizerBuildPromptIDs verifies Canary prompt construction.
func TestAggregateTokenizerBuildPromptIDs(t *testing.T) {
	stub := &stubSPE{vocabSize: 1024}
	specials := tokenizer.CanarySpecialTokens([]string{"en", "de", "es"})
	agg, err := tokenizer.NewAggregateTokenizer(stub, specials)
	require.NoError(t, err)

	t.Run("transcribe_en_pnc", func(t *testing.T) {
		ids, err := agg.BuildPromptIDs("en", "transcribe", "en", true)
		require.NoError(t, err)
		// Expect: <|startoftranscript|>, <|en|>, <|transcribe|>, <|pnc|>
		require.Len(t, ids, 4)
		startTok, _ := agg.TokenID("<|startoftranscript|>")
		enTok, _ := agg.TokenID("<|en|>")
		transcribeTok, _ := agg.TokenID("<|transcribe|>")
		pncTok, _ := agg.TokenID("<|pnc|>")
		assert.Equal(t, []int32{startTok, enTok, transcribeTok, pncTok}, ids)
	})

	t.Run("translate_en_to_de", func(t *testing.T) {
		ids, err := agg.BuildPromptIDs("en", "translate", "de", false)
		require.NoError(t, err)
		// Expect: <|startoftranscript|>, <|en|>, <|translate|>, <|de|>, <|nopnc|>
		require.Len(t, ids, 5)
		translateTok, _ := agg.TokenID("<|translate|>")
		deTok, _ := agg.TokenID("<|de|>")
		nopncTok, _ := agg.TokenID("<|nopnc|>")
		startTok, _ := agg.TokenID("<|startoftranscript|>")
		enTok, _ := agg.TokenID("<|en|>")
		assert.Equal(t, []int32{startTok, enTok, translateTok, deTok, nopncTok}, ids)
	})

	t.Run("unknown_language_returns_error", func(t *testing.T) {
		_, err := agg.BuildPromptIDs("xx", "transcribe", "xx", true)
		assert.Error(t, err)
	})
}

// TestAggregateTokenizerDecode verifies special tokens are stripped from output.
func TestAggregateTokenizerDecode(t *testing.T) {
	stub := &stubSPE{
		vocabSize: 10,
		pieces:    []string{"he", "llo", "▁world", "▁", "!", "▁foo", "▁bar", "a", "b", "c"},
	}
	specials := []string{"<|startoftranscript|>", "<|endoftext|>", "<|en|>"}
	agg, err := tokenizer.NewAggregateTokenizer(stub, specials)
	require.NoError(t, err)

	// Token IDs: prompt(10,12,11) + text(0,1) + eos(11)
	bosID := int32(10) // <|startoftranscript|>
	eosID := int32(11) // <|endoftext|>
	enID := int32(12)  // <|en|>

	ids := []int32{bosID, enID, 0, 1, eosID}
	text, err := agg.Decode(ids)
	require.NoError(t, err)
	assert.Equal(t, "hello", text)
}

// TestSentencePieceLoadReal loads a real .model file if TEST_SPM_PATH is set.
func TestSentencePieceLoadReal(t *testing.T) {
	path := os.Getenv("TEST_SPM_PATH")
	if path == "" {
		t.Skip("set TEST_SPM_PATH=/path/to/tokenizer.model to run this test")
	}
	tok, err := tokenizer.LoadSentencePiece(path, -1)
	require.NoError(t, err)
	assert.Greater(t, tok.VocabSize(), 0)

	ids, err := tok.Encode("hello world")
	require.NoError(t, err)
	assert.NotEmpty(t, ids)

	text, err := tok.Decode(ids)
	require.NoError(t, err)
	assert.Equal(t, "hello world", text)
}

// --- stub SentencePieceTokenizer for unit tests ---

type stubSPE struct {
	vocabSize int
	pieces    []string // if set, Decode joins these by ID
}

func (s *stubSPE) Encode(text string) ([]int32, error) { return nil, nil }
func (s *stubSPE) Decode(ids []int32) (string, error) {
	if s.pieces == nil {
		return "", nil
	}
	result := ""
	for _, id := range ids {
		if int(id) < len(s.pieces) {
			result += s.pieces[id]
		}
	}
	// Replace ▁ with space and trim.
	r := ""
	for _, ch := range result {
		if ch == '▁' {
			r += " "
		} else {
			r += string(ch)
		}
	}
	if len(r) > 0 && r[0] == ' ' {
		r = r[1:]
	}
	return r, nil
}
func (s *stubSPE) VocabSize() int                     { return s.vocabSize }
func (s *stubSPE) BlankID() int                       { return s.vocabSize }
func (s *stubSPE) BOSID() int                         { return -1 }
func (s *stubSPE) EOSID() int                         { return -1 }
func (s *stubSPE) TokenID(tok string) (int32, bool)   { return -1, false }
