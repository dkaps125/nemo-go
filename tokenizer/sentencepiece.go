package tokenizer

import (
	"fmt"
	"math"
	"os"
	"strings"
	"unicode/utf8"

	"google.golang.org/protobuf/encoding/protowire"
)

// SPE piece types matching SentencePiece's Type enum.
const (
	pieceNormal      = 1
	pieceUnknown     = 2
	pieceControl     = 3 // BOS, EOS, PAD
	pieceUserDefined = 4 // user-added special tokens
	pieceByte        = 6
)

type spePiece struct {
	piece string
	score float32
	typ   int
}

// SentencePieceTokenizer is a pure-Go SentencePiece tokenizer.
// It supports both BPE and unigram models for decoding, and greedy
// longest-match encoding (sufficient for Canary prompt construction).
type SentencePieceTokenizer struct {
	pieces    []spePiece        // indexed by token ID
	pieceToID map[string]int32  // piece text → token ID
	blankID   int               // index of blank token (last in NeMo BPE CTC vocab)
	bosID     int
	eosID     int
	padID     int
	unkID     int
}

// LoadSentencePiece loads a SentencePiece .model file.
// blankOverride sets the blank token ID for CTC models; pass -1 to use
// vocab_size (NeMo convention: blank = last token).
func LoadSentencePiece(path string, blankOverride int) (*SentencePieceTokenizer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("sentencepiece: read %s: %w", path, err)
	}
	pieces, err := parseModelProto(data)
	if err != nil {
		return nil, fmt.Errorf("sentencepiece: parse %s: %w", path, err)
	}
	if len(pieces) == 0 {
		return nil, fmt.Errorf("sentencepiece: empty vocabulary in %s", path)
	}

	t := &SentencePieceTokenizer{
		pieces:    pieces,
		pieceToID: make(map[string]int32, len(pieces)),
		bosID:     -1, eosID: -1, padID: -1, unkID: -1,
	}
	for i, p := range pieces {
		t.pieceToID[p.piece] = int32(i)
		switch p.piece {
		case "<s>", "<BOS>", "<bos>":
			t.bosID = i
		case "</s>", "<EOS>", "<eos>":
			t.eosID = i
		case "<pad>", "<PAD>":
			t.padID = i
		case "<unk>", "<UNK>":
			t.unkID = i
		}
		// Control pieces are typically BOS/EOS/PAD
		if p.typ == pieceControl {
			switch i {
			case 0:
				t.unkID = i
			case 1:
				t.bosID = i
			case 2:
				t.eosID = i
			}
		}
	}

	if blankOverride >= 0 {
		t.blankID = blankOverride
	} else {
		// NeMo convention: blank token is the last token.
		t.blankID = len(pieces)
	}

	return t, nil
}

// Encode tokenizes text using greedy longest-match BPE.
// Prepends a ▁ (U+2581) space prefix to mirror NeMo's add_dummy_prefix behavior.
func (t *SentencePieceTokenizer) Encode(text string) ([]int32, error) {
	if text == "" {
		return nil, nil
	}
	// NeMo SentencePiece models use ▁ as a word-boundary marker.
	normalized := "▁" + strings.ReplaceAll(text, " ", "▁")
	return t.greedyEncode(normalized), nil
}

// greedyEncode implements greedy longest-match tokenization.
func (t *SentencePieceTokenizer) greedyEncode(s string) []int32 {
	var ids []int32
	for len(s) > 0 {
		best := -1
		bestLen := 0
		// Try to find the longest matching piece starting at s[0].
		r := []rune(s)
		for end := len(r); end > 0; end-- {
			candidate := string(r[:end])
			if id, ok := t.pieceToID[candidate]; ok {
				best = int(id)
				bestLen = len(candidate)
				break
			}
		}
		if best == -1 {
			// Unknown: emit one UTF-8 character as UNK or fall back to byte pieces.
			_, size := utf8.DecodeRuneInString(s)
			if t.unkID >= 0 {
				ids = append(ids, int32(t.unkID))
			}
			s = s[size:]
		} else {
			ids = append(ids, int32(best))
			s = s[bestLen:]
		}
	}
	return ids
}

// Decode converts token IDs to text by joining pieces and replacing ▁ with spaces.
// Skips BOS, EOS, PAD, and UNK tokens.
func (t *SentencePieceTokenizer) Decode(ids []int32) (string, error) {
	var sb strings.Builder
	for _, id := range ids {
		if int(id) < 0 || int(id) >= len(t.pieces) {
			continue
		}
		p := t.pieces[id]
		// Skip control tokens.
		if p.typ == pieceControl || p.typ == pieceUnknown {
			continue
		}
		// Skip BOS/EOS/PAD by ID too.
		if int(id) == t.bosID || int(id) == t.eosID || int(id) == t.padID {
			continue
		}
		sb.WriteString(p.piece)
	}
	// ▁ (U+2581) marks word boundaries; replace with space and trim leading space.
	result := strings.ReplaceAll(sb.String(), "▁", " ")
	return strings.TrimPrefix(result, " "), nil
}

func (t *SentencePieceTokenizer) VocabSize() int   { return len(t.pieces) }
func (t *SentencePieceTokenizer) BlankID() int      { return t.blankID }
func (t *SentencePieceTokenizer) BOSID() int        { return t.bosID }
func (t *SentencePieceTokenizer) EOSID() int        { return t.eosID }
func (t *SentencePieceTokenizer) TokenID(tok string) (int32, bool) {
	id, ok := t.pieceToID[tok]
	return id, ok
}

// parseModelProto parses the SentencePiece ModelProto binary protobuf.
// It extracts only the pieces (field 1 of ModelProto), each of which
// contains piece (field 1), score (field 2), and type (field 3).
func parseModelProto(data []byte) ([]spePiece, error) {
	var pieces []spePiece
	b := data
	for len(b) > 0 {
		num, typ, n := protowire.ConsumeTag(b)
		if n < 0 {
			return nil, fmt.Errorf("bad proto tag")
		}
		b = b[n:]
		if num == 1 && typ == protowire.BytesType {
			// SentencePiece message (repeated, field 1).
			msgBytes, n := protowire.ConsumeBytes(b)
			if n < 0 {
				return nil, fmt.Errorf("bad bytes field")
			}
			b = b[n:]
			p, err := parsePiece(msgBytes)
			if err != nil {
				return nil, err
			}
			pieces = append(pieces, p)
		} else {
			// Skip other fields.
			n := protowire.ConsumeFieldValue(num, typ, b)
			if n < 0 {
				return nil, fmt.Errorf("bad field value (field %d, type %d)", num, typ)
			}
			b = b[n:]
		}
	}
	return pieces, nil
}

func parsePiece(data []byte) (spePiece, error) {
	var p spePiece
	p.typ = pieceNormal
	b := data
	for len(b) > 0 {
		num, typ, n := protowire.ConsumeTag(b)
		if n < 0 {
			return p, fmt.Errorf("bad piece tag")
		}
		b = b[n:]
		switch {
		case num == 1 && typ == protowire.BytesType: // piece string
			s, n := protowire.ConsumeBytes(b)
			if n < 0 {
				return p, fmt.Errorf("bad piece string")
			}
			p.piece = string(s)
			b = b[n:]
		case num == 2 && typ == protowire.Fixed32Type: // score (float32)
			v, n := protowire.ConsumeFixed32(b)
			if n < 0 {
				return p, fmt.Errorf("bad piece score")
			}
			p.score = math.Float32frombits(v)
			b = b[n:]
		case num == 3 && typ == protowire.VarintType: // type enum
			v, n := protowire.ConsumeVarint(b)
			if n < 0 {
				return p, fmt.Errorf("bad piece type")
			}
			p.typ = int(v)
			b = b[n:]
		default:
			n := protowire.ConsumeFieldValue(num, typ, b)
			if n < 0 {
				return p, fmt.Errorf("bad piece field")
			}
			b = b[n:]
		}
	}
	return p, nil
}
