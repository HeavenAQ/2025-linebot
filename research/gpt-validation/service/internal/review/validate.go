package review

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"strconv"
	"unicode/utf8"
)

// Request limits.
const (
	MaxBodyBytes     = 64 << 10
	MaxCommentRunes  = 2000
	MaxClientSeconds = 24 * 60 * 60
)

// ValidationError is a client error with a message safe to return.
type ValidationError struct{ Msg string }

func (e *ValidationError) Error() string { return e.Msg }

func invalid(format string, args ...any) error {
	return &ValidationError{Msg: fmt.Sprintf(format, args...)}
}

type step1Body struct {
	NeedsImprovement map[string]json.RawMessage `json:"needs_improvement"`
	Seconds          *float64                   `json:"seconds"`
}

type step2Body struct {
	CueRatings   map[string]json.RawMessage `json:"cue_ratings"`
	OverallScore json.RawMessage            `json:"overall_score"`
	Comment      *string                    `json:"comment"`
	Seconds      *float64                   `json:"seconds"`
}

// decodeStrict decodes exactly one JSON object with no unknown fields.
func decodeStrict(r io.Reader, dst any) error {
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()
	if err := dec.Decode(dst); err != nil {
		var tooLarge *http.MaxBytesError
		if errors.As(err, &tooLarge) {
			return err
		}
		return invalid("malformed JSON body")
	}
	if dec.More() {
		return invalid("unexpected data after JSON body")
	}
	if _, err := dec.Token(); err != io.EOF {
		return invalid("unexpected data after JSON body")
	}
	return nil
}

func validSeconds(p *float64) (float64, error) {
	if p == nil {
		return 0, invalid("seconds is required")
	}
	s := *p
	if math.IsNaN(s) || math.IsInf(s, 0) || s < 0 {
		return 0, invalid("seconds must be a non-negative number")
	}
	// A page left open overnight should still submit; cap rather than reject.
	return math.Min(s, MaxClientSeconds), nil
}

// ParseStep1 validates a step 1 body against the item's rubric: the keys must
// be exactly the item's criterion ids and every value a boolean.
func ParseStep1(item Item, body io.Reader) (map[string]bool, float64, error) {
	var b step1Body
	if err := decodeStrict(body, &b); err != nil {
		return nil, 0, err
	}
	if b.NeedsImprovement == nil {
		return nil, 0, invalid("needs_improvement is required")
	}
	ids := item.CriterionIDs()
	if len(b.NeedsImprovement) != len(ids) {
		return nil, 0, invalid("needs_improvement must have exactly %d criteria", len(ids))
	}
	out := make(map[string]bool, len(ids))
	for _, id := range ids {
		raw, ok := b.NeedsImprovement[id]
		if !ok {
			return nil, 0, invalid("needs_improvement is missing criterion %q", id)
		}
		switch string(bytes.TrimSpace(raw)) {
		case "true":
			out[id] = true
		case "false":
			out[id] = false
		default:
			return nil, 0, invalid("needs_improvement[%q] must be a boolean", id)
		}
	}
	seconds, err := validSeconds(b.Seconds)
	if err != nil {
		return nil, 0, err
	}
	return out, seconds, nil
}

// ParseStep2 validates a step 2 body: one rating per GPT cue (keyed by cue
// index), an integer overall score 0-3, an optional comment and seconds.
func ParseStep2(item Item, body io.Reader) (Step2, error) {
	var b step2Body
	if err := decodeStrict(body, &b); err != nil {
		return Step2{}, err
	}
	if b.CueRatings == nil {
		if len(item.GPTCues) > 0 {
			return Step2{}, invalid("cue_ratings is required")
		}
		b.CueRatings = map[string]json.RawMessage{}
	}
	if len(b.CueRatings) != len(item.GPTCues) {
		return Step2{}, invalid("cue_ratings must have exactly %d entries", len(item.GPTCues))
	}
	cues := make(map[string]string, len(item.GPTCues))
	for _, cue := range item.GPTCues {
		key := strconv.Itoa(cue.Index)
		raw, ok := b.CueRatings[key]
		if !ok {
			return Step2{}, invalid("cue_ratings is missing cue %s", key)
		}
		var v string
		if err := json.Unmarshal(raw, &v); err != nil {
			return Step2{}, invalid("cue_ratings[%s] must be a string", key)
		}
		if _, ok := CueScore(v); !ok {
			return Step2{}, invalid("cue_ratings[%s] must be correct, partial or incorrect", key)
		}
		cues[key] = v
	}

	scoreRaw := bytes.TrimSpace(b.OverallScore)
	if len(scoreRaw) == 0 || string(scoreRaw) == "null" {
		return Step2{}, invalid("overall_score is required")
	}
	score, err := strconv.Atoi(string(scoreRaw))
	if err != nil || score < 0 || score > 3 {
		return Step2{}, invalid("overall_score must be an integer from 0 to 3")
	}

	comment := ""
	if b.Comment != nil {
		comment = *b.Comment
	}
	if !utf8.ValidString(comment) || utf8.RuneCountInString(comment) > MaxCommentRunes {
		return Step2{}, invalid("comment must be at most %d characters", MaxCommentRunes)
	}

	seconds, err := validSeconds(b.Seconds)
	if err != nil {
		return Step2{}, err
	}
	return Step2{CueRatings: cues, OverallScore: score, Comment: comment, Seconds: seconds}, nil
}
