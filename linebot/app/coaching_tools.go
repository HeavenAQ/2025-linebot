package app

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/HeavenAQ/nstc-linebot-2025/api/db"
	"github.com/HeavenAQ/nstc-linebot-2025/api/gpt"
)

// skillEnum lists the strokes a tool may be asked about. Anything else is
// refused by OpenAI before the handler runs.
var skillEnum = []any{db.Serve.String(), db.Smash.String(), db.Clear.String(), db.Lift.String()}

// object builds a closed JSON schema. Strict tools require every property to
// be listed as required and forbid extras, so a malformed call never reaches
// Firestore.
func object(properties map[string]any) map[string]any {
	required := make([]any, 0, len(properties))
	for name := range properties {
		required = append(required, name)
	}
	return map[string]any{
		"type":                 "object",
		"properties":           properties,
		"required":             required,
		"additionalProperties": false,
	}
}

func encode(value any) (string, error) {
	raw, err := json.Marshal(value)
	if err != nil {
		return "", fmt.Errorf("encode result: %w", err)
	}
	return string(raw), nil
}

// coachingToolArgs is every field any tool takes. Each tool declares only the
// ones it uses; this is just the decoding target.
type coachingToolArgs struct {
	Skill string `json:"skill"`
	Limit int    `json:"limit"`
	Date  string `json:"date"`
	Week  string `json:"week"`
}

func parseArgs(arguments string) (coachingToolArgs, error) {
	var args coachingToolArgs
	if strings.TrimSpace(arguments) == "" {
		return args, nil
	}
	if err := json.Unmarshal([]byte(arguments), &args); err != nil {
		return args, fmt.Errorf("arguments are not valid JSON")
	}
	args.Skill = strings.ToLower(strings.TrimSpace(args.Skill))
	if args.Skill != "" && !db.IsSupportedSkill(args.Skill) {
		return args, fmt.Errorf("unsupported skill %q", args.Skill)
	}
	if args.Limit <= 0 || args.Limit > 20 {
		args.Limit = 5
	}
	return args, nil
}

// CoachingTools are the lookups the coach may use while answering userID.
//
// The learner is fixed here, by the caller, and is never a tool argument: a
// question -- or anything quoted inside it -- must not be able to name someone
// else's record. Everything is read-only, and class comparisons come back
// without any identity attached.
func (app *App) CoachingTools(userID string) []gpt.Tool {
	return []gpt.Tool{
		{
			Name:        "get_recent_scores",
			Description: "The learner's own recent graded attempts for one stroke, newest first, with each criterion's score.",
			Parameters: object(map[string]any{
				"skill": map[string]any{"type": "string", "enum": skillEnum},
				"limit": map[string]any{"type": "integer", "description": "How many attempts, 1-20."},
			}),
			Handler: func(ctx context.Context, arguments string) (string, error) {
				args, err := parseArgs(arguments)
				if err != nil {
					return "", err
				}
				scores, err := app.FirestoreClient.GetRecentSkillScores(userID, args.Skill, args.Limit)
				if err != nil {
					return "", err
				}
				return encode(map[string]any{"skill": args.Skill, "attempts": scores})
			},
		},
		{
			Name:        "get_attempt",
			Description: "One of the learner's own attempts in full: every criterion, the coach's cues from that analysis, and the expert it was compared against. Use the date reported by get_recent_scores.",
			Parameters: object(map[string]any{
				"skill": map[string]any{"type": "string", "enum": skillEnum},
				"date":  map[string]any{"type": "string", "description": "The attempt's date key, exactly as get_recent_scores reported it."},
			}),
			Handler: func(ctx context.Context, arguments string) (string, error) {
				args, err := parseArgs(arguments)
				if err != nil {
					return "", err
				}
				user, err := app.FirestoreClient.GetUserData(userID)
				if err != nil {
					return "", err
				}
				work, ok := user.Portfolio.GetSkillPortfolio(args.Skill)[args.Date]
				if !ok {
					return encode(map[string]any{"error": "no attempt with that date", "skill": args.Skill})
				}
				return encode(map[string]any{
					"skill":            args.Skill,
					"date":             args.Date,
					"total_grade":      work.GradingOutcome.TotalGrade,
					"criteria":         work.GradingOutcome.GradingDetails,
					"coaching_cues":    work.CoachingCues,
					"overall_feedback": work.AINote,
					"expert":           work.Expert.DisplayName,
					"handedness":       work.Handedness,
				})
			},
		},
		{
			Name:        "get_class_standing",
			Description: "Where the learner's best attempt stands in the class for one stroke: rank, percentile, how many classmates have attempted it, the class mean and the class best total. No classmate is named.",
			Parameters: object(map[string]any{
				"skill": map[string]any{"type": "string", "enum": skillEnum},
			}),
			Handler: func(ctx context.Context, arguments string) (string, error) {
				args, err := parseArgs(arguments)
				if err != nil {
					return "", err
				}
				standing, err := app.FirestoreClient.ClassStandingFor(ctx, userID, args.Skill)
				if err != nil {
					return "", err
				}
				return encode(standing)
			},
		},
		{
			Name:        "get_class_best_attempt",
			Description: "The strongest attempt anyone in the class has recorded for one stroke, as a per-criterion breakdown. It is anonymous: use it to show what a high score looks like on each criterion, and never suggest whose attempt it is.",
			Parameters: object(map[string]any{
				"skill": map[string]any{"type": "string", "enum": skillEnum},
			}),
			Handler: func(ctx context.Context, arguments string) (string, error) {
				args, err := parseArgs(arguments)
				if err != nil {
					return "", err
				}
				best, err := app.FirestoreClient.ClassBestAttemptFor(ctx, args.Skill)
				if err != nil {
					return "", err
				}
				return encode(best)
			},
		},
		{
			Name:        "get_weekly_notes",
			Description: "What the learner wrote themselves: their weekly reflections and pre-class notes, newest first.",
			Parameters: object(map[string]any{
				"limit": map[string]any{"type": "integer", "description": "How many weeks, 1-20."},
			}),
			Handler: func(ctx context.Context, arguments string) (string, error) {
				args, err := parseArgs(arguments)
				if err != nil {
					return "", err
				}
				weekly, err := app.FirestoreClient.ListWeeklyReflections(userID)
				if err != nil {
					return "", err
				}
				weeks := make([]map[string]any, 0, len(weekly))
				for week, reflection := range weekly {
					weeks = append(weeks, map[string]any{
						"week": week, "note": reflection.Note, "preview": reflection.Preview,
					})
				}
				// ISO week labels sort newest-last as strings, and the map
				// they came from has no order at all.
				sort.Slice(weeks, func(i, j int) bool {
					return weeks[i]["week"].(string) > weeks[j]["week"].(string)
				})
				if len(weeks) > args.Limit {
					weeks = weeks[:args.Limit]
				}
				return encode(map[string]any{"weeks": weeks})
			},
		},
	}
}
