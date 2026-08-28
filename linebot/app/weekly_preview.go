package app

import (
	"fmt"
	"strings"
	"time"

	"github.com/HeavenAQ/nstc-linebot-2025/api/db"
	"github.com/HeavenAQ/nstc-linebot-2025/commons"
)

// How many graded attempts per skill the weekly preview reasons over.
const weeklyPreviewScoreLimit = 5

// PreviewStatus is what happened to one learner in a weekly preview run.
type PreviewStatus string

const (
	PreviewPushed      PreviewStatus = "pushed"
	PreviewPrepared    PreviewStatus = "prepared"
	PreviewAlreadySent PreviewStatus = "already_sent"
	PreviewNoHistory   PreviewStatus = "no_history"
	// PreviewNoSupportedSkill is a learner who has only ever attempted skills
	// the course is not running this semester. There is nothing useful to
	// preview, so they are skipped rather than pushed a note about a skill they
	// cannot practise. Reported separately from PreviewNoHistory so a run makes
	// the reason visible.
	PreviewNoSupportedSkill PreviewStatus = "no_supported_skill"
	PreviewFailed           PreviewStatus = "failed"
)

// PreviewOutcome reports one learner's result so a run can be inspected
// without reading the logs.
type PreviewOutcome struct {
	UserID string        `json:"user_id"`
	Name   string        `json:"name"`
	Skill  string        `json:"skill,omitempty"`
	Note   string        `json:"note,omitempty"`
	Status PreviewStatus `json:"status"`
	Error  string        `json:"error,omitempty"`
}

// supportedSkillHistory drops the skills the course is not running this
// semester. A learner's stored history still spans every skill they have ever
// attempted, but a preview may only ever focus on — and reason from — a skill
// they can actually practise next lesson.
func supportedSkillHistory(history []commons.SkillHistory) []commons.SkillHistory {
	supported := make([]commons.SkillHistory, 0, len(history))
	for _, candidate := range history {
		if len(candidate.Scores) == 0 || !db.IsSupportedSkill(candidate.Skill) {
			continue
		}
		supported = append(supported, candidate)
	}
	return supported
}

// focusSkill picks the skill a learner should work on this week: the one whose
// latest attempt scored lowest, since that is what they are carrying into the
// next lesson. Ties go to the weaker recent average, then to SkillOrder so the
// choice never depends on map ordering. Callers pass history already narrowed
// by supportedSkillHistory.
func focusSkill(history []commons.SkillHistory) commons.SkillHistory {
	best := history[0]
	for _, candidate := range history[1:] {
		current, currentAverage := best.Scores[0].TotalGrade, averageGrade(best)
		next, nextAverage := candidate.Scores[0].TotalGrade, averageGrade(candidate)
		if next < current || (next == current && nextAverage < currentAverage) {
			best = candidate
		}
	}
	return best
}

func averageGrade(skill commons.SkillHistory) float64 {
	if len(skill.Scores) == 0 {
		return 0
	}
	total := 0.0
	for _, score := range skill.Scores {
		total += score.TotalGrade
	}
	return total / float64(len(skill.Scores))
}

// previewFocus reads a learner's history and picks the skill a preview should
// be about. It reports false when there is nothing to preview, having recorded
// the reason on the outcome, so the scheduled run and the on-demand button
// choose the same skill and report the same statuses.
func (app *App) previewFocus(user db.UserData, outcome *PreviewOutcome) (db.BadmintonSkill, []commons.SkillHistory, bool) {
	history, err := app.FirestoreClient.LearningHistory(user.ID, weeklyPreviewScoreLimit)
	if err != nil {
		outcome.Status = PreviewFailed
		outcome.Error = fmt.Sprintf("read learning history: %v", err)
		return 0, nil, false
	}
	if len(history) == 0 {
		outcome.Status = PreviewNoHistory
		return 0, nil, false
	}

	// Narrowed before the focus is chosen and before the note is written, so
	// neither the headline skill nor the reasoning behind it can mention a
	// skill the learner has no way to practise this semester.
	history = supportedSkillHistory(history)
	if len(history) == 0 {
		outcome.Status = PreviewNoSupportedSkill
		return 0, nil, false
	}

	focus := focusSkill(history)
	outcome.Skill = focus.Skill
	// ChnString indexes a fixed array, so an unrecognised skill would panic and
	// take the rest of the roster down with it. Fail this learner instead.
	skill := db.SkillStrToEnum(focus.Skill)
	if !skill.Supported() {
		outcome.Status = PreviewFailed
		outcome.Error = fmt.Sprintf("unsupported skill: %s", focus.Skill)
		return 0, nil, false
	}
	return skill, history, true
}

// WeeklyPreviewOnDemand writes the note behind the 產生課前預習 button. It shares
// the scheduled run's skill choice and history depth, but deliberately neither
// dedupes nor records: the learner asked for this one, so a second tap is a
// second answer, and it must not consume that week's scheduled push.
func (app *App) WeeklyPreviewOnDemand(user db.UserData) PreviewOutcome {
	outcome := PreviewOutcome{UserID: user.ID, Name: user.Name}

	skill, history, ok := app.previewFocus(user, &outcome)
	if !ok {
		return outcome
	}

	note, err := app.GPTClient.WeeklyPreview(user.Name, skill.ChnString(), history)
	if err != nil {
		outcome.Status = PreviewFailed
		outcome.Error = fmt.Sprintf("generate preview: %v", err)
		return outcome
	}
	outcome.Note = strings.TrimSpace(note)
	outcome.Status = PreviewPrepared
	return outcome
}

// WeeklyPreviewForUser writes one learner's 課前預習 note and pushes it.
//
// A learner is messaged at most once per ISO week: the run is expected to be
// retried and rescheduled, and a duplicate push reads as a bug to a student.
// With dryRun set the note is written but not delivered or recorded, so a run
// can be reviewed before any student sees it.
func (app *App) WeeklyPreviewForUser(user db.UserData, week string, dryRun bool) PreviewOutcome {
	outcome := PreviewOutcome{UserID: user.ID, Name: user.Name}

	skill, history, ok := app.previewFocus(user, &outcome)
	if !ok {
		return outcome
	}

	if !dryRun {
		sent, err := app.FirestoreClient.GetWeeklyPreview(user.ID, week)
		if err != nil {
			outcome.Status = PreviewFailed
			outcome.Error = fmt.Sprintf("read weekly preview: %v", err)
			return outcome
		}
		if sent != nil {
			outcome.Status = PreviewAlreadySent
			outcome.Note = sent.Message
			outcome.Skill = sent.Skill
			return outcome
		}
	}

	note, err := app.GPTClient.WeeklyPreview(user.Name, skill.ChnString(), history)
	if err != nil {
		outcome.Status = PreviewFailed
		outcome.Error = fmt.Sprintf("generate preview: %v", err)
		return outcome
	}
	outcome.Note = strings.TrimSpace(note)

	if dryRun {
		outcome.Status = PreviewPrepared
		return outcome
	}

	if err := app.LineBot.PushWeeklyPreview(user.ID, skill, outcome.Note); err != nil {
		outcome.Status = PreviewFailed
		outcome.Error = fmt.Sprintf("push preview: %v", err)
		return outcome
	}
	// Recorded only after delivery, so a failed push is retried next run rather
	// than silently marked done.
	if err := app.FirestoreClient.SetWeeklyPreview(user.ID, week, outcome.Skill, outcome.Note); err != nil {
		app.Logger.Warn.Printf("[preview.weekly] user_id=%s record_failed err=%v", user.ID, err)
	}
	outcome.Status = PreviewPushed
	return outcome
}

// RunWeeklyPreview previews every listed learner, or the whole roster when no
// IDs are given. One learner's failure never stops the rest of the run.
func (app *App) RunWeeklyPreview(at time.Time, dryRun bool, userIDs []string) ([]PreviewOutcome, error) {
	week := db.ISOWeek(at)
	users, err := app.selectPreviewUsers(userIDs)
	if err != nil {
		return nil, err
	}

	outcomes := make([]PreviewOutcome, 0, len(users))
	for _, user := range users {
		outcome := app.WeeklyPreviewForUser(user, week, dryRun)
		if outcome.Status == PreviewFailed {
			app.Logger.Error.Printf(
				"[preview.weekly] user_id=%s week=%s status=%s error=%s",
				outcome.UserID, week, outcome.Status, outcome.Error,
			)
		} else {
			app.Logger.Info.Printf(
				"[preview.weekly] user_id=%s week=%s status=%s skill=%s",
				outcome.UserID, week, outcome.Status, outcome.Skill,
			)
		}
		outcomes = append(outcomes, outcome)
	}
	return outcomes, nil
}

func (app *App) selectPreviewUsers(userIDs []string) ([]db.UserData, error) {
	if len(userIDs) == 0 {
		all, err := app.FirestoreClient.ListUsers()
		if err != nil {
			return nil, fmt.Errorf("list users: %w", err)
		}
		return *all, nil
	}
	users := make([]db.UserData, 0, len(userIDs))
	for _, userID := range userIDs {
		user, err := app.FirestoreClient.GetUserData(userID)
		if err != nil {
			return nil, fmt.Errorf("user %s: %w", userID, err)
		}
		users = append(users, *user)
	}
	return users, nil
}
