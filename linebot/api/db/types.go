package db

import (
	"errors"
	"strings"
)

type UserState int8

// UserState represents the action that a user is currently taking
const (
	WritingNotes UserState = iota
	ViewingExpertVideos
	ViewingPortfoilo
	AnalyzingVideo
	ReadingInstruction
	None
)

func (s UserState) String() string {
	return [...]string{"writing_notes", "viewing_expert_videos", "viewing_portfolio", "analyzing_video", "reading_instruction", "none"}[s]
}

func (s UserState) ChnString() string {
	return [...]string{"預習及反思", "專家影片", "學習歷程", "動作分析", "使用說明", "無"}[s]
}

func UserStateChnStrToEnum(str string) (UserState, error) {
	switch str {
	case "預習及反思":
		return WritingNotes, nil
	case "專家影片":
		return ViewingExpertVideos, nil
	case "學習歷程":
		return ViewingPortfoilo, nil
	case "動作分析":
		return AnalyzingVideo, nil
	case "使用說明":
		return ReadingInstruction, nil
	case "無":
		return None, nil
	default:
		return -1, errors.New("invalid user state")
	}
}

// ActionStep represents the step of the action that a user is currently taking
type ActionStep int8

const (
	SelectingSkill ActionStep = iota
	SelectingHandedness
	WritingPreviewNote
	WritingReflection
	UploadingVideo
	SelectingPortfolio
	Empty
)

func ActionStepStrToEnum(str string) (ActionStep, error) {
	switch str {
	case "selecting_skill":
		return SelectingSkill, nil
	case "selecting_handedness":
		return SelectingHandedness, nil
	case "writing_preview_note":
		return WritingPreviewNote, nil
	case "writing_reflection":
		return WritingReflection, nil
	case "uploading_video":
		return UploadingVideo, nil
	case "selecting_portfolio":
		return SelectingPortfolio, nil
	case "empty":
		return Empty, nil
	default:
		return -1, errors.New("invalid action step")
	}
}

func (s ActionStep) String() string {
	return [...]string{"selecting_skill", "selecting_handedness", "writing_preview_note", "writing_reflection", "uploading_video", "selecting_portfolio", "empty"}[s]
}

// Handedness represents the handedness of a player
type Handedness int8

const (
	Left Handedness = iota
	Right
)

func (h Handedness) String() string {
	return [...]string{"left", "right"}[h]
}

func (h Handedness) ChnString() string {
	return [...]string{"左手", "右手"}[h]
}

func HandednessStrToEnum(str string) (Handedness, error) {
	switch str {
	case "left":
		return Left, nil
	case "right":
		return Right, nil
	default:
		return -1, errors.New("invalid handedness")
	}
}

// SkillOrder is the order skills are reported in, so a learner's records read
// the same way every time regardless of Go's map iteration. It is derived from
// the skill enum so the two cannot drift apart.
var SkillOrder = [...]string{
	Serve.String(),
	Smash.String(),
	Clear.String(),
	Lift.String(),
}

// Badminton skill types
type BadmintonSkill int8

const (
	Serve BadmintonSkill = iota
	Smash
	Clear
	Lift
)

func (s BadmintonSkill) String() string {
	return [...]string{"serve", "smash", "clear", "lift"}[s]
}

func (s BadmintonSkill) ChnString() string {
	return [...]string{"發球", "殺球", "高遠球", "挑球"}[s]
}

func SkillStrToEnum(str string) BadmintonSkill {
	switch str {
	case "serve":
		return Serve
	case "smash":
		return Smash
	case "clear":
		return Clear
	case "lift":
		return Lift
	default:
		return -1
	}
}

// Valid reports whether the value names a real skill. SkillStrToEnum returns -1
// for anything it does not recognise, and String/ChnString index a fixed array,
// so an unchecked value would panic.
func (s BadmintonSkill) Valid() bool {
	return s >= Serve && s <= Lift
}

// unsupportedSkills are the skills the course is not running this semester.
//
// PER-SEMESTER TOGGLE: this map is the single source of truth for which skills
// students may start new work on. To offer a skill again next semester, delete
// its line here — the selection UI, the flow guards and the weekly preview all
// read from it, so nothing else needs to change. Removing a skill only blocks
// NEW activity; historical portfolio entries, scores and analyses for it stay
// readable everywhere.
var unsupportedSkills = map[BadmintonSkill]bool{
	Clear: true,
	Lift:  true,
}

// Supported reports whether a student may start new work on this skill this
// semester. An unrecognised skill is never supported.
func (s BadmintonSkill) Supported() bool {
	return s.Valid() && !unsupportedSkills[s]
}

// IsSupportedSkill reports whether a raw skill string (a postback payload, a
// stored session field, a query parameter) names a skill offered this semester.
func IsSupportedSkill(str string) bool {
	return SkillStrToEnum(str).Supported()
}

// SupportedSkills lists this semester's skills in enum order, so every place a
// student is offered a choice presents the same set in the same order.
func SupportedSkills() []BadmintonSkill {
	skills := make([]BadmintonSkill, 0, len(SkillOrder))
	for skill := Serve; skill <= Lift; skill++ {
		if skill.Supported() {
			skills = append(skills, skill)
		}
	}
	return skills
}

// SupportedSkillsChnString names this semester's skills for a learner-facing
// message, e.g. "【發球】、【殺球】".
func SupportedSkillsChnString() string {
	names := make([]string, 0, len(SkillOrder))
	for _, skill := range SupportedSkills() {
		names = append(names, "【"+skill.ChnString()+"】")
	}
	return strings.Join(names, "、")
}
