package db

import "errors"

type enum interface {
	String() string
	ChnString() string
}

type UserState int8

// UserState represents the action that a user is currently taking
const (
	WritingNotes UserState = iota
	ChattingWithGPT
	ChattingWithTeacher
	ViewingDashboard
	ViewingExpertVideos
	ViewingPortfoilo
	AnalyzingVideo
	None
)

func (state UserState) String() string {
	return [...]string{"writing reflection", "writing preview note", "chatting with GPT", "chatting with teacher", "viewing dashboard", "viewing expert videos", "viewing portfolio", "analyzing video", "none"}[state]
}

// ActionStep represents the step of the action that a user is currently taking
type ActionStep int8

const (
	SelectingSkill ActionStep = iota
	Writing
	Chatting
	ChoosingVideoUploadMethod
	Empty
)

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

// Badminton skill types
type BadmintonSkill int8

const (
	Serve BadmintonSkill = iota
	Smash
	Clear
)

func (s BadmintonSkill) String() string {
	return [...]string{"serve", "smash", "clear"}[s]
}

func (s BadmintonSkill) ChnString() string {
	return [...]string{"發球", "殺球", "高遠球"}[s]
}

func SkillStrToEnum(str string) BadmintonSkill {
	switch str {
	case "serve":
		return Serve
	case "smash":
		return Smash
	case "clear":
		return Clear
	default:
		return -1
	}
}
