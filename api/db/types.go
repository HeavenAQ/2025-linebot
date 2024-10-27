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
	ViewingExpertVideos
	ViewingPortfoilo
	AnalyzingVideo
	ReadingInstruction
	None
)

func (s UserState) String() string {
	return [...]string{"writing_notes", "chatting_with_gpt", "viewing_expert_videos", "viewing_portfolio", "analyzing_video", "reading_instruction", "none"}[s]
}

func (s UserState) ChnString() string {
	return [...]string{"預習及反思", "GPT對談", "專家影片", "學習歷程", "動作分析", "使用說明", "無"}[s]
}

func UserStateChnStrToEnum(str string) (UserState, error) {
	switch str {
	case "預習及反思":
		return WritingNotes, nil
	case "GPT對談":
		return ChattingWithGPT, nil
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
	Chatting
	ChoosingVideoUploadMethod
	Empty
)

func (s ActionStep) String() string {
	return [...]string{"selecting_skill", "selecting_handedness", "writing_preview_note", "writing_reflection", "chatting", "choosing_video_upload_method", "empty"}[s]
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
