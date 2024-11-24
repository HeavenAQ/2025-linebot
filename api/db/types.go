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
	UploadingVideo
	ReadingInstruction
	None
)

func (s UserState) String() string {
	return [...]string{"writing_notes", "chatting_with_gpt", "viewing_expert_videos", "viewing_portfolio", "uploading_video", "reading_instruction", "none"}[s]
}

func (s UserState) ChnString() string {
	return [...]string{"學習反思", "GPT", "專家影片", "學習歷程", "影片上傳", "使用說明", "無"}[s]
}

func UserStateChnStrToEnum(str string) (UserState, error) {
	switch str {
	case "學習反思":
		return WritingNotes, nil
	case "GPT":
		return ChattingWithGPT, nil
	case "專家影片":
		return ViewingExpertVideos, nil
	case "學習歷程":
		return ViewingPortfoilo, nil
	case "影片上傳":
		return UploadingVideo, nil
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
	WritingReflection
	SelectingVideoUploadMethod
	Chatting
	SelectingPortfolio
	Empty
)

func ActionStepStrToEnum(str string) (ActionStep, error) {
	switch str {
	case "selecting_skill":
		return SelectingSkill, nil
	case "selecting_handedness":
		return SelectingHandedness, nil
	case "writing_reflection":
		return WritingReflection, nil
	case "selecting_video_upload_method":
		return SelectingVideoUploadMethod, nil
	case "chatting":
		return Chatting, nil
	case "selecting_portfolio":
		return SelectingPortfolio, nil
	case "empty":
		return Empty, nil
	default:
		return -1, errors.New("invalid action step")
	}
}

func (s ActionStep) String() string {
	return [...]string{"selecting_skill", "selecting_handedness", "writing_reflection", "selecting_video_upload_method", "chatting", "selecting_portfolio", "empty"}[s]
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
	Smash BadmintonSkill = iota
	Drive
	NetKill
	BackCourtFootwork
	FrontCourtFootwork
	DoublesRotation
)

func BadmintonSkillSlice() []BadmintonSkill {
	return []BadmintonSkill{Smash, Drive, NetKill, BackCourtFootwork, FrontCourtFootwork, DoublesRotation}
}

func (s BadmintonSkill) String() string {
	return [...]string{"smash", "drive", "netkill", "back_court_footwork", "front_court_footwork", "doubles_rotation"}[s]
}

func (s BadmintonSkill) ChnString() string {
	return [...]string{"殺球", "抽球", "壓球", "後場腳步", "前場腳步", "雙打輪轉"}[s]
}

func SkillStrToEnum(str string) BadmintonSkill {
	switch str {
	case "smash":
		return Smash
	case "drive":
		return Drive
	case "netkill":
		return NetKill
	case "back_court_footwork":
		return BackCourtFootwork
	case "front_court_footwork":
		return FrontCourtFootwork
	case "doubles_rotation":
		return DoublesRotation
	default:
		return -1
	}
}
