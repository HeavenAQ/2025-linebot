package line

type PostbackData interface {
	// PostbackData is an interface that defines the methods that a postback data struct should implement
	isPostbackData()
}

type VideoPostback struct {
	WorkDate string `json:"work_date" validate:"required"`
	Skill    string `json:"skill" validate:"required"`
}

type WritingNotePostback struct {
	State      string `json:"state" validate:"required"`
	WorkDate   string `json:"work_date" validate:"required"`
	ActionStep string `json:"action_step" validate:"required"`
	Skill      string `json:"skill" validate:"required"`
}

type SelectingSkillPostback struct {
	State string `json:"state" validate:"required"`
	Skill string `json:"skill" validate:"required"`
}

type SelectingHandednessPostback struct {
	Handedness string `json:"handedness" validate:"required"`
}

type StopGPTPostback struct {
	Stop bool `json:"stop" validate:"required"`
}

// WeeklyPreviewPostback is the 產生課前預習 button on the weekly review card. It
// carries no arguments -- the focus skill is chosen from the learner's own
// history -- but postback payloads are told apart by their exact field set, so
// it needs a field of its own.
type WeeklyPreviewPostback struct {
	Preview bool `json:"preview" validate:"required"`
}

// Implement the marker interface for each struct
func (VideoPostback) isPostbackData()               {}
func (WritingNotePostback) isPostbackData()         {}
func (SelectingSkillPostback) isPostbackData()      {}
func (SelectingHandednessPostback) isPostbackData() {}
func (StopGPTPostback) isPostbackData()             {}
func (WeeklyPreviewPostback) isPostbackData()       {}
