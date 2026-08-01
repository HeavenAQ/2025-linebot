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

type AnalyzingWithGPTPostback struct {
	Handedness string `json:"handedness" validate:"required"`
	WorkDate   string `json:"work_date" validate:"required"`
	Skill      string `json:"skill" validate:"required"`
}

type StopGPTPostback struct {
	Stop bool `json:"stop" validate:"required"`
}

// Implement the marker interface for each struct
func (VideoPostback) isPostbackData()               {}
func (WritingNotePostback) isPostbackData()         {}
func (SelectingSkillPostback) isPostbackData()      {}
func (SelectingHandednessPostback) isPostbackData() {}
func (AnalyzingWithGPTPostback) isPostbackData()    {}
func (StopGPTPostback) isPostbackData()             {}
