package line

type PostbackData interface {
	// PostbackData is an interface that defines the methods that a postback data struct should implement
	isPostbackData()
}

type VideoPostback struct {
	VideoID     string `json:"video_id"`
	ThumbnailID string `json:"thumbnail_id"`
}

type WritingNotePostback struct {
	State      string `json:"state"`
	WorkDate   string `json:"work_date"`
	ActionStep string `json:"action_step"`
}

type SelectingSkillPostback struct {
	State string `json:"state"`
	Skill string `json:"skill"`
}

type SelectingHandednessPostback struct {
	Handedness string `json:"handedness"`
}

// Implement the marker interface for each struct
func (VideoPostback) isPostbackData()               {}
func (WritingNotePostback) isPostbackData()         {}
func (SelectingSkillPostback) isPostbackData()      {}
func (SelectingHandednessPostback) isPostbackData() {}
