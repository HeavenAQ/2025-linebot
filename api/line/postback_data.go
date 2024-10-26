package line

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
