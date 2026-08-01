package commons

type GradingDetail struct {
	CriterionID string  `json:"criterion_id" firestore:"criterion_id"`
	Description string  `json:"description" firestore:"description"`
	Grade       float64 `json:"grade" firestore:"grade"`
	Maximum     float64 `json:"maximum" firestore:"maximum"`
}

type GradingOutcome struct {
	GradingDetails []GradingDetail `json:"grading_details" firestore:"grading_details"`
	TotalGrade     float64         `json:"total_grade" firestore:"total_grade"`
	ScoreStatus    string          `json:"score_status" firestore:"score_status"`
}

type MediaRef struct {
	ObjectPath       string  `json:"object_path" firestore:"object_path"`
	GCSURI           string  `json:"gcs_uri" firestore:"gcs_uri"`
	SignedURL        string  `json:"signed_url" firestore:"signed_url"`
	SignedURLExpires int64   `json:"signed_url_expires_at_unix" firestore:"signed_url_expires_at_unix"`
	DurationSeconds  float64 `json:"duration_seconds" firestore:"duration_seconds"`
	FPS              float64 `json:"fps" firestore:"fps"`
	Width            int32   `json:"width" firestore:"width"`
	Height           int32   `json:"height" firestore:"height"`
}

type ExpertMatch struct {
	ExpertID          string   `json:"expert_id" firestore:"expert_id"`
	DisplayName       string   `json:"display_name" firestore:"display_name"`
	EuclideanDistance float64  `json:"euclidean_distance" firestore:"euclidean_distance"`
	Video             MediaRef `json:"video" firestore:"video"`
}

type PhaseMarker struct {
	ID                 string  `json:"id" firestore:"id"`
	Label              string  `json:"label" firestore:"label"`
	NormalizedFrame    int32   `json:"normalized_frame" firestore:"normalized_frame"`
	NormalizedPosition float64 `json:"normalized_position" firestore:"normalized_position"`
	TimestampSeconds   float64 `json:"timestamp_seconds" firestore:"timestamp_seconds"`
}

type CoachingCue struct {
	Title                   string  `json:"title" firestore:"title"`
	Feedback                string  `json:"feedback" firestore:"feedback"`
	NormalizedFrame         int32   `json:"normalized_frame" firestore:"normalized_frame"`
	NormalizedPosition      float64 `json:"normalized_position" firestore:"normalized_position"`
	StudentTimestampSeconds float64 `json:"student_timestamp_seconds" firestore:"student_timestamp_seconds"`
	PauseDurationSeconds    float64 `json:"pause_duration_seconds" firestore:"pause_duration_seconds"`
	JointIDs                []int32 `json:"joint_ids" firestore:"joint_ids"`
}

type AnalysisOutcome struct {
	AnalysisID      string             `json:"analysis_id" firestore:"analysis_id"`
	Skill           string             `json:"skill" firestore:"skill"`
	Handedness      string             `json:"handedness" firestore:"handedness"`
	Grade           GradingOutcome     `json:"grade" firestore:"grade"`
	StudentVideo    MediaRef           `json:"student_video" firestore:"student_video"`
	Expert          ExpertMatch        `json:"expert" firestore:"expert"`
	Timeline        []PhaseMarker      `json:"timeline" firestore:"timeline"`
	CoachingCues    []CoachingCue      `json:"coaching_cues" firestore:"coaching_cues"`
	OverallFeedback string             `json:"overall_feedback" firestore:"overall_feedback"`
	Diagnostics     map[string]float64 `json:"diagnostics" firestore:"diagnostics"`
}
