package commons

type GradingDetail struct {
	Description string  `json:"description"`
	Grade       float64 `json:"grade"`
}

type GradingOutcome struct {
	GradingDetails []GradingDetail `json:"grading_details"`
	TotalGrade     float64         `json:"total_grade"`
}
