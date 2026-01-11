package commons

type GradingDetail struct {
    Description string  `json:"description" firestore:"description"`
    Grade       float64 `json:"grade" firestore:"grade"`
}

type GradingOutcome struct {
    GradingDetails []GradingDetail `json:"grading_details" firestore:"grading_details"`
    TotalGrade     float64         `json:"total_grade" firestore:"total_grade"`
}
