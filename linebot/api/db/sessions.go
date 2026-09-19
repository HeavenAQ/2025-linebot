package db

import (
	"fmt"
	"time"
)

type UserSession struct {
	Skill       string    `json:"skill" firestore:"skill"`
	UpdatedDate string    `json:"updated_date" firestore:"updated_date"`
	UserState   UserState `json:"user_state" firestore:"user_state"`
	// ChatQuestion is the learner's last question in coach chat, kept only
	// long enough for a video sent just after it to be answered together.
	ChatQuestion   string    `json:"chat_question" firestore:"chat_question"`
	ChatQuestionAt time.Time `json:"chat_question_at" firestore:"chat_question_at"`
	// PendingAnswer is an analysis answer whose reply window closed before it
	// was ready. It rides along with the learner's next message rather than
	// costing a push; the video is watched in the dashboard instead.
	PendingAnswer string     `json:"pending_answer" firestore:"pending_answer"`
	ActionStep    ActionStep `json:"action_step" firestore:"action_step"`
}

func (client *FirestoreClient) GetUserSession(userID string) (*UserSession, error) {
	session, err := client.Sessions.Doc(userID).Get(*client.Ctx)
	if err != nil {
		return nil, fmt.Errorf("error getting user session: %w", err)
	}

	var userSessioon UserSession
	err = session.DataTo(&userSessioon)
	if err != nil {
		return nil, fmt.Errorf("error converting user session data: %w", err)
	}
	return &userSessioon, nil
}

func (client *FirestoreClient) UpdateUserSession(userID string, newSessionContent UserSession) error {
	_, err := client.Sessions.Doc(userID).Set(*client.Ctx, newSessionContent)
	if err != nil {
		return fmt.Errorf("error updating user session: %w", err)
	}
	return nil
}

func (client *FirestoreClient) CreateUserSession(userID string) (*UserSession, error) {
	newSession := UserSession{
		UserState:   None,
		Skill:       "",
		ActionStep:  Empty,
		UpdatedDate: "",
	}
	err := client.UpdateUserSession(userID, newSession)
	if err != nil {
		return nil, err
	}
	return &newSession, nil
}

func (client *FirestoreClient) UpdateSessionUserState(userID string, state UserState, step ActionStep) error {
	userSession, err := client.GetUserSession(userID)
	if err != nil {
		return err
	}

	userSession.UserState = state
	userSession.ActionStep = step
	return client.UpdateUserSession(userID, *userSession)
}

func (client *FirestoreClient) ResetSession(userID string) error {
	userSession := UserSession{
		Skill:       "",
		UserState:   None,
		ActionStep:  Empty,
		UpdatedDate: "",
	}
	err := client.UpdateUserSession(userID, userSession)
	if err != nil {
		return err
	}
	return nil
}

// SetChatQuestion remembers what the learner just asked the coach, so a video
// they send straight afterwards is answered together with the question.
func (client *FirestoreClient) SetChatQuestion(userID, question string) error {
	session, err := client.GetUserSession(userID)
	if err != nil {
		return err
	}
	session.ChatQuestion = question
	session.ChatQuestionAt = time.Now()
	return client.UpdateUserSession(userID, *session)
}

// ChatQuestionWindow is how long a question waits for a video. Long enough to
// find the clip and send it, short enough that last week's question is not
// answered against today's upload.
const ChatQuestionWindow = 10 * time.Minute

// PendingChatQuestion returns the learner's recent question, if it is still
// recent enough to belong with a video arriving now.
func (session *UserSession) PendingChatQuestion() string {
	if session == nil || session.ChatQuestion == "" {
		return ""
	}
	if time.Since(session.ChatQuestionAt) > ChatQuestionWindow {
		return ""
	}
	return session.ChatQuestion
}

// SetPendingAnswer stores an answer that missed its reply window, to be
// delivered with the learner's next message.
func (client *FirestoreClient) SetPendingAnswer(userID, answer string) error {
	session, err := client.GetUserSession(userID)
	if err != nil {
		return err
	}
	session.PendingAnswer = answer
	return client.UpdateUserSession(userID, *session)
}

// TakePendingAnswer returns the waiting answer and clears it.
func (client *FirestoreClient) TakePendingAnswer(userID string) (string, error) {
	session, err := client.GetUserSession(userID)
	if err != nil || session.PendingAnswer == "" {
		return "", err
	}
	answer := session.PendingAnswer
	session.PendingAnswer = ""
	return answer, client.UpdateUserSession(userID, *session)
}
