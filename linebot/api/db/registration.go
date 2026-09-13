package db

import (
	"context"
	"errors"
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"

	"cloud.google.com/go/firestore"
)

const RegistrationInstructions = "請先在 LINE 聊天室輸入「實驗編號 姓名」完成登記，才能使用功能。\n例如：01 王小明\n編號和姓名中間加一個空格即可；請填寫真實姓名。"
const RegistrationFormatError = "格式不正確，請輸入「實驗編號 姓名」。\n例如：01 王小明\n編號可用數字或英文字母加數字（例如 EG01）；姓名不可含數字、表情符號或其他說明。"

var ErrRegistrationFormat = errors.New("invalid experiment registration format")
var ErrRegistrationLocked = errors.New("experiment registration is already completed")
var experimentNumberPattern = regexp.MustCompile(`^[A-Z]{0,4}[0-9]{1,6}$`)

type ExperimentRegistration struct {
	ExperimentNumber string
	RealName         string
}

// ParseExperimentRegistration validates syntax, not a person's legal identity.
// Keep leading zeroes: an experiment number is an identifier, not an integer.
func ParseExperimentRegistration(text string) (ExperimentRegistration, error) {
	invalid := ExperimentRegistration{}
	if !utf8.ValidString(text) || len(text) > 300 || strings.ContainsAny(text, "\r\n") {
		return invalid, ErrRegistrationFormat
	}
	text = strings.Map(func(r rune) rune {
		if r >= '！' && r <= '～' {
			return r - 0xFEE0
		}
		return r
	}, text)
	fields := strings.Fields(text)
	if len(fields) < 2 {
		return invalid, ErrRegistrationFormat
	}
	number := strings.ToUpper(fields[0])
	if !experimentNumberPattern.MatchString(number) {
		return invalid, ErrRegistrationFormat
	}
	name := strings.Join(fields[1:], " ")
	count, letters := utf8.RuneCountInString(name), 0
	if count < 2 || count > 60 {
		return invalid, ErrRegistrationFormat
	}
	for _, r := range name {
		if unicode.IsLetter(r) {
			letters++
			continue
		}
		if unicode.IsMark(r) || strings.ContainsRune(" '-’·・", r) {
			continue
		}
		return invalid, ErrRegistrationFormat
	}
	runes := []rune(name)
	if letters < 2 || !unicode.IsLetter(runes[0]) ||
		!(unicode.IsLetter(runes[len(runes)-1]) || unicode.IsMark(runes[len(runes)-1])) {
		return invalid, ErrRegistrationFormat
	}
	return ExperimentRegistration{ExperimentNumber: number, RealName: name}, nil
}

func (user *UserData) HasExperimentRegistration() bool {
	if user == nil || user.RegistrationVersion != 1 || user.RegistrationCompletedAt.IsZero() {
		return false
	}
	parsed, err := ParseExperimentRegistration(user.ExperimentNumber + " " + user.RealName)
	return err == nil && parsed.ExperimentNumber == user.ExperimentNumber && parsed.RealName == user.RealName
}

// Registration is completed only through a signed LINE event. Field updates
// preserve portfolios; a transaction makes webhook retries idempotent and prevents
// conflicting concurrent messages from silently changing an existing identity.
func (client *FirestoreClient) RegisterExperiment(userID string, registration ExperimentRegistration) (*UserData, error) {
	parsed, err := ParseExperimentRegistration(registration.ExperimentNumber + " " + registration.RealName)
	if err != nil || parsed != registration {
		return nil, ErrRegistrationFormat
	}
	ref := client.Data.Doc(userID)
	err = client.Client.RunTransaction(*client.Ctx, func(ctx context.Context, tx *firestore.Transaction) error {
		snapshot, err := tx.Get(ref)
		if err != nil {
			return err
		}
		var user UserData
		if err := snapshot.DataTo(&user); err != nil {
			return err
		}
		if user.HasExperimentRegistration() {
			if user.RealName == registration.RealName && user.ExperimentNumber == registration.ExperimentNumber {
				return nil
			}
			return ErrRegistrationLocked
		}
		return tx.Update(ref, []firestore.Update{
			{Path: "real_name", Value: registration.RealName},
			{Path: "experiment_number", Value: registration.ExperimentNumber},
			{Path: "registration_version", Value: 1},
			{Path: "registration_completed_at", Value: firestore.ServerTimestamp},
		})
	})
	if err != nil {
		return nil, err
	}
	return client.GetUserData(userID)
}
