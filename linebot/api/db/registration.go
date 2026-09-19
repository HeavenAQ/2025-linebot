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

const RegistrationInstructions = "請先在學習網頁完成實驗登記，填寫實驗編號、姓名與慣用手後，才能使用功能。"
const RegistrationFormatError = "資料格式不正確。實驗編號可用數字，或英文字母加數字（例如 01、EG01）；姓名請填真實姓名，不可含數字、表情符號或其他說明。"

var ErrRegistrationFormat = errors.New("invalid experiment registration format")
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

// SaveExperimentRegistration writes a learner's own details, first time or as
// an edit from the profile page. It does not lock the first answer: the
// learner is signed in as themselves and may fix a typo in their own name or
// number. The completion time is stamped once and then left alone, so the
// record still says when the learner joined the experiment.
func (client *FirestoreClient) SaveExperimentRegistration(userID string, registration ExperimentRegistration) (*UserData, error) {
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
		updates := []firestore.Update{
			{Path: "real_name", Value: registration.RealName},
			{Path: "experiment_number", Value: registration.ExperimentNumber},
			{Path: "registration_version", Value: 1},
		}
		if !user.HasExperimentRegistration() {
			updates = append(updates, firestore.Update{
				Path: "registration_completed_at", Value: firestore.ServerTimestamp,
			})
		}
		return tx.Update(ref, updates)
	})
	if err != nil {
		return nil, err
	}
	return client.GetUserData(userID)
}
