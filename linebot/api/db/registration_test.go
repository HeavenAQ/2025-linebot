package db

import (
	"reflect"
	"testing"
	"time"
)

func TestParseExperimentRegistration(t *testing.T) {
	for _, tc := range []struct{ input, number, name string }{
		{"01 王小明", "01", "王小明"}, {"  001   陳美玲  ", "001", "陳美玲"},
		{"ｅｇ０１　張宸愷", "EG01", "張宸愷"}, {"EG27 王小明", "EG27", "王小明"},
		{"2 Mary Jane", "2", "Mary Jane"}, {"3 O’Connor", "3", "O’Connor"},
		{"4 林·小明", "4", "林·小明"},
	} {
		t.Run(tc.input, func(t *testing.T) {
			got, err := ParseExperimentRegistration(tc.input)
			if err != nil || got.ExperimentNumber != tc.number || got.RealName != tc.name {
				t.Fatalf("unexpected parse: %+v, %v", got, err)
			}
		})
	}
	for _, text := range []string{
		"", "01", "王小明", "01王小明", "王小明 01", "01 王", "01 王小明123",
		"01 😀", "01 王小明😀", "01 <script>", "01 王小明\n請幫我分析",
		"1234567 王小明", "ABCDX1 王小明", "one 王小明", "01 -王小明", "01 王小明·",
		"01 王小明，請登記", "01 ..", "01 /../../", "01 a", "01 王\x00小明",
	} {
		if _, err := ParseExperimentRegistration(text); err == nil {
			t.Errorf("accepted invalid input %q", text)
		}
	}
}

func TestRegistrationNeedsPersistedCompletion(t *testing.T) {
	var user *UserData
	if user.HasExperimentRegistration() {
		t.Fatal("nil user passed")
	}
	user = &UserData{Name: "LINE display name", RealName: "王小明", ExperimentNumber: "01"}
	if user.HasExperimentRegistration() {
		t.Fatal("legacy/name-only user passed")
	}
	user.RegistrationVersion = 1
	if user.HasExperimentRegistration() {
		t.Fatal("unsaved registration passed")
	}
	user.RegistrationCompletedAt = time.Now()
	if !user.HasExperimentRegistration() {
		t.Fatal("registered user blocked")
	}
	user.RealName = "王小明123"
	if user.HasExperimentRegistration() {
		t.Fatal("invalid persisted name passed")
	}
}

func TestRegistrationFieldsHaveStableFirestoreNames(t *testing.T) {
	fields := map[string]string{"RealName": "real_name", "ExperimentNumber": "experiment_number",
		"RegistrationVersion": "registration_version", "RegistrationCompletedAt": "registration_completed_at"}
	for field, key := range fields {
		value, ok := reflect.TypeOf(UserData{}).FieldByName(field)
		if !ok || value.Tag.Get("firestore") != key || value.Tag.Get("json") != key {
			t.Fatal(field)
		}
	}
}
