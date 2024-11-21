package line

import (
	"encoding/json"
	"errors"
	"fmt"
	"reflect"

	"github.com/go-playground/validator"
)

// Helper function to get JSON field names from struct tags for exact matching
func getFieldNames[T any]() map[string]bool {
	var t T
	typ := reflect.TypeOf(t)
	fieldMap := make(map[string]bool, typ.NumField())
	for i := 0; i < typ.NumField(); i++ {
		jsonTag := typ.Field(i).Tag.Get("json")
		if jsonTag != "" {
			fieldMap[jsonTag] = true
		}
	}
	return fieldMap
}

// Exact match validation function
func handlePostbackData[T PostbackData](rawData string) (*T, error) {
	validate := validator.New()

	// Unmarshal JSON to map to check for extra fields
	var rawMap map[string]interface{}
	if err := json.Unmarshal([]byte(rawData), &rawMap); err != nil {
		return nil, err
	}

	// Get the exact fields expected for the struct type
	expectedFields := getFieldNames[T]()

	// Check if rawMap matches expectedFields exactly
	for field := range rawMap {
		if !expectedFields[field] {
			return nil, fmt.Errorf("unexpected field: %s", field)
		}
	}
	if len(rawMap) != len(expectedFields) {
		return nil, errors.New("missing or extra fields")
	}

	// Unmarshal into the actual struct if fields match
	var data T
	if err := json.Unmarshal([]byte(rawData), &data); err != nil {
		return nil, err
	}

	// Validate required fields using `validator`
	if err := validate.Struct(data); err != nil {
		return nil, err
	}

	return &data, nil
}

func (client *Client) HandleSelectingSkillPostbackData(rawData string) (*SelectingSkillPostback, error) {
	return handlePostbackData[SelectingSkillPostback](rawData)
}

func (client *Client) HandleSelectingHandednessPostbackData(rawData string) (*SelectingHandednessPostback, error) {
	return handlePostbackData[SelectingHandednessPostback](rawData)
}

func (client *Client) HandleWritingNotePostbackData(rawData string) (*WritingNotePostback, error) {
	return handlePostbackData[WritingNotePostback](rawData)
}

func (client *Client) HandleVideoPostbackData(rawData string) (*VideoPostback, error) {
	return handlePostbackData[VideoPostback](rawData)
}

func (client *Client) HandleStopGPTPostbackData(rawData string) (*StopGPTPostback, error) {
	return handlePostbackData[StopGPTPostback](rawData)
}
