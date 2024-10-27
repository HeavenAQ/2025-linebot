package line

import (
	"encoding/json"
)

func handlePostbackData[T PostbackData]() func(string) (*T, error) {
	var data T
	return func(rawData string) (*T, error) {
		if err := json.Unmarshal([]byte(rawData), &data); err != nil {
			return nil, err
		}
		return &data, nil
	}
}

func (client *Client) HandleSelectingSkill(rawData string) (*SelectingSkillPostback, error) {
	return handlePostbackData[SelectingSkillPostback]()(rawData)
}

func (client *Client) HandleSelectingHandedness(rawData string) (*SelectingHandednessPostback, error) {
	return handlePostbackData[SelectingHandednessPostback]()(rawData)
}
