package line

import (
	"encoding/json"

	"github.com/HeavenAQ/nstc-linebot-2025/api/db"
)

func (client *Client) getActionUrls(hand db.Handedness, skill db.BadmintonSkill) []string {
	actionUrls := map[db.Handedness]map[db.BadmintonSkill][]string{
		db.Right: {
			db.Serve: []string{
				"https://youtu.be/uE-EHVX1LrA",
			},
			db.Smash: []string{
				"https://youtu.be/K7EEhEF2vMo",
			},
			db.Clear: []string{
				"https://youtu.be/K7EEhEF2vMo",
			},
		},
		db.Left: {
			db.Serve: []string{
				"https://youtu.be/7i0KvbJ4rEE",
				"https://youtu.be/LiQWE6i3bbI",
			},
			db.Smash: []string{
				"https://youtu.be/yyjC-xXOsdg",
				"https://youtu.be/AzF44kouBBQ",
			},
			db.Clear: []string{
				"https://youtu.be/yyjC-xXOsdg",
				"https://youtu.be/AzF44kouBBQ",
			},
		},
	}
	return actionUrls[hand][skill]
}

func (client *Client) HandleSelectingSkill(rawData string) (*SelectingSkillPostback, error) {
	var data SelectingSkillPostback
	if err := json.Unmarshal([]byte(rawData), &data); err != nil {
		return nil, err
	}
	return &data, nil
}
