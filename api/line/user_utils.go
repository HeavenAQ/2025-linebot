package line

import "fmt"

func (client *Client) GetUserName(userID string) (string, error) {
	profile, err := client.bot.GetProfile(userID).Do()
	if err != nil {
		return "", fmt.Errorf("failed to get user profile: %w", err)
	}

	return profile.DisplayName, nil
}
