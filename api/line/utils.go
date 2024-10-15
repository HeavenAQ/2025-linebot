package line

func (client *LineBot) GetUserName(userId string) (string, error) {
	profile, err := client.bot.GetProfile(userId).Do()
	if err != nil {
		return "", err
	}
	return profile.DisplayName, nil
}
