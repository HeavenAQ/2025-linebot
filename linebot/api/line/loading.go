package line

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

const loadingEndpoint = "https://api.line.me/v2/bot/chat/loading/start"

// ShowLoading animates the typing indicator in the learner's chat for up to a
// minute, so a wait with no message in it does not look like the bot ignored
// them. It costs nothing against the message quota, unlike a push.
//
// seconds is rounded to the 5-second steps LINE accepts, and capped at 60.
func (client *Client) ShowLoading(ctx context.Context, userID string, seconds int) error {
	if seconds < 5 {
		seconds = 5
	}
	if seconds > 60 {
		seconds = 60
	}
	seconds = seconds / 5 * 5
	body, err := json.Marshal(map[string]any{"chatId": userID, "loadingSeconds": seconds})
	if err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, loadingEndpoint, bytes.NewReader(body))
	if err != nil {
		return err
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Authorization", "Bearer "+client.channelToken)
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode/100 != 2 {
		return fmt.Errorf("loading indicator: %s", response.Status)
	}
	return nil
}
