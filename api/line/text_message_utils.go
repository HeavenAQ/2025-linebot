package line

import (
	"encoding/json"
	"errors"
	"fmt"

	"github.com/HeavenAQ/nstc-linebot-2025/api/db"
	"github.com/line/line-bot-sdk-go/v7/linebot"
)

func (client *Client) SendReply(replyToken string, msg string) (*linebot.BasicResponse, error) {
	res, err := client.bot.ReplyMessage(replyToken, linebot.NewTextMessage(msg)).Do()
	return res, fmt.Errorf("failed to reply message: %w", err)
}

func (client *Client) SendDefaultReply(replyToken string) (*linebot.BasicResponse, error) {
	return client.SendReply(replyToken, "請點選選單的項目")
}

func (client *Client) SendDefaultErrorReply(replyToken string) (*linebot.BasicResponse, error) {
	return client.SendReply(replyToken, "發生錯誤，請重新操作")
}

func (client *Client) SendWelcomeReply(event *linebot.Event) (*linebot.BasicResponse, error) {
	username, err := client.GetUserName(event.Source.UserID)
	if err != nil {
		return nil, err
	}
	welcomMsg := "Hi " + username + "! 歡迎加入羽球教室🏸\n" + "已建立您的使用者資料🎉🎊 請點選選單的項目開始使用"
	return client.SendReply(event.ReplyToken, welcomMsg)
}

func (client *Client) SendGPTChattingModeReply(replyToken string, msg string) (*linebot.BasicResponse, error) {
	data, err := json.Marshal(StopGPTPostback{Stop: true})
	if err != nil {
		return nil, err
	}

	return client.bot.ReplyMessage(replyToken, linebot.NewTextMessage(
		msg,
	).WithQuickReplies(&linebot.QuickReplyItems{
		Items: []*linebot.QuickReplyButton{
			linebot.NewQuickReplyButton(
				"",
				linebot.NewPostbackAction(
					"結束對話",
					string(data),
					"",
					"結束對話",
					"OpenRichMenu",
					"",
				),
			),
		},
	})).Do()
}

func (client *Client) SendVideoUploadedReply(
	replyToken string,
	skill string,
	videoFolder string,
) (*linebot.BasicResponse, error) {
	s := db.SkillStrToEnum(skill)
	skillFolder := "https://drive.google.com/drive/u/0/folders/" + videoFolder
	return client.bot.ReplyMessage(
		replyToken,
		linebot.NewTextMessage("已成功上傳影片!"),
		linebot.NewTextMessage("以下為【"+s.ChnString()+"】的影片資料夾：\n"+skillFolder),
	).Do()
}

func (client *Client) SendNoPortfolioReply(replyToken string, skill db.BadmintonSkill) error {
	_, err := client.bot.ReplyMessage(
		replyToken,
		linebot.NewTextMessage(
			fmt.Sprintf("尚未上傳【%v】的學習反思及影片", skill.ChnString()),
		),
	).Do()
	if err != nil {
		return fmt.Errorf("failed to reply message: %w", err)
	}
	return nil
}

// ReplyMessage wraps the linebot.Client's ReplyMessage method
func (client *Client) ReplyMessage(
	replyToken string,
	messages ...linebot.SendingMessage,
) (*linebot.BasicResponse, error) {
	res, err := client.bot.ReplyMessage(replyToken, messages...).Do()
	if err != nil {
		return nil, fmt.Errorf("failed to reply message: %w", err)
	}
	return res, nil
}

func (client *Client) SendTypeErrorReply(replyToken string) (*linebot.BasicResponse, error) {
	res, err := client.bot.ReplyMessage(replyToken, linebot.NewTextMessage("抱歉，您所輸入的訊息格式目前並未支援，請重試一次！")).Do()
	if err != nil {
		return nil, fmt.Errorf("failed to reply message: %w", err)
	}
	return res, nil
}

func (client *Client) SendInstruction(replyToken string) (*linebot.BasicResponse, error) {
	const welcome = "歡迎加入羽球教室🏸，以下為選單的使用說明:\n\n"
	const instruction = "➡️ 使用說明：呼叫選單各個項目的解說\n\n"
	const addReflection = "➡️ 學習反思：新增每周各動作的學習反思\n\n"
	const portfolio = "➡️ 學習歷程：查看個人每周的學習歷程記錄\n\n"
	const analyzeRecording = "➡️ 影片上傳：上傳個人動作錄影，LINE Bot將自動產生學習歷程\n\n"
	const chatWithGPT = "➡️ 與GPT對談：與GPT對話，獲取羽球相關資訊\n\n"
	const expertVideo = "➡️ 專家影片：觀看專家示範影片\n\n"
	const note1 = "✅ 如需查看課程大綱，請輸入「課程大綱」\n\n"
	const note2 = "⚠️ 每周的學習歷程都需有【影片】才能建檔 \n\n"
	const note3 = "⚠️ 如需和老師對話，請在發送訊息前確認自己已退出和GPT對談"
	const msg = welcome + instruction + addReflection + portfolio + analyzeRecording + chatWithGPT + expertVideo + note1 + note2 + note3
	return client.bot.ReplyMessage(replyToken, linebot.NewTextMessage(msg)).Do()
}

func (client *Client) SendSyllabus(replyToken string) (*linebot.BasicResponse, error) {
	const syllabus = "課程大綱：\n"

	const msg = syllabus + "https://drive.google.com/open?id=1PeWkePHtq30ArcGqZwzWP64olL9F7Tqw&usp=drive_fs"

	res, err := client.bot.ReplyMessage(replyToken, linebot.NewTextMessage(msg)).Do()
	return res, fmt.Errorf("failed to reply message: %w", err)
}

func (client *Client) getSkillQuickReplyItems(userState db.UserState) *linebot.QuickReplyItems {
	items := []*linebot.QuickReplyButton{}
	quickReplyAction := client.getQuickReplyAction()

	skills := db.BadmintonSkillSlice()
	for _, skill := range skills {
		items = append(items, linebot.NewQuickReplyButton(
			"",
			quickReplyAction(userState, skill),
		))
	}
	return linebot.NewQuickReplyItems(items...)
}

func (client *Client) PromptSkillSelection(
	replyToken string,
	userState db.UserState,
	prompt string,
) (*linebot.BasicResponse, error) {
	msg := linebot.NewTextMessage(prompt).WithQuickReplies(
		client.getSkillQuickReplyItems(userState),
	)
	return client.bot.ReplyMessage(replyToken, msg).Do()
}

func (client *Client) PromptHandednessSelection(event *linebot.Event) error {
	msg := linebot.NewTextMessage("請選擇左手或右手").WithQuickReplies(
		client.getHandednessQuickReplyItems(),
	)
	_, err := client.bot.ReplyMessage(event.ReplyToken, msg).Do()
	return err
}

func (client *Client) SendVideoMessage(replyToken string, video *VideoPostback) (*linebot.BasicResponse, error) {
	videoLink := "https://drive.google.com/uc?export=download&id=" + video.VideoID
	thumbnailLink := "https://drive.usercontent.google.com/download?id=" + video.ThumbnailID
	return client.bot.ReplyMessage(
		replyToken,
		linebot.NewVideoMessage(videoLink, thumbnailLink),
	).Do()
}

type NoPortfolioError struct {
	Err   error
	Skill db.BadmintonSkill
}

func (e *NoPortfolioError) Error() string {
	return fmt.Sprintf("No portfolio found for skill %v: %v", e.Skill, e.Err)
}

func (client *Client) SendPortfolio(
	event *linebot.Event,
	user *db.UserData,
	skill db.BadmintonSkill,
	userState db.UserState,
	textMsg string,
	showBtns bool,
) error {
	// get works from user portfolio
	works := user.Portfolio.GetSkillPortfolio(skill.String())
	if len(works) == 0 {
		return &NoPortfolioError{Skill: skill, Err: errors.New("No portfolio found")}
	}

	// generate carousels from works
	carousels, err := client.getCarousels(works, skill.String(), showBtns)
	if err != nil {
		client.SendDefaultErrorReply(event.ReplyToken)
		return errors.New("Error getting carousels: " + err.Error())
	}

	// turn carousels into sending messages
	var sendMsgs []linebot.SendingMessage
	sendMsgs = append(sendMsgs, linebot.NewTextMessage(textMsg))
	for _, msg := range carousels {
		sendMsgs = append(sendMsgs, msg)
	}

	_, err = client.bot.ReplyMessage(
		event.ReplyToken,
		sendMsgs...,
	).Do()
	if err != nil {
		client.SendDefaultErrorReply(event.ReplyToken)
		return err
	}
	return nil
}

func (client *Client) getSkillUrls(hand db.Handedness, skill db.BadmintonSkill) []string {
	actionUrls := map[db.Handedness]map[db.BadmintonSkill][]string{
		db.Right: {
			db.Lift: []string{
				"https://youtu.be/6T6zMCKc6Mw",
				"https://youtu.be/k9RejtgoatA",
				"https://youtu.be/4XVJKG6KwlI",
				"https://youtu.be/g58fyhMkRD4",
			},
			db.Drop: []string{
				"https://youtu.be/ST5citEQZps",
			},
			db.Netplay: []string{
				"https://youtu.be/mklLfEWPG_U",
			},
			db.Clear: []string{
				"https://youtu.be/K7EEhEF2vMo",
			},
			db.Footwork: []string{
				"https://youtu.be/IPl7-mCESfs",
			},
			db.Strategy: []string{
				"https://youtu.be/7i0KvbJ4rEE",
			},
		},
		db.Left: {
			db.Lift: []string{
				"https://youtu.be/ah9ZE9KNFpI",
				"https://youtu.be/JKbQSG27vkk",
				"https://youtu.be/ah9ZE9KNFpI",
				"https://youtu.be/JKbQSG27vkk",
			},
			db.Drop: []string{
				"https://youtu.be/zatTzMKNUgY",
				"https://youtu.be/BKpO9u9Ci14",
			},
			db.Netplay: []string{
				"https://youtu.be/lWnLgTaiSAY",
				"https://youtu.be/KkAfJBuYx00",
			},
			db.Clear: []string{
				"https://youtu.be/yyjC-xXOsdg",
				"https://youtu.be/AzF44kouBBQ",
			},
			db.Footwork: []string{
				"https://youtu.be/9i_5PgCYgts",
				"https://youtu.be/AZtvW9faDA8",
			},
			db.Strategy: []string{
				"https://youtu.be/7i0KvbJ4rEE",
			},
		},
	}
	return actionUrls[hand][skill]
}

func (client *Client) SendExpertVideos(handedness db.Handedness, skill db.BadmintonSkill, replyToken string) error {
	urls := client.getSkillUrls(handedness, skill)

	// create messages
	msgs := []linebot.SendingMessage{
		linebot.NewTextMessage(
			fmt.Sprintf("以下是【%v】-【%v】的專家示範影片：",
				handedness.ChnString(),
				skill.ChnString()),
		),
	}

	// append video urls to messages
	for i, url := range urls {
		msg := fmt.Sprintf("專家影片%v：\n%v", i+1, url)
		msgs = append(msgs, linebot.NewTextMessage(msg))
	}

	// Send messages
	_, err := client.bot.ReplyMessage(replyToken, msgs...).Do()
	if err != nil {
		return err
	}
	return nil
}
