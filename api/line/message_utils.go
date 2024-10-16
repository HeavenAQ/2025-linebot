package line

import (
	"github.com/HeavenAQ/nstc-linebot-2025/api/db"
	"github.com/line/line-bot-sdk-go/v7/linebot"
)

func (client *LineBot) SendReply(replyToken string, msg string) (*linebot.BasicResponse, error) {
	return client.bot.ReplyMessage(replyToken, linebot.NewTextMessage(msg)).Do()
}

func (client *LineBot) SendDefaultReply(replyToken string) (*linebot.BasicResponse, error) {
	return client.SendReply(replyToken, "請點選選單的項目")
}

func (client *LineBot) SendDefaultErrorReply(replyToken string) (*linebot.BasicResponse, error) {
	return client.SendReply(replyToken, "發生錯誤，請重新操作")
}

func (client *LineBot) SendWelcomeReply(event *linebot.Event) (*linebot.BasicResponse, error) {
	username, err := client.GetUserName(event.Source.UserID)
	if err != nil {
		return nil, err
	}
	welcomMsg := "Hi " + username + "! 歡迎加入羽球教室🏸\n" + "已建立您的使用者資料🎉🎊 請點選選單的項目開始使用"
	return client.SendReply(event.ReplyToken, welcomMsg)
}

func (client *LineBot) SendVideoUploadedReply(replyToken string, skill string, videoFolder string) (*linebot.BasicResponse, error) {
	s := db.SkillStrToEnum(skill)
	skillFolder := "https://drive.google.com/drive/u/0/folders/" + videoFolder
	return client.bot.ReplyMessage(
		replyToken,
		linebot.NewTextMessage("已成功上傳影片!"),
		linebot.NewTextMessage("以下為【"+s.ChnString()+"】的影片資料夾：\n"+skillFolder),
	).Do()
}

func (client *LineBot) SendInstruction(replyToken string) (*linebot.BasicResponse, error) {
	const welcome = "歡迎加入羽球教室🏸，以下為選單的使用說明:\n\n"
	const instruction = "➡️ 使用說明：呼叫選單各個項目的解說\n\n"
	const portfolio = "➡️ 學習歷程：查看個人每周的學習歷程記錄\n\n"
	const expertVideo = "➡️ 專家影片：觀看專家示範影片\n\n"
	const addPreviewNote = "➡️ 課前動作檢測：課前預習上週動作，並記錄需進步的要點\n\n"
	const analyzeRecording = "➡️ 分析影片：上傳個人動作錄影，系統將自動產生分析結果\n\n"
	const addReflection = "➡️ 本週學習反思：新增每周各動作的學習反思\n\n"
	const note1 = "✅ 如需查看課程大綱，請輸入「課程大綱」\n\n"
	const note2 = "⚠️ 每周的學習歷程都需有【影片】才能建檔"
	const msg = welcome + instruction + portfolio + expertVideo + addPreviewNote + analyzeRecording + addReflection + note1 + note2
	return client.bot.ReplyMessage(replyToken, linebot.NewTextMessage(msg)).Do()
}

func (client *LineBot) SendSyllabus(replyToken string) (*linebot.BasicResponse, error) {
	const syllabus = "課程大綱：\n"
	const msg = syllabus + "https://drive.google.com/open?id=1PeWkePHtq30ArcGqZwzWP64olL9F7Tqw&usp=drive_fs"
	return client.bot.ReplyMessage(replyToken, linebot.NewTextMessage(msg)).Do()
}

func (client *LineBot) getSkillQuickReplyItems(userState db.UserState) *linebot.QuickReplyItems {
	items := []*linebot.QuickReplyButton{}
	quickReplyAction := client.getQuickReplyAction()

	for _, skill := range []db.BadmintonSkill{db.Serve, db.Smash, db.Clear} {
		items = append(items, linebot.NewQuickReplyButton(
			"",
			quickReplyAction(userState, skill),
		))
	}
	return linebot.NewQuickReplyItems(items...)
}

func (client *LineBot) PromptSkillSelection(replyToken string, userState db.UserState, prompt string) (*linebot.BasicResponse, error) {
	msg := linebot.NewTextMessage(prompt).WithQuickReplies(
		client.getSkillQuickReplyItems(userState),
	)
	return client.bot.ReplyMessage(replyToken, msg).Do()
}

func (client *LineBot) PromptHandednessSelection(replyToken string) (*linebot.BasicResponse, error) {
	msg := linebot.NewTextMessage("請選擇左手或右手").WithQuickReplies(
		client.getHandednessQuickReplyItems(),
	)
	return client.bot.ReplyMessage(replyToken, msg).Do()
}

func (handler *LineBot) SendVideoMessage(replyToken string, video VideoInfo) (*linebot.BasicResponse, error) {
	videoLink := "https://drive.google.com/uc?export=download&id=" + video.VideoId
	thumbnailLink := "https://drive.usercontent.google.com/download?id=" + video.ThumbnailId
	return handler.bot.ReplyMessage(
		replyToken,
		linebot.NewVideoMessage(videoLink, thumbnailLink),
	).Do()
}
