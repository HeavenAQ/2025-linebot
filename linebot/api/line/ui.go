package line

import (
	"encoding/json"
	"fmt"
	"net/url"
	"sort"

	"github.com/HeavenAQ/nstc-linebot-2025/api/db"
	"github.com/line/line-bot-sdk-go/v7/linebot"
	"golang.org/x/exp/maps"
)

// getPortfolioRating creates the star rating component
func (client *Client) getPortfolioRating(work db.Work) *linebot.BoxComponent {
	if work.AnalysisStatus == "pending" || work.AnalysisStatus == "failed" {
		text := "影片分析中 · 點擊查看最新結果"
		if work.AnalysisStatus == "failed" {
			text = work.AnalysisError
		}
		return &linebot.BoxComponent{Type: "box", Layout: "horizontal", Contents: []linebot.FlexComponent{
			&linebot.TextComponent{Type: "text", Text: text, Wrap: true, Size: "sm"},
		}}
	}
	rating := work.GradingOutcome.TotalGrade
	contents := []linebot.FlexComponent{}
	for i := 0; i < 5; i++ {
		url := "https://scdn.line-apps.com/n/channel_devcenter/img/fx/review_gray_star_28.png"
		if rating >= 20 {
			url = "https://scdn.line-apps.com/n/channel_devcenter/img/fx/review_gold_star_28.png"
		}
		contents = append(contents, &linebot.IconComponent{
			Type: "icon",
			Size: "sm",
			URL:  url,
		})
		rating -= 20
	}
	contents = append(contents, &linebot.TextComponent{
		Type:   "text",
		Text:   fmt.Sprintf("%.2f", work.GradingOutcome.TotalGrade),
		Size:   "sm",
		Color:  "#8c8c8c",
		Margin: "md",
		Flex:   linebot.IntPtr(0),
	})
	return &linebot.BoxComponent{
		Type:     "box",
		Layout:   "baseline",
		Margin:   "md",
		Contents: contents,
	}
}

// workReviewURL points the review tab at one recorded attempt. The configured
// URL already carries a query of its own (?tab=review), so the work is merged
// into it rather than appended behind a second "?".
func workReviewURL(reviewURL string, skill string, workDate string) string {
	parsed, err := url.Parse(reviewURL)
	if err != nil {
		// Nothing to build on, but the plain review tab still gets the learner
		// to their reflections -- they just have to pick the video themselves.
		return reviewURL
	}
	query := parsed.Query()
	query.Set("skill", skill)
	query.Set("date", workDate)
	parsed.RawQuery = query.Encode()
	return parsed.String()
}

// createButtonActions generates the buttons for reflection and video actions
func (client *Client) createButtonActions(work db.Work, skill string) ([]linebot.FlexComponent, error) {
	videoData, err := json.Marshal(VideoPostback{
		WorkDate: work.DateTime,
		Skill:    skill,
	})
	if err != nil {
		return nil, err
	}

	return []linebot.FlexComponent{
		&linebot.ButtonComponent{
			Type:   "button",
			Style:  "primary",
			Height: "sm",
			// The reflection is written in the web app, alongside the video and
			// the AI feedback for this same attempt, so the button opens that
			// attempt directly instead of starting a note over chat.
			Action: linebot.NewURIAction(
				"更新學習反思",
				workReviewURL(client.reviewURL, skill, work.DateTime),
			),
		},
		&linebot.ButtonComponent{
			Type:   "button",
			Style:  "link",
			Height: "sm",
			Action: linebot.NewPostbackAction(
				"查看影片",
				string(videoData),
				"",
				"",
				"",
				"",
			),
		},
	}, nil
}

// createNotesSection generates the notes sections for AI Note and Reflection
func createNotesSection(label string, content string) *linebot.BoxComponent {
	// If content is empty, provide a default placeholder text
	if content == "" {
		content = "無內容" // You can replace this with any placeholder text
	}
	return &linebot.BoxComponent{
		Type:    "box",
		Layout:  "vertical",
		Spacing: "sm",
		Contents: []linebot.FlexComponent{
			&linebot.TextComponent{
				Type:   "text",
				Text:   label,
				Color:  "#000000",
				Size:   "md",
				Flex:   linebot.IntPtr(1),
				Weight: "bold",
			},
			&linebot.TextComponent{
				Type:  "text",
				Text:  content,
				Wrap:  true,
				Color: "#666666",
				Size:  "sm",
				Flex:  linebot.IntPtr(5),
			},
		},
	}
}

// getCarouselItem constructs the carousel item using helper functions
func (client *Client) getCarouselItem(work db.Work, skill string, showBtns bool) *linebot.BubbleContainer {
	dateTime, _ := db.ParseWorkTime(work.DateTime)
	formattedDate := dateTime.Format("2006-01-02")
	rating := client.getPortfolioRating(work)
	buttons, err := client.createButtonActions(work, skill)
	if err != nil {
		return nil
	}

	item := &linebot.BubbleContainer{
		Type: "bubble",
		Hero: &linebot.ImageComponent{
			Type:        "image",
			URL:         client.assetURL(work.Thumbnail),
			Size:        "full",
			AspectRatio: "20:13",
			AspectMode:  "cover",
		},
		Body: &linebot.BoxComponent{
			Type:   "box",
			Layout: "vertical",
			Contents: []linebot.FlexComponent{
				&linebot.TextComponent{
					Type:   "text",
					Text:   "🗓️ " + formattedDate,
					Weight: "bold",
					Size:   "xl",
				},
				rating,
				createNotesSection("需調整細節：", work.AINote),
				createNotesSection("課前檢視要點：", work.Preview),
				createNotesSection("學習反思：", work.Reflection),
			},
		},
		// Outside the update flow the card is a record to look at, so it offers
		// playback only -- buttons[0] is the reflection link.
		Footer: &linebot.BoxComponent{
			Type:     "box",
			Layout:   "vertical",
			Spacing:  "sm",
			Contents: buttons[1:],
		},
	}

	if work.Thumbnail == "" {
		item.Hero = nil
	}
	if showBtns {
		item.Footer = &linebot.BoxComponent{
			Type:     "box",
			Layout:   "vertical",
			Spacing:  "sm",
			Contents: buttons,
		}
	}
	return item
}

func (client *Client) sortWorks(works map[string]db.Work) []db.Work {
	workValues := maps.Values(works)
	sort.Slice(workValues, func(i, j int) bool {
		dateTimeI, _ := db.ParseWorkTime(workValues[i].DateTime)
		dateTimeJ, _ := db.ParseWorkTime(workValues[j].DateTime)
		return dateTimeI.After(dateTimeJ)
	})

	sortedWorks := []db.Work{}
	for _, workValue := range workValues {
		sortedWorks = append(sortedWorks, workValue)
	}
	return sortedWorks
}

// latestPortfolioWorks keeps LINE replies compact and useful. A Flex carousel
// accepts ten bubbles, and returning the complete history can also push the
// surrounding reply past LINE's five-message limit.
const latestPortfolioWorkLimit = 10

func (client *Client) latestPortfolioWorks(works map[string]db.Work) []db.Work {
	sortedWorks := client.sortWorks(works)
	if len(sortedWorks) > latestPortfolioWorkLimit {
		sortedWorks = sortedWorks[:latestPortfolioWorkLimit]
	}
	return sortedWorks
}

// portfolioCarousel shows the latest works as one Flex carousel; the work limit
// keeps it within the ten bubbles LINE allows in a carousel.
func (client *Client) portfolioCarousel(works map[string]db.Work, skill string, showBtns bool) *linebot.FlexMessage {
	items := []*linebot.BubbleContainer{}
	for _, work := range client.latestPortfolioWorks(works) {
		items = append(items, client.getCarouselItem(work, skill, showBtns))
	}
	return linebot.NewFlexMessage("portfolio", &linebot.CarouselContainer{Type: "carousel", Contents: items})
}

// PortfolioWorksForDisplay shares the card limit/order with media signing, so
// old records not shown in the carousel cannot block a current portfolio.
func (client *Client) PortfolioWorksForDisplay(works map[string]db.Work) []db.Work {
	return client.latestPortfolioWorks(works)
}
