package line

import (
	"encoding/json"
	"fmt"
	"net/url"
	"slices"
	"sort"
	"time"

	"github.com/HeavenAQ/nstc-linebot-2025/api/db"
	"github.com/line/line-bot-sdk-go/v7/linebot"
	"golang.org/x/exp/maps"
)

// getPortfolioRating creates the star rating component
func (client *Client) getPortfolioRating(work db.Work) *linebot.BoxComponent {
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

// The 每週回顧 sub-tabs, one per weekly menu entry. These are the values the web
// app matches on, so both ends have to spell them the same way.
const (
	previewSection    = "preview"
	reflectionSection = "reflection"
)

// withReviewQuery adds parameters to the configured review URL. That URL
// already carries a query of its own (?tab=review), so they are merged into it
// rather than appended behind a second "?".
func withReviewQuery(reviewURL string, params map[string]string) string {
	parsed, err := url.Parse(reviewURL)
	if err != nil {
		// Nothing to build on, but the plain review tab still gets the learner
		// to their notes -- they just have to find their way from there.
		return reviewURL
	}
	query := parsed.Query()
	for key, value := range params {
		query.Set(key, value)
	}
	parsed.RawQuery = query.Encode()
	return parsed.String()
}

// workReviewURL points the review tab at one recorded attempt.
func workReviewURL(reviewURL string, skill string, workDate string) string {
	return withReviewQuery(reviewURL, map[string]string{"skill": skill, "date": workDate})
}

// reviewSectionURL points the review tab at one of its sub-tabs.
func reviewSectionURL(reviewURL string, section string) string {
	return withReviewQuery(reviewURL, map[string]string{"section": section})
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
			// The reflection is written in the web app, alongside the video
			// and the grades for this same attempt, so the button opens that
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

// createNotesSection generates the notes section for the reflection
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
	dateTime, _ := time.Parse("2006-01-02-15-04", work.DateTime)
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

func (client *Client) insertCarousel(carouselItems []*linebot.FlexMessage, items []*linebot.BubbleContainer) []*linebot.FlexMessage {
	return append(carouselItems,
		linebot.NewFlexMessage("portfolio",
			&linebot.CarouselContainer{
				Type:     "carousel",
				Contents: items,
			},
		),
	)
}

func (client *Client) sortWorks(works map[string]db.Work) []db.Work {
	workValues := maps.Values(works)
	sort.Slice(workValues, func(i, j int) bool {
		dateTimeI, _ := time.Parse("2006-01-02-15-04", workValues[i].DateTime)
		dateTimeJ, _ := time.Parse("2006-01-02-15-04", workValues[j].DateTime)
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

func (client *Client) getCarousels(works map[string]db.Work, skill string, showBtns bool) ([]*linebot.FlexMessage, error) {
	items := []*linebot.BubbleContainer{}
	carouselItems := []*linebot.FlexMessage{}
	sortedWorks := client.latestPortfolioWorks(works)
	for _, work := range sortedWorks {
		items = append(items, client.getCarouselItem(work, skill, showBtns))

		// since the carousel can only contain 10 items, we need to split the works into multiple carousels in order to display all of them
		if len(items) == 10 {
			carouselItems = client.insertCarousel(carouselItems, items)
			items = []*linebot.BubbleContainer{}
		}
	}

	// insert the last carousel
	if len(items) > 0 {
		carouselItems = client.insertCarousel(carouselItems, items)
	}

	// latest work will be displayed last
	slices.Reverse(carouselItems)
	return carouselItems, nil
}
