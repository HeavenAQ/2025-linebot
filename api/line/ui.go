package line

import (
	"encoding/json"
	"slices"
	"sort"
	"time"

	"github.com/HeavenAQ/nstc-linebot-2025/api/db"
	"github.com/line/line-bot-sdk-go/v7/linebot"
	"golang.org/x/exp/maps"
)

// createButtonActions generates the buttons for preview and reflection actions
func (client *Client) createButtonActions(work db.Work, skill string, showReflectionBtn bool) ([]linebot.FlexComponent, error) {
	reflectionData, err := json.Marshal(WritingNotePostback{
		State:      db.WritingNotes.String(),
		WorkDate:   work.DateTime,
		ActionStep: db.WritingReflection.String(),
		Skill:      skill,
	})
	if err != nil {
		return nil, err
	}

	videoData, err := json.Marshal(VideoPostback{
		VideoID:     work.Video,
		ThumbnailID: work.Thumbnail,
	})
	if err != nil {
		return nil, err
	}

	var btnComponents []linebot.FlexComponent
	if showReflectionBtn {
		btnComponents = []linebot.FlexComponent{
			&linebot.ButtonComponent{
				Type:   "button",
				Style:  "primary",
				Height: "sm",
				Action: linebot.NewPostbackAction(
					"更新學習反思",
					string(reflectionData),
					"",
					"",
					"openKeyboard",
					"",
				),
			},
		}
	}

	btnComponents = append(btnComponents, &linebot.ButtonComponent{
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
	})
	return btnComponents, nil
}

// createNotesSection generates the notes sections for AI Note, Preview Note, and Reflection
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
	buttons, err := client.createButtonActions(work, skill, showBtns)
	if err != nil {
		return nil
	}

	item := &linebot.BubbleContainer{
		Type: "bubble",
		Hero: &linebot.ImageComponent{
			Type:        "image",
			URL:         "https://storage.googleapis.com/moe-linebot-2026-storage/" + work.Thumbnail,
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
				createNotesSection("學習反思：", work.Reflection),
			},
		},
		Footer: &linebot.BoxComponent{
			Type:     "box",
			Layout:   "vertical",
			Spacing:  "sm",
			Contents: buttons,
		},
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

func (client *Client) getCarousels(works map[string]db.Work, skill string, showBtns bool) ([]*linebot.FlexMessage, error) {
	items := []*linebot.BubbleContainer{}
	carouselItems := []*linebot.FlexMessage{}
	sortedWorks := client.sortWorks(works)
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
