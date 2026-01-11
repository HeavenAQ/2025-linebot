package line

import (
	"encoding/json"
	"fmt"
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

// createButtonActions generates the buttons for preview and reflection actions
func (client *Client) createButtonActions(work db.Work, skill string, handedness string) ([]linebot.FlexComponent, error) {
	previewData, err := json.Marshal(WritingNotePostback{
		State:      db.WritingNotes.String(),
		WorkDate:   work.DateTime,
		ActionStep: db.WritingPreviewNote.String(),
		Skill:      skill,
	})
	if err != nil {
		return nil, err
	}

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
        VideoID:     client.assetURL(work.SkeletonVideo),
        ThumbnailID: client.assetURL(work.Thumbnail),
    })
    if err != nil {
        return nil, err
    }

    // Optional: comparison video
    var compareButton linebot.FlexComponent
    if work.SkeletonComparisonVideo != "" {
        compareData, err := json.Marshal(VideoPostback{
            VideoID:     client.assetURL(work.SkeletonComparisonVideo),
            ThumbnailID: client.assetURL(work.Thumbnail),
        })
        if err != nil {
            return nil, err
        }
        compareButton = &linebot.ButtonComponent{
            Type:   "button",
            Style:  "link",
            Height: "sm",
            Action: linebot.NewPostbackAction(
                "查看比較影片",
                string(compareData),
                "",
                "",
                "",
                "",
            ),
        }
    }

	askedAIForHelpData, err := json.Marshal(AnalyzingWithGPTPostback{
		Handedness: handedness,
		WorkDate:   work.DateTime,
		Skill:      skill,
	})

    return []linebot.FlexComponent{
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
		&linebot.ButtonComponent{
			Type:   "button",
			Style:  "primary",
			Height: "sm",
			Action: linebot.NewPostbackAction(
				"更新課前動作檢測要點",
				string(previewData),
				"",
				"",
				"openKeyboard",
				"",
			),
		},
		&linebot.ButtonComponent{
			Type:   "button",
			Style:  "primary",
			Height: "sm",
			Action: linebot.NewPostbackAction(
				"詢問AI建議",
				string(askedAIForHelpData),
				"",
				"",
				"",
				"",
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
        // Conditionally rendered comparison video button
        compareButton,
    }, nil
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
func (client *Client) getCarouselItem(work db.Work, skill string, handedness string, showBtns bool) *linebot.BubbleContainer {
	dateTime, _ := time.Parse("2006-01-02-15-04", work.DateTime)
	formattedDate := dateTime.Format("2006-01-02")
	rating := client.getPortfolioRating(work)
	buttons, err := client.createButtonActions(work, skill, handedness)
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
				createNotesSection("課前動作檢測要點：", work.PreviewNote),
				createNotesSection("學習反思：", work.Reflection),
			},
		},
		Footer: &linebot.BoxComponent{
			Type:     "box",
			Layout:   "vertical",
			Spacing:  "sm",
			Contents: buttons[2:],
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

func (client *Client) getCarousels(works map[string]db.Work, skill string, handedness string, showBtns bool) ([]*linebot.FlexMessage, error) {
	items := []*linebot.BubbleContainer{}
	carouselItems := []*linebot.FlexMessage{}
	sortedWorks := client.sortWorks(works)
	for _, work := range sortedWorks {
		items = append(items, client.getCarouselItem(work, skill, handedness, showBtns))

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
