package main

import (
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/HeavenAQ/nstc-linebot-2025/app"
	"github.com/gin-gonic/gin"
)

// (runtime snake_case conversion removed; DB is migrated instead)

func main() {
	application := app.NewApp(".env")

	r := gin.New()
	r.Use(gin.Recovery())

	// Routes (parity with previous net/http handlers)
	r.POST("/callback", func(c *gin.Context) {
		handler := application.LineWebhookHandler()
		handler(c.Writer, c.Request)
	})
	r.GET("/test", func(c *gin.Context) { c.String(http.StatusOK, "Hello, World!") })

	// Backend APIs for chat history and summarization
	r.GET("/api/chat/history", func(c *gin.Context) {
		start := time.Now()
		userID := c.Query("userId")
		if userID == "" {
			application.Logger.Warn.Println("[chat.history] missing userId")
			c.JSON(http.StatusBadRequest, gin.H{"error": "missing userId"})
			return
		}
		skill := strings.ToLower(strings.TrimSpace(c.Query("skill")))
		application.Logger.Info.Printf("[chat.history] userId=%s skill=%s", userID, skill)
		history, err := application.FirestoreClient.GetChatHistory(userID)
		if err != nil {
			application.Logger.Error.Printf("[chat.history] userId=%s error=%v", userID, err)
			c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to fetch chat history"})
			return
		}
		messages := history.Messages
		if skill != "" {
			filtered := make([]interface{}, 0, len(messages))
			for _, m := range messages {
				if strings.ToLower(m.Skill) == skill {
					filtered = append(filtered, m)
				}
			}
			application.Logger.Info.Printf("[chat.history] userId=%s skill=%s count=%d took=%s", userID, skill, len(filtered), time.Since(start))
			c.JSON(http.StatusOK, gin.H{"data": filtered})
			return
		}
		application.Logger.Info.Printf("[chat.history] userId=%s count=%d took=%s", userID, len(messages), time.Since(start))
		c.JSON(http.StatusOK, gin.H{"data": messages})
	})

	type summarizeReq struct {
		Content string `json:"content"`
	}
	r.POST("/api/chat/summarize", func(c *gin.Context) {
		start := time.Now()
		var req summarizeReq
		if err := c.BindJSON(&req); err != nil || strings.TrimSpace(req.Content) == "" {
			application.Logger.Warn.Printf("[chat.summarize] invalid body len=%d", len(req.Content))
			c.JSON(http.StatusBadRequest, gin.H{"error": "invalid body"})
			return
		}
		application.Logger.Info.Printf("[chat.summarize] content_len=%d", len(req.Content))
		sum, err := application.GPTClient.Summarize(req.Content)
		if err != nil {
			application.Logger.Error.Printf("[chat.summarize] error=%v took=%s", err, time.Since(start))
			c.JSON(http.StatusBadGateway, gin.H{"error": "failed to summarize"})
			return
		}
		application.Logger.Info.Printf("[chat.summarize] ok summary_len=%d took=%s", len(sum), time.Since(start))
		c.JSON(http.StatusOK, gin.H{"summary": sum})
	})

	// DB convenience endpoints
	r.GET("/api/db/user", func(c *gin.Context) {
		start := time.Now()
		userID := c.Query("user_id")
		if userID == "" {
			application.Logger.Warn.Println("[db.user] missing userId")
			c.JSON(http.StatusBadRequest, gin.H{"error": "missing userId"})
			return
		}
		application.Logger.Info.Printf("[db.user] user_id=%s", userID)
		user, err := application.FirestoreClient.GetUserData(userID)
		if err != nil {
			c.JSON(http.StatusNotFound, gin.H{"error": "user not found"})
			application.Logger.Warn.Printf("[db.user] user_id=%s not found took=%s", userID, time.Since(start))
			return
		}
		application.Logger.Info.Printf("[db.user] user_id=%s ok took=%s", userID, time.Since(start))
		c.JSON(http.StatusOK, user)
	})

	r.GET("/api/db/users", func(c *gin.Context) {
		start := time.Now()
		application.Logger.Info.Println("[db.users] list")
		all, err := application.FirestoreClient.ListUsers()
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		application.Logger.Info.Printf("[db.users] count=%d took=%s", len(*all), time.Since(start))
		c.JSON(http.StatusOK, *all)
	})

	// Stats endpoints
	r.GET("/api/db/stats/users/:id", func(c *gin.Context) {
		start := time.Now()
		id := c.Param("id")
		skill := strings.ToLower(strings.TrimSpace(c.Query("skill")))
		if id == "" || skill == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "missing id or skill"})
			return
		}
		stats, err := application.FirestoreClient.GetUserSkillStats(id, skill)
		if err != nil {
			application.Logger.Error.Printf("[db.stats.user] id=%s skill=%s err=%v", id, skill, err)
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		application.Logger.Info.Printf("[db.stats.user] id=%s skill=%s took=%s", id, skill, time.Since(start))
		c.JSON(http.StatusOK, stats)
	})

	r.GET("/api/db/stats/class", func(c *gin.Context) {
		start := time.Now()
		skill := strings.ToLower(strings.TrimSpace(c.Query("skill")))
		if skill == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "missing skill"})
			return
		}
		stats, err := application.FirestoreClient.GetClassSkillStats(skill)
		if err != nil {
			application.Logger.Error.Printf("[db.stats.class] skill=%s err=%v", skill, err)
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		application.Logger.Info.Printf("[db.stats.class] skill=%s took=%s", skill, time.Since(start))
		c.JSON(http.StatusOK, stats)
	})

	// HTTP server with timeouts
	const (
		DefaultReadTimeout  = 100 * time.Second
		DefaultWriteTimeout = 100 * time.Second
		DefaultIdleTimeout  = 120 * time.Second
	)
	srv := &http.Server{
		Addr:         "0.0.0.0:" + application.Config.Port,
		Handler:      r,
		ReadTimeout:  DefaultReadTimeout,
		WriteTimeout: DefaultWriteTimeout,
		IdleTimeout:  DefaultIdleTimeout,
	}

	application.Logger.Info.Println("\n\tServer started on port " + application.Config.Port)
	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatal(err)
	}
}
