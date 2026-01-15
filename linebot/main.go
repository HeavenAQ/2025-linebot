package main

import (
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/HeavenAQ/nstc-linebot-2025/app"
	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
)

// (runtime snake_case conversion removed; DB is migrated instead)

func main() {
	gin.SetMode(gin.ReleaseMode)
	application := app.NewApp(".env")

	r := gin.New()
	r.Use(gin.Recovery())

	// Middleware for routing
	r.Use(cors.New(cors.Config{
		AllowOrigins: []string{
			"https://linebot-liff-nstc-2025.heavian.work",
			"http://localhost:3000",
		},
		AllowMethods: []string{"GET", "POST", "PUT", "DELETE"},
		AllowHeaders: []string{"Origin", "Content-Type", "Authorization"},
	}))

	// Routes (parity with previous net/http handlers)
	r.POST("/callback", func(c *gin.Context) {
		handler := application.LineWebhookHandler()
		handler(c.Writer, c.Request)
	})
	r.GET("/test", func(c *gin.Context) { c.String(http.StatusOK, "Hello, World!") })

	// Backend APIs for chat history and summarization
	r.GET("/api/chat/history", func(c *gin.Context) {
		start := time.Now()
		userID := c.Query("user_id")
		if userID == "" {
			application.Logger.Warn.Println("[chat.history] missing user_id")
			c.JSON(http.StatusBadRequest, gin.H{"error": "missing user_id"})
			return
		}
		skill := strings.ToLower(strings.TrimSpace(c.Query("skill")))
		application.Logger.Info.Printf("[chat.history] user_id=%s skill=%s", userID, skill)
		history, err := application.FirestoreClient.GetChatHistory(userID)
		if err != nil {
			application.Logger.Error.Printf("[chat.history] user_id=%s error=%v", userID, err)
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
			application.Logger.Info.Printf("[chat.history] user_id=%s skill=%s count=%d took=%s", userID, skill, len(filtered), time.Since(start))
			c.JSON(http.StatusOK, gin.H{"data": filtered})
			return
		}
		application.Logger.Info.Printf("[chat.history] user_id=%s count=%d took=%s", userID, len(messages), time.Since(start))
		c.JSON(http.StatusOK, gin.H{"data": messages})
	})

	type summarizeReq struct {
		Content string `json:"content"`
		UserID  string `json:"user_id"`
		Skill   string `json:"skill"`
	}
	r.POST("/api/chat/summarize", func(c *gin.Context) {
		start := time.Now()
		var req summarizeReq
		if err := c.BindJSON(&req); err != nil || strings.TrimSpace(req.Content) == "" || strings.TrimSpace(req.UserID) == "" || strings.TrimSpace(req.Skill) == "" {
			application.Logger.Warn.Printf("[chat.summarize] invalid body content_len=%d user_id_present=%t skill_present=%t", len(req.Content), strings.TrimSpace(req.UserID) != "", strings.TrimSpace(req.Skill) != "")
			c.JSON(http.StatusBadRequest, gin.H{"error": "invalid body"})
			return
		}

		// Determine today's date in server local time (YYYY-MM-DD)
		today := time.Now().Format("2006-01-02")

		// Compute current chat message count for the user+skill
		history, err := application.FirestoreClient.GetChatHistory(req.UserID)
		if err != nil {
			application.Logger.Error.Printf("[chat.summarize] user_id=%s history_error=%v", req.UserID, err)
			c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to fetch chat history"})
			return
		}
		skillLower := strings.ToLower(strings.TrimSpace(req.Skill))
		currentCount := 0
		if skillLower == "" {
			currentCount = len(history.Messages)
		} else {
			for _, m := range history.Messages {
				if strings.ToLower(m.Skill) == skillLower {
					currentCount++
				}
			}
		}

		// Try cache first
		cached, err := application.FirestoreClient.GetDailySummary(req.UserID, today, skillLower)
		if err == nil && cached != nil && cached.LastCount == currentCount && strings.TrimSpace(cached.Summary) != "" {
			application.Logger.Info.Printf("[chat.summarize] cache_hit user_id=%s date=%s skill=%s count=%d took=%s", req.UserID, today, skillLower, currentCount, time.Since(start))
			c.JSON(http.StatusOK, gin.H{"summary": cached.Summary, "cached": true})
			return
		}

		// Cache miss or count changed; generate new summary
		application.Logger.Info.Printf("[chat.summarize] cache_miss user_id=%s date=%s skill=%s count=%d content_len=%d", req.UserID, today, skillLower, currentCount, len(req.Content))
		sum, err := application.GPTClient.Summarize(req.Content)
		if err != nil {
			application.Logger.Error.Printf("[chat.summarize] error=%v took=%s", err, time.Since(start))
			c.JSON(http.StatusBadGateway, gin.H{"error": "failed to summarize"})
			return
		}

		// Store/Update cache
		if err := application.FirestoreClient.SetDailySummary(req.UserID, today, skillLower, sum, currentCount); err != nil {
			application.Logger.Warn.Printf("[chat.summarize] failed_cache_store user_id=%s date=%s skill=%s err=%v", req.UserID, today, skillLower, err)
		}

		application.Logger.Info.Printf("[chat.summarize] ok summary_len=%d took=%s", len(sum), time.Since(start))
		c.JSON(http.StatusOK, gin.H{"summary": sum, "cached": false})
	})

	// DB convenience endpoints
	r.GET("/api/db/user", func(c *gin.Context) {
		start := time.Now()
		userID := c.Query("user_id")
		if userID == "" {
			application.Logger.Warn.Println("[db.user] missing user_id")
			c.JSON(http.StatusBadRequest, gin.H{"error": "missing user_id"})
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
