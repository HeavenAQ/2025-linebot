package main

import (
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/HeavenAQ/nstc-linebot-2025/api/auth"
	"github.com/HeavenAQ/nstc-linebot-2025/api/db"
	"github.com/HeavenAQ/nstc-linebot-2025/app"
	"github.com/HeavenAQ/nstc-linebot-2025/commons"
	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
)

// (runtime snake_case conversion removed; DB is migrated instead)

// authenticatedUserKey holds the LINE user ID proven by the caller's ID token.
const authenticatedUserKey = "authenticatedUserID"

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

	// Every learner-facing route below identifies its caller from a verified
	// LINE ID token. A user ID in a query string or body proves nothing --
	// anyone can send anyone's -- so the ID comes from the token's subject and
	// request-supplied IDs are only ever compared against it.
	verifier := auth.NewVerifier(application.Config.Line.LoginChannelID)
	if verifier == nil {
		application.Logger.Warn.Println(
			"[auth] LINE_LOGIN_CHANNEL_ID is not set; learner API routes will refuse every request",
		)
	}
	requireLearner := func(c *gin.Context) {
		if verifier == nil {
			c.AbortWithStatusJSON(http.StatusServiceUnavailable, gin.H{"error": "authentication is not configured"})
			return
		}
		userID, err := verifier.UserID(c.Request.Context(), auth.BearerToken(c.GetHeader("Authorization")))
		if err != nil {
			application.Logger.Warn.Printf("[auth] rejected a request: %v", err)
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{"error": "unauthorized"})
			return
		}
		c.Set(authenticatedUserKey, userID)
		c.Next()
	}
	// learnerID is the only identity these handlers may act on.
	learnerID := func(c *gin.Context) string { return c.GetString(authenticatedUserKey) }

	// Weekly reflections, written by learners in the LIFF review tab.
	r.GET("/api/db/weekly-reflections", requireLearner, func(c *gin.Context) {
		start := time.Now()
		userID := learnerID(c)
		reflections, err := application.FirestoreClient.ListWeeklyReflections(userID)
		if err != nil {
			application.Logger.Error.Printf("[db.reflections] user_id=%s err=%v", userID, err)
			c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to fetch reflections"})
			return
		}
		application.Logger.Info.Printf(
			"[db.reflections] user_id=%s count=%d took=%s", userID, len(reflections), time.Since(start),
		)
		c.JSON(http.StatusOK, gin.H{"data": reflections})
	})

	// A week's record holds two notes -- the reflection and the learner's own
	// 課前檢視要點 -- written from separate editors in the review tab. Both are
	// optional here and only the ones actually sent are written, so a save from
	// one editor cannot blank what the other holds.
	type weeklyReflectionReq struct {
		UserID  string  `json:"user_id"`
		Week    string  `json:"week"`
		Note    *string `json:"note"`
		Preview *string `json:"preview"`
	}
	r.PUT("/api/db/weekly-reflection", requireLearner, func(c *gin.Context) {
		start := time.Now()
		var req weeklyReflectionReq
		if err := c.BindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "invalid body"})
			return
		}
		userID := learnerID(c)
		week := strings.TrimSpace(req.Week)
		if userID == "" || !db.ValidWeek(week) {
			application.Logger.Warn.Printf(
				"[db.reflection] rejected user_id_present=%t week=%q", userID != "", week,
			)
			c.JSON(http.StatusBadRequest, gin.H{"error": "missing user_id or malformed week"})
			return
		}
		notes := map[db.WeeklyNoteField]string{}
		if req.Note != nil {
			notes[db.ReflectionNote] = *req.Note
		}
		if req.Preview != nil {
			notes[db.PreviewNote] = *req.Preview
		}
		if len(notes) == 0 {
			c.JSON(http.StatusBadRequest, gin.H{"error": "no note to save"})
			return
		}
		// A note is stored as written, including blank to clear it, but a
		// runaway paste is refused rather than silently truncated.
		for _, text := range notes {
			if len(text) > db.MaxReflectionLength {
				c.JSON(http.StatusRequestEntityTooLarge, gin.H{"error": "note is too long"})
				return
			}
		}
		reflection, err := application.FirestoreClient.SetWeeklyReflectionNotes(userID, week, notes)
		if err != nil {
			application.Logger.Error.Printf("[db.reflection] user_id=%s week=%s err=%v", userID, week, err)
			c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to save reflection"})
			return
		}
		application.Logger.Info.Printf(
			"[db.reflection] saved user_id=%s week=%s note_length=%d preview_length=%d took=%s",
			userID, week, len(reflection.Note), len(reflection.Preview), time.Since(start),
		)
		c.JSON(http.StatusOK, reflection)
	})

	// DB convenience endpoints
	r.GET("/api/db/user", requireLearner, func(c *gin.Context) {
		start := time.Now()
		userID := learnerID(c)
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

	r.GET("/api/db/playback", requireLearner, func(c *gin.Context) {
		// Playback hands out signed URLs to practice video, so it serves the
		// caller's own analyses only.
		userID := learnerID(c)
		skill := strings.ToLower(strings.TrimSpace(c.Query("skill")))
		workDate := strings.TrimSpace(c.Query("work_date"))
		if skill == "" || workDate == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "missing skill or work_date"})
			return
		}
		user, err := application.FirestoreClient.GetUserData(userID)
		if err != nil {
			c.JSON(http.StatusNotFound, gin.H{"error": "user not found"})
			return
		}
		portfolio := user.Portfolio.GetSkillPortfolio(skill)
		work, ok := portfolio[workDate]
		if !ok {
			c.JSON(http.StatusNotFound, gin.H{"error": "analysis not found"})
			return
		}
		if work.StudentVideo.ObjectPath == "" {
			c.JSON(http.StatusConflict, gin.H{"error": "analysis predates synchronized playback"})
			return
		}
		// Signed here rather than through the analysis service: that service
		// scales to zero, and a student opening a video should never wait for a
		// GPU to boot just to be handed a URL.
		sign := func(media *commons.MediaRef) error {
			if media.ObjectPath == "" {
				return nil
			}
			signed, err := application.StorageClient.SignPlaybackURL(
				media.ObjectPath, application.Config.GCP.ServiceAccountEmail,
			)
			if err != nil {
				return err
			}
			media.SignedURL = signed.SignedURL
			media.SignedURLExpires = signed.SignedURLExpires
			return nil
		}
		for _, media := range []*commons.MediaRef{
			&work.StudentVideo, &work.SkeletonOverlayVideo, &work.Expert.Video,
		} {
			if err := sign(media); err != nil {
				application.Logger.Error.Printf(
					"[db.playback] signing failed user=%s skill=%s date=%s err=%v",
					userID, skill, workDate, err,
				)
				c.JSON(http.StatusBadGateway, gin.H{"error": "failed to refresh playback URLs"})
				return
			}
		}
		work.FeedbackVideo = work.StudentVideo
		c.JSON(http.StatusOK, gin.H{
			"analysis_id":            work.AnalysisID,
			"handedness":             work.Handedness,
			"student_video":          work.StudentVideo,
			"feedback_video":         work.FeedbackVideo,
			"skeleton_overlay_video": work.SkeletonOverlayVideo,
			"expert":                 work.Expert,
			"timeline":               work.Timeline,
			"grade":                  work.GradingOutcome,
		})
	})

	// Stats endpoints
	r.GET("/api/db/stats/users/:id", requireLearner, func(c *gin.Context) {
		start := time.Now()
		// The path still carries an ID so existing links keep working, but a
		// learner may only read their own scores.
		id := learnerID(c)
		if requested := c.Param("id"); requested != "" && requested != id {
			application.Logger.Warn.Printf("[db.stats.user] refused cross-user read of %s", requested)
			c.JSON(http.StatusForbidden, gin.H{"error": "forbidden"})
			return
		}
		skill := strings.ToLower(strings.TrimSpace(c.Query("skill")))
		if skill == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "missing skill"})
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

	r.GET("/api/db/stats/class", requireLearner, func(c *gin.Context) {
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
