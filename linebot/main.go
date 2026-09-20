package main

import (
	"context"
	"errors"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/HeavenAQ/nstc-linebot-2025/api/auth"
	"github.com/HeavenAQ/nstc-linebot-2025/api/db"
	"github.com/HeavenAQ/nstc-linebot-2025/api/obs"
	"github.com/HeavenAQ/nstc-linebot-2025/api/ratelimit"
	"github.com/HeavenAQ/nstc-linebot-2025/api/storage"
	"github.com/HeavenAQ/nstc-linebot-2025/app"
	"github.com/HeavenAQ/nstc-linebot-2025/commons"
	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
)

// authenticatedUserKey holds the LINE user ID proven by the caller's credential.
const authenticatedUserKey = "authenticatedUserID"

// maxLearnerRequestBody bounds learner API bodies; the largest legitimate one
// is a 4 KB weekly reflection.
const maxLearnerRequestBody = 64 << 10

// Per-instance budgets for the learner API (see api/ratelimit). The IP limit
// is generous because a classroom shares one campus NAT address, and the
// failed authentication limit is small so forged credentials cost little.
var (
	ipLimiter          = ratelimit.New(1800, 300)
	authFailureLimiter = ratelimit.New(30, 20)
	learnerLimiter     = ratelimit.New(240, 60)
)

func tooManyRequests(c *gin.Context, limiter *ratelimit.Limiter, name string) {
	obs.Event(c.Request.Context(), obs.Warning, "rate limited", map[string]any{
		"limiter": name, "route": c.FullPath(), "client_ip": ratelimit.ClientIP(c.Request),
	})
	c.Header("Retry-After", limiter.RetryAfter())
	c.AbortWithStatusJSON(http.StatusTooManyRequests, gin.H{"error": "請求過於頻繁，請稍後再試。"})
}

func main() {
	gin.SetMode(gin.ReleaseMode)
	application := app.NewApp(".env")
	obs.Configure(application.Config.GCP.ProjectID)

	r := gin.New()
	// Carry the front end's trace and a request ID through every handler, so
	// structured logs and calls to the analysis service share one trace.
	r.Use(func(c *gin.Context) {
		request := obs.FromHeaders(c.GetHeader(obs.TraceHeader), c.GetHeader(obs.RequestIDHeader))
		c.Request = c.Request.WithContext(obs.WithRequest(c.Request.Context(), request))
		c.Header(obs.RequestIDHeader, request.ID)
		c.Next()
	})
	r.Use(gin.CustomRecovery(func(c *gin.Context, recovered any) {
		obs.Event(c.Request.Context(), obs.Error, "panic recovered", map[string]any{
			"panic": recovered, "route": c.FullPath(),
		})
		c.AbortWithStatus(http.StatusInternalServerError)
	}))

	// The browser calls this API from the deployment's own web app, so the
	// allowed origin is derived from the review URL rather than written out.
	// A hard-coded origin silently belongs to whichever deployment was built
	// first: this variant serves a different site, and every learner request
	// from it was refused with 403 while the LINE webhook, which sends no
	// Origin header, kept working and hid the fault.
	allowedOrigins := []string{"http://localhost:3000"}
	if reviewURL, err := url.Parse(application.Config.ReviewURL()); err == nil &&
		reviewURL.Scheme != "" && reviewURL.Host != "" {
		allowedOrigins = append(allowedOrigins, reviewURL.Scheme+"://"+reviewURL.Host)
	} else {
		application.Logger.Warn.Println(
			"[cors] LIFF_REVIEW_URL is unusable; the web app will be refused",
		)
	}
	r.Use(cors.New(cors.Config{
		AllowOrigins:  allowedOrigins,
		AllowMethods:  []string{"GET", "POST", "PUT", "DELETE"},
		AllowHeaders:  []string{"Origin", "Content-Type", "Authorization", auth.AccessTokenHeader},
		ExposeHeaders: []string{"Retry-After", obs.RequestIDHeader},
	}))

	// Routes
	r.POST("/callback", func(c *gin.Context) {
		handler := application.LineWebhookHandler()
		handler(c.Writer, c.Request)
	})
	r.GET("/test", func(c *gin.Context) { c.String(http.StatusOK, "Hello, World!") })
	r.POST("/internal/analysis/task", func(c *gin.Context) { application.HandleAnalysisTask(c.Writer, c.Request) })
	r.POST("/internal/analysis/outbox", func(c *gin.Context) { application.HandleAnalysisOutbox(c.Writer, c.Request) })
	r.POST("/internal/analysis/warmup", func(c *gin.Context) { application.HandleAnalysisWarmup(c.Writer, c.Request) })
	r.POST("/internal/analysis/capacity", func(c *gin.Context) { application.HandleAnalysisCapacity(c.Writer, c.Request) })
	r.POST("/internal/stats/rebuild", func(c *gin.Context) { application.HandleClassStatsRebuild(c.Writer, c.Request) })

	// Every learner-facing route below identifies its caller from a verified
	// LINE credential (ID token, or the LIFF access token once that expires).
	// A user ID in a query string or body proves nothing -- anyone can send
	// anyone's -- so the ID comes from the credential and request-supplied IDs
	// are only ever compared against it.
	verifier := auth.NewVerifier(application.Config.Line.LoginChannelID)
	if verifier == nil {
		application.Logger.Warn.Println(
			"[auth] LINE_LOGIN_CHANNEL_ID is not set; learner API routes will refuse every request",
		)
	}
	// authenticate runs the limits and the credential check, and reports
	// whether the request may go on. Registration itself has to be reachable
	// before a learner is registered, so the registration check below is a
	// separate step rather than part of this one.
	authenticate := func(c *gin.Context) bool {
		if verifier == nil {
			c.AbortWithStatusJSON(http.StatusServiceUnavailable, gin.H{"error": "authentication is not configured"})
			return false
		}
		clientIP := ratelimit.ClientIP(c.Request)
		if !ipLimiter.Allow(clientIP) {
			tooManyRequests(c, ipLimiter, "client_ip")
			return false
		}
		// An address that keeps presenting bad credentials is refused before
		// any verification work is spent on it.
		if authFailureLimiter.Exhausted(clientIP) {
			tooManyRequests(c, authFailureLimiter, "auth_failures")
			return false
		}
		c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, maxLearnerRequestBody)
		userID, err := verifier.Authenticate(c.Request.Context(), c.Request)
		if err != nil {
			if !errors.Is(err, auth.ErrUnauthorized) {
				// LINE unreachable is not the learner's fault; a 401 would tell
				// the page its session expired.
				obs.Event(c.Request.Context(), obs.Error, "authentication unavailable", map[string]any{"error": err})
				c.AbortWithStatusJSON(http.StatusServiceUnavailable, gin.H{"error": "暫時無法驗證登入，請稍後再試。"})
				return false
			}
			authFailureLimiter.Allow(clientIP)
			obs.Event(c.Request.Context(), obs.Warning, "authentication rejected", map[string]any{
				"error": err, "client_ip": clientIP, "route": c.FullPath(),
			})
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{"error": "unauthorized"})
			return false
		}
		if !learnerLimiter.Allow(userID) {
			tooManyRequests(c, learnerLimiter, "learner")
			return false
		}
		c.Set(authenticatedUserKey, userID)
		return true
	}
	// Signed in, registered or not. Only the registration form may use this.
	requireSignIn := func(c *gin.Context) {
		if !authenticate(c) {
			return
		}
		c.Next()
	}
	requireLearner := func(c *gin.Context) {
		if !authenticate(c) {
			return
		}
		if !auth.AllowRegisteredLearner(c, c.GetString(authenticatedUserKey), application.FirestoreClient.GetUserData) {
			return
		}
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

	// The learner's own profile: their experiment number, real name and
	// handedness. Signing in is enough, because this is also the form that
	// registers them -- every other learner route needs registration first.
	r.PUT("/api/db/profile", requireSignIn, func(c *gin.Context) {
		start := time.Now()
		userID := learnerID(c)
		var body struct {
			RealName         string `json:"real_name"`
			ExperimentNumber string `json:"experiment_number"`
			Handedness       string `json:"handedness"`
		}
		if err := c.ShouldBindJSON(&body); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": db.RegistrationFormatError})
			return
		}
		registration, err := db.ParseExperimentRegistration(body.ExperimentNumber + " " + body.RealName)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": db.RegistrationFormatError})
			return
		}
		handedness, err := db.HandednessStrToEnum(strings.ToLower(strings.TrimSpace(body.Handedness)))
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "handedness must be left or right"})
			return
		}
		// A learner can open the dashboard before ever messaging the bot, so
		// their record and storage folders may not exist yet.
		user, err := application.EnsureUserData(userID)
		if err != nil {
			application.Logger.Error.Printf("[db.profile] user_id=%s create err=%v", userID, err)
			c.JSON(http.StatusServiceUnavailable, gin.H{"error": "暫時無法儲存資料，請稍後再試。"})
			return
		}
		saved, err := application.FirestoreClient.SaveExperimentRegistration(userID, registration)
		if err != nil {
			status := http.StatusServiceUnavailable
			message := "暫時無法儲存資料，請稍後再試。"
			if errors.Is(err, db.ErrRegistrationFormat) {
				status, message = http.StatusBadRequest, db.RegistrationFormatError
			}
			application.Logger.Error.Printf("[db.profile] user_id=%s save err=%v", userID, err)
			c.JSON(status, gin.H{"error": message})
			return
		}
		if saved.Handedness != handedness {
			if err := application.FirestoreClient.UpdateUserHandedness(saved, handedness); err != nil {
				application.Logger.Error.Printf("[db.profile] user_id=%s handedness err=%v", userID, err)
				c.JSON(http.StatusServiceUnavailable, gin.H{"error": "暫時無法儲存資料，請稍後再試。"})
				return
			}
		}
		application.Logger.Info.Printf(
			"[db.profile] saved user_id=%s first_time=%t took=%s",
			userID, !user.HasExperimentRegistration(), time.Since(start),
		)
		c.JSON(http.StatusOK, saved)
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
		if work.AnalysisStatus == "pending" {
			c.Header("Retry-After", "3")
			c.JSON(http.StatusAccepted, gin.H{"status": "pending", "error": "影片分析中，完成後會自動顯示。"})
			return
		}
		if work.AnalysisStatus == "failed" {
			c.JSON(http.StatusUnprocessableEntity, gin.H{"status": "failed", "error": work.AnalysisError})
			return
		}
		if work.StudentVideo.ObjectPath == "" {
			c.JSON(http.StatusConflict, gin.H{"error": "analysis predates synchronized playback"})
			return
		}
		// Signed here rather than through the analysis service, so a student
		// opening a video never waits on the GPU service's capacity.
		sign := func(media *commons.MediaRef) error {
			if media.ObjectPath == "" {
				return nil
			}
			// Sign in the bucket that actually holds the object. Sharing the
			// analysis service means sharing the bucket it writes to, which is
			// not this deployment's own, and signing against the wrong bucket
			// yields a URL that 404s rather than an error here.
			signed, err := application.StorageClient.SignPlaybackURLIn(
				storage.BucketFromGCSURI(media.GCSURI),
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
		// The poster the player shows until the video has enough data to paint
		// its first frame, which is otherwise a blank rectangle. A thumbnail
		// that predates thumbnailing, or has since been removed, simply leaves
		// the poster out rather than failing the whole playback.
		var thumbnail commons.MediaRef
		if work.Thumbnail != "" {
			signed, err := application.StorageClient.SignThumbnailURL(
				work.Thumbnail, application.Config.GCP.ServiceAccountEmail,
			)
			if err != nil {
				application.Logger.Warn.Printf(
					"[db.playback] thumbnail unavailable user=%s skill=%s date=%s err=%v",
					userID, skill, workDate, err,
				)
			} else {
				thumbnail = signed
			}
		}
		c.JSON(http.StatusOK, gin.H{
			"analysis_id":            work.AnalysisID,
			"handedness":             work.Handedness,
			"student_video":          work.StudentVideo,
			"feedback_video":         work.FeedbackVideo,
			"skeleton_overlay_video": work.SkeletonOverlayVideo,
			"thumbnail":              thumbnail,
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
		DefaultReadTimeout = 100 * time.Second
		// Matches the production service, so the two deployments time out
		// slow responses identically.
		DefaultWriteTimeout = 150 * time.Second
		DefaultIdleTimeout  = 180 * time.Second
	)
	srv := &http.Server{
		Addr:         "0.0.0.0:" + application.Config.Port,
		Handler:      r,
		ReadTimeout:  DefaultReadTimeout,
		WriteTimeout: DefaultWriteTimeout,
		IdleTimeout:  DefaultIdleTimeout,
	}

	// Cloud Run sends SIGTERM and allows 10 seconds before killing the
	// instance: stop accepting requests, let in-flight ones finish, then close
	// the clients. A queued analysis cut off here is retried by Cloud Tasks.
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGTERM, syscall.SIGINT)
	defer stop()
	serverErrors := make(chan error, 1)
	go func() {
		application.Logger.Info.Println("Server started on port " + application.Config.Port)
		serverErrors <- srv.ListenAndServe()
	}()
	select {
	case err := <-serverErrors:
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			application.Logger.Error.Printf("server stopped: %v", err)
			os.Exit(1)
		}
	case <-ctx.Done():
		application.Logger.Info.Println("shutdown signal received; draining requests")
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 8*time.Second)
		defer cancel()
		if err := srv.Shutdown(shutdownCtx); err != nil {
			application.Logger.Warn.Printf("graceful shutdown incomplete: %v", err)
		}
		application.Close()
		application.Logger.Info.Println("shutdown complete")
	}
}
