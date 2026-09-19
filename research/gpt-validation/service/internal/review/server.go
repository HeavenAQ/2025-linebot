package review

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"io/fs"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// Login failure budget per client IP.
const (
	MaxAuthFailures   = 10
	AuthFailureWindow = 10 * time.Minute
)

// ContentSecurityPolicy is sent on every response.
const ContentSecurityPolicy = "default-src 'self'; media-src https://storage.googleapis.com; img-src 'self' data:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; object-src 'none'"

// Server wires handlers to the store, signer and authenticator.
type Server struct {
	batchID string
	store   Store
	signer  Signer
	auth    *Authenticator
	limiter *FailureLimiter
	log     *slog.Logger
	static  fs.FS
	now     func() time.Time
}

// NewServer builds a server. static holds index.html, app.css and app.js.
func NewServer(batchID string, store Store, signer Signer, auth *Authenticator, static fs.FS, log *slog.Logger) *Server {
	if log == nil {
		log = slog.New(slog.NewJSONHandler(io.Discard, nil))
	}
	return &Server{
		batchID: batchID,
		store:   store,
		signer:  signer,
		auth:    auth,
		limiter: NewFailureLimiter(MaxAuthFailures, AuthFailureWindow),
		log:     log,
		static:  static,
		now:     time.Now,
	}
}

// Handler returns the full HTTP handler with middleware.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /api/login", s.handleLogin)
	mux.HandleFunc("GET /api/items", s.expert(s.handleListItems))
	mux.HandleFunc("GET /api/items/{item_id}", s.expert(s.handleGetItem))
	mux.HandleFunc("PUT /api/items/{item_id}/step1", s.expert(s.handleStep1))
	mux.HandleFunc("PUT /api/items/{item_id}/step2", s.expert(s.handleStep2))
	mux.HandleFunc("GET /api/admin/progress", s.admin(s.handleProgress))
	mux.HandleFunc("GET /api/admin/export/{name}", s.admin(s.handleExport))
	var files http.Handler = http.NotFoundHandler()
	if s.static != nil {
		files = http.FileServerFS(s.static)
	}
	mux.Handle("GET /", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/api/") {
			writeError(w, http.StatusNotFound, "not_found", "no such endpoint")
			return
		}
		w.Header().Set("Cache-Control", "no-cache")
		files.ServeHTTP(w, r)
	}))
	return s.logRequests(securityHeaders(mux))
}

// --- middleware -----------------------------------------------------------

func securityHeaders(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		h := w.Header()
		h.Set("Content-Security-Policy", ContentSecurityPolicy)
		h.Set("X-Content-Type-Options", "nosniff")
		h.Set("Referrer-Policy", "no-referrer")
		h.Set("X-Frame-Options", "DENY")
		if strings.HasPrefix(r.URL.Path, "/api/") {
			h.Set("Cache-Control", "no-store")
		}
		next.ServeHTTP(w, r)
	})
}

type statusRecorder struct {
	http.ResponseWriter
	status int
}

func (r *statusRecorder) WriteHeader(code int) {
	r.status = code
	r.ResponseWriter.WriteHeader(code)
}

func (s *Server) logRequests(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(rec, r)
		level := slog.LevelInfo
		if rec.status >= 500 {
			level = slog.LevelError
		} else if rec.status >= 400 {
			level = slog.LevelWarn
		}
		s.log.LogAttrs(r.Context(), level, "request",
			slog.String("method", r.Method),
			slog.String("path", r.URL.Path),
			slog.Int("status", rec.status),
			slog.Int64("duration_ms", time.Since(start).Milliseconds()),
		)
	})
}

type principalHandler func(w http.ResponseWriter, r *http.Request, p Principal)

// authenticate checks X-Review-Code under the per-IP failure budget.
func (s *Server) authenticate(w http.ResponseWriter, r *http.Request, code string) (Principal, bool) {
	ip := ClientIP(r)
	if blocked, retry := s.limiter.Blocked(ip); blocked {
		w.Header().Set("Retry-After", strconv.Itoa(int(retry.Round(time.Second).Seconds())))
		writeError(w, http.StatusTooManyRequests, "too_many_attempts", "too many failed attempts; try again later")
		return Principal{}, false
	}
	p, ok := s.auth.Authenticate(code)
	if !ok {
		if code != "" {
			s.limiter.RecordFailure(ip)
			s.log.Warn("access code rejected", slog.String("client_ip", ip))
		}
		writeError(w, http.StatusUnauthorized, "unauthorized", "invalid access code")
		return Principal{}, false
	}
	return p, true
}

func (s *Server) withRole(role Role, next principalHandler) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		p, ok := s.authenticate(w, r, r.Header.Get("X-Review-Code"))
		if !ok {
			return
		}
		if p.Role != role {
			writeError(w, http.StatusForbidden, "forbidden", "not allowed for this role")
			return
		}
		next(w, r, p)
	}
}

func (s *Server) expert(next principalHandler) http.HandlerFunc { return s.withRole(RoleExpert, next) }
func (s *Server) admin(next principalHandler) http.HandlerFunc  { return s.withRole(RoleAdmin, next) }

// --- helpers --------------------------------------------------------------

func writeJSON(w http.ResponseWriter, status int, v any) {
	var buf bytes.Buffer
	if err := json.NewEncoder(&buf).Encode(v); err != nil {
		http.Error(w, `{"error":"internal"}`, http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_, _ = w.Write(buf.Bytes())
}

func writeError(w http.ResponseWriter, status int, code, message string) {
	writeJSON(w, status, map[string]string{"error": code, "message": message})
}

func (s *Server) internalError(w http.ResponseWriter, r *http.Request, what string, err error) {
	s.log.ErrorContext(r.Context(), what, slog.String("error", err.Error()), slog.String("path", r.URL.Path))
	writeError(w, http.StatusInternalServerError, "internal", "internal error")
}

func (s *Server) writeBodyError(w http.ResponseWriter, r *http.Request, err error) {
	var tooLarge *http.MaxBytesError
	var verr *ValidationError
	switch {
	case errors.As(err, &tooLarge):
		writeError(w, http.StatusRequestEntityTooLarge, "too_large", "request body too large")
	case errors.As(err, &verr):
		writeError(w, http.StatusBadRequest, "invalid", verr.Msg)
	default:
		s.internalError(w, r, "read body", err)
	}
}

func validItemID(id string) bool {
	if id == "" || len(id) > 128 || strings.Contains(id, "__") {
		return false
	}
	for _, c := range id {
		switch {
		case c >= 'a' && c <= 'z', c >= 'A' && c <= 'Z', c >= '0' && c <= '9', c == '_', c == '-', c == '.':
		default:
			return false
		}
	}
	return id != "." && id != ".."
}

// servableItem loads an item the expert may see, writing 404 otherwise.
func (s *Server) servableItem(w http.ResponseWriter, r *http.Request) (Item, bool) {
	id := r.PathValue("item_id")
	if !validItemID(id) {
		writeError(w, http.StatusNotFound, "not_found", "item not found")
		return Item{}, false
	}
	item, err := s.store.GetItem(r.Context(), id)
	if errors.Is(err, ErrNotFound) || (err == nil && !item.ServableTo(s.batchID)) {
		writeError(w, http.StatusNotFound, "not_found", "item not found")
		return Item{}, false
	}
	if err != nil {
		s.internalError(w, r, "get item", err)
		return Item{}, false
	}
	return item, true
}

// ownRating returns the caller's rating or nil when there is none.
func (s *Server) ownRating(ctx context.Context, itemID, expertID string) (*Rating, error) {
	rating, err := s.store.GetRating(ctx, itemID, expertID)
	if errors.Is(err, ErrNotFound) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return &rating, nil
}

func (s *Server) signFunc(ctx context.Context) func(string) string {
	return func(path string) string {
		url, err := s.signer.SignedURL(path)
		if err != nil {
			s.log.ErrorContext(ctx, "sign media URL", slog.String("error", err.Error()))
			return ""
		}
		return url
	}
}

// --- handlers -------------------------------------------------------------

func (s *Server) handleLogin(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, MaxBodyBytes)
	var body struct {
		Code string `json:"code"`
	}
	if err := decodeStrict(r.Body, &body); err != nil {
		s.writeBodyError(w, r, err)
		return
	}
	code := strings.TrimSpace(body.Code)
	if code == "" {
		// Still subject to the limiter so a blocked client sees 429.
		code = "\x00empty"
	}
	p, ok := s.authenticate(w, r, code)
	if !ok {
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"expert_id": p.ID, "role": string(p.Role)})
}

func (s *Server) handleListItems(w http.ResponseWriter, r *http.Request, p Principal) {
	items, err := s.store.ListItems(r.Context(), s.batchID)
	if err != nil {
		s.internalError(w, r, "list items", err)
		return
	}
	ratings, err := s.store.ListRatings(r.Context(), s.batchID, p.ID)
	if err != nil {
		s.internalError(w, r, "list ratings", err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"items": BuildItemSummaries(p.ID, s.batchID, items, ratings)})
}

func (s *Server) handleGetItem(w http.ResponseWriter, r *http.Request, p Principal) {
	item, ok := s.servableItem(w, r)
	if !ok {
		return
	}
	rating, err := s.ownRating(r.Context(), item.ItemID, p.ID)
	if err != nil {
		s.internalError(w, r, "get rating", err)
		return
	}
	writeJSON(w, http.StatusOK, BuildItemView(item, rating, s.signFunc(r.Context())))
}

func (s *Server) handleStep1(w http.ResponseWriter, r *http.Request, p Principal) {
	item, ok := s.servableItem(w, r)
	if !ok {
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, MaxBodyBytes)
	needs, seconds, err := ParseStep1(item, r.Body)
	if err != nil {
		s.writeBodyError(w, r, err)
		return
	}
	now := s.now().UTC()
	rating := Rating{
		ItemID:           item.ItemID,
		ExpertID:         p.ID,
		Skill:            item.Skill,
		BatchID:          item.BatchID,
		NeedsImprovement: needs,
		Step1SubmittedAt: &now,
		Step1Seconds:     seconds,
	}
	switch err := s.store.SubmitStep1(r.Context(), rating); {
	case errors.Is(err, ErrStep1Locked):
		writeError(w, http.StatusConflict, "step1_locked", "step 1 was already submitted")
		return
	case err != nil:
		s.internalError(w, r, "submit step 1", err)
		return
	}
	saved, err := s.ownRating(r.Context(), item.ItemID, p.ID)
	if err != nil || saved == nil {
		saved = &rating
	}
	writeJSON(w, http.StatusOK, BuildItemView(item, saved, s.signFunc(r.Context())))
}

func (s *Server) handleStep2(w http.ResponseWriter, r *http.Request, p Principal) {
	item, ok := s.servableItem(w, r)
	if !ok {
		return
	}
	existing, err := s.ownRating(r.Context(), item.ItemID, p.ID)
	if err != nil {
		s.internalError(w, r, "get rating", err)
		return
	}
	if !existing.Step1Done() {
		writeError(w, http.StatusConflict, "step1_required", "submit step 1 first")
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, MaxBodyBytes)
	step2, err := ParseStep2(item, r.Body)
	if err != nil {
		s.writeBodyError(w, r, err)
		return
	}
	switch err := s.store.SubmitStep2(r.Context(), item.ItemID, p.ID, step2, s.now().UTC()); {
	case errors.Is(err, ErrStep1Required):
		writeError(w, http.StatusConflict, "step1_required", "submit step 1 first")
		return
	case err != nil:
		s.internalError(w, r, "submit step 2", err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"item_id": item.ItemID, "completed": true})
}

func (s *Server) loadExportData(ctx context.Context) (ExportData, error) {
	items, err := s.store.ListItems(ctx, s.batchID)
	if err != nil {
		return ExportData{}, err
	}
	ratings, err := s.store.ListRatings(ctx, s.batchID, "")
	if err != nil {
		return ExportData{}, err
	}
	return ExportData{BatchID: s.batchID, ExpertIDs: s.auth.ExpertIDs(), Items: items, Ratings: ratings}, nil
}

func (s *Server) handleProgress(w http.ResponseWriter, r *http.Request, _ Principal) {
	d, err := s.loadExportData(r.Context())
	if err != nil {
		s.internalError(w, r, "load progress", err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"batch_id": s.batchID,
		"experts":  BuildProgress(d.ExpertIDs, s.batchID, d.Items, d.Ratings),
	})
}

var exporters = map[string]func(io.Writer, ExportData) error{
	"items.csv":    WriteItemsCSV,
	"criteria.csv": WriteCriteriaCSV,
	"cues.csv":     WriteCuesCSV,
	"overall.csv":  WriteOverallCSV,
}

func (s *Server) handleExport(w http.ResponseWriter, r *http.Request, _ Principal) {
	name := r.PathValue("name")
	export, ok := exporters[name]
	if !ok {
		writeError(w, http.StatusNotFound, "not_found", "no such export")
		return
	}
	d, err := s.loadExportData(r.Context())
	if err != nil {
		s.internalError(w, r, "load export", err)
		return
	}
	var buf bytes.Buffer
	if err := export(&buf, d); err != nil {
		s.internalError(w, r, "write export", err)
		return
	}
	w.Header().Set("Content-Type", "text/csv; charset=utf-8")
	w.Header().Set("Content-Disposition", `attachment; filename="`+name+`"`)
	_, _ = w.Write(buf.Bytes())
}

// NewLogger writes Cloud Logging structured JSON ("severity", "message").
func NewLogger(w io.Writer) *slog.Logger {
	return slog.New(slog.NewJSONHandler(w, &slog.HandlerOptions{
		ReplaceAttr: func(groups []string, a slog.Attr) slog.Attr {
			if len(groups) > 0 {
				return a
			}
			switch a.Key {
			case slog.MessageKey:
				a.Key = "message"
			case slog.LevelKey:
				a.Key = "severity"
				switch level := a.Value.Any().(slog.Level); {
				case level >= slog.LevelError:
					a.Value = slog.StringValue("ERROR")
				case level >= slog.LevelWarn:
					a.Value = slog.StringValue("WARNING")
				case level >= slog.LevelInfo:
					a.Value = slog.StringValue("INFO")
				default:
					a.Value = slog.StringValue("DEBUG")
				}
			}
			return a
		},
	}))
}
