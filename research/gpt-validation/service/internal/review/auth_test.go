package review

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestAuthenticateRoles(t *testing.T) {
	a := NewAuthenticator(testCodes)
	cases := []struct {
		code     string
		wantOK   bool
		wantID   string
		wantRole Role
	}{
		{"code-alpha-1234", true, "expert_a", RoleExpert},
		{"code-bravo-5678", true, "expert_b", RoleExpert},
		{"code-admin-9999", true, "admin", RoleAdmin},
		{"", false, "", ""},
		{"code-alpha-123", false, "", ""},
		{"code-alpha-12345", false, "", ""},
		{"CODE-ALPHA-1234", false, "", ""},
		{"expert_a", false, "", ""},
	}
	for _, c := range cases {
		p, ok := a.Authenticate(c.code)
		if ok != c.wantOK || p.ID != c.wantID || p.Role != c.wantRole {
			t.Errorf("Authenticate(%q) = %+v, %v; want %s/%s, %v", c.code, p, ok, c.wantID, c.wantRole, c.wantOK)
		}
	}
	if got := a.ExpertIDs(); strings.Join(got, ",") != "expert_a,expert_b" {
		t.Errorf("ExpertIDs = %v", got)
	}
}

func TestParseAccessCodes(t *testing.T) {
	bad := []string{
		``,
		`not json`,
		`{}`,
		`{"admin":"x1"}`,
		`{"a":""}`,
		`{"a":" padded "}`,
		`{"a/b":"code"}`,
		`{"a":"same","b":"same"}`,
		`["a"]`,
	}
	for _, raw := range bad {
		if _, err := ParseAccessCodes(raw); err == nil {
			t.Errorf("ParseAccessCodes(%q) accepted", raw)
		}
	}
	codes, err := ParseAccessCodes(`{"expert_1":"abc","admin":"def"}`)
	if err != nil || len(codes) != 2 {
		t.Fatalf("valid codes rejected: %v", err)
	}
}

func TestLoadConfigFailsFast(t *testing.T) {
	env := map[string]string{}
	_, err := LoadConfig(func(k string) string { return env[k] })
	if err == nil {
		t.Fatal("empty environment accepted")
	}
	for _, name := range []string{"GCP_PROJECT_ID", "GCS_BUCKET_NAME", "GCP_SERVICE_ACCOUNT_EMAIL", "BATCH_ID", "ACCESS_CODES"} {
		if !strings.Contains(err.Error(), name) {
			t.Errorf("error does not mention %s: %v", name, err)
		}
	}
	env = map[string]string{
		"GCP_PROJECT_ID": "p", "GCS_BUCKET_NAME": "b", "GCP_SERVICE_ACCOUNT_EMAIL": "sa@p.iam.gserviceaccount.com",
		"BATCH_ID": "batch", "ACCESS_CODES": `{"e1":"c1"}`,
	}
	cfg, err := LoadConfig(func(k string) string { return env[k] })
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Port != "8080" || cfg.BatchID != "batch" || cfg.AccessCodes["e1"] != "c1" {
		t.Errorf("unexpected config %+v", cfg)
	}
}

func TestLoginEndpoint(t *testing.T) {
	env := newTestEnv(t)
	rec := env.do("POST", "/api/login", "", map[string]string{"code": "code-admin-9999"})
	if rec.Code != 200 {
		t.Fatalf("login status %d: %s", rec.Code, rec.Body)
	}
	m := decodeMap(t, rec)
	if m["expert_id"] != "admin" || m["role"] != "admin" {
		t.Errorf("login body %v", m)
	}
	rec = env.do("POST", "/api/login", "", map[string]string{"code": "code-alpha-1234"})
	if m := decodeMap(t, rec); m["role"] != "expert" || m["expert_id"] != "expert_a" {
		t.Errorf("expert login body %v", m)
	}
	if rec := env.do("POST", "/api/login", "", map[string]string{"code": "wrong"}); rec.Code != 401 {
		t.Errorf("wrong code status %d", rec.Code)
	}
}

func TestRoleEnforcement(t *testing.T) {
	env := newTestEnv(t, testItem("serve-a", 1))
	if rec := env.do("GET", "/api/items", "", nil); rec.Code != 401 {
		t.Errorf("no code: %d", rec.Code)
	}
	if rec := env.do("GET", "/api/items", "nope", nil); rec.Code != 401 {
		t.Errorf("bad code: %d", rec.Code)
	}
	if rec := env.do("GET", "/api/items", "code-admin-9999", nil); rec.Code != 403 {
		t.Errorf("admin on expert endpoint: %d", rec.Code)
	}
	if rec := env.do("GET", "/api/admin/progress", "code-alpha-1234", nil); rec.Code != 403 {
		t.Errorf("expert on admin endpoint: %d", rec.Code)
	}
	if rec := env.do("GET", "/api/admin/export/items.csv", "code-alpha-1234", nil); rec.Code != 403 {
		t.Errorf("expert on export: %d", rec.Code)
	}
	if rec := env.do("GET", "/api/admin/progress", "code-admin-9999", nil); rec.Code != 200 {
		t.Errorf("admin progress: %d", rec.Code)
	}
}

func TestFailedLoginRateLimit(t *testing.T) {
	env := newTestEnv(t)
	for i := 0; i < MaxAuthFailures; i++ {
		if rec := env.do("POST", "/api/login", "", map[string]string{"code": "wrong"}); rec.Code != 401 {
			t.Fatalf("attempt %d: status %d", i, rec.Code)
		}
	}
	rec := env.do("POST", "/api/login", "", map[string]string{"code": "code-alpha-1234"})
	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("after budget: status %d", rec.Code)
	}
	if rec.Header().Get("Retry-After") == "" {
		t.Error("missing Retry-After")
	}
	// Other API calls from the same IP are blocked too.
	if rec := env.do("GET", "/api/items", "code-alpha-1234", nil); rec.Code != http.StatusTooManyRequests {
		t.Errorf("items after budget: %d", rec.Code)
	}
	// A different client IP is unaffected.
	req := httptest.NewRequest("POST", "/api/login", strings.NewReader(`{"code":"code-alpha-1234"}`))
	req.RemoteAddr = "198.51.100.1:1"
	rr := httptest.NewRecorder()
	env.handler.ServeHTTP(rr, req)
	if rr.Code != 200 {
		t.Errorf("other IP: %d", rr.Code)
	}
}

func TestFailureLimiterWindow(t *testing.T) {
	now := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	l := NewFailureLimiter(3, 10*time.Minute)
	l.now = func() time.Time { return now }
	for i := 0; i < 3; i++ {
		l.RecordFailure("ip")
		now = now.Add(time.Minute)
	}
	blocked, retry := l.Blocked("ip")
	if !blocked || retry != 7*time.Minute {
		t.Fatalf("blocked=%v retry=%v", blocked, retry)
	}
	now = now.Add(8 * time.Minute)
	if blocked, _ := l.Blocked("ip"); blocked {
		t.Error("still blocked after oldest failure left window")
	}
}

func TestClientIP(t *testing.T) {
	r := httptest.NewRequest("GET", "/", nil)
	r.RemoteAddr = "10.0.0.1:1234"
	if got := ClientIP(r); got != "10.0.0.1" {
		t.Errorf("remote addr: %s", got)
	}
	r.Header.Set("X-Forwarded-For", "1.1.1.1, 2.2.2.2 ,  3.3.3.3")
	if got := ClientIP(r); got != "3.3.3.3" {
		t.Errorf("xff: %s", got)
	}
}

func TestSecurityHeaders(t *testing.T) {
	env := newTestEnv(t)
	for _, path := range []string{"/", "/app.js", "/api/items"} {
		rec := env.do("GET", path, "", nil)
		h := rec.Header()
		if h.Get("Content-Security-Policy") != ContentSecurityPolicy {
			t.Errorf("%s: CSP %q", path, h.Get("Content-Security-Policy"))
		}
		if h.Get("X-Content-Type-Options") != "nosniff" || h.Get("Referrer-Policy") != "no-referrer" {
			t.Errorf("%s: missing headers %v", path, h)
		}
		isAPI := strings.HasPrefix(path, "/api/")
		if isAPI && h.Get("Cache-Control") != "no-store" {
			t.Errorf("%s: Cache-Control %q", path, h.Get("Cache-Control"))
		}
	}
	if rec := env.do("GET", "/", "", nil); rec.Code != 200 || !strings.Contains(rec.Body.String(), "<title>") {
		t.Errorf("index: %d %s", rec.Code, rec.Body)
	}
	if rec := env.do("GET", "/api/nope", "", nil); rec.Code != 404 {
		t.Errorf("unknown api: %d", rec.Code)
	}
}
