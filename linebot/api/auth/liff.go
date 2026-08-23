// Package auth establishes who is calling the learner-facing API.
//
// The LIFF client knows its own LINE user ID, but a client-supplied ID proves
// nothing: anyone can send any ID. LINE issues a signed ID token for the logged
// in user, and this verifies that token with LINE and takes the learner's
// identity from the verified subject instead.
package auth

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"
)

// VerifyEndpoint is LINE's ID token verification API.
const VerifyEndpoint = "https://api.line.me/oauth2/v2.1/verify"

// verifyTimeout bounds a single verification call.
const verifyTimeout = 5 * time.Second

// cacheTTL caps how long a verified token is trusted without re-checking.
// Tokens are short lived and every LIFF page load makes several requests, so
// re-verifying each one would add a round trip to LINE per request.
const cacheTTL = 5 * time.Minute

// Claims is the part of LINE's response this service relies on.
type Claims struct {
	Subject  string `json:"sub"`
	Audience string `json:"aud"`
	Expires  int64  `json:"exp"`
	Name     string `json:"name"`
}

type cacheEntry struct {
	subject   string
	expiresAt time.Time
}

// Verifier checks LIFF ID tokens against LINE.
type Verifier struct {
	channelID string
	client    *http.Client
	endpoint  string

	mu    sync.Mutex
	cache map[string]cacheEntry
}

// NewVerifier returns a verifier for one LINE Login channel. An empty channel
// ID yields nil: the caller decides whether to run without authentication,
// rather than this silently accepting every token.
func NewVerifier(channelID string) *Verifier {
	if strings.TrimSpace(channelID) == "" {
		return nil
	}
	return &Verifier{
		channelID: strings.TrimSpace(channelID),
		client:    &http.Client{Timeout: verifyTimeout},
		endpoint:  VerifyEndpoint,
		cache:     map[string]cacheEntry{},
	}
}

// UserID returns the LINE user ID the token was issued for.
//
// The audience check is what stops a token minted for some other LINE channel
// from being replayed here, so a token that verifies but names a different
// channel is rejected.
func (v *Verifier) UserID(ctx context.Context, idToken string) (string, error) {
	token := strings.TrimSpace(idToken)
	if token == "" {
		return "", fmt.Errorf("no id token supplied")
	}
	if subject, ok := v.cached(token); ok {
		return subject, nil
	}

	form := url.Values{"id_token": {token}, "client_id": {v.channelID}}
	request, err := http.NewRequestWithContext(
		ctx, http.MethodPost, v.endpoint, strings.NewReader(form.Encode()),
	)
	if err != nil {
		return "", fmt.Errorf("build verification request: %w", err)
	}
	request.Header.Set("Content-Type", "application/x-www-form-urlencoded")

	response, err := v.client.Do(request)
	if err != nil {
		return "", fmt.Errorf("verify id token: %w", err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return "", fmt.Errorf("id token rejected by LINE (status %d)", response.StatusCode)
	}

	var claims Claims
	if err := json.NewDecoder(response.Body).Decode(&claims); err != nil {
		return "", fmt.Errorf("decode verification response: %w", err)
	}
	if claims.Audience != v.channelID {
		return "", fmt.Errorf("id token was issued for another channel")
	}
	if claims.Subject == "" {
		return "", fmt.Errorf("id token has no subject")
	}
	if claims.Expires > 0 && time.Now().After(time.Unix(claims.Expires, 0)) {
		return "", fmt.Errorf("id token has expired")
	}

	v.remember(token, claims.Subject, claims.Expires)
	return claims.Subject, nil
}

func (v *Verifier) cached(token string) (string, bool) {
	v.mu.Lock()
	defer v.mu.Unlock()
	entry, ok := v.cache[token]
	if !ok || time.Now().After(entry.expiresAt) {
		delete(v.cache, token)
		return "", false
	}
	return entry.subject, true
}

func (v *Verifier) remember(token, subject string, expires int64) {
	until := time.Now().Add(cacheTTL)
	// Never trust a token past its own expiry, however long the cache allows.
	if expires > 0 {
		if tokenExpiry := time.Unix(expires, 0); tokenExpiry.Before(until) {
			until = tokenExpiry
		}
	}
	v.mu.Lock()
	defer v.mu.Unlock()
	if len(v.cache) > 1024 {
		v.cache = map[string]cacheEntry{}
	}
	v.cache[token] = cacheEntry{subject: subject, expiresAt: until}
}

// BearerToken pulls the credential out of an Authorization header.
func BearerToken(header string) string {
	fields := strings.Fields(strings.TrimSpace(header))
	if len(fields) != 2 || !strings.EqualFold(fields[0], "Bearer") {
		return ""
	}
	return fields[1]
}
