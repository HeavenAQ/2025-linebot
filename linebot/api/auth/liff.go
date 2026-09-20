// Package auth establishes who is calling the learner-facing API. A
// client-supplied user ID proves nothing, so the learner's identity comes from
// the subject of a LINE ID token: verified locally against LINE's published
// ES256 keys, so a forged one never costs a request to LINE. Those expire after
// an hour, after which a page falls back to the access token, verified remotely.
package auth

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/big"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"
)

// VerifyEndpoint is LINE's ID token verification API, used for tokens that are
// not ES256-signed.
const VerifyEndpoint = "https://api.line.me/oauth2/v2.1/verify"

// KeysEndpoint publishes the ES256 keys LINE signs LIFF ID tokens with.
const KeysEndpoint = "https://api.line.me/oauth2/v2.1/certs"

// ProfileEndpoint returns the user an access token belongs to.
const ProfileEndpoint = "https://api.line.me/v2/profile"

// Issuer is the iss claim LINE puts in its ID tokens.
const Issuer = "https://access.line.me"

// verifyTimeout bounds a single call to LINE.
const verifyTimeout = 5 * time.Second

// cacheTTL caps how long a remotely verified credential is trusted without
// re-checking. Every LIFF page load makes several requests, so re-verifying
// each one would add a round trip to LINE per request.
const cacheTTL = 5 * time.Minute

// rejectionTTL remembers credentials LINE refused, so replaying one does not
// make this service call LINE again for every attempt.
const rejectionTTL = time.Minute

// clockSkew tolerates small clock differences when checking exp and iat.
const clockSkew = 30 * time.Second

// keysTTL is how long LINE's signing keys are reused before refetching; an
// unknown key ID refetches sooner, at most once per keysRefetchInterval.
const (
	keysTTL             = time.Hour
	keysRefetchInterval = time.Minute
)

const maxCacheEntries = 4096

// ErrUnauthorized marks every credential failure; callers map it to 401.
var ErrUnauthorized = errors.New("unauthorized")

// Claims is the part of an ID token this service relies on.
type Claims struct {
	Subject  string `json:"sub"`
	Audience string `json:"aud"`
	Issuer   string `json:"iss"`
	Expires  int64  `json:"exp"`
	IssuedAt int64  `json:"iat"`
	Name     string `json:"name"`
}

type cacheEntry struct {
	subject   string
	expiresAt time.Time
	rejected  bool
}

// Verifier checks LIFF credentials for one LINE Login channel.
type Verifier struct {
	channelID       string
	client          *http.Client
	endpoint        string
	keysEndpoint    string
	profileEndpoint string
	now             func() time.Time

	mu    sync.Mutex
	cache map[string]cacheEntry

	keysMu        sync.Mutex
	keys          map[string]*ecdsa.PublicKey
	keysFetchedAt time.Time
}

// NewVerifier returns a verifier for one LINE Login channel. An empty channel
// ID yields nil: the caller decides whether to run without authentication,
// rather than this silently accepting every token.
func NewVerifier(channelID string) *Verifier {
	if strings.TrimSpace(channelID) == "" {
		return nil
	}
	return &Verifier{
		channelID:       strings.TrimSpace(channelID),
		client:          &http.Client{Timeout: verifyTimeout},
		endpoint:        VerifyEndpoint,
		keysEndpoint:    KeysEndpoint,
		profileEndpoint: ProfileEndpoint,
		now:             time.Now,
		cache:           map[string]cacheEntry{},
	}
}

// Authenticate identifies the caller of one HTTP request: the ID token in
// "Authorization: Bearer", or else the LIFF access token in X-Line-Access-Token.
func (v *Verifier) Authenticate(ctx context.Context, r *http.Request) (string, error) {
	if token := BearerToken(r.Header.Get("Authorization")); token != "" {
		return v.UserID(ctx, token)
	}
	if token := strings.TrimSpace(r.Header.Get(AccessTokenHeader)); token != "" {
		return v.UserIDFromAccessToken(ctx, token)
	}
	return "", fmt.Errorf("%w: no credential supplied", ErrUnauthorized)
}

// AccessTokenHeader carries the LIFF access token once the ID token expired.
const AccessTokenHeader = "X-Line-Access-Token"

// UserID returns the LINE user ID an ID token was issued for.
//
// The audience check is what stops a token minted for some other LINE channel
// from being replayed here, so a token that verifies but names a different
// channel is rejected.
func (v *Verifier) UserID(ctx context.Context, idToken string) (string, error) {
	token := strings.TrimSpace(idToken)
	if token == "" {
		return "", fmt.Errorf("%w: no id token supplied", ErrUnauthorized)
	}
	if header, ok := jwtHeader(token); ok && header.Algorithm == "ES256" {
		return v.verifyLocally(ctx, token, header.KeyID)
	}
	key := cacheKey("id", token)
	if subject, err, ok := v.cached(key); ok {
		return subject, err
	}
	subject, expires, err := v.verifyRemotely(ctx, token)
	v.remember(key, subject, expires, err)
	return subject, err
}

// UserIDFromAccessToken returns the LINE user a LIFF access token belongs to,
// after LINE confirms the token was issued to this channel.
func (v *Verifier) UserIDFromAccessToken(ctx context.Context, accessToken string) (string, error) {
	token := strings.TrimSpace(accessToken)
	if token == "" {
		return "", fmt.Errorf("%w: no access token supplied", ErrUnauthorized)
	}
	key := cacheKey("access", token)
	if subject, err, ok := v.cached(key); ok {
		return subject, err
	}
	subject, expires, err := v.verifyAccessToken(ctx, token)
	v.remember(key, subject, expires, err)
	return subject, err
}

type header struct {
	Algorithm string `json:"alg"`
	KeyID     string `json:"kid"`
}

func jwtHeader(token string) (header, bool) {
	parts := strings.Split(token, ".")
	if len(parts) != 3 {
		return header{}, false
	}
	raw, err := base64.RawURLEncoding.DecodeString(parts[0])
	if err != nil {
		return header{}, false
	}
	var h header
	if json.Unmarshal(raw, &h) != nil {
		return header{}, false
	}
	return h, true
}

func (v *Verifier) verifyLocally(ctx context.Context, token, keyID string) (string, error) {
	parts := strings.Split(token, ".")
	signature, err := base64.RawURLEncoding.DecodeString(parts[2])
	if err != nil || len(signature) != 64 {
		return "", fmt.Errorf("%w: malformed signature", ErrUnauthorized)
	}
	key, err := v.key(ctx, keyID)
	if err != nil {
		return "", err
	}
	digest := sha256.Sum256([]byte(parts[0] + "." + parts[1]))
	r := new(big.Int).SetBytes(signature[:32])
	s := new(big.Int).SetBytes(signature[32:])
	if !ecdsa.Verify(key, digest[:], r, s) {
		return "", fmt.Errorf("%w: bad signature", ErrUnauthorized)
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return "", fmt.Errorf("%w: malformed payload", ErrUnauthorized)
	}
	var claims Claims
	if err := json.Unmarshal(payload, &claims); err != nil {
		return "", fmt.Errorf("%w: malformed claims", ErrUnauthorized)
	}
	if err := v.checkClaims(claims, true); err != nil {
		return "", err
	}
	return claims.Subject, nil
}

func (v *Verifier) checkClaims(claims Claims, requireIssuer bool) error {
	now := v.now()
	switch {
	case claims.Audience != v.channelID:
		return fmt.Errorf("%w: id token was issued for another channel", ErrUnauthorized)
	case requireIssuer && claims.Issuer != Issuer:
		return fmt.Errorf("%w: unexpected issuer", ErrUnauthorized)
	case claims.Subject == "":
		return fmt.Errorf("%w: id token has no subject", ErrUnauthorized)
	case claims.Expires <= 0 && requireIssuer:
		return fmt.Errorf("%w: id token has no expiry", ErrUnauthorized)
	case claims.Expires > 0 && now.After(time.Unix(claims.Expires, 0).Add(clockSkew)):
		return fmt.Errorf("%w: id token has expired", ErrUnauthorized)
	case claims.IssuedAt > 0 && time.Unix(claims.IssuedAt, 0).After(now.Add(clockSkew)):
		return fmt.Errorf("%w: id token issued in the future", ErrUnauthorized)
	}
	return nil
}

// key returns LINE's signing key by ID, refreshing the key set when it is
// stale or the ID is unknown (keys rotate).
func (v *Verifier) key(ctx context.Context, keyID string) (*ecdsa.PublicKey, error) {
	v.keysMu.Lock()
	defer v.keysMu.Unlock()
	age := v.now().Sub(v.keysFetchedAt)
	if key, ok := v.keys[keyID]; ok && age < keysTTL {
		return key, nil
	}
	if v.keys == nil || age >= keysRefetchInterval {
		keys, err := v.fetchKeys(ctx)
		if err != nil {
			if key, ok := v.keys[keyID]; ok {
				return key, nil // LINE unreachable: keep using a known key.
			}
			return nil, fmt.Errorf("fetch LINE signing keys: %w", err)
		}
		v.keys, v.keysFetchedAt = keys, v.now()
	}
	if key, ok := v.keys[keyID]; ok {
		return key, nil
	}
	return nil, fmt.Errorf("%w: unknown signing key", ErrUnauthorized)
}

func (v *Verifier) fetchKeys(ctx context.Context) (map[string]*ecdsa.PublicKey, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, v.keysEndpoint, nil)
	if err != nil {
		return nil, err
	}
	response, err := v.client.Do(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("status %d", response.StatusCode)
	}
	var set struct {
		Keys []struct {
			KeyType string `json:"kty"`
			Curve   string `json:"crv"`
			KeyID   string `json:"kid"`
			X       string `json:"x"`
			Y       string `json:"y"`
		} `json:"keys"`
	}
	if err := json.NewDecoder(http.MaxBytesReader(nil, response.Body, 1<<20)).Decode(&set); err != nil {
		return nil, err
	}
	keys := map[string]*ecdsa.PublicKey{}
	for _, k := range set.Keys {
		if k.KeyType != "EC" || k.Curve != "P-256" {
			continue
		}
		x, errX := base64.RawURLEncoding.DecodeString(k.X)
		y, errY := base64.RawURLEncoding.DecodeString(k.Y)
		if errX != nil || errY != nil {
			continue
		}
		key := &ecdsa.PublicKey{Curve: elliptic.P256(), X: new(big.Int).SetBytes(x), Y: new(big.Int).SetBytes(y)}
		if !key.Curve.IsOnCurve(key.X, key.Y) {
			continue
		}
		keys[k.KeyID] = key
	}
	if len(keys) == 0 {
		return nil, errors.New("no usable P-256 keys")
	}
	return keys, nil
}

func (v *Verifier) verifyRemotely(ctx context.Context, token string) (string, time.Time, error) {
	form := url.Values{"id_token": {token}, "client_id": {v.channelID}}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, v.endpoint, strings.NewReader(form.Encode()))
	if err != nil {
		return "", time.Time{}, fmt.Errorf("build verification request: %w", err)
	}
	request.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	response, err := v.client.Do(request)
	if err != nil {
		return "", time.Time{}, fmt.Errorf("verify id token: %w", err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		var failure struct {
			Description string `json:"error_description"`
		}
		// Never log the upstream body, token, claims, or learner identity.
		_ = json.NewDecoder(io.LimitReader(response.Body, 4096)).Decode(&failure)
		reason := "line_rejected"
		description := strings.ToLower(failure.Description)
		switch {
		case strings.Contains(description, "expir"):
			reason = "token_expired"
		case strings.Contains(description, "audience") || strings.Contains(description, "client_id"):
			reason = "channel_mismatch"
		case strings.Contains(description, "signature"):
			reason = "invalid_signature"
		}
		return "", time.Time{}, fmt.Errorf("%w: id token rejected by LINE (status %d, reason=%s)", ErrUnauthorized, response.StatusCode, reason)
	}
	var claims Claims
	if err := json.NewDecoder(response.Body).Decode(&claims); err != nil {
		return "", time.Time{}, fmt.Errorf("decode verification response: %w", err)
	}
	if err := v.checkClaims(claims, false); err != nil {
		return "", time.Time{}, err
	}
	var expires time.Time
	if claims.Expires > 0 {
		expires = time.Unix(claims.Expires, 0)
	}
	return claims.Subject, expires, nil
}

func (v *Verifier) verifyAccessToken(ctx context.Context, token string) (string, time.Time, error) {
	verifyURL := v.endpoint + "?" + url.Values{"access_token": {token}}.Encode()
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, verifyURL, nil)
	if err != nil {
		return "", time.Time{}, err
	}
	response, err := v.client.Do(request)
	if err != nil {
		return "", time.Time{}, fmt.Errorf("verify access token: %w", err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return "", time.Time{}, fmt.Errorf("%w: access token rejected by LINE (status %d)", ErrUnauthorized, response.StatusCode)
	}
	var verified struct {
		ClientID  string `json:"client_id"`
		ExpiresIn int64  `json:"expires_in"`
	}
	if err := json.NewDecoder(response.Body).Decode(&verified); err != nil {
		return "", time.Time{}, fmt.Errorf("decode access token verification: %w", err)
	}
	if verified.ClientID != v.channelID {
		return "", time.Time{}, fmt.Errorf("%w: access token was issued for another channel", ErrUnauthorized)
	}
	if verified.ExpiresIn <= 0 {
		return "", time.Time{}, fmt.Errorf("%w: access token has expired", ErrUnauthorized)
	}

	profileRequest, err := http.NewRequestWithContext(ctx, http.MethodGet, v.profileEndpoint, nil)
	if err != nil {
		return "", time.Time{}, err
	}
	profileRequest.Header.Set("Authorization", "Bearer "+token)
	profileResponse, err := v.client.Do(profileRequest)
	if err != nil {
		return "", time.Time{}, fmt.Errorf("read LINE profile: %w", err)
	}
	defer profileResponse.Body.Close()
	if profileResponse.StatusCode != http.StatusOK {
		return "", time.Time{}, fmt.Errorf("%w: profile rejected the access token (status %d)", ErrUnauthorized, profileResponse.StatusCode)
	}
	var profile struct {
		UserID string `json:"userId"`
	}
	if err := json.NewDecoder(profileResponse.Body).Decode(&profile); err != nil || profile.UserID == "" {
		return "", time.Time{}, fmt.Errorf("%w: profile has no user", ErrUnauthorized)
	}
	return profile.UserID, v.now().Add(time.Duration(verified.ExpiresIn) * time.Second), nil
}

// Credentials are cached by hash, so raw tokens never sit in memory longer
// than the request that carried them.
func cacheKey(kind, token string) string {
	sum := sha256.Sum256([]byte(token))
	return kind + ":" + hex.EncodeToString(sum[:])
}

func (v *Verifier) cached(key string) (string, error, bool) {
	v.mu.Lock()
	defer v.mu.Unlock()
	entry, ok := v.cache[key]
	if !ok || v.now().After(entry.expiresAt) {
		delete(v.cache, key)
		return "", nil, false
	}
	if entry.rejected {
		return "", fmt.Errorf("%w: credential was recently rejected", ErrUnauthorized), true
	}
	return entry.subject, nil, true
}

// remember caches a verified credential (never past its own expiry) or a
// rejection. Failures to reach LINE are not cached: they say nothing about the
// credential.
func (v *Verifier) remember(key, subject string, expires time.Time, err error) {
	now := v.now()
	entry := cacheEntry{subject: subject, expiresAt: now.Add(cacheTTL)}
	if err != nil {
		if !errors.Is(err, ErrUnauthorized) {
			return
		}
		entry = cacheEntry{rejected: true, expiresAt: now.Add(rejectionTTL)}
	} else if !expires.IsZero() && expires.Before(entry.expiresAt) {
		entry.expiresAt = expires
	}
	v.mu.Lock()
	defer v.mu.Unlock()
	if len(v.cache) >= maxCacheEntries {
		for k, e := range v.cache {
			if now.After(e.expiresAt) {
				delete(v.cache, k)
			}
		}
		if len(v.cache) >= maxCacheEntries {
			v.cache = map[string]cacheEntry{}
		}
	}
	v.cache[key] = entry
}

// BearerToken pulls the credential out of an Authorization header.
func BearerToken(header string) string {
	fields := strings.Fields(strings.TrimSpace(header))
	if len(fields) != 2 || !strings.EqualFold(fields[0], "Bearer") {
		return ""
	}
	return fields[1]
}
