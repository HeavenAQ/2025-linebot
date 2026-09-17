package auth

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

type lineStub struct {
	key          *ecdsa.PrivateKey
	keyID        string
	keyFetches   atomic.Int32
	verifyCalls  atomic.Int32
	profileCalls atomic.Int32
	accessClient string
}

func newLineStub(t *testing.T) (*lineStub, *Verifier) {
	t.Helper()
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	require.NoError(t, err)
	stub := &lineStub{key: key, keyID: "kid-1", accessClient: "channel-1"}
	mux := http.NewServeMux()
	mux.HandleFunc("/certs", func(w http.ResponseWriter, _ *http.Request) {
		stub.keyFetches.Add(1)
		json.NewEncoder(w).Encode(map[string]any{"keys": []map[string]string{{
			"kty": "EC", "crv": "P-256", "alg": "ES256", "kid": stub.keyID,
			"x": base64.RawURLEncoding.EncodeToString(key.X.FillBytes(make([]byte, 32))),
			"y": base64.RawURLEncoding.EncodeToString(key.Y.FillBytes(make([]byte, 32))),
		}}})
	})
	mux.HandleFunc("/verify", func(w http.ResponseWriter, r *http.Request) {
		stub.verifyCalls.Add(1)
		switch r.URL.Query().Get("access_token") {
		case "good-access":
			w.Write([]byte(`{"scope":"profile openid","client_id":"` + stub.accessClient + `","expires_in":43200}`))
		case "":
			w.WriteHeader(http.StatusBadRequest)
		default:
			w.WriteHeader(http.StatusBadRequest)
		}
	})
	mux.HandleFunc("/profile", func(w http.ResponseWriter, r *http.Request) {
		stub.profileCalls.Add(1)
		require.Equal(t, "Bearer good-access", r.Header.Get("Authorization"))
		w.Write([]byte(`{"userId":"U-access"}`))
	})
	server := httptest.NewServer(mux)
	t.Cleanup(server.Close)
	v := NewVerifier("channel-1")
	v.keysEndpoint = server.URL + "/certs"
	v.endpoint = server.URL + "/verify"
	v.profileEndpoint = server.URL + "/profile"
	return stub, v
}

func (s *lineStub) sign(t *testing.T, claims map[string]any, keyID string) string {
	t.Helper()
	header, _ := json.Marshal(map[string]string{"alg": "ES256", "typ": "JWT", "kid": keyID})
	payload, _ := json.Marshal(claims)
	signingInput := base64.RawURLEncoding.EncodeToString(header) + "." + base64.RawURLEncoding.EncodeToString(payload)
	digest := sha256.Sum256([]byte(signingInput))
	r, sig, err := ecdsa.Sign(rand.Reader, s.key, digest[:])
	require.NoError(t, err)
	signature := append(r.FillBytes(make([]byte, 32)), sig.FillBytes(make([]byte, 32))...)
	return signingInput + "." + base64.RawURLEncoding.EncodeToString(signature)
}

func validClaims() map[string]any {
	now := time.Now()
	return map[string]any{
		"iss": Issuer, "sub": "U-local", "aud": "channel-1",
		"exp": now.Add(time.Hour).Unix(), "iat": now.Unix(),
	}
}

func TestES256IDTokenIsVerifiedWithoutCallingLine(t *testing.T) {
	stub, v := newLineStub(t)
	token := stub.sign(t, validClaims(), stub.keyID)

	for i := 0; i < 3; i++ {
		id, err := v.UserID(context.Background(), token)
		require.NoError(t, err)
		require.Equal(t, "U-local", id)
	}
	require.EqualValues(t, 1, stub.keyFetches.Load(), "keys are fetched once and reused")
	require.Zero(t, stub.verifyCalls.Load(), "a valid ES256 token never reaches LINE's verify API")
}

func TestES256TokenChecks(t *testing.T) {
	stub, v := newLineStub(t)
	cases := map[string]func(map[string]any){
		"another channel": func(c map[string]any) { c["aud"] = "other" },
		"issuer":          func(c map[string]any) { c["iss"] = "https://evil.example" },
		"expired":         func(c map[string]any) { c["exp"] = time.Now().Add(-time.Hour).Unix() },
		"subject":         func(c map[string]any) { c["sub"] = "" },
		"future":          func(c map[string]any) { c["iat"] = time.Now().Add(time.Hour).Unix() },
	}
	for name, mutate := range cases {
		claims := validClaims()
		mutate(claims)
		_, err := v.UserID(context.Background(), stub.sign(t, claims, stub.keyID))
		require.ErrorIs(t, err, ErrUnauthorized, name)
	}
}

func TestTamperedOrForeignSignatureIsRejected(t *testing.T) {
	stub, v := newLineStub(t)
	token := stub.sign(t, validClaims(), stub.keyID)
	// Swap in a payload naming someone else, keeping the original signature.
	claims := validClaims()
	claims["sub"] = "U-victim"
	forged := stub.sign(t, claims, stub.keyID)
	parts := splitToken(token)
	forgedParts := splitToken(forged)
	_, err := v.UserID(context.Background(), parts[0]+"."+forgedParts[1]+"."+parts[2])
	require.ErrorIs(t, err, ErrUnauthorized)

	otherKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	require.NoError(t, err)
	impostor := &lineStub{key: otherKey}
	_, err = v.UserID(context.Background(), impostor.sign(t, validClaims(), stub.keyID))
	require.ErrorIs(t, err, ErrUnauthorized)
}

func TestUnknownKeyRefetchesAtMostOncePerInterval(t *testing.T) {
	stub, v := newLineStub(t)
	_, err := v.UserID(context.Background(), stub.sign(t, validClaims(), stub.keyID))
	require.NoError(t, err)

	for i := 0; i < 5; i++ {
		_, err = v.UserID(context.Background(), stub.sign(t, validClaims(), "rotated-away"))
		require.ErrorIs(t, err, ErrUnauthorized)
	}
	require.EqualValues(t, 1, stub.keyFetches.Load(), "garbage key IDs must not hammer LINE's key endpoint")

	// After the refetch interval a rotated key is picked up.
	now := time.Now().Add(2 * keysRefetchInterval)
	v.now = func() time.Time { return now }
	stub.keyID = "kid-2"
	claims := validClaims()
	_, err = v.UserID(context.Background(), stub.sign(t, claims, "kid-2"))
	require.NoError(t, err)
	require.EqualValues(t, 2, stub.keyFetches.Load())
}

func TestAccessTokenIdentifiesTheProfileUserAndIsCached(t *testing.T) {
	stub, v := newLineStub(t)
	for i := 0; i < 3; i++ {
		id, err := v.UserIDFromAccessToken(context.Background(), "good-access")
		require.NoError(t, err)
		require.Equal(t, "U-access", id)
	}
	require.EqualValues(t, 1, stub.verifyCalls.Load())
	require.EqualValues(t, 1, stub.profileCalls.Load())
}

func TestAccessTokenForAnotherChannelIsRejected(t *testing.T) {
	stub, v := newLineStub(t)
	stub.accessClient = "someone-elses-channel"
	_, err := v.UserIDFromAccessToken(context.Background(), "good-access")
	require.ErrorIs(t, err, ErrUnauthorized)
	require.Zero(t, stub.profileCalls.Load())
}

// A replayed bad credential is answered from the rejection cache instead of
// turning every attempt into a call to LINE.
func TestRejectedCredentialsAreNotReverifiedImmediately(t *testing.T) {
	stub, v := newLineStub(t)
	for i := 0; i < 5; i++ {
		_, err := v.UserIDFromAccessToken(context.Background(), "garbage")
		require.ErrorIs(t, err, ErrUnauthorized)
	}
	require.EqualValues(t, 1, stub.verifyCalls.Load())
}

func TestAuthenticatePrefersTheIDToken(t *testing.T) {
	stub, v := newLineStub(t)
	request := httptest.NewRequest(http.MethodGet, "/api/db/user", nil)
	request.Header.Set("Authorization", "Bearer "+stub.sign(t, validClaims(), stub.keyID))
	request.Header.Set(AccessTokenHeader, "good-access")
	id, err := v.Authenticate(context.Background(), request)
	require.NoError(t, err)
	require.Equal(t, "U-local", id)

	request.Header.Del("Authorization")
	id, err = v.Authenticate(context.Background(), request)
	require.NoError(t, err)
	require.Equal(t, "U-access", id)

	request.Header.Del(AccessTokenHeader)
	_, err = v.Authenticate(context.Background(), request)
	require.ErrorIs(t, err, ErrUnauthorized)
}

func splitToken(token string) []string {
	parts := make([]string, 0, 3)
	start := 0
	for i := range token {
		if token[i] == '.' {
			parts = append(parts, token[start:i])
			start = i + 1
		}
	}
	return append(parts, token[start:])
}
