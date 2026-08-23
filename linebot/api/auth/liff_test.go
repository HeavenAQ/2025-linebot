package auth

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func verifierAgainst(t *testing.T, handler http.HandlerFunc) *Verifier {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	v := NewVerifier("channel-1")
	v.endpoint = server.URL
	return v
}

func TestUserIDComesFromTheVerifiedSubject(t *testing.T) {
	t.Parallel()

	v := verifierAgainst(t, func(w http.ResponseWriter, r *http.Request) {
		require.NoError(t, r.ParseForm())
		require.Equal(t, "channel-1", r.PostForm.Get("client_id"))
		require.Equal(t, "token-abc", r.PostForm.Get("id_token"))
		w.Write([]byte(`{"sub":"U1234","aud":"channel-1","exp":99999999999}`))
	})

	id, err := v.UserID(context.Background(), "token-abc")
	require.NoError(t, err)
	require.Equal(t, "U1234", id)
}

// A token minted for a different LINE channel must not be replayable here,
// which is the whole point of checking the audience.
func TestTokenForAnotherChannelIsRejected(t *testing.T) {
	t.Parallel()

	v := verifierAgainst(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte(`{"sub":"U1234","aud":"someone-elses-channel","exp":99999999999}`))
	})

	_, err := v.UserID(context.Background(), "token-abc")
	require.ErrorContains(t, err, "another channel")
}

func TestExpiredTokenIsRejected(t *testing.T) {
	t.Parallel()

	v := verifierAgainst(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte(`{"sub":"U1234","aud":"channel-1","exp":1}`))
	})

	_, err := v.UserID(context.Background(), "token-abc")
	require.ErrorContains(t, err, "expired")
}

func TestTokenLineRejectsIsRejected(t *testing.T) {
	t.Parallel()

	v := verifierAgainst(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
	})

	_, err := v.UserID(context.Background(), "forged")
	require.Error(t, err)
}

func TestEmptyTokenNeverReachesLine(t *testing.T) {
	t.Parallel()

	called := false
	v := verifierAgainst(t, func(w http.ResponseWriter, _ *http.Request) { called = true })

	_, err := v.UserID(context.Background(), "   ")
	require.Error(t, err)
	require.False(t, called)
}

func TestVerifiedTokensAreCachedButNotPastTheirExpiry(t *testing.T) {
	t.Parallel()

	calls := 0
	v := verifierAgainst(t, func(w http.ResponseWriter, _ *http.Request) {
		calls++
		w.Write([]byte(`{"sub":"U1234","aud":"channel-1","exp":99999999999}`))
	})

	for i := 0; i < 3; i++ {
		id, err := v.UserID(context.Background(), "token-abc")
		require.NoError(t, err)
		require.Equal(t, "U1234", id)
	}
	require.Equal(t, 1, calls, "a verified token should not be re-checked on every request")

	// An entry past its expiry is discarded rather than served.
	v.mu.Lock()
	v.cache["token-abc"] = cacheEntry{subject: "U1234", expiresAt: time.Now().Add(-time.Second)}
	v.mu.Unlock()
	_, err := v.UserID(context.Background(), "token-abc")
	require.NoError(t, err)
	require.Equal(t, 2, calls)
}

// Without a channel ID there is nothing to check a token against, so the
// caller is handed nil and must decide explicitly what that means.
func TestNoChannelYieldsNoVerifier(t *testing.T) {
	t.Parallel()

	require.Nil(t, NewVerifier(""))
	require.Nil(t, NewVerifier("   "))
	require.NotNil(t, NewVerifier("channel-1"))
}

func TestBearerTokenParsing(t *testing.T) {
	t.Parallel()

	require.Equal(t, "abc", BearerToken("Bearer abc"))
	require.Equal(t, "abc", BearerToken("bearer abc"))
	require.Equal(t, "abc", BearerToken("  Bearer   abc  "))
	for _, header := range []string{"", "abc", "Basic abc", "Bearer", "Bearer a b"} {
		require.Empty(t, BearerToken(header), header)
	}
}
