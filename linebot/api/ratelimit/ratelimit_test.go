package ratelimit

import (
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestBurstThenRefill(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	l := New(60, 3) // one per second, burst three
	l.now = func() time.Time { return now }

	for i := 0; i < 3; i++ {
		require.True(t, l.Allow("ip-a"))
	}
	require.False(t, l.Allow("ip-a"))
	require.True(t, l.Allow("ip-b"), "keys are independent")

	now = now.Add(time.Second)
	require.True(t, l.Allow("ip-a"))
	require.False(t, l.Allow("ip-a"))
}

func TestExhaustedDoesNotSpend(t *testing.T) {
	l := New(60, 1)
	require.False(t, l.Exhausted("k"))
	require.False(t, l.Exhausted("k"))
	require.True(t, l.Allow("k"))
	require.True(t, l.Exhausted("k"))
}

func TestKeyTableIsBounded(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	l := New(60, 1)
	l.now = func() time.Time { return now }
	for i := 0; i < maxKeys+10; i++ {
		l.Allow(itoa(i + 1))
	}
	require.LessOrEqual(t, len(l.buckets), maxKeys)
}

func TestRetryAfter(t *testing.T) {
	require.Equal(t, "2", New(60, 1).RetryAfter())
	require.Equal(t, "11", New(6, 1).RetryAfter())
}

func TestClientIPUsesTheFrontEndAddress(t *testing.T) {
	r := httptest.NewRequest("GET", "/", nil)
	r.RemoteAddr = "169.254.1.1:5555"
	require.Equal(t, "169.254.1.1", ClientIP(r))

	r.Header.Set("X-Forwarded-For", "6.6.6.6, 203.0.113.7")
	require.Equal(t, "203.0.113.7", ClientIP(r), "a client-supplied first entry is ignored")
}
