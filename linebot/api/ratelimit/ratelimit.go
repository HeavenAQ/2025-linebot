// Package ratelimit keeps one client from exhausting the learner API.
//
// Limits are per Cloud Run instance, not global: an attacker spread over many
// instances gets more budget in total. That still bounds what one client can
// cost per instance, and the service's max-instances setting bounds the rest.
// A global limit belongs at the edge (Cloud Armor), not in application memory.
package ratelimit

import (
	"net/http"
	"strings"
	"sync"
	"time"

	"golang.org/x/time/rate"
)

const maxKeys = 10000

// Limiter hands out a token bucket per key (a client IP or a learner ID).
type Limiter struct {
	limit rate.Limit
	burst int
	now   func() time.Time

	mu      sync.Mutex
	buckets map[string]*bucket
}

type bucket struct {
	limiter  *rate.Limiter
	lastSeen time.Time
}

// New allows each key `perMinute` requests per minute on average, with bursts
// of up to `burst`.
func New(perMinute float64, burst int) *Limiter {
	return &Limiter{
		limit:   rate.Limit(perMinute / 60),
		burst:   burst,
		now:     time.Now,
		buckets: map[string]*bucket{},
	}
}

func (l *Limiter) get(key string) *rate.Limiter {
	l.mu.Lock()
	defer l.mu.Unlock()
	now := l.now()
	if b, ok := l.buckets[key]; ok {
		b.lastSeen = now
		return b.limiter
	}
	if len(l.buckets) >= maxKeys {
		// Drop buckets idle long enough to have refilled; they would start
		// full again anyway.
		for k, b := range l.buckets {
			if now.Sub(b.lastSeen) > time.Duration(float64(l.burst)/float64(l.limit)*float64(time.Second))+time.Minute {
				delete(l.buckets, k)
			}
		}
		if len(l.buckets) >= maxKeys {
			l.buckets = map[string]*bucket{}
		}
	}
	b := &bucket{limiter: rate.NewLimiter(l.limit, l.burst), lastSeen: now}
	l.buckets[key] = b
	return b.limiter
}

// Allow spends one token for key and reports whether one was available.
func (l *Limiter) Allow(key string) bool {
	return l.get(key).AllowN(l.now(), 1)
}

// Exhausted reports whether key has no token left, without spending one.
func (l *Limiter) Exhausted(key string) bool {
	return l.get(key).TokensAt(l.now()) < 1
}

// RetryAfter is a whole-second hint for the Retry-After header.
func (l *Limiter) RetryAfter() string {
	seconds := 1
	if l.limit > 0 {
		if s := int(1/float64(l.limit)) + 1; s > seconds {
			seconds = s
		}
	}
	return itoa(seconds)
}

func itoa(n int) string {
	if n <= 0 {
		return "1"
	}
	digits := []byte{}
	for n > 0 {
		digits = append([]byte{byte('0' + n%10)}, digits...)
		n /= 10
	}
	return string(digits)
}

// ClientIP returns the address Google's front end saw. Cloud Run appends it to
// X-Forwarded-For after anything the client sent, so the last entry is the one
// a client cannot forge.
func ClientIP(r *http.Request) string {
	if forwarded := r.Header.Get("X-Forwarded-For"); forwarded != "" {
		parts := strings.Split(forwarded, ",")
		for i := len(parts) - 1; i >= 0; i-- {
			if ip := strings.TrimSpace(parts[i]); ip != "" {
				return ip
			}
		}
	}
	host := r.RemoteAddr
	if i := strings.LastIndex(host, ":"); i > 0 && !strings.HasSuffix(host, "]") {
		host = host[:i]
	}
	return strings.Trim(host, "[]")
}
