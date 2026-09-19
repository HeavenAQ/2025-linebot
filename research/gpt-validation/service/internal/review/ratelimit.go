package review

import (
	"sync"
	"time"
)

// FailureLimiter counts failed authentication attempts per key (client IP) in
// a sliding window. It is in-memory: with at most two instances an attacker
// gets at most twice the budget, which is acceptable for random access codes.
type FailureLimiter struct {
	mu        sync.Mutex
	max       int
	window    time.Duration
	now       func() time.Time
	failures  map[string][]time.Time
	lastSweep time.Time
}

// NewFailureLimiter allows max failures per key within window.
func NewFailureLimiter(max int, window time.Duration) *FailureLimiter {
	return &FailureLimiter{max: max, window: window, now: time.Now, failures: map[string][]time.Time{}}
}

// Blocked reports whether key is over budget and, if so, how long until the
// oldest counted failure leaves the window.
func (l *FailureLimiter) Blocked(key string) (bool, time.Duration) {
	l.mu.Lock()
	defer l.mu.Unlock()
	now := l.now()
	recent := l.prune(key, now)
	if len(recent) < l.max {
		return false, 0
	}
	retry := recent[0].Add(l.window).Sub(now)
	if retry < time.Second {
		retry = time.Second
	}
	return true, retry
}

// RecordFailure counts one failed attempt for key.
func (l *FailureLimiter) RecordFailure(key string) {
	l.mu.Lock()
	defer l.mu.Unlock()
	now := l.now()
	recent := l.prune(key, now)
	if len(recent) >= l.max {
		recent = recent[len(recent)-l.max+1:]
	}
	l.failures[key] = append(recent, now)
	if now.Sub(l.lastSweep) > l.window {
		l.sweep(now)
	}
}

func (l *FailureLimiter) prune(key string, now time.Time) []time.Time {
	times := l.failures[key]
	cut := 0
	for cut < len(times) && now.Sub(times[cut]) >= l.window {
		cut++
	}
	times = times[cut:]
	if len(times) == 0 {
		delete(l.failures, key)
		return nil
	}
	l.failures[key] = times
	return times
}

func (l *FailureLimiter) sweep(now time.Time) {
	l.lastSweep = now
	for key := range l.failures {
		l.prune(key, now)
	}
}
