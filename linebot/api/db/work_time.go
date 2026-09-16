package db

import "time"

// Async attempts append seconds and a unique suffix to prevent concurrent
// uploads from overwriting the same minute's portfolio entry.
func ParseWorkTime(key string) (time.Time, error) {
	if len(key) >= 19 {
		return time.Parse("2006-01-02-15-04-05", key[:19])
	}
	return time.Parse("2006-01-02-15-04", key)
}
