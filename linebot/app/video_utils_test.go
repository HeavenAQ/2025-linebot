package app

import "testing"

func TestThumbnailObjectPathUsesAnalysisPrefix(t *testing.T) {
	got := thumbnailObjectPath("U123", "2026-08-30-12-00")
	want := "analyses/thumbnail/U123/2026-08-30-12-00.jpeg"
	if got != want {
		t.Fatalf("thumbnailObjectPath() = %q, want %q", got, want)
	}
}
