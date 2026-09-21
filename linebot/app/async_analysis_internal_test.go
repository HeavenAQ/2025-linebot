package app

import "testing"

func TestAnalysisResultMessageExplainsAFailureAndDropsTheButtons(t *testing.T) {
	text, showButtons := analysisResultMessage("揮拍結束後請保持姿勢…")

	if text != "揮拍結束後請保持姿勢…" {
		t.Fatalf("a failure must reach the learner verbatim, got %q", text)
	}
	if showButtons {
		t.Fatal("a failed analysis has nothing to tap through to")
	}
}

func TestAnalysisResultMessageHandsBackThePortfolioOnSuccess(t *testing.T) {
	text, showButtons := analysisResultMessage("")

	if text == "" || !showButtons {
		t.Fatalf("a graded upload shows its portfolio card, got %q buttons=%v", text, showButtons)
	}
}
