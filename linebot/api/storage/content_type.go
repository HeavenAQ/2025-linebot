package storage

import "strings"

func playbackContentType(path string) string {
	switch strings.ToLower(path[strings.LastIndex(path, ".")+1:]) {
	case "jpg", "jpeg":
		return "image/jpeg"
	case "png":
		return "image/png"
	case "webp":
		return "image/webp"
	case "mov":
		return "video/quicktime"
	default:
		return "video/mp4"
	}
}
