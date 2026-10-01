// Package prompts holds every GPT prompt the bot sends, one text file each.
package prompts

import (
	"embed"
	"strings"
)

//go:embed *.txt
var files embed.FS

// Text returns the prompt <name>.txt without its trailing newline.
func Text(name string) string {
	data, err := files.ReadFile(name + ".txt")
	if err != nil {
		panic("missing prompt " + name + ": " + err.Error())
	}
	return strings.TrimSuffix(string(data), "\n")
}
