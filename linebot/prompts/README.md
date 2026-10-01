# GPT prompts

Every prompt the bot sends to GPT, one text file each, embedded into the
binary at build time. `prompts.Text(name)` returns `<name>.txt` without its
trailing newline.

| File | Used by |
|---|---|
| `coach.txt` | The coaching agent's system instructions (chat) |
| `rewrite_query.txt` | Rewriting a follow-up question into a standalone one |
| `summary.txt` | Learner progress summaries |
| `weekly_preview.txt` | The weekly pre-class preview |
| `upload_result_header.txt`, `upload_result_note.txt`, `upload_default_question.txt` | The message sent to the coach after a video is analysed in chat |
