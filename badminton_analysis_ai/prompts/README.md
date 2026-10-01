# GPT prompts

Every prompt the analysis service sends to GPT, one text file each. Edit the
text here; `badminton_analysis.prompts.prompt(name, **fields)` loads
`<name>.txt` and fills its `{placeholders}`. A single trailing newline is ignored.

| File | Sent as | Placeholders |
|---|---|---|
| `coach/system.txt` | Coach system instructions, shared part | `description`, `rule_count`, `name` |
| `<skill>/coach_system.txt` | Appended to the system instructions for that skill | — |
| `coach/task.txt` | The per-video request, before the analysis JSON | `criterion_count`, `name`, `slug`, `maximum_problem_count`, `minimum_problem_count`, `required_priority_criteria`, `feedback_candidate_criteria` |
| `coach/frame.txt` | Caption above each evidence frame | `frame_index`, `phase`, `source_frame_index`, `timestamp_seconds`, `checkpoint_role` |
| `coach/retry.txt` | Sent when an answer fails validation | `validation_error` |
| `coach/handedness.txt`, `coach/handedness_unknown.txt` | Joint-side note | `hand`, `side` |
| `coach/score_method.txt`, `<skill>/score_method.txt` | How the grade was computed (serve appends, smash replaces) | — |
| `coach/score_warning.txt`, `smash/score_warning.txt` | What the grade does and does not prove | — |
