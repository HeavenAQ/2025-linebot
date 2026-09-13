# Required experiment registration (AI and no-AI)

Before any learner feature can be used, the student sends one message to the
bot's **one-to-one LINE chat**:

```
01 王小明
```

The format is `experiment-number real-name`, separated by whitespace. No command,
colon, comma, or multi-step form is needed.

- Numbers accept 1–6 digits, optionally preceded by up to four English letters:
  `01`, `001`, `EG01`. Prefixes are stored uppercase; leading zeroes stay intact.
- Ordinary/full-width spaces, full-width digits/Latin letters, and repeated
  spaces are normalized.
- Names accept 2–60 characters with at least two letters: Chinese and other
  Unicode letters, combining marks, internal spaces, apostrophes, hyphens, and
  middle dots. Digits, emoji, markup, line breaks, and explanatory punctuation
  are rejected with a Traditional Chinese format-error reply and an example.
- This validates **syntax**, not legal identity or membership in a teacher's
  roster. Experiment numbers are not forced to be globally unique.
- Existing accounts must register too; their LINE display name is not proof of
  registration. Existing portfolios and notes are preserved.
- Repeating the same completed registration is safe. Self-service changes to
  an existing registered identity are not supported; students contact the teacher.

## Firestore

The user document remains at the existing configured user-data collection,
keyed by the verified LINE user ID. The no-AI branch retains its own configured
Firestore database/collection; it does not read registration from the AI one.

New fields:

| Field | Stored type |
| --- | --- |
| `real_name` | string |
| `experiment_number` | string |
| `registration_version` | integer, currently 1 |
| `registration_completed_at` | server timestamp |

`name` stays the LINE display name. A field-scoped transaction saves registration
only after a valid signed LINE event and successful user initialization.
Concurrent conflicting registrations cannot overwrite a completed identity.
Generic portfolio/profile writes exclude these four registration fields, so a
stale in-flight write cannot erase enrollment. New-user creation uses Firestore
Create, not an overwrite of a potentially existing document.

## Enforcement

- The LINE event gate runs before sessions, menu/postback handling, video
  downloads, analysis, coaching, and notes.
- Invalid text receives `格式不正確…`. A video, follow, or postback receives the
  registration instructions. Group/room requests are directed to one-to-one chat.
- Every learner HTTP API checks registration **after** existing LINE ID-token
  authentication, using the token's subject, never an input user ID.
- Unregistered requests return HTTP 403 with `code: registration_required`.
  Database failures return 503 and do not create/re-register a user.
- LIFF hides navigation and all feature content behind the registration screen.
  Registration is entered in LINE, not in LIFF. “已在 LINE 登記，重新確認” rechecks
  the backend. Even a 200 response from an older backend without registration
  fields does not unlock the frontend.
- AI weekly preview pushes skip unregistered users.
- Health checks, signed LINE callbacks, and authenticated administrative scheduler
  endpoints retain their existing roles. This does not revoke already-issued
  signed media URLs; those keep their existing expiry.

## Tests and release

Both variants: `cd linebot && go test ./...`; `cd liff && npm test`,
`npx tsc --noEmit`, and `npm run build`.

Registration storage tests use a local in-memory Firestore protocol server:
no real student records, LINE messages, or production credentials are used.
They verify persistence, untouched portfolios, server timestamp, safe retries,
conflict rejection, and preservation across stale portfolio writes.

Deploy only the registration commits to AI `main` and `variant/no-ai`; do not
include the separate unfinished smash scoring/coaching candidate. Both bot and
LIFF deployments are needed.
