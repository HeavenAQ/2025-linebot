# Required experiment registration (LLM and no-LLM)

Before any learner feature can be used, the student registers on the **learning
dashboard** (LIFF): the first time they open it, the page asks for their
experiment number, real name and handedness. The bot itself only hands out a
link to that page; no identifying detail is ever typed into the chat.

They can change the same three fields later under 個人資料 in the dashboard menu.

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
- Repeating the same completed registration is safe. A student may correct
  their own number, name or handedness from the profile page; the time they
  first registered is kept.

## Firestore

The user document remains at the existing configured user-data collection,
keyed by the verified LINE user ID. The no-LLM branch retains its own configured
Firestore database/collection; it does not read registration from the AI one.

New fields:

| Field | Stored type |
| --- | --- |
| `real_name` | string |
| `experiment_number` | string |
| `registration_version` | integer, currently 1 |
| `registration_completed_at` | server timestamp |

`name` stays the LINE display name. A field-scoped transaction saves the
registration under the ID token's own subject, creating the record and storage
folders first if the student has never messaged the bot. Generic portfolio/profile writes exclude these four registration fields, so a
stale in-flight write cannot erase enrollment. New-user creation uses Firestore
Create, not an overwrite of a potentially existing document.

## Enforcement

- The LINE event gate runs before sessions, menu/postback handling, video
  downloads, analysis, coaching, and notes.
- An unregistered learner receives a card linking to the registration form,
  whatever they send. Group/room requests are directed to one-to-one chat.
- Every learner HTTP API checks registration **after** existing LINE ID-token
  authentication, using the token's subject, never an input user ID.
- Unregistered requests return HTTP 403 with `code: registration_required`.
  Database failures return 503 and do not create/re-register a user.
- LIFF hides all feature content behind the registration form, which saves
  through `PUT /api/db/profile`. That is the one learner route that needs a
  signed-in learner but not a registered one; every other route still requires
  registration. Even a 200 response from an older backend without registration
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
They verify persistence, untouched portfolios, server timestamp, later edits
keeping the original join time, and preservation across stale portfolio writes.

Deploy only the registration commits to LLM `main` and `variant/no-llm`; do not
include the separate unfinished smash scoring/coaching candidate. Both bot and
LIFF deployments are needed.
