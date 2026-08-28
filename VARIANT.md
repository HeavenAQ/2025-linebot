# The no-AI variant

`variant/no-ai` is the badminton coaching product with every large-language-model
layer removed, so it can run beside the original and be compared against it.

**Nothing has been provisioned.** No bucket, no Firestore database, no Cloud Run
service, no Netlify site and no GitHub secret was created. Everything below is a
command for you to run deliberately, once you have decided you want the resource
and the bill that comes with it.

**Never merge `variant/no-ai` into `main`.** It is a parallel product line.

## Topology

Same GCP project, same service account, **separate Cloud Run services**, its own
Netlify site, its own bucket and its own Firestore databases. The GPU analysis
service is **shared with production**.

Sharing the analysis service is safe because of `skip_coaching`. Coaching is the
only stage of an analysis that leaves the machine -- it uploads sampled JPEG
frames of the learner to a third-party model as `input_image`. The service
branches *before* that call, so with the flag set no image of a learner is sent
anywhere, and the response comes back without `coaching_cues` or
`overall_feedback`. Everything that decides the grade -- pose, the diffusion
correction, the rubric, checkpoints, expert matching -- is local and runs either
way.

The bot sets it deployment-wide from `ANALYSIS_SKIP_COACHING=true` and puts it on
every request. **That flag is the whole of what makes this deployment GPT-free on
the analysis side.** If it is ever unset, this deployment silently stops being
GPT-free: nothing else fails, and learner imagery starts going to a third party
again. The CD workflow reads it back off the deployed revision for that reason.

`badminton_analysis_ai/` and `proto/` on this branch are therefore byte-identical
to `main`, and must stay that way.

## What is gone, and what is not

The analysis was never GPT. Pose estimation, the EIMD diffusion correction,
grading, the checkpoint timeline and expert matching are local models, and all of
them are unchanged. What is removed is the natural-language layer that sat on
top.

Removed:

| Area | Gone |
|---|---|
| Go | `api/gpt/` in full; the `ChattingWithGPT` state and its rich-menu, postback and quick-reply entries; chat history (`api/db/chat_history.go`) and `daily_summaries` (`api/db/daily_summary.go`); the weekly 課前預習 push (`app/weekly_preview.go`, `api/db/weekly_preview.go`, `PushWeeklyPreview`); `/api/chat/history`, `/api/chat/summarize` and `/api/preview/weekly`; `OPENAI_*` and `WEEKLY_PREVIEW_TOKEN` config; `gpt_conversation_ids` and `ai_note` on the user record; the 詢問AI建議 portfolio button |
| Python | Nothing. `badminton_analysis_ai/` is identical to `main`; the coaching stage is switched off per deployment by `ANALYSIS_SKIP_COACHING` rather than deleted, so one GPU service serves both products |
| Web app | `SkillSummary.tsx`, `useSkillSummary.ts`, the `gpt-chat` page and its menu entry; the chat-history panel in `WeeklyReview.tsx`; the coaching-cue markers, cue list and pause legend in `VideoComparison.tsx` |

Kept, deliberately: uploads, grading and the score charts, the checkpoint
timeline, the expert comparison with checkpoint alignment and segmental warping,
weekly reflections, stats, the portfolio carousel, and both `student_video` and
`skeleton_overlay_video` fields on the response.

The rich menu differs too. `main` keeps one **預習及反思** entry whose card offers
both the review page and a GPT-written 課前預習 note on demand. This variant has
no such note, so the entry is split in two: **課前預習** opens the review tab's
預習 sub-tab and **學習反思** its 反思 sub-tab. Both products let a student write
their own 課前檢視要點 in 預習 — that note is theirs, not a model's.

The rich menu is not defined in code. The two tappable areas are created in the
LINE console and must send exactly these texts, which are what
`UserStateChnStrToEnum` matches:

| Menu area | Text it sends | Where the card's button goes |
|---|---|---|
| 課前預習 | `課前預習` | `${LIFF_REVIEW_URL}` + `section=preview` |
| 學習反思 | `學習反思` | `${LIFF_REVIEW_URL}` + `section=reflection` |

With `LIFF_REVIEW_URL=https://liff-nstc-2025-noai.netlify.app/personal?tab=review`
those resolve to
`https://liff-nstc-2025-noai.netlify.app/personal?section=preview&tab=review` and
`https://liff-nstc-2025-noai.netlify.app/personal?section=reflection&tab=review`.

`coaching_cues` and `overall_feedback` remain in the protobuf contract and come
back empty for this deployment because the service skips the stage that fills
them. The Go client and the web app no longer read either field.

## Separate data plane

The variant must never read or write the original's data. Every value that
selects a data store is configuration-driven with no default:

- `GCS_BUCKET_NAME`, `FIREBASE_DATA_DB`, `FIREBASE_SESSION_DB` and
  `GCP_PROJECT_ID` are read from the environment and have no fallback.
- `GCP_ENV_SECRET_NAME` selects which Secret Manager secret the bot downloads its
  `.env` from at boot. This used to be the hardcoded `2025-linebot-env`; it is now
  required, because a fallback there would have quietly handed the variant the
  live bucket and databases.
- `LIFF_REVIEW_URL` has no default either. It used to fall back to the live web
  app, which would have sent the variant's learners into the original's UI.

`.env.example` at the repo root lists every variable with a one-line note on what
it is and where it comes from.

## Creating the resources

Set these first. Pick names you cannot confuse with the live ones — the live
bucket is `nstc-2025-storage` and the live Cloud Run services are
`nstc-linebot-2025` (asia-east1) and `badminton-analysis-ai` (asia-southeast1).

```bash
export PROJECT_ID=<your project>
export REGION=asia-east1              # bot; keep data and bucket in one region
export BUCKET=nstc-2025-storage-noai
export SA_EMAIL=nstc-linebot-2025@${PROJECT_ID}.iam.gserviceaccount.com  # shared with production
```

### Service account

**There is no new service account.** The variant runs as the existing
`nstc-linebot-2025@nstc-linebot-2025.iam.gserviceaccount.com`, which is already
the sole `roles/run.invoker` on the analysis service, so **no IAM change is
needed** to let the variant call it.

Be clear about what one shared identity costs, because it is the real price of
this topology:

- **The two deployments are indistinguishable at the identity layer.** Cloud Run
  audit logs show the same principal for both, so "which product did this?"
  cannot be answered from IAM.
- **You cannot revoke one without revoking the other.** Any binding you remove to
  cut the variant off also cuts off production.
- **Nothing but configuration keeps the variant out of production's data.** That
  account holds project-wide Firestore and Storage permissions, so a wrong
  `GCS_BUCKET_NAME` or `FIREBASE_DATA_DB` does not fail -- it silently reads and
  writes the live product's learner data, and the first sign of it is two
  cohorts' work mixed in one portfolio.

There is no permission boundary between the two products. The separation is
entirely in the four values below, which is why the deploy reads them back off
the running revision instead of trusting that it set them.

### GCS bucket

```bash
gcloud storage buckets create "gs://${BUCKET}" \
  --project "${PROJECT_ID}" \
  --location "${REGION}" \
  --uniform-bucket-level-access \
  --soft-delete-duration=7d
```

Then decide the read policy deliberately, because the two halves of this product
want different things:

- **Rendered practice video and expert clips are served through signed URLs**
  (`SignPlaybackURL` in the Go bot, and `RefreshPlaybackUrls` on the analysis
  service). These need no public read at all.
- **Portfolio carousel thumbnails are fetched by LINE's servers with no
  credentials.** If nothing in the bucket is anonymously readable, every carousel
  bubble renders with a broken image.

So grant anonymous read to the thumbnail prefix only, and never to the bucket as
a whole. With uniform bucket-level access on, an IAM condition can scope it:

```bash
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member allUsers \
  --role roles/storage.legacyObjectReader \
  --condition "expression=resource.name.startsWith('projects/_/buckets/${BUCKET}/objects/users/') && resource.name.endsWith('thumbnail'),title=thumbnails-only,description=LINE fetches carousel thumbnails unauthenticated"
```

Two things matter here:

- `roles/storage.legacyObjectReader` grants `storage.objects.get` only.
  `roles/storage.objectViewer` also grants `storage.objects.list`, which makes a
  bucket anonymously enumerable — every object walkable by anyone with no
  credentials and no knowledge of any path. For a bucket holding students'
  practice video that difference is the whole ballgame. Use the narrow role.
- Public access prevention cannot be `enforced` while any `allUsers` binding
  exists. If you would rather enforce it, you must first move thumbnails to
  signed URLs too; that is a code change in `api/line/ui.go` (`assetURL`), not a
  configuration one.

Verify both halves before putting a class on it:

```bash
# A thumbnail must be fetchable with no credentials (LINE depends on this).
curl -s -o /dev/null -w '%{http_code}\n' \
  "https://storage.googleapis.com/${BUCKET}/users/<user>/thumbnail"

# A rendered analysis must NOT be.
curl -s -o /dev/null -w '%{http_code}\n' \
  "https://storage.googleapis.com/${BUCKET}/analyses/v1/<user>/<request>/student_corrected.mp4"

# The bucket must NOT be anonymously listable.
curl -s -o /dev/null -w '%{http_code}\n' \
  "https://storage.googleapis.com/storage/v1/b/${BUCKET}/o"
```

Expect `200`, `403`, `403`. Anything else means the binding is broader than
intended — re-read the condition before continuing.

### Firestore databases

Two named databases, in the same region as the bucket:

```bash
gcloud firestore databases create \
  --project "${PROJECT_ID}" \
  --database nstc-linebot-noai-data \
  --location "${REGION}" \
  --type firestore-native

gcloud firestore databases create \
  --project "${PROJECT_ID}" \
  --database nstc-linebot-noai-sessions \
  --location "${REGION}" \
  --type firestore-native
```

The bot creates the `weekly_reflections` collection on first write; nothing else
needs seeding.

### Secret Manager

The bot downloads its whole `.env` from one secret at boot:

```bash
cp .env.example .env      # then fill it in; .env is gitignored
gcloud secrets create 2025-linebot-noai-env --project "${PROJECT_ID}" --replication-policy automatic
gcloud secrets versions add 2025-linebot-noai-env --project "${PROJECT_ID}" --data-file .env
gcloud secrets add-iam-policy-binding 2025-linebot-noai-env --project "${PROJECT_ID}" \
  --member "serviceAccount:${SA_EMAIL}" --role roles/secretmanager.secretAccessor
```

No second gRPC key is created. The analysis service accepts exactly one
configured `x-api-key`, so the variant presents the existing
`analysis-grpc-api-key` -- put that same value in the `ANALYSIS_GRPC_API_KEY`
line of `.env` before uploading it, and read it back with:

```bash
gcloud secrets versions access latest --secret analysis-grpc-api-key
```

### Netlify site

```bash
npx --yes netlify-cli@26.1.0 login              # opens a browser
npx --yes netlify-cli@26.1.0 sites:create \
  --name nstc-linebot-liff-noai \
  --account-slug <your team slug>
```

The site ID it prints is `NOAI_NETLIFY_SITE_ID`, and the site URL is what goes in
`LIFF_REVIEW_URL` and `NOAI_BACKEND_BASE_URL`'s sibling
`NEXT_PUBLIC_BACKEND_BASE_URL`. Creating a site needs a Netlify auth token that
is not in this repo, so this is the one step that cannot be scripted from CI.

### The analysis service

Nothing to create. The variant points at the existing `badminton-analysis-ai` in
asia-southeast1 and sets `ANALYSIS_SKIP_COACHING=true`.

**Reuse the existing `analysis-grpc-api-key`.** The service compares the
`x-api-key` metadata against a single configured value with
`secrets.compare_digest`; it has no notion of several valid keys, so a
variant-specific key would simply be rejected. Both deployments present the same
one.

No invoker binding is needed either: the variant runs as the account that is
already the only `roles/run.invoker` on that service.

The Go client attaches an OIDC token audienced to the **service** URL, not a tag
URL; Cloud Run refuses a token minted for a tag URL with a bare 401.

## GitHub secrets to create

Repository or environment secrets, referenced by name only in the workflows. The
variant's CD workflows use an environment named `nstc-linebot-2025-noai`.

| Secret | What it is |
|---|---|
| `GCP_PROJECT_ID` | Project both deployments live in (shared with production's workflows) |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | OIDC provider for GitHub → GCP auth (shared) |
| `NETLIFY_AUTH_TOKEN` | Netlify personal access token (shared) |
| `GCP_SA_EMAIL` | The shared runtime service account (same as production) |
| `NOAI_ENV_SECRET_NAME` | `2025-linebot-noai-env` |
| `NOAI_GCS_BUCKET_NAME` | `${BUCKET}` |
| `NOAI_NEXT_PUBLIC_LIFF_ID` | The variant's LIFF ID, `<login channel id>-<suffix>` |
| `NOAI_BACKEND_BASE_URL` | The variant's Cloud Run bot URL, with `https://` |
| `NOAI_LIFF_REVIEW_URL` | The variant's Netlify site + `/personal?tab=review` |
| `NOAI_NETLIFY_SITE_ID` | The variant's Netlify site ID |

## CD

Three workflows are added, all firing only on `variant/no-ai` and each with the
narrowest path filter that can change what it deploys:

- `.github/workflows/ci-noai.yml` — build, vet, tests, `staticcheck -checks U1000`,
  a check that refuses a reintroduced `openai`/`gpt` reference in variant-owned
  code, and a check that the deploy still sets `ANALYSIS_SKIP_COACHING=true`.
- `.github/workflows/cd-linebot-noai.yml` — `linebot/**`; Cloud Run service
  `nstc-linebot-2025-noai`, image `gcr.io/<project>/nstc-linebot-2025-noai`. After
  deploying it reads the running revision's env back and fails unless
  `ANALYSIS_SKIP_COACHING` is `true`, `GCP_ENV_SECRET_NAME` is not production's,
  and the bucket and database names inside that secret are the variant's own.
- `.github/workflows/cd-liff-noai.yml` — `liff/**`; Netlify site
  `NOAI_NETLIFY_SITE_ID`, and it refuses to ship a build that has production's bot
  URL compiled into it.

No GPU deploy workflow is added: the analysis service is shared, so the variant
has nothing to build or deploy there. `cd-motion-analysis.yml` stays production's
alone.

Production's `ci.yml`, `cd-linebot.yml`, `cd-liff.yml` and `cd-motion-analysis.yml`
are inherited on this branch and left byte-identical. They all filter
`branches: [main]`, so they are inert here, and leaving them untouched means the
branch can never delete production's CD. They still carry production names,
including `GCS_BUCKET_NAME: nstc-2025-storage` — which is correct for them and
must not be edited on this branch.

## Verification

```bash
cd linebot && go build ./... && go vet ./... && go test ./...
cd linebot && PATH="$PATH:$(go env GOPATH)/bin" staticcheck -checks 'U1000' ./...
cd badminton_analysis_ai && PYTHONPATH=.:generated .venv/bin/python -m pytest -q tests
cd badminton_analysis_ai && PYTHONPATH=.:generated .venv/bin/python -c "import service.server"
cd liff && npx tsc --noEmit    # three pre-existing TS5097 errors are expected
cd liff && npm test && npm run build

# Variant-owned code carries no LLM reference. badminton_analysis_ai/ and the
# generated stubs under api/analysis/v1 are the shared contract and do, correctly.
grep -rniE "openai|gpt" --include='*.go' --include='*.py' --include='*.ts' \
  --include='*.tsx' --exclude-dir=v1 --exclude-dir=node_modules linebot/ liff/

# badminton_analysis_ai/ and proto/ must not drift from main.
git diff --stat main -- badminton_analysis_ai/ proto/ linebot/api/analysis/v1/
```

The last two must print nothing.

After a deploy, confirm the running service is pointed at the variant's own data
plane -- with one shared service account this is the only thing separating the
two products:

```bash
gcloud run services describe nstc-linebot-2025-noai --region asia-east1 --format=json \
  | jq -r '.spec.template.spec.containers[0].env[]? | "\(.name)=\(.value // "")"'
# expect ANALYSIS_SKIP_COACHING=true and a GCP_ENV_SECRET_NAME that is not 2025-linebot-env

gcloud secrets versions access latest --secret 2025-linebot-noai-env \
  | grep -E '^(GCS_BUCKET_NAME|FIREBASE_DATA_DB|FIREBASE_SESSION_DB)='
# none of these may be the live bucket (nstc-2025-storage) or the live databases
```
