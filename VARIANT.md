# The no-AI variant

`variant/no-ai` is the badminton coaching product with every large-language-model
layer removed, so it can run beside the original and be compared against it.

**Nothing has been provisioned.** No bucket, no Firestore database, no Cloud Run
service, no Netlify site and no GitHub secret was created. Everything below is a
command for you to run deliberately, once you have decided you want the resource
and the bill that comes with it.

**Never merge `variant/no-ai` into `main`.** It is a parallel product line.

## What is gone, and what is not

The analysis was never GPT. Pose estimation, the EIMD diffusion correction,
grading, the checkpoint timeline and expert matching are local models, and all of
them are unchanged. What is removed is the natural-language layer that sat on
top.

Removed:

| Area | Gone |
|---|---|
| Go | `api/gpt/` in full; the `ChattingWithGPT` state and its rich-menu, postback and quick-reply entries; chat history (`api/db/chat_history.go`) and `daily_summaries` (`api/db/daily_summary.go`); the weekly 課前預習 push (`app/weekly_preview.go`, `api/db/weekly_preview.go`, `PushWeeklyPreview`); `/api/chat/history`, `/api/chat/summarize` and `/api/preview/weekly`; `OPENAI_*` and `WEEKLY_PREVIEW_TOKEN` config; `gpt_conversation_ids` and `ai_note` on the user record; the 詢問AI建議 portfolio button |
| Python | `service/coaching.py` and `badminton_analysis/ml/clear_feedback.py` (the prompt, sampling and response-schema layer); the coaching-cue render pass and its 2-second pauses; `OPENAI_*` env vars and the `openai` dependency; `coaching_joints` and `as_prompt_dict` on the rule spec, which only ever fed a prompt |
| Web app | `SkillSummary.tsx`, `useSkillSummary.ts`, the `gpt-chat` page and its menu entry; the chat-history panel in `WeeklyReview.tsx`; the coaching-cue markers, cue list and pause legend in `VideoComparison.tsx` |

Kept, deliberately: uploads, grading and the score charts, the checkpoint
timeline, the expert comparison with checkpoint alignment and segmental warping,
weekly reflections, stats, the portfolio carousel, and both `student_video` and
`skeleton_overlay_video` fields on the response.

`coaching_cues` and `overall_feedback` remain in the protobuf contract and are
always empty. They are not populated anywhere.

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
export SA_EMAIL=nstc-linebot-noai@${PROJECT_ID}.iam.gserviceaccount.com
```

### Service account

```bash
gcloud iam service-accounts create nstc-linebot-noai \
  --project "${PROJECT_ID}" \
  --display-name "No-AI variant runtime"

for role in roles/datastore.user roles/storage.objectAdmin \
            roles/secretmanager.secretAccessor roles/iam.serviceAccountTokenCreator; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member "serviceAccount:${SA_EMAIL}" --role "${role}"
done
```

`roles/iam.serviceAccountTokenCreator` is what lets the account sign playback
URLs through IAM: on Cloud Run the metadata credentials carry no private key.

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

And the analysis gRPC key, which both services check:

```bash
openssl rand -hex 32 | tr -d '\n' | \
  gcloud secrets create analysis-grpc-api-key-noai --project "${PROJECT_ID}" \
    --replication-policy automatic --data-file -
gcloud secrets add-iam-policy-binding analysis-grpc-api-key-noai --project "${PROJECT_ID}" \
  --member "serviceAccount:${SA_EMAIL}" --role roles/secretmanager.secretAccessor
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

You have a choice here, and it is a real trade rather than a detail.

**Option A — run the variant's own GPU service.** The service in this branch has
no OpenAI call in it at all. This is the only option that makes the side-by-side
comparison honest, because in Option B every analysis in *both* arms still goes
through GPT. It costs a second NVIDIA L4 on the bill, though it scales to zero
(`--min 0`), so the cost is per analysis rather than per week.

Deploy it with distinct names — `badminton-analysis-ai-noai` in
asia-southeast1, image `gcr.io/${PROJECT_ID}/badminton-analysis-noai` — mirroring
`.github/workflows/cd-motion-analysis.yml` but dropping its `OPENAI_COACHING_MODEL`,
`COACHING_PAUSE_SECONDS`, `COACHING_NO_SUGGESTION_*` env vars and its
`OPENAI_API_KEY` secret, which this code no longer reads. Copy the expert
reference bank, expert clips and the RF-DETR TensorRT engines into the variant
bucket first; the deploy verifies itself against two expert fixtures that have to
be there. No CD workflow for this is shipped on the branch, deliberately: standing
up a second L4 is your call, not a push trigger's.

**Option B — share the already-deployed `badminton-analysis-ai`.** Set
`ANALYSIS_GRPC_TARGET` to that service's host and use the existing
`analysis-grpc-api-key`. Nothing needs deploying. Be clear about what you are
buying: the shared service still calls OpenAI on every analysis, so every
student's video frames still reach OpenAI in both arms, the API bill is unchanged,
and "with AI" versus "without AI" no longer distinguishes the two deployments —
only what the learner is shown differs. If that is acceptable, then consume
`skeleton_overlay_video` rather than `student_video`, since the shared service
burns coaching captions and 2-second pauses into `student_video`'s pixels, which
no consumer-side filtering can remove. The checkpoint timeline is computed as
`frame / fps` with no pause offset (`_qualitative_phase_results` in
`service/pipeline.py`), so it is already expressed in overlay-render time and
lines up against that video without adjustment.

Either way, the service accepts only one invoker. If the variant's bot runs as
its own service account, add it:

```bash
gcloud run services add-iam-policy-binding <analysis service> \
  --region asia-southeast1 \
  --member "serviceAccount:${SA_EMAIL}" \
  --role roles/run.invoker
```

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
| `NOAI_GCP_SA_EMAIL` | `${SA_EMAIL}` above — the variant's runtime service account |
| `NOAI_ENV_SECRET_NAME` | `2025-linebot-noai-env` |
| `NOAI_ANALYSIS_API_KEY_SECRET` | `analysis-grpc-api-key-noai` |
| `NOAI_GCS_BUCKET_NAME` | `${BUCKET}` |
| `NOAI_NEXT_PUBLIC_LIFF_ID` | The variant's LIFF ID, `<login channel id>-<suffix>` |
| `NOAI_BACKEND_BASE_URL` | The variant's Cloud Run bot URL, with `https://` |
| `NOAI_LIFF_REVIEW_URL` | The variant's Netlify site + `/personal?tab=review` |
| `NOAI_NETLIFY_SITE_ID` | The variant's Netlify site ID |

## CD

Three workflows are added, all firing only on `variant/no-ai` and each with the
narrowest path filter that can change what it deploys:

- `.github/workflows/ci-noai.yml` — build, vet, tests, `staticcheck -checks U1000`,
  and a check that refuses any reintroduced `openai`/`gpt` reference.
- `.github/workflows/cd-linebot-noai.yml` — `linebot/**`; Cloud Run service
  `nstc-linebot-2025-noai`, image `gcr.io/<project>/nstc-linebot-2025-noai`.
- `.github/workflows/cd-liff-noai.yml` — `liff/**`; Netlify site
  `NOAI_NETLIFY_SITE_ID`, and it refuses to ship a build that has production's bot
  URL compiled into it.

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
cd liff && npx tsc --noEmit    # two pre-existing TS5097 errors are expected
cd liff && npm test && npm run build
grep -rniE "openai|gpt" --include='*.go' --include='*.py' --include='*.ts' --include='*.tsx' .
```

The last one must print nothing.
