# GPT validation review service

Web app where two badminton experts rate the GPT coaching feedback, blind and on
their own. The contract (Firestore schema, API, CSV columns) is in
[`../DESIGN.md`](../DESIGN.md).

- `main.go`: config, wiring, graceful shutdown
- `internal/review/`: auth and rate limiting, validation, the store (Firestore
  plus an in-memory fake), signed URLs, response filtering, CSV export, HTTP
  handlers
- `web/`: the single-page front end in Traditional Chinese, embedded in the
  binary. It is plain HTML, CSS and JS, with no build step.

## Configuration

| Env | Required | Notes |
|---|---|---|
| `PORT` | no | default `8080` |
| `GCP_PROJECT_ID` | yes | Firestore project |
| `GCS_BUCKET_NAME` | yes | e.g. `nstc-2025-storage` |
| `GCP_SERVICE_ACCOUNT_EMAIL` | yes | signs V4 URLs through IAM when there is no key file |
| `BATCH_ID` | yes | the service only serves items with this `batch_id` |
| `ACCESS_CODES` | yes | JSON `{"expert_1":"…","expert_2":"…","admin":"…"}` |
| `LOCAL_FAKE_DATA` | no | `1` serves sample data from memory with no GCP access (no videos). Local only. |

The service exits at startup if any setting is missing or invalid.

## Run locally

With sample data and no cloud access:

```sh
LOCAL_FAKE_DATA=1 \
ACCESS_CODES='{"expert_a":"aaa-111","expert_b":"bbb-222","admin":"adm-999"}' \
go run .
# open http://localhost:8080 and log in with aaa-111 (expert) or adm-999 (admin)
```

Against real Firestore/GCS, using your Application Default Credentials. Local
URL signing needs a service-account key, or your user needs
`roles/iam.serviceAccountTokenCreator` on the service account:

```sh
gcloud auth application-default login
GCP_PROJECT_ID=nstc-linebot-2025 \
GCS_BUCKET_NAME=nstc-2025-storage \
GCP_SERVICE_ACCOUNT_EMAIL=nstc-linebot-2025@nstc-linebot-2025.iam.gserviceaccount.com \
BATCH_ID=beginners-2026-09 \
ACCESS_CODES='{"expert_a":"…","expert_b":"…","admin":"…"}' \
go run .
```

## Test

```sh
go test ./...
go vet ./...
gofmt -l .
node --check web/app.js
```

## Deploy

`deploy.sh` builds the image with Cloud Build and deploys the Cloud Run service
`gpt-validation` in asia-east1. `ACCESS_CODES` comes from the Secret Manager
secret `gpt-validation-access-codes`.

```sh
BATCH_ID=beginners-2026-09 ./deploy.sh
```
