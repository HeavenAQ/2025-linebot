#!/usr/bin/env bash
# Build and deploy the GPT validation expert review service to Cloud Run.
#
# Usage: BATCH_ID=beginners-2026-09 ./deploy.sh
#
# The service itself, its service account, its IAM and the secret that holds
# the access codes are declared in infra/ and created by `terraform apply`.
# This script only builds an image and deploys a revision -- the same split the
# two bots have. What it still needs from you is the secret's *value*, which
# Terraform deliberately never sees:
#
#   gcloud secrets versions add gpt-validation-access-codes --data-file=codes.json
#
# holding {"<expert_id>": "<code>", ..., "admin": "<code>"}.
set -euo pipefail

PROJECT="${GCP_PROJECT_ID:-nstc-linebot-2025}"
REGION="asia-east1"
SERVICE="gpt-validation"
BUCKET="${GCS_BUCKET_NAME:-nstc-2025-storage}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-nstc-linebot-2025@${PROJECT}.iam.gserviceaccount.com}"
BATCH_ID="${BATCH_ID:?set BATCH_ID, e.g. BATCH_ID=beginners-2026-09}"
IMAGE="gcr.io/${PROJECT}/${SERVICE}"

cd "$(dirname "$0")"

gcloud builds submit --project "${PROJECT}" --tag "${IMAGE}" .

gcloud run deploy "${SERVICE}" \
  --project "${PROJECT}" \
  --region "${REGION}" \
  --image "${IMAGE}" \
  --platform managed \
  --min-instances 0 \
  --max-instances 2 \
  --service-account "${SERVICE_ACCOUNT}" \
  --set-env-vars "GCP_PROJECT_ID=${PROJECT},GCS_BUCKET_NAME=${BUCKET},GCP_SERVICE_ACCOUNT_EMAIL=${SERVICE_ACCOUNT},BATCH_ID=${BATCH_ID}" \
  --set-secrets "ACCESS_CODES=gpt-validation-access-codes:latest"

URL="$(gcloud run services describe "${SERVICE}" --project "${PROJECT}" --region "${REGION}" --format 'value(status.url)')"
echo "Deployed ${SERVICE}: ${URL}"
