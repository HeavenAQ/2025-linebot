#!/usr/bin/env bash
# Idempotent infrastructure setup. Run after both bot workers are deployed.
# llm/no-llm name the queues and schedulers; service URLs, Firestore collections
# and storage prefixes are unchanged.
set -euo pipefail

project="${GCP_PROJECT_ID:-nstc-linebot-2025}"
region="asia-east1"
account="nstc-linebot-2025@${project}.iam.gserviceaccount.com"
accept="${ANALYSIS_ASYNC_ACCEPT:-false}"

gcloud projects add-iam-policy-binding "$project" \
  --member="serviceAccount:${account}" --role=roles/cloudtasks.enqueuer --condition=None --quiet >/dev/null

upsert_scheduler() {
  local name="$1"; shift
  local operation=create
  if gcloud scheduler jobs describe "$name" --location "$region" --project "$project" >/dev/null 2>&1; then operation=update; fi
  local args=()
  for arg in "$@"; do
    if [[ "$operation" == update && "$arg" == --headers=* ]]; then
      args+=("--update-headers=${arg#--headers=}")
    else
      args+=("$arg")
    fi
  done
  gcloud scheduler jobs "$operation" http "$name" --location "$region" --project "$project" \
    --time-zone=Asia/Taipei --quiet "${args[@]}"
}

for variant in llm no-llm; do
  service=nstc-linebot-2025
  if [[ "$variant" == no-llm ]]; then service=nstc-linebot-2025-noai; fi
  queue="analysis-${variant}"
  operation=create
  if gcloud tasks queues describe "$queue" --location "$region" --project "$project" >/dev/null 2>&1; then operation=update; fi
  # Two queues together admit at most four GPU requests to one L4 instance.
  # Workers acknowledge only after durable completion; Cloud Tasks owns retry.
  gcloud tasks queues "$operation" "$queue" --location "$region" --project "$project" \
    --max-concurrent-dispatches=2 --max-dispatches-per-second=4 \
    --max-attempts=-1 --max-retry-duration=86400s --min-backoff=10s --max-backoff=120s --quiet
  worker_url="$(gcloud run services describe "$service" --region "$region" --project "$project" --format='value(status.url)')"
  gcloud run services update "$service" --region "$region" --project "$project" --timeout=1200 \
    --update-env-vars "ANALYSIS_TASKS_QUEUE=projects/${project}/locations/${region}/queues/${queue},ANALYSIS_WORKER_URL=${worker_url},ANALYSIS_TASK_SERVICE_ACCOUNT=${account},ANALYSIS_ASYNC_ACCEPT=${accept}" --quiet
  upsert_scheduler "analysis-${variant}-outbox" --schedule='* * * * *' \
    --uri="${worker_url}/internal/analysis/outbox" --http-method=POST \
    --oidc-service-account-email="$account" --oidc-token-audience="$worker_url" --attempt-deadline=180s
  if [[ "$variant" == llm ]]; then
    llm_worker_url="$worker_url"
    # Monday class is 14:00–18:00 Taiwan; reserve at 13:45 and load the actual
    # inference engine at 13:50 (a generic HTTP ping would not warm lazy models).
    upsert_scheduler analysis-gpu-warmup --schedule='50,55 13 * * 1' \
      --uri="${worker_url}/internal/analysis/warmup" --http-method=POST \
      --oidc-service-account-email="$account" --oidc-token-audience="$worker_url" --attempt-deadline=600s
    upsert_scheduler analysis-gpu-class-warmup --schedule='*/10 14-17 * * 1' \
      --uri="${worker_url}/internal/analysis/warmup" --http-method=POST \
      --oidc-service-account-email="$account" --oidc-token-audience="$worker_url" --attempt-deadline=600s
  fi
done

upsert_scheduler analysis-gpu-monday-start --schedule='45 13 * * 1' \
  --uri="${llm_worker_url}/internal/analysis/capacity" --http-method=POST \
  --oidc-service-account-email="$account" --oidc-token-audience="$llm_worker_url" \
  --headers=Content-Type=application/json --message-body='{"minimum_instances":1}'
upsert_scheduler analysis-gpu-monday-stop --schedule='10 18 * * 1' \
  --uri="${llm_worker_url}/internal/analysis/capacity" --http-method=POST \
  --oidc-service-account-email="$account" --oidc-token-audience="$llm_worker_url" \
  --headers=Content-Type=application/json --message-body='{"minimum_instances":0}'
