#!/usr/bin/env bash
# Idempotent log-based metrics built on the structured JSON logs of the Go bot
# (message/jsonPayload fields written by linebot/api/obs) and the Python
# analysis service (service/logging_config.py). Cloud Run already exports
# request count, latency and instance metrics; these add application meaning.
set -euo pipefail

project="${GCP_PROJECT_ID:-nstc-linebot-2025}"
workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT

upsert_metric() {
  local name="$1" file="$2"
  if gcloud logging metrics describe "$name" --project "$project" >/dev/null 2>&1; then
    gcloud logging metrics update "$name" --project "$project" --config-from-file="$file" --quiet >/dev/null
  else
    gcloud logging metrics create "$name" --project "$project" --config-from-file="$file" --quiet >/dev/null
  fi
  echo "metric ${name} ready"
}

cat >"$workdir/analysis_jobs.yaml" <<'YAML'
description: Queued analysis jobs by outcome (completed, retry, failed) and skill.
filter: >-
  resource.type="cloud_run_revision"
  jsonPayload.message="analysis job finished"
metricDescriptor:
  metricKind: DELTA
  valueType: INT64
  labels:
    - key: outcome
    - key: skill
    - key: service
labelExtractors:
  outcome: EXTRACT(jsonPayload.outcome)
  skill: EXTRACT(jsonPayload.skill)
  service: EXTRACT(resource.labels.service_name)
YAML
upsert_metric analysis_jobs "$workdir/analysis_jobs.yaml"

cat >"$workdir/analysis_job_duration.yaml" <<'YAML'
description: Wall time of one queued analysis attempt as seen by the Go worker, in seconds.
filter: >-
  resource.type="cloud_run_revision"
  jsonPayload.message="analysis job finished"
valueExtractor: EXTRACT(jsonPayload.duration_seconds)
metricDescriptor:
  metricKind: DELTA
  valueType: DISTRIBUTION
  unit: s
  labels:
    - key: outcome
labelExtractors:
  outcome: EXTRACT(jsonPayload.outcome)
bucketOptions:
  exponentialBuckets:
    numFiniteBuckets: 20
    growthFactor: 1.5
    scale: 1
YAML
upsert_metric analysis_job_duration "$workdir/analysis_job_duration.yaml"

for stage in service llm_inference pose; do
  cat >"$workdir/analysis_latency_${stage}.yaml" <<YAML
description: GPU analysis service latency_${stage}_seconds per completed analysis.
filter: >-
  resource.type="cloud_run_revision"
  resource.labels.service_name="badminton-analysis-ai"
  jsonPayload.message="analysis completed"
valueExtractor: EXTRACT(jsonPayload.latency_${stage}_seconds)
metricDescriptor:
  metricKind: DELTA
  valueType: DISTRIBUTION
  unit: s
  labels:
    - key: skill
labelExtractors:
  skill: EXTRACT(jsonPayload.skill)
bucketOptions:
  exponentialBuckets:
    numFiniteBuckets: 20
    growthFactor: 1.5
    scale: 0.1
YAML
  upsert_metric "analysis_latency_${stage}" "$workdir/analysis_latency_${stage}.yaml"
done

cat >"$workdir/analysis_failures.yaml" <<'YAML'
description: Analyses the GPU service rejected or failed, by gRPC code and error type.
filter: >-
  resource.type="cloud_run_revision"
  resource.labels.service_name="badminton-analysis-ai"
  jsonPayload.message="analysis failed"
metricDescriptor:
  metricKind: DELTA
  valueType: INT64
  labels:
    - key: grpc_code
    - key: error_type
labelExtractors:
  grpc_code: EXTRACT(jsonPayload.grpc_code)
  error_type: EXTRACT(jsonPayload.error_type)
YAML
upsert_metric analysis_failures "$workdir/analysis_failures.yaml"

cat >"$workdir/learner_api_rejections.yaml" <<'YAML'
description: Learner API requests refused by rate limits or failed authentication.
filter: >-
  resource.type="cloud_run_revision"
  (jsonPayload.message="rate limited" OR jsonPayload.message="authentication rejected")
metricDescriptor:
  metricKind: DELTA
  valueType: INT64
  labels:
    - key: reason
    - key: limiter
    - key: service
labelExtractors:
  reason: EXTRACT(jsonPayload.message)
  limiter: EXTRACT(jsonPayload.limiter)
  service: EXTRACT(resource.labels.service_name)
YAML
upsert_metric learner_api_rejections "$workdir/learner_api_rejections.yaml"
