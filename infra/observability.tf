# Log-based metrics, built from the structured logs both services already
# write. They are project-wide and carry a `service` label, so one set covers
# both products; the no-LLM branch does not redeclare them.
#
# Everything here reads fields the code logs deliberately. If a field is
# renamed, the metric goes quiet rather than wrong, which is why the field
# names appear in tests on the Go side.

# Queued analyses by how they ended. The outcome label separates a completed
# attempt from one that will be retried and one that failed for good.
resource "google_logging_metric" "analysis_jobs" {
  project     = var.project_id
  name        = "analysis_jobs"
  description = "Queued analysis jobs by outcome (completed, retry, failed) and skill."
  filter      = <<-EOT
    resource.type="cloud_run_revision"
    jsonPayload.message="analysis job finished"
  EOT

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"

    labels {
      key = "outcome"
    }
    labels {
      key = "skill"
    }
    labels {
      key = "service"
    }
  }

  label_extractors = {
    outcome = "EXTRACT(jsonPayload.outcome)"
    skill   = "EXTRACT(jsonPayload.skill)"
    service = "EXTRACT(resource.labels.service_name)"
  }
}

# How long one attempt took end to end, as the worker saw it: queue wait is
# not included, GPU time and storage are.
resource "google_logging_metric" "analysis_job_duration" {
  project     = var.project_id
  name        = "analysis_job_duration"
  description = "Wall time of one queued analysis attempt as seen by the Go worker, in seconds."
  filter      = <<-EOT
    resource.type="cloud_run_revision"
    jsonPayload.message="analysis job finished"
  EOT

  value_extractor = "EXTRACT(jsonPayload.duration_seconds)"

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "DISTRIBUTION"
    unit        = "s"

    labels {
      key = "outcome"
    }
  }

  label_extractors = {
    outcome = "EXTRACT(jsonPayload.outcome)"
  }

  bucket_options {
    exponential_buckets {
      num_finite_buckets = 20
      growth_factor      = 1.5
      scale              = 1
    }
  }
}

# The GPU pipeline's own stages. Watching these separately is what shows
# whether a slow analysis was pose extraction, the coaching call or rendering.
resource "google_logging_metric" "analysis_stage_latency" {
  for_each = toset(["service", "llm_inference", "pose"])

  project     = var.project_id
  name        = "analysis_latency_${each.value}"
  description = "GPU analysis service latency_${each.value}_seconds per completed analysis."
  filter      = <<-EOT
    resource.type="cloud_run_revision"
    resource.labels.service_name="${google_cloud_run_v2_service.analysis.name}"
    jsonPayload.message="analysis completed"
  EOT

  value_extractor = "EXTRACT(jsonPayload.latency_${each.value}_seconds)"

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "DISTRIBUTION"
    unit        = "s"
  }

  bucket_options {
    exponential_buckets {
      num_finite_buckets = 20
      growth_factor      = 1.5
      scale              = 0.1
    }
  }
}

# Analyses the GPU service turned away, by gRPC code and reason: a video that
# does not match the chosen stroke looks very different from an outage.
resource "google_logging_metric" "analysis_failures" {
  project     = var.project_id
  name        = "analysis_failures"
  description = "Analyses the GPU service rejected or failed, by gRPC code and error type."
  filter      = <<-EOT
    resource.type="cloud_run_revision"
    resource.labels.service_name="${google_cloud_run_v2_service.analysis.name}"
    jsonPayload.message="analysis failed"
  EOT

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"

    labels {
      key = "grpc_code"
    }
    labels {
      key = "error_type"
    }
  }

  label_extractors = {
    grpc_code  = "EXTRACT(jsonPayload.grpc_code)"
    error_type = "EXTRACT(jsonPayload.error_type)"
  }
}

# Learner requests refused before they reached a handler. A rise here is
# either an attack or a bug in the app's retry behaviour.
resource "google_logging_metric" "learner_api_refusals" {
  project     = var.project_id
  name        = "learner_api_refusals"
  description = "Learner API requests refused by rate limits or failed authentication."
  filter      = <<-EOT
    resource.type="cloud_run_revision"
    (jsonPayload.message="rate limited" OR jsonPayload.message="authentication rejected")
  EOT

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"

    labels {
      key = "service"
    }
  }

  label_extractors = {
    service = "EXTRACT(resource.labels.service_name)"
  }
}
