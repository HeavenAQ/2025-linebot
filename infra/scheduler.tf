# This variant's scheduled work, calling its own worker with a Google-signed
# OIDC token for that worker's URL. The GPU warm-up and capacity jobs live on
# `main`: one L4 serves both products, so it is reserved once.
locals {
  worker_url = google_cloud_run_v2_service.bot.uri
}

# Sweeps up jobs recorded but never published (a crash between the two), and
# expires anything still unpublished after a day.
resource "google_cloud_scheduler_job" "outbox" {
  project          = var.project_id
  name             = "analysis-no-llm-outbox"
  region           = var.region
  schedule         = "* * * * *"
  time_zone        = var.class_timezone
  attempt_deadline = "180s"

  http_target {
    uri         = "${local.worker_url}/internal/analysis/outbox"
    http_method = "POST"

    oidc_token {
      service_account_email = data.google_service_account.bot.email
      audience              = local.worker_url
    }
  }

  retry_config {
    retry_count          = 0
    max_retry_duration   = "0s"
    min_backoff_duration = "5s"
    max_backoff_duration = "3600s"
    max_doublings        = 5
  }
}

# Attempts fold into this variant's class aggregates as they complete; this
# recomputes them nightly, outside class hours, to correct any drift.
resource "google_cloud_scheduler_job" "class_stats_rebuild" {
  project          = var.project_id
  name             = "class-stats-no-llm-rebuild"
  region           = var.region
  schedule         = "30 3 * * *"
  time_zone        = var.class_timezone
  attempt_deadline = "600s"

  http_target {
    uri         = "${local.worker_url}/internal/stats/rebuild"
    http_method = "POST"

    oidc_token {
      service_account_email = data.google_service_account.bot.email
      audience              = local.worker_url
    }
  }

  retry_config {
    retry_count          = 0
    max_retry_duration   = "0s"
    min_backoff_duration = "5s"
    max_backoff_duration = "3600s"
    max_doublings        = 5
  }
}
