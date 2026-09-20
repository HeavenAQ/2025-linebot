# Scheduled work, calling the bot's internal endpoints with a Google-signed
# OIDC token whose audience is the worker URL, read back from the service so a
# redeploy cannot leave a job pointing at a dead host. The outbox and rebuild
# are per product; the GPU jobs are shared, since one L4 serves both.
locals {
  worker_url = google_cloud_run_v2_service.bot.uri
}

# Sweeps up jobs recorded but never published (a crash between the two), and
# expires anything still unpublished after a day.
resource "google_cloud_scheduler_job" "outbox" {
  project          = var.project_id
  name             = "analysis-llm-outbox"
  region           = var.region
  schedule         = "* * * * *"
  time_zone        = var.class_timezone
  attempt_deadline = "180s"

  http_target {
    uri         = "${local.worker_url}/internal/analysis/outbox"
    http_method = "POST"

    oidc_token {
      service_account_email = google_service_account.bot.email
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

# Attempts fold into the class aggregates as they complete; this recomputes
# them nightly, outside class hours, to correct any drift.
resource "google_cloud_scheduler_job" "class_stats_rebuild" {
  project          = var.project_id
  name             = "class-stats-llm-rebuild"
  region           = var.region
  schedule         = "30 3 * * *"
  time_zone        = var.class_timezone
  attempt_deadline = "600s"

  http_target {
    uri         = "${local.worker_url}/internal/stats/rebuild"
    http_method = "POST"

    oidc_token {
      service_account_email = google_service_account.bot.email
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

# Class is Monday 14:00-18:00 Taipei and a cold L4 costs the first student a
# minute, so capacity is reserved at 13:45 and released at 18:10 -- an idle GPU
# is billed by the second.
resource "google_cloud_scheduler_job" "gpu_capacity" {
  for_each = {
    "analysis-gpu-monday-start" = { schedule = "45 13 * * 1", minimum = 1 }
    "analysis-gpu-monday-stop"  = { schedule = "10 18 * * 1", minimum = 0 }
  }

  project   = var.project_id
  name      = each.key
  region    = var.region
  schedule  = each.value.schedule
  time_zone = var.class_timezone

  http_target {
    uri         = "${local.worker_url}/internal/analysis/capacity"
    http_method = "POST"
    headers     = { "Content-Type" = "application/json" }
    body        = base64encode(jsonencode({ minimum_instances = each.value.minimum }))

    oidc_token {
      service_account_email = google_service_account.bot.email
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

# The models load lazily, so these run a real inference rather than pinging:
# the pre-class pair wakes the GPU, the in-class sweep keeps it warm.
resource "google_cloud_scheduler_job" "gpu_warmup" {
  for_each = {
    "analysis-gpu-warmup"       = "50,55 13 * * 1"
    "analysis-gpu-class-warmup" = "*/10 14-17 * * 1"
  }

  project          = var.project_id
  name             = each.key
  region           = var.region
  schedule         = each.value
  time_zone        = var.class_timezone
  attempt_deadline = "600s"

  http_target {
    uri         = "${local.worker_url}/internal/analysis/warmup"
    http_method = "POST"

    oidc_token {
      service_account_email = google_service_account.bot.email
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
