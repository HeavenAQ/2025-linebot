# The weekly class report: one score per student per stroke, pushed to the
# instructors over LINE. A service account cannot send mail as a consumer Gmail
# account, and both instructors already have the bot in their chat list.

resource "google_storage_bucket" "function_source" {
  project                     = var.project_id
  name                        = "${var.project_id}-functions"
  location                    = upper(var.region)
  storage_class               = "STANDARD"
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  # Only the deploy artefact lives here, and only the newest matters.
  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "Delete"
    }
  }

  depends_on = [google_project_service.enabled]
}

data "archive_file" "class_report" {
  type        = "zip"
  source_dir  = "${path.module}/../functions/class-report"
  output_path = "${path.module}/.build/class-report.zip"
  excludes    = ["test_main.py", "__pycache__", ".gcloudignore"]
}

# The object name carries the hash, so a changed function is a new object and
# Cloud Functions redeploys; an unchanged one is not rebuilt on every apply.
resource "google_storage_bucket_object" "class_report" {
  name   = "class-report/${data.archive_file.class_report.output_md5}.zip"
  bucket = google_storage_bucket.function_source.name
  source = data.archive_file.class_report.output_path
}

resource "google_cloudfunctions2_function" "class_report" {
  project  = var.project_id
  name     = "class-report"
  location = var.region

  build_config {
    runtime     = "python312"
    entry_point = "class_report"

    source {
      storage_source {
        bucket = google_storage_bucket.function_source.name
        object = google_storage_bucket_object.class_report.name
      }
    }
  }

  service_config {
    service_account_email = google_service_account.bot.email
    available_memory      = "512Mi"
    # Reading both databases and rendering the chart, with room for a cold
    # start on a function that runs once a week.
    timeout_seconds                = 300
    max_instance_count             = 2
    ingress_settings               = "ALLOW_ALL"
    all_traffic_on_latest_revision = true

    environment_variables = {
      GCP_PROJECT_ID      = var.project_id
      CLASS_REPORT_BUCKET = google_storage_bucket.learner_media.name
      CLASS_REPORT_WEEKS  = "8"
      CLASS_REPORT_PRODUCTS = jsonencode([
        { label = "llm", database = "(default)" },
        { label = "no-llm", database = "nstc-linebot-noai" },
      ])
      CLASS_REPORT_RECIPIENTS       = var.class_report_recipients
      CLASS_STATS_EXCLUDED_USER_IDS = var.class_stats_excluded_user_ids
    }

    # The bots keep their whole .env in one secret and the function needs one
    # line of it, which is better than a second copy of the channel token to
    # rotate.
    secret_environment_variables {
      key        = "BOT_ENV"
      project_id = var.project_id
      secret     = "2025-linebot-env"
      version    = "latest"
    }
  }

  depends_on = [google_project_service.enabled]
}

# Second-generation functions are Cloud Run services, so the scheduler needs
# the invoker role there rather than on the function resource.
resource "google_cloud_run_v2_service_iam_member" "class_report_invoker" {
  project  = var.project_id
  location = var.region
  name     = google_cloudfunctions2_function.class_report.name
  role     = "roles/run.invoker"
  member   = google_service_account.bot.member
}

resource "google_cloud_scheduler_job" "class_report" {
  project     = var.project_id
  name        = "class-report-weekly"
  region      = var.region
  description = "Push the weekly class report to the instructors over LINE."
  # Monday 18:30 Taipei: both classes end at 18:00, and the GPU releases its
  # capacity at 18:10, so by now the week's uploads have been graded.
  schedule         = "30 18 * * 1"
  time_zone        = var.class_timezone
  attempt_deadline = "320s"

  http_target {
    http_method = "POST"
    uri         = google_cloudfunctions2_function.class_report.service_config[0].uri

    oidc_token {
      service_account_email = google_service_account.bot.email
      audience              = google_cloudfunctions2_function.class_report.service_config[0].uri
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
