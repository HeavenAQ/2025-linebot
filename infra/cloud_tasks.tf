# The LLM product's analysis queue; the variant has its own on its own branch.
resource "google_cloud_tasks_queue" "analysis" {
  project  = var.project_id
  name     = "analysis-llm"
  location = var.region

  rate_limits {
    max_concurrent_dispatches = 2
    max_dispatches_per_second = 4
  }

  retry_config {
    max_attempts       = -1
    max_retry_duration = "86400s"
    min_backoff        = "10s"
    max_backoff        = "120s"
  }

  depends_on = [google_project_service.enabled]
}
