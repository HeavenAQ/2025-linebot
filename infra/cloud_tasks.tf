# This variant's analysis queue.
#
# Two concurrent dispatches, because the two products' queues together must not
# put more than four requests on the single L4 that serves both. The worker
# acknowledges a task only after the attempt is durably recorded, so Cloud
# Tasks owns every retry and nothing is lost to a crash mid-analysis.
resource "google_cloud_tasks_queue" "analysis" {
  project  = var.project_id
  name     = "analysis-no-llm"
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
}
