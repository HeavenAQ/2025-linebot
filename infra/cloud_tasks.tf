# The LLM product's analysis queue. The no-LLM variant has its own, declared
# on its own branch.
#
# Two concurrent dispatches, because the two variants' queues together must not
# put more than four requests on the single L4 that serves both. The worker
# acknowledges a task only after the attempt is durably recorded, so Cloud
# Tasks owns every retry and nothing is lost to a crash mid-analysis.
#
# Retries never give up on a schedule (`max_attempts = -1`) but do give up
# after a day: a video that cannot be analysed in 24 hours is a bug to fix,
# not a task to keep running.
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
