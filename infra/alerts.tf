# The read side of observability.tf: a student stuck, the study losing data, or
# money going out. Thresholds are loose on purpose -- an alert that fires on one
# transient failure gets muted, and a muted alert is worse than none.

resource "google_monitoring_notification_channel" "email" {
  project      = var.project_id
  display_name = "Operator email"
  type         = "email"
  description  = "Where every alert in this project goes."

  labels = {
    email_address = var.alert_email
  }

  depends_on = [google_project_service.enabled]
}

# The failure a student notices first: the webhook 500s and no reply arrives.
# Both bots are matched by name, since alert policies are project-wide.
resource "google_monitoring_alert_policy" "bot_errors" {
  project      = var.project_id
  display_name = "LINE bot serving 5xx"
  combiner     = "OR"

  documentation {
    content   = "The bot backend returned server errors. Check the service's logs for `analysis job finished` with outcome=failed, and the GPU service's health."
    mime_type = "text/markdown"
  }

  conditions {
    display_name = "5xx responses over five minutes"

    condition_threshold {
      filter = join(" AND ", [
        "metric.type=\"run.googleapis.com/request_count\"",
        "resource.type=\"cloud_run_revision\"",
        "metric.label.response_code_class=\"5xx\"",
        "resource.label.service_name=one_of(\"${google_cloud_run_v2_service.bot.name}\",\"nstc-linebot-2025-noai\")",
      ])
      comparison      = "COMPARISON_GT"
      threshold_value = 3
      duration        = "0s"

      aggregations {
        alignment_period   = "300s"
        per_series_aligner = "ALIGN_SUM"
      }
    }
  }

  notification_channels = [google_monitoring_notification_channel.email.id]

  alert_strategy {
    auto_close = "3600s"
  }
}

# A handful a day is students uploading the wrong stroke; a burst is the service.
resource "google_monitoring_alert_policy" "analysis_failures" {
  project      = var.project_id
  display_name = "Analyses failing"
  combiner     = "OR"

  documentation {
    content   = "The GPU analysis service is failing requests. The `analysis_failures` metric's grpc_code and error_type labels separate a rejected video from an outage."
    mime_type = "text/markdown"
  }

  conditions {
    display_name = "More than three failures in ten minutes"

    condition_threshold {
      filter          = "metric.type=\"logging.googleapis.com/user/${google_logging_metric.analysis_failures.name}\" AND resource.type=\"cloud_run_revision\""
      comparison      = "COMPARISON_GT"
      threshold_value = 3
      duration        = "0s"

      aggregations {
        alignment_period   = "600s"
        per_series_aligner = "ALIGN_SUM"
      }
    }
  }

  notification_channels = [google_monitoring_notification_channel.email.id]

  alert_strategy {
    auto_close = "3600s"
  }
}

# A scheduled job is failing. The outbox is how a queued analysis ever reaches
# the GPU, and the Monday stop job is what releases the reserved L4 -- both
# fail silently, which is exactly why they are worth an alert.
resource "google_monitoring_alert_policy" "scheduler_failures" {
  project      = var.project_id
  display_name = "Scheduled job failing"
  combiner     = "OR"

  documentation {
    content   = "A Cloud Scheduler job has been failing. If it is `analysis-llm-outbox` or `analysis-no-llm-outbox`, queued analyses are not being dispatched. If it is `analysis-gpu-monday-stop`, the GPU is still reserved and being billed."
    mime_type = "text/markdown"
  }

  conditions {
    display_name = "Repeated failures in fifteen minutes"

    condition_threshold {
      filter          = "metric.type=\"logging.googleapis.com/user/${google_logging_metric.scheduler_failures.name}\" AND resource.type=\"cloud_scheduler_job\""
      comparison      = "COMPARISON_GT"
      threshold_value = 5
      duration        = "0s"

      aggregations {
        alignment_period     = "900s"
        per_series_aligner   = "ALIGN_SUM"
        cross_series_reducer = "REDUCE_SUM"
        group_by_fields      = ["metric.label.job_id"]
      }
    }
  }

  notification_channels = [google_monitoring_notification_channel.email.id]

  alert_strategy {
    auto_close = "3600s"
  }
}

# The GPU never went back to sleep. Class reserves capacity from 13:45 to
# 18:10, so anything still running six hours later means the stop job did not
# take effect -- an L4 held overnight is the most expensive failure here.
resource "google_monitoring_alert_policy" "gpu_left_warm" {
  project      = var.project_id
  display_name = "GPU instance held for six hours"
  combiner     = "OR"

  documentation {
    content   = "The analysis service has had an instance for six hours straight. Class reserves one from 13:45 to 18:10 Taipei; outside that window it should scale to zero. Check `analysis-gpu-monday-stop` and the service's minimum instance count."
    mime_type = "text/markdown"
  }

  conditions {
    display_name = "An instance exists continuously"

    condition_threshold {
      filter = join(" AND ", [
        "metric.type=\"run.googleapis.com/container/instance_count\"",
        "resource.type=\"cloud_run_revision\"",
        "resource.label.service_name=\"${google_cloud_run_v2_service.analysis.name}\"",
      ])
      comparison      = "COMPARISON_GT"
      threshold_value = 0
      duration        = "21600s"

      aggregations {
        alignment_period     = "300s"
        per_series_aligner   = "ALIGN_MAX"
        cross_series_reducer = "REDUCE_SUM"
      }
    }
  }

  notification_channels = [google_monitoring_notification_channel.email.id]

  alert_strategy {
    auto_close = "7200s"
  }
}

# Learner requests being refused in bulk. A student hitting their own limit is
# ordinary; fifty refusals in five minutes is either someone probing the API or
# a retry loop in the app.
resource "google_monitoring_alert_policy" "learner_refusals" {
  project      = var.project_id
  display_name = "Learner API refusing requests in bulk"
  combiner     = "OR"

  documentation {
    content   = "Rate limits or authentication are refusing learner requests in bulk. The `limiter` label is empty on authentication failures, which distinguishes a probe from an app retry loop."
    mime_type = "text/markdown"
  }

  conditions {
    display_name = "Fifty refusals in five minutes"

    condition_threshold {
      filter          = "metric.type=\"logging.googleapis.com/user/${google_logging_metric.learner_api_refusals.name}\" AND resource.type=\"cloud_run_revision\""
      comparison      = "COMPARISON_GT"
      threshold_value = 50
      duration        = "0s"

      aggregations {
        alignment_period   = "300s"
        per_series_aligner = "ALIGN_SUM"
      }
    }
  }

  notification_channels = [google_monitoring_notification_channel.email.id]

  alert_strategy {
    auto_close = "3600s"
  }
}

# NT$1,000 a month, mailing at every full multiple of it. It caps nothing; it
# exists because the L4 bills by the second whenever an instance is up.
resource "google_billing_budget" "monthly" {
  provider        = google.billing
  billing_account = var.billing_account_id
  display_name    = "Badminton coaching — NT$1,000 steps"

  budget_filter {
    projects               = ["projects/${data.google_project.this.number}"]
    calendar_period        = "MONTH"
    credit_types_treatment = "INCLUDE_ALL_CREDITS"
  }

  amount {
    specified_amount {
      currency_code = "TWD"
      units         = "1000"
    }
  }

  dynamic "threshold_rules" {
    # 100%, 200%, … 500% of NT$1,000. Extend the range rather than editing a
    # list; each step is one more email, not a new kind of alert.
    for_each = range(1, 6)

    content {
      threshold_percent = threshold_rules.value
      spend_basis       = "CURRENT_SPEND"
    }
  }

  all_updates_rule {
    monitoring_notification_channels = [google_monitoring_notification_channel.email.id]
    # Billing administrators keep getting the mail as well; this adds the
    # address alerts already go to, so spend and failures arrive together.
    disable_default_iam_recipients = false
  }

  depends_on = [google_project_service.enabled]
}
