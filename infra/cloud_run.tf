# Cloud Run services.
#
# Terraform creates each service and owns what rarely changes: its identity,
# its region, who may call it, and the shape of the machine. GitHub Actions
# owns what changes on every push -- the image and the environment it is
# deployed with -- so every `template` here is ignored after creation. Without
# that, `terraform apply` would roll production back to whatever image this
# file last named.
#
# On a fresh project the placeholder image starts, the deploy workflow replaces
# it, and Terraform stops looking.

# The LLM product's LINE webhook, learner API and analysis job worker. The
# no-LLM bot is its own service, declared on its own branch.
#
# Public, because LINE has to reach the webhook; every learner route
# authenticates the caller's own LINE credential, and the internal worker
# routes verify a Google-signed OIDC token.
resource "google_cloud_run_v2_service" "bot" {
  project             = var.project_id
  name                = "nstc-linebot-2025"
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = true

  template {
    service_account = google_service_account.bot.email
    timeout         = "1200s"

    containers {
      image = var.placeholder_image
    }
  }

  lifecycle {
    ignore_changes = [template, client, client_version, traffic]
  }

  depends_on = [google_project_service.enabled]
}

resource "google_cloud_run_v2_service_iam_member" "bot_public" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.bot.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# The GPU analysis service, shared by both products and owned here.
#
# Sharing it is safe because coaching -- the only stage that sends anything to
# a third party -- is switched off per request by the no-LLM deployment.
# Private: only the bots' service account may call it, on top of the API key
# the gRPC handlers check themselves.
#
# One instance with four concurrent requests is deliberate. The pose model
# batches frames from concurrent analyses into one fixed-size TensorRT batch,
# and a second instance would mean a second cold start and a second engine load
# rather than more throughput.
resource "google_cloud_run_v2_service" "analysis" {
  project             = var.project_id
  name                = "badminton-analysis-ai"
  location            = var.gpu_region
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = true

  template {
    service_account                  = google_service_account.bot.email
    timeout                          = "3600s"
    max_instance_request_concurrency = 4
    gpu_zonal_redundancy_disabled    = true

    scaling {
      min_instance_count = 0
      max_instance_count = 1
    }

    node_selector {
      accelerator = "nvidia-l4"
    }

    containers {
      image = var.placeholder_image

      resources {
        limits = {
          cpu              = "4"
          memory           = "16Gi"
          "nvidia.com/gpu" = "1"
        }
        # A GPU that is throttled between requests reloads its engine on the
        # next one, so CPU stays allocated for the life of the instance.
        cpu_idle          = false
        startup_cpu_boost = true
      }
    }
  }

  lifecycle {
    ignore_changes = [template, client, client_version, traffic]
  }

  depends_on = [google_project_service.enabled]
}

resource "google_cloud_run_v2_service_iam_member" "analysis_invoker" {
  project  = var.project_id
  location = var.gpu_region
  name     = google_cloud_run_v2_service.analysis.name
  role     = "roles/run.invoker"
  member   = google_service_account.bot.member
}

# The expert review site for the GPT feedback validation study. Public, gated
# by the access codes in Secret Manager; it scales to zero between reviews.
resource "google_cloud_run_v2_service" "gpt_validation" {
  project             = var.project_id
  name                = "gpt-validation"
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = false

  template {
    service_account = google_service_account.bot.email

    scaling {
      min_instance_count = 0
      max_instance_count = 2
    }

    containers {
      image = var.placeholder_image
    }
  }

  lifecycle {
    ignore_changes = [template, client, client_version, traffic]
  }

  depends_on = [google_project_service.enabled]
}

resource "google_cloud_run_v2_service_iam_member" "gpt_validation_public" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.gpt_validation.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
