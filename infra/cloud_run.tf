# Terraform owns each service's existence, region, ingress and invokers; the
# deploy workflow owns every revision. `template` is ignored after creation, so
# an apply can never roll production back to the placeholder image below.

# The LLM product's webhook, learner API and job worker; the variant's bot is
# declared on its own branch. Public because LINE must reach the webhook --
# learner routes check a LINE credential, worker routes a Google OIDC token.
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
    ignore_changes = [template, scaling, client, client_version, traffic]
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

# The GPU analysis service, shared by both products: safe because coaching --
# the only stage that leaves the machine -- is switched off per request by the
# no-LLM deployment. One instance at concurrency four is deliberate: the pose
# model batches concurrent analyses into one TensorRT batch, so a second
# instance would buy a second cold start rather than throughput.
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
    ignore_changes = [template, scaling, client, client_version, traffic]
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
  project  = var.project_id
  name     = "gpt-validation"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"
  # The study outlives the site, but the site holds the experts' only way in
  # while they are reviewing; take it down on purpose, not on an apply.
  deletion_protection = true

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
    ignore_changes = [template, scaling, client, client_version, traffic]
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
