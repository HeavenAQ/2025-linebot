# This variant's LINE webhook, learner API and analysis job worker.
#
# Terraform creates the service and owns what rarely changes: its identity,
# its region, who may call it. GitHub Actions owns the image and the
# environment, so the template is ignored after creation -- otherwise an apply
# would roll production back to whatever image this file last named.
#
# Public, because LINE has to reach the webhook; every learner route
# authenticates the caller's own LINE credential, and the internal worker
# routes verify a Google-signed OIDC token.
resource "google_cloud_run_v2_service" "bot" {
  project             = var.project_id
  name                = "nstc-linebot-2025-noai"
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = true

  template {
    service_account = data.google_service_account.bot.email
    timeout         = "1200s"

    containers {
      image = var.placeholder_image
    }
  }

  lifecycle {
    ignore_changes = [template, scaling, client, client_version, traffic]
  }
}

resource "google_cloud_run_v2_service_iam_member" "bot_public" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.bot.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
