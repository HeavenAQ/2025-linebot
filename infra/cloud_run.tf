# This variant's webhook, learner API and job worker. Terraform owns the
# service's existence, region, ingress and invokers; the deploy workflow owns
# every revision, so `template` is ignored after creation. Public because LINE
# must reach the webhook -- learner routes check a LINE credential, worker
# routes a Google OIDC token.
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
