# One service account for everything: the variants are kept apart by their own
# databases, buckets and prefixes, not by identity.
resource "google_service_account" "bot" {
  project      = var.project_id
  account_id   = var.service_account_id
  display_name = "NSTC LINE Bot (2025)"
  description  = "Runs the LINE bots, the GPU analysis service and their scheduled jobs."

  depends_on = [google_project_service.enabled]
}

# Project-level roles. run.invoker is granted per service, in cloud_run.tf.
locals {
  project_roles = [
    "roles/artifactregistry.writer",      # the deploy workflows push images
    "roles/cloudbuild.builds.editor",     # and build them
    "roles/cloudtasks.enqueuer",          # the bot publishes its own analysis jobs
    "roles/datastore.user",               # Firestore
    "roles/iam.serviceAccountUser",       # act as itself to deploy and mint OIDC tokens
    "roles/run.admin",                    # the capacity jobs move the GPU's minimum instances
    "roles/secretmanager.secretAccessor", # the .env each bot loads at startup
    "roles/storage.admin",                # learner media
  ]
}

resource "google_project_iam_member" "bot" {
  for_each = toset(local.project_roles)

  project = var.project_id
  role    = each.value
  member  = google_service_account.bot.member
}

# Signing playback URLs without a downloaded key: the account signs as itself
# through the IAM credentials API, so no private key file has to exist.
resource "google_service_account_iam_member" "bot_signs_as_itself" {
  service_account_id = google_service_account.bot.name
  role               = "roles/iam.serviceAccountTokenCreator"
  member             = google_service_account.bot.member
}
