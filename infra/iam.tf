# One service account runs the bots, the analysis service and every scheduled
# job. A second identity per service would buy isolation the deployment does
# not have anyway: the two product variants already share this account and are
# kept apart by their own databases, buckets and prefixes.
resource "google_service_account" "bot" {
  project      = var.project_id
  account_id   = var.service_account_id
  display_name = "NSTC LINE Bot (2025)"
  description  = "Runs the LINE bots, the GPU analysis service and their scheduled jobs."

  depends_on = [google_project_service.enabled]
}

# Project-level roles. Each one is here because something breaks without it:
#
#   run.admin            the capacity scheduler raises and drops the GPU
#                        service's minimum instances around class
#   run.invoker          is granted per service below, not project-wide
#   cloudtasks.enqueuer  the bot publishes its own analysis jobs
#   datastore.user       Firestore reads and writes
#   storage.admin        learner media in the variant buckets
#   secretmanager.secretAccessor  the .env each bot loads at startup
#   iam.serviceAccountUser        lets the account act as itself when
#                        deploying revisions and minting OIDC tokens
#   artifactregistry.writer, cloudbuild.builds.editor
#                        the deploy workflows build and push images
locals {
  project_roles = [
    "roles/artifactregistry.writer",
    "roles/cloudbuild.builds.editor",
    "roles/cloudtasks.enqueuer",
    "roles/datastore.user",
    "roles/iam.serviceAccountUser",
    "roles/run.admin",
    "roles/secretmanager.secretAccessor",
    "roles/storage.admin",
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
