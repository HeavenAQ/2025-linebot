# Only the APIs this project actually calls. `disable_on_destroy = false`
# because tearing down this configuration should not break anything else that
# happens to share the project.
resource "google_project_service" "enabled" {
  for_each = toset([
    "artifactregistry.googleapis.com",
    "billingbudgets.googleapis.com",
    "cloudbuild.googleapis.com",
    "cloudscheduler.googleapis.com",
    "cloudtasks.googleapis.com",
    "firestore.googleapis.com",
    "iam.googleapis.com",
    "iamcredentials.googleapis.com",
    "logging.googleapis.com",
    "monitoring.googleapis.com",
    "run.googleapis.com",
    "secretmanager.googleapis.com",
    "storage.googleapis.com",
  ])

  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}
