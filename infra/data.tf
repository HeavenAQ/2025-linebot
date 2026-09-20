# Owned by the LLM product's configuration on `main`, read here.
#
# The two products share one identity, one GPU analysis service and one
# Artifact Registry; they share nothing that holds a learner's data. Declaring
# the shared pieces twice would mean two configurations fighting over them, so
# this branch only ever reads them.
data "google_service_account" "bot" {
  project    = var.project_id
  account_id = var.service_account_id
}
