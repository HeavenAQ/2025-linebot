# How GitHub Actions becomes the deploy service account.
#
# This is the most security-relevant thing in the project, and until now it
# existed only in the console. The service account these bindings hand out has
# `storage.admin` and `datastore.user`: whoever can mint a token as it can read
# every learner's recordings and scores. What stops that from being anyone with
# a GitHub account is one line -- the provider's attribute condition -- so that
# line belongs in review, not in a web form.
#
# No key file is involved anywhere. GitHub presents its own OIDC token, this
# provider decides whether to believe it, and the binding below says which
# repository may exchange it for the service account.

data "google_project" "this" {
  project_id = var.project_id
}

resource "google_iam_workload_identity_pool" "github" {
  project                   = var.project_id
  workload_identity_pool_id = "git-actions-pool"
  display_name              = "Git Actions Pool"

  lifecycle {
    # A deleted pool is recoverable for 30 days and its ID cannot be reused
    # before then, which would leave every deploy unable to authenticate.
    prevent_destroy = true
  }

  depends_on = [google_project_service.enabled]
}

resource "google_iam_workload_identity_pool_provider" "github" {
  project                            = var.project_id
  workload_identity_pool_id          = google_iam_workload_identity_pool.github.workload_identity_pool_id
  workload_identity_pool_provider_id = "git-actions-provider"
  display_name                       = "GitHub Identity Provider"

  # Without this condition the provider would accept a token from any GitHub
  # repository in the world, and the binding below would hand each of them the
  # service account. Narrow it further (a branch, an environment) by adding to
  # the expression -- never widen it.
  attribute_condition = "assertion.repository=='${var.github_repository}'"

  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.actor"      = "assertion.actor"
    "attribute.repository" = "assertion.repository"
    "attribute.ref"        = "assertion.ref"
  }

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }

  lifecycle {
    prevent_destroy = true
  }
}

# The repository, and nothing else, may act as the deploy service account.
# `workloadIdentityUser` is what lets the exchange happen at all;
# `serviceAccountTokenCreator` is what lets the workflow mint the identity
# tokens it sends to the GPU service during the end-to-end test.
locals {
  github_principal = "principalSet://iam.googleapis.com/projects/${data.google_project.this.number}/locations/global/workloadIdentityPools/${google_iam_workload_identity_pool.github.workload_identity_pool_id}/attribute.repository/${var.github_repository}"
}

resource "google_service_account_iam_member" "github_impersonates_bot" {
  service_account_id = google_service_account.bot.name
  role               = "roles/iam.workloadIdentityUser"
  member             = local.github_principal
}

resource "google_service_account_iam_member" "github_mints_tokens_as_bot" {
  service_account_id = google_service_account.bot.name
  role               = "roles/iam.serviceAccountTokenCreator"
  member             = local.github_principal
}
