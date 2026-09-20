# Terraform and provider versions are pinned: an infrastructure change should
# come from this repository, never from a provider upgrade nobody asked for.
terraform {
  required_version = ">= 1.9"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }

  # State lives in its own bucket so two people (or a laptop and CI) cannot
  # apply at once. Create that bucket first -- see README.md -- then run
  # `terraform init -migrate-state`.
  backend "gcs" {
    bucket = "nstc-linebot-2025-tfstate"
    prefix = "infra"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# The budget is the one resource billed against a *billing account* rather than
# the project. Requests for it must name a project to charge the API quota to,
# which user credentials do not carry by default -- without this the API replies
# that the service is disabled on a project nobody recognises. Scoped to an
# alias so every other resource keeps the plain provider, which needs no extra
# permission.
provider "google" {
  alias                 = "billing"
  project               = var.project_id
  region                = var.region
  user_project_override = true
  billing_project       = var.project_id
}
