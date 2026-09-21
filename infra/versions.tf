# Terraform and provider versions are pinned: an infrastructure change should
# come from this repository, never from a provider upgrade nobody asked for.
terraform {
  required_version = ">= 1.9"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.4"
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

# Only the budget needs this: requests to the billing API must name a project
# to charge quota to, which user credentials do not carry by default.
provider "google" {
  alias                 = "billing"
  project               = var.project_id
  region                = var.region
  user_project_override = true
  billing_project       = var.project_id
}
