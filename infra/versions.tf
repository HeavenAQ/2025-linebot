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

  # Same bucket as the LLM product, different prefix: one project, two
  # configurations that must never write each other's state.
  backend "gcs" {
    bucket = "nstc-linebot-2025-tfstate"
    prefix = "infra-no-llm"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}
