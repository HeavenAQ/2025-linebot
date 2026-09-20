# The LLM product's learner media. The no-LLM variant has its own bucket,
# declared on its own branch, so a mistake in one cannot touch the other's
# recordings. Private: everything is served through V4 signed URLs minted by
# the Go backend.
resource "google_storage_bucket" "learner_media" {
  project                     = var.project_id
  name                        = "nstc-2025-storage"
  location                    = upper(var.region)
  storage_class               = "STANDARD"
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  # Student recordings are the study's raw data: nothing here deletes them on a
  # timer. Add a lifecycle rule deliberately, after the study, if ever.

  versioning {
    enabled = false
  }

  depends_on = [google_project_service.enabled]
}

# Terraform's own state. Created by hand before the first apply (see README),
# so this block exists to keep it described and versioned rather than to make
# it: adopt it with `terraform import`.
resource "google_storage_bucket" "terraform_state" {
  project                     = var.project_id
  name                        = "${var.project_id}-tfstate"
  location                    = upper(var.region)
  storage_class               = "STANDARD"
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  # State history is the only way back from a bad apply.
  versioning {
    enabled = true
  }

  lifecycle {
    prevent_destroy = true
  }
}
