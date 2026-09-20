# This variant's learner media, separate from the LLM product's bucket so a
# mistake in one cannot touch the other's recordings. Private: everything is
# served through V4 signed URLs minted by the Go backend.
resource "google_storage_bucket" "learner_media" {
  project                     = var.project_id
  name                        = "nstc-2025-storage-noai"
  location                    = upper(var.region)
  storage_class               = "STANDARD"
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  # Student recordings are the study's raw data: nothing here deletes them on
  # a timer.
}
