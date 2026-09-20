# This variant's own .env: its LINE channel, its database, its bucket, its
# storage prefix. The shared secrets -- the gRPC API key and the OpenAI key
# the GPU service uses for the other product -- are declared on `main`.
#
# The container only. A value written from Terraform would be readable in
# state and in every plan; add versions with `gcloud secrets versions add`.
resource "google_secret_manager_secret" "bot_env" {
  project   = var.project_id
  secret_id = "2025-linebot-noai-env"

  replication {
    auto {}
  }

  lifecycle {
    prevent_destroy = true
  }
}
