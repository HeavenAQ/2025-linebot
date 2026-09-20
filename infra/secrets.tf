# Containers only -- a value written here would be readable in state and in
# every plan. Add versions with `gcloud secrets versions add`; README.md says
# what each holds. The variant's own .env is declared on its branch.
resource "google_secret_manager_secret" "secrets" {
  for_each = toset([
    # The whole .env the LLM bot downloads at startup (LINE channel
    # credentials, Firestore database and collection names, OpenAI key).
    "2025-linebot-env",
    # Shared between the bots and the GPU service, so only the bots may call it.
    "analysis-grpc-api-key",
    # Read by the GPU service for the coaching pass.
    "openai-api-key",
    # The expert review site for the feedback validation study.
    "gpt-validation-access-codes",
  ])

  project   = var.project_id
  secret_id = each.value

  replication {
    auto {}
  }

  lifecycle {
    prevent_destroy = true
  }

  depends_on = [google_project_service.enabled]
}
