# The LLM product's default database; the variant's is named, on its branch, so
# neither can read the other's students. Collections are the bot's: Firestore
# creates them on first write.
resource "google_firestore_database" "llm" {
  project                 = var.project_id
  name                    = "(default)"
  location_id             = var.region
  type                    = "FIRESTORE_NATIVE"
  delete_protection_state = "DELETE_PROTECTION_ENABLED"

  lifecycle {
    prevent_destroy = true
  }

  depends_on = [google_project_service.enabled]
}
