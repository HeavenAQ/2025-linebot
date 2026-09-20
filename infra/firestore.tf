# This variant's own named database, so neither product can read the other's
# students. Collections are the bot's: Firestore creates them on first write.
resource "google_firestore_database" "no_llm" {
  project                 = var.project_id
  name                    = "nstc-linebot-noai"
  location_id             = var.region
  type                    = "FIRESTORE_NATIVE"
  delete_protection_state = "DELETE_PROTECTION_ENABLED"

  lifecycle {
    prevent_destroy = true
  }
}
