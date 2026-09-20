# This variant's learners live in their own named database, so the two
# products can never read each other's students.
#
# Collections are not declared: Firestore creates them on first write, and the
# bot owns their shape.
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
