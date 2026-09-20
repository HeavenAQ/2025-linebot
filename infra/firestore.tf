# The LLM product's learners live in the project's default database. The
# no-LLM variant has its own named database, declared on its own branch, so
# the two can never read each other's students.
#
# Collections are not declared: Firestore creates them on first write, and the
# bot owns their shape.
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
