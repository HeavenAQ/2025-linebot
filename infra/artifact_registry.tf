# Images and the TensorRT engine; Cloud Storage holds learner data only. The
# engine is a generic artifact built once on an L4 and baked into each image,
# because building it at startup would sit on every cold start.
resource "google_artifact_registry_repository" "model_artifacts" {
  project       = var.project_id
  location      = var.gpu_region
  repository_id = "model-artifacts"
  format        = "GENERIC"
  description   = "RF-DETR TensorRT engines, pinned per GPU architecture and TensorRT version."

  depends_on = [google_project_service.enabled]
}
