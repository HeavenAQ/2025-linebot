# Container images and the TensorRT engine both live in Artifact Registry;
# Cloud Storage holds learner data only.
#
# The engine is a generic artifact rather than a layer in the image because it
# is built once on an L4 (about two minutes) and then baked into every image
# by the deploy workflow. Building it at startup would add that to every cold
# start on a service that scales to zero.
resource "google_artifact_registry_repository" "model_artifacts" {
  project       = var.project_id
  location      = var.gpu_region
  repository_id = "model-artifacts"
  format        = "GENERIC"
  description   = "RF-DETR TensorRT engines, pinned per GPU architecture and TensorRT version."

  depends_on = [google_project_service.enabled]
}
