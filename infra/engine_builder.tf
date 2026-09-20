# Builds the RF-DETR TensorRT engine on the same L4 the service runs on: an
# engine is bound to the GPU, driver and TensorRT version it was compiled
# against. Terraform cannot build the image this job runs, so the recipe is
# `cloudbuild/engine-bootstrap.yaml`; README.md has the two commands, and the
# flag stays false until that image exists, because Cloud Run checks for it.
resource "google_cloud_run_v2_job" "engine_builder" {
  count = var.create_engine_builder_job ? 1 : 0

  project             = var.project_id
  name                = "rfdetr-engine-builder"
  location            = var.gpu_region
  deletion_protection = false

  template {
    template {
      service_account = google_service_account.bot.email
      # Compiling is minutes, not hours; a stuck build should stop rather than
      # hold an L4.
      timeout     = "3600s"
      max_retries = 0

      containers {
        image   = "gcr.io/${var.project_id}/badminton-analysis:engine-builder"
        command = ["python", "build_rfdetr_engine.py"]

        resources {
          limits = {
            cpu              = "4"
            memory           = "16Gi"
            "nvidia.com/gpu" = "1"
          }
        }
      }

      node_selector {
        accelerator = "nvidia-l4"
      }
    }
  }

  lifecycle {
    # The image tag is rebuilt in place by the bootstrap build above.
    ignore_changes = [template[0].template[0].containers[0].image, client, client_version]
  }

  depends_on = [google_project_service.enabled]
}
