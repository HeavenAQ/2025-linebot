# Builds the two pose TensorRT engines -- RF-DETR Medium for the player's box
# and ViTPose++-L for the joints -- on the same L4 the service runs on: an
# engine is bound to the GPU, driver and TensorRT version it was compiled
# against. Terraform cannot build the image this job runs, so the recipe is
# `cloudbuild/engine-bootstrap.yaml`; README.md has the two commands, and the
# flag stays false until that image exists, because Cloud Run checks for it.
resource "google_cloud_run_v2_job" "engine_builder" {
  count = var.create_engine_builder_job ? 1 : 0

  project             = var.project_id
  name                = "pose-engine-builder"
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
        command = ["python", "build_pose_engines.py"]

        # Where the builder writes the engines before publishing them, and the
        # same path the service reads them from, so the detector's GPU-named
        # subdirectory is identical on both sides.
        env {
          name  = "BADMINTON_TRT_CACHE_DIR"
          value = "/app/models/trt-engines"
        }

        env {
          name  = "GCP_PROJECT_ID"
          value = var.project_id
        }

        resources {
          limits = {
            # Exporting ViTPose to ONNX before compiling it needs more headroom
            # than the detector alone did, and Cloud Run ties the memory
            # ceiling to the CPU count: 4 CPU tops out at 16Gi.
            cpu              = "8"
            memory           = "24Gi"
            "nvidia.com/gpu" = "1"
          }
        }
      }

      # Cloud Run cannot offer GPU jobs with zonal redundancy, the same
      # constraint the GPU service runs under.
      gpu_zonal_redundancy_disabled = true

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
