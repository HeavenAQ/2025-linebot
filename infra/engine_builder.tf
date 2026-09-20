# The TensorRT engine builder.
#
# A serialized engine is bound to the exact GPU architecture, driver and
# TensorRT version it was built against, so it has to be produced on the same
# L4 the service runs on. It is built rarely -- when RF-DETR, TensorRT or the
# GPU generation changes -- and takes about two minutes, after which the
# artifact is published to Artifact Registry and baked into every image.
#
# Terraform declares the job so that run is reproducible rather than assembled
# by hand each time. It does not declare how the image is built: the recipe
# lives with the code it builds, in
# `badminton_analysis_ai/cloudbuild-engine-bootstrap.yaml`, which produces the
# same image as production with the engine assertion switched off (it cannot
# hold before the engine exists).
#
# To rebuild an engine:
#
#   cd badminton_analysis_ai
#   gcloud builds submit --config cloudbuild-engine-bootstrap.yaml \
#     --project <project>
#   gcloud run jobs execute rfdetr-engine-builder --region <gpu region> \
#     --project <project> --wait
#
# The job uploads the engine itself; see `build_rfdetr_engine.py` and
# `models/trt-engine.env` for the artifact version it publishes.
resource "google_cloud_run_v2_job" "engine_builder" {
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
