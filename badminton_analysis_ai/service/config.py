from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    port: int
    max_video_bytes: int
    grpc_api_key: str
    gcp_project_id: str
    gcs_bucket_name: str
    gcp_service_account_email: str
    signed_url_minutes: int
    expert_motion_model_root: Path
    device: str

    @classmethod
    def from_env(cls) -> "Settings":
        root = Path(__file__).resolve().parents[1]
        values = cls(
            port=int(os.getenv("PORT", "8080")),
            max_video_bytes=int(os.getenv("MAX_VIDEO_BYTES", str(150 * 1024 * 1024))),
            grpc_api_key=os.getenv("ANALYSIS_GRPC_API_KEY", ""),
            gcp_project_id=os.getenv("GCP_PROJECT_ID", ""),
            gcs_bucket_name=os.getenv("GCS_BUCKET_NAME", ""),
            gcp_service_account_email=os.getenv("GCP_SERVICE_ACCOUNT_EMAIL", ""),
            signed_url_minutes=int(os.getenv("SIGNED_URL_MINUTES", "60")),
            expert_motion_model_root=Path(
                os.getenv(
                    "EXPERT_MOTION_MODEL_ROOT",
                    str(root / "models" / "error_isolated_motion"),
                )
            ),
            device=os.getenv(
                "EXPERT_MOTION_DEVICE", os.getenv("SKELETON_DEVICE", "auto")
            ),
        )
        missing = [
            name
            for name, value in (
                ("ANALYSIS_GRPC_API_KEY", values.grpc_api_key),
                ("GCP_PROJECT_ID", values.gcp_project_id),
                ("GCS_BUCKET_NAME", values.gcs_bucket_name),
            )
            if not value
        ]
        if missing:
            raise ValueError(f"missing required environment variables: {', '.join(missing)}")
        if values.signed_url_minutes < 1 or values.signed_url_minutes > 10080:
            raise ValueError("SIGNED_URL_MINUTES must be between 1 and 10080")
        return values
