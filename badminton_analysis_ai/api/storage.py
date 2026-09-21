from __future__ import annotations

import mimetypes
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import google.auth
from google.auth.exceptions import TransportError
from google.auth.transport.requests import Request
from google.cloud import storage

# Signing goes through the IAM signBytes API, which answers 5xx now and then.
# Without a retry that blip throws away an analysis the GPU has already run.
SIGN_ATTEMPTS = 3
SIGN_RETRY_SECONDS = 0.2


@dataclass(frozen=True)
class SignedObject:
    object_path: str
    gcs_uri: str
    signed_url: str
    expires_at_unix: int


class ObjectStorage:
    def __init__(
        self,
        project_id: str,
        bucket_name: str,
        *,
        service_account_email: str = "",
        signed_url_minutes: int = 60,
    ) -> None:
        self.client = storage.Client(project=project_id)
        self.bucket = self.client.bucket(bucket_name)
        self.bucket_name = bucket_name
        self.service_account_email = service_account_email
        self.signed_url_minutes = signed_url_minutes

    def upload_file(
        self, source: Path, object_path: str, *, content_type: str | None = None
    ) -> SignedObject:
        blob = self.bucket.blob(object_path)
        resolved_type = content_type or mimetypes.guess_type(source.name)[0]
        blob.upload_from_filename(str(source), content_type=resolved_type)
        return self.sign(object_path)

    def sign(self, object_path: str) -> SignedObject:
        blob = self.bucket.blob(object_path)
        expires_at = datetime.now(timezone.utc) + timedelta(
            minutes=self.signed_url_minutes
        )
        kwargs: dict[str, object] = {
            "version": "v4",
            "expiration": expires_at,
            "method": "GET",
            # Older expert objects have no GCS Content-Type. LIFF's embedded
            # browser does not reliably sniff an octet-stream as video.
            "query_parameters": {
                "response-content-type": (
                    "video/quicktime"
                    if object_path.lower().endswith(".mov")
                    else "video/mp4"
                )
            },
        }
        credentials, _ = google.auth.default()
        if hasattr(credentials, "sign_bytes"):
            kwargs["credentials"] = credentials
        else:
            credentials.refresh(Request())
            service_account_email = self.service_account_email or getattr(
                credentials, "service_account_email", ""
            )
            if not service_account_email or not credentials.token:
                raise RuntimeError(
                    "GCP_SERVICE_ACCOUNT_EMAIL is required for IAM signed URLs"
                )
            kwargs["service_account_email"] = service_account_email
            kwargs["access_token"] = credentials.token
        return SignedObject(
            object_path=object_path,
            gcs_uri=f"gs://{self.bucket_name}/{object_path}",
            signed_url=self._signed_url(blob, kwargs),
            expires_at_unix=int(expires_at.timestamp()),
        )

    @staticmethod
    def _signed_url(blob: storage.Blob, kwargs: dict[str, object]) -> str:
        for attempt in range(SIGN_ATTEMPTS):
            try:
                return blob.generate_signed_url(**kwargs)
            except TransportError:
                if attempt == SIGN_ATTEMPTS - 1:
                    raise
                time.sleep(SIGN_RETRY_SECONDS * (attempt + 1))
        raise AssertionError("unreachable")
