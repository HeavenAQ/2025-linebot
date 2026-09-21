from __future__ import annotations

import pytest
from google.auth.exceptions import TransportError

from api import storage as storage_module
from api.storage import ObjectStorage


class _Blob:
    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    def generate_signed_url(self, **_: object) -> str:
        self.calls += 1
        if self.calls <= self.failures:
            raise TransportError("Error calling the IAM signBytes API: 503")
        return "https://signed.test/video.mp4"


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(storage_module.time, "sleep", lambda _: None)


def test_signing_retries_a_transient_iam_failure() -> None:
    blob = _Blob(failures=1)

    assert ObjectStorage._signed_url(blob, {}) == "https://signed.test/video.mp4"
    assert blob.calls == 2


def test_signing_gives_up_after_the_last_attempt() -> None:
    blob = _Blob(failures=storage_module.SIGN_ATTEMPTS)

    with pytest.raises(TransportError):
        ObjectStorage._signed_url(blob, {})
    assert blob.calls == storage_module.SIGN_ATTEMPTS
