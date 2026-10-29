from datetime import datetime, timezone

import pytest

import s3_storage


def test_s3_client_uses_default_credential_chain(monkeypatch):
    calls = {}
    monkeypatch.setattr(s3_storage, "_setting", lambda name, default=None: "eu-west-1")
    monkeypatch.setattr(
        s3_storage.boto3,
        "client",
        lambda service, **kwargs: calls.update(service=service, **kwargs) or object(),
    )

    s3_storage.get_s3_client()

    assert calls == {"service": "s3", "region_name": "eu-west-1"}


def test_upload_bytes_encrypts_image_and_uses_private_upload_prefix(monkeypatch):
    calls = {}
    monkeypatch.setattr(s3_storage, "get_s3_bucket", lambda: "private-bucket")

    class FakeS3:
        def put_object(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.setattr(s3_storage, "get_s3_client", lambda: FakeS3())

    key = s3_storage.upload_bytes(b"image-bytes", "xray image.png", "image/png")

    assert key.startswith("uploads/")
    assert calls["Bucket"] == "private-bucket"
    assert calls["Body"] == b"image-bytes"
    assert calls["ContentType"] == "image/png"
    assert calls["ServerSideEncryption"] == "AES256"


def test_upload_bytes_separates_pdf_reports(monkeypatch):
    monkeypatch.setattr(s3_storage, "get_s3_bucket", lambda: "private-bucket")

    class FakeS3:
        def put_object(self, **kwargs):
            pass

    monkeypatch.setattr(s3_storage, "get_s3_client", lambda: FakeS3())

    key = s3_storage.upload_bytes(b"pdf-bytes", "report.pdf", "application/pdf")

    assert key.startswith("reports/")


def test_list_files_identifies_reports_by_prefix(monkeypatch):
    monkeypatch.setattr(s3_storage, "get_s3_bucket", lambda: "private-bucket")
    modified = datetime(2026, 1, 2, tzinfo=timezone.utc)

    class FakeS3:
        def list_objects_v2(self, **kwargs):
            assert kwargs == {"Bucket": "private-bucket", "MaxKeys": 50}
            return {
                "Contents": [
                    {"Key": "uploads/report-image.jpg", "LastModified": modified},
                    {"Key": "reports/analysis.pdf", "LastModified": modified},
                ]
            }

    monkeypatch.setattr(s3_storage, "get_s3_client", lambda: FakeS3())

    assert s3_storage.list_files() == [
        {"type": "image", "name": "report-image.jpg", "date": "2026-01-02 00:00"},
        {"type": "report", "name": "analysis.pdf", "date": "2026-01-02 00:00"},
    ]


def test_missing_bucket_has_actionable_error(monkeypatch):
    monkeypatch.setattr(s3_storage, "_setting", lambda name, default=None: None)

    with pytest.raises(RuntimeError, match="S3_BUCKET_NAME"):
        s3_storage.get_s3_bucket()
