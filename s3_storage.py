import os
import uuid
from datetime import datetime, timezone

import boto3
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError


def _setting(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value:
        return value

    try:
        return st.secrets.get(name, default)
    except StreamlitSecretNotFoundError:
        return default


def get_s3_bucket() -> str:
    bucket = _setting("S3_BUCKET_NAME") or _setting("AWS_S3_BUCKET")
    if not bucket:
        raise RuntimeError(
            "S3 is not configured. Set S3_BUCKET_NAME to enable cloud storage."
        )
    return bucket


def get_s3_region() -> str:
    return (
        _setting("AWS_REGION")
        or _setting("AWS_DEFAULT_REGION")
        or "us-east-1"
    )


def get_s3_client():
    # boto3 uses its standard credential chain, including IAM roles and profiles.
    return boto3.client("s3", region_name=get_s3_region())


def upload_bytes(data: bytes, filename: str, content_type: str) -> str:
    """Upload bytes to the private S3 bucket and return the object key."""
    bucket = get_s3_bucket()

    safe_name = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    safe_name = safe_name.replace(" ", "_")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = "reports" if content_type == "application/pdf" else "uploads"
    object_key = f"{prefix}/{timestamp}_{uuid.uuid4().hex}_{safe_name}"

    get_s3_client().put_object(
        Bucket=bucket,
        Key=object_key,
        Body=data,
        ContentType=content_type,
        ServerSideEncryption="AES256",
    )

    return object_key


def list_files():
    """List stored files from the private S3 bucket."""
    bucket = get_s3_bucket()
    s3 = get_s3_client()

    response = s3.list_objects_v2(Bucket=bucket, MaxKeys=50)
    files = []

    if "Contents" in response:
        for obj in response["Contents"]:
            key = obj["Key"]
            last_modified = obj["LastModified"].strftime("%Y-%m-%d %H:%M")
            lower_key = key.lower()
            file_type = (
                "report"
                if lower_key.startswith("reports/") or lower_key.endswith(".pdf")
                else "image"
            )
            files.append({
                "type": file_type,
                "name": key.split("/")[-1],
                "date": last_modified
            })

    return files