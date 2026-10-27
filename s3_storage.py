
import uuid
from datetime import datetime, timezone

import boto3
import streamlit as st


def get_s3_client():
    return boto3.client(
        "s3",
        region_name=st.secrets["AWS_REGION"],
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
    )


def upload_bytes(data: bytes, filename: str, content_type: str) -> str:
    """Upload bytes to the private S3 bucket and return the object key."""
    bucket = st.secrets["S3_BUCKET_NAME"]
    safe_name = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    safe_name = safe_name.replace(" ", "_")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    object_key = f"uploads/{timestamp}_{uuid.uuid4().hex}_{safe_name}"

    get_s3_client().put_object(
        Bucket=bucket,
        Key=object_key,
        Body=data,
        ContentType=content_type,
        ServerSideEncryption="AES256",
    )

    return object_key
