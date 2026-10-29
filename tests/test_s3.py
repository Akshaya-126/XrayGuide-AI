
import sys
from pathlib import Path

import streamlit as st

# Add the project root to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from s3_storage import get_s3_bucket, get_s3_client

st.title("XrayGuide AI — S3 Connection Test")

if st.button("Test private S3 connection"):
    try:
        get_s3_client().head_bucket(
            Bucket=get_s3_bucket()
        )
        st.success("S3 bucket connection successful.")
    except Exception:
        st.error(
            "Connection failed. Check Streamlit Secrets, "
            "the bucket name, region, and IAM permissions."
        )
