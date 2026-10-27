
import streamlit as st
from s3_storage import get_s3_client

st.title("XrayGuide AI — S3 connection test")

if st.button("Test private S3 connection"):
    try:
        response = get_s3_client().head_bucket(
            Bucket=st.secrets["S3_BUCKET_NAME"]
        )
        st.success("S3 bucket connection successful.")
    except Exception:
        st.error(
            "Connection failed. Check the Streamlit Secrets, "
            "bucket name, region, and IAM permissions."
        )
