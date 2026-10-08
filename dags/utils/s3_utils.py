"""
Generic S3/MinIO utility functions for MLOps pipeline.
"""

import io
import os

import boto3
import pandas as pd
from botocore.client import Config
from botocore.exceptions import ClientError


def get_s3_client():
    """
    Create and return an S3/MinIO client with proper configuration.

    Uses environment variables:
    - MLFLOW_S3_ENDPOINT_URL: MinIO/S3 endpoint
    - AWS_ACCESS_KEY_ID: Access key
    - AWS_SECRET_ACCESS_KEY: Secret key

    Returns:
        boto3.client: Configured S3 client
    """
    minio_url = os.environ["MLFLOW_S3_ENDPOINT_URL"]
    access_key = os.environ["AWS_ACCESS_KEY_ID"]
    secret_key = os.environ["AWS_SECRET_ACCESS_KEY"]

    print(f"Conectando a S3/MinIO en {minio_url} con usuario {access_key}")
    return boto3.client(
        "s3",
        endpoint_url=minio_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def s3_file_exists(s3_client, bucket: str, file_key: str) -> bool:
    """
    Check if a file exists in S3/MinIO.

    Args:
        s3_client: Boto3 S3 client
        bucket: Bucket name
        file_key: Object key/path

    Returns:
        bool: True if file exists, False otherwise

    Raises:
        ClientError: If there's a permission or network error (not 404)
    """
    try:
        s3_client.head_object(Bucket=bucket, Key=file_key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise


def s3_to_dataframe(s3_client, bucket: str, key: str) -> pd.DataFrame:
    """
    Download a CSV file from S3/MinIO and return as DataFrame.

    Args:
        s3_client: Boto3 S3 client
        bucket: Bucket name
        key: Object key/path to CSV file

    Returns:
        pd.DataFrame: DataFrame containing CSV data
    """
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))


def dataframe_to_s3(df: pd.DataFrame, s3_client, bucket: str, key: str) -> None:
    """
    Upload a DataFrame as CSV to S3/MinIO.

    Args:
        df: DataFrame to upload
        s3_client: Boto3 S3 client
        bucket: Bucket name
        key: Object key/path for CSV file
    """
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    s3_client.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue().encode("utf-8"))
