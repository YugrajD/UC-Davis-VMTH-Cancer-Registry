"""The real boto3 S3 client (default credential chain, so AWS_PROFILE works). Kept apart so tests inject a fake."""

from __future__ import annotations

import config


def make_client():
    import boto3

    return boto3.client("s3", region_name=config.S3_REGION)
