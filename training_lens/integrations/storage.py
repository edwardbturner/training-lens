"""Storage backends for checkpoints and artifacts."""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..utils.helpers import ensure_dir, load_file, safe_save
from ..utils.logging import get_logger

logger = get_logger(__name__)


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def save_checkpoint(self, checkpoint_data: Dict[str, Any], checkpoint_id: str) -> str:
        """Save a checkpoint.

        Args:
            checkpoint_data: Checkpoint data to save
            checkpoint_id: Unique identifier for the checkpoint

        Returns:
            Storage location identifier
        """

    @abstractmethod
    def load_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """Load a checkpoint.

        Args:
            checkpoint_id: Checkpoint identifier

        Returns:
            Loaded checkpoint data
        """

    @abstractmethod
    def list_checkpoints(self) -> List[str]:
        """List available checkpoints.

        Returns:
            List of checkpoint identifiers
        """

    @abstractmethod
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint.

        Args:
            checkpoint_id: Checkpoint identifier

        Returns:
            True if deleted successfully
        """


class LocalStorage(StorageBackend):
    """Local filesystem storage backend."""

    def __init__(self, base_path: Union[str, Path]):
        """Initialize local storage.

        Args:
            base_path: Base directory for storage
        """
        self.base_path = Path(base_path)
        ensure_dir(self.base_path)

        logger.info(f"LocalStorage initialized at {self.base_path}")

    def save_checkpoint(self, checkpoint_data: Dict[str, Any], checkpoint_id: str) -> str:
        """Save checkpoint to local filesystem."""
        checkpoint_path = self.base_path / f"{checkpoint_id}.json"

        try:
            safe_save(checkpoint_data, checkpoint_path, format="json")
            logger.debug(f"Checkpoint {checkpoint_id} saved to {checkpoint_path}")
            return str(checkpoint_path)

        except Exception as e:
            logger.error(f"Failed to save checkpoint {checkpoint_id}: {e}")
            raise

    def load_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """Load checkpoint from local filesystem."""
        checkpoint_path = self.base_path / f"{checkpoint_id}.json"

        try:
            data = load_file(checkpoint_path, format="json")
            logger.debug(f"Checkpoint {checkpoint_id} loaded from {checkpoint_path}")
            return data

        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
            raise

    def list_checkpoints(self) -> List[str]:
        """List available checkpoints."""
        try:
            checkpoint_files = list(self.base_path.glob("*.json"))
            checkpoint_ids = [f.stem for f in checkpoint_files]
            checkpoint_ids.sort()

            logger.debug(f"Found {len(checkpoint_ids)} checkpoints")
            return checkpoint_ids

        except Exception as e:
            logger.error(f"Failed to list checkpoints: {e}")
            return []

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete checkpoint from local filesystem."""
        checkpoint_path = self.base_path / f"{checkpoint_id}.json"

        try:
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                logger.debug(f"Checkpoint {checkpoint_id} deleted")
                return True
            else:
                logger.warning(f"Checkpoint {checkpoint_id} not found")
                return False

        except Exception as e:
            logger.error(f"Failed to delete checkpoint {checkpoint_id}: {e}")
            return False


class S3Storage(StorageBackend):
    """AWS S3 storage backend."""

    def __init__(
        self,
        bucket_name: str,
        prefix: str = "training-lens-checkpoints",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region_name: str = "us-east-1",
    ):
        """Initialize S3 storage.

        Args:
            bucket_name: S3 bucket name
            prefix: Key prefix for checkpoints
            aws_access_key_id: AWS access key ID
            aws_secret_access_key: AWS secret access key
            region_name: AWS region name
        """
        try:
            import boto3  # type: ignore
        except ImportError:
            raise ImportError("boto3 is required for S3 storage. Install with: pip install boto3")

        self.bucket_name = bucket_name
        self.prefix = prefix

        # Initialize S3 client
        session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
        )
        self.s3_client = session.client("s3")

        logger.info(f"S3Storage initialized for bucket {bucket_name}")

    def save_checkpoint(self, checkpoint_data: Dict[str, Any], checkpoint_id: str) -> str:
        """Save checkpoint to S3."""
        key = f"{self.prefix}/{checkpoint_id}.json"

        try:
            # Convert to JSON string
            checkpoint_json = json.dumps(checkpoint_data, indent=2, default=str)

            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=checkpoint_json,
                ContentType="application/json",
            )

            location = f"s3://{self.bucket_name}/{key}"
            logger.debug(f"Checkpoint {checkpoint_id} saved to {location}")
            return location

        except Exception as e:
            logger.error(f"Failed to save checkpoint {checkpoint_id} to S3: {e}")
            raise

    def load_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """Load checkpoint from S3."""
        key = f"{self.prefix}/{checkpoint_id}.json"

        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            checkpoint_json = response["Body"].read().decode("utf-8")
            data = json.loads(checkpoint_json)

            logger.debug(f"Checkpoint {checkpoint_id} loaded from S3")
            return data

        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_id} from S3: {e}")
            raise

    def list_checkpoints(self) -> List[str]:
        """List available checkpoints in S3."""
        try:
            paginator = self.s3_client.get_paginator("list_objects_v2")
            page_iterator = paginator.paginate(Bucket=self.bucket_name, Prefix=self.prefix + "/")

            checkpoint_ids = []
            for page in page_iterator:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        key = obj["Key"]
                        if key.endswith(".json"):
                            # Extract checkpoint ID from key
                            checkpoint_id = Path(key).stem
                            checkpoint_ids.append(checkpoint_id)

            checkpoint_ids.sort()
            logger.debug(f"Found {len(checkpoint_ids)} checkpoints in S3")
            return checkpoint_ids

        except Exception as e:
            logger.error(f"Failed to list checkpoints from S3: {e}")
            return []

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete checkpoint from S3."""
        key = f"{self.prefix}/{checkpoint_id}.json"

        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=key)
            logger.debug(f"Checkpoint {checkpoint_id} deleted from S3")
            return True

        except Exception as e:
            logger.error(f"Failed to delete checkpoint {checkpoint_id} from S3: {e}")
            return False


class GCSStorage(StorageBackend):
    """Google Cloud Storage backend."""

    def __init__(
        self,
        bucket_name: str,
        prefix: str = "training-lens-checkpoints",
        credentials_path: Optional[str] = None,
    ):
        """Initialize GCS storage.

        Args:
            bucket_name: GCS bucket name
            prefix: Object prefix for checkpoints
            credentials_path: Path to service account credentials JSON
        """
        try:
            from google.cloud import storage  # type: ignore
        except ImportError:
            raise ImportError(
                "google-cloud-storage is required for GCS storage. Install with: pip install google-cloud-storage"
            )

        self.bucket_name = bucket_name
        self.prefix = prefix

        # Initialize GCS client
        if credentials_path:
            self.client = storage.Client.from_service_account_json(credentials_path)
        else:
            self.client = storage.Client()  # Use default credentials

        self.bucket = self.client.bucket(bucket_name)

        logger.info(f"GCSStorage initialized for bucket {bucket_name}")

    def save_checkpoint(self, checkpoint_data: Dict[str, Any], checkpoint_id: str) -> str:
        """Save checkpoint to GCS."""
        blob_name = f"{self.prefix}/{checkpoint_id}.json"

        try:
            blob = self.bucket.blob(blob_name)
            checkpoint_json = json.dumps(checkpoint_data, indent=2, default=str)
            blob.upload_from_string(checkpoint_json, content_type="application/json")

            location = f"gs://{self.bucket_name}/{blob_name}"
            logger.debug(f"Checkpoint {checkpoint_id} saved to {location}")
            return location

        except Exception as e:
            logger.error(f"Failed to save checkpoint {checkpoint_id} to GCS: {e}")
            raise

    def load_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """Load checkpoint from GCS."""
        blob_name = f"{self.prefix}/{checkpoint_id}.json"

        try:
            blob = self.bucket.blob(blob_name)
            checkpoint_json = blob.download_as_text()
            data = json.loads(checkpoint_json)

            logger.debug(f"Checkpoint {checkpoint_id} loaded from GCS")
            return data

        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_id} from GCS: {e}")
            raise

    def list_checkpoints(self) -> List[str]:
        """List available checkpoints in GCS."""
        try:
            blobs = self.client.list_blobs(self.bucket_name, prefix=self.prefix + "/")

            checkpoint_ids = []
            for blob in blobs:
                if blob.name.endswith(".json"):
                    checkpoint_id = Path(blob.name).stem
                    checkpoint_ids.append(checkpoint_id)

            checkpoint_ids.sort()
            logger.debug(f"Found {len(checkpoint_ids)} checkpoints in GCS")
            return checkpoint_ids

        except Exception as e:
            logger.error(f"Failed to list checkpoints from GCS: {e}")
            return []

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete checkpoint from GCS."""
        blob_name = f"{self.prefix}/{checkpoint_id}.json"

        try:
            blob = self.bucket.blob(blob_name)
            blob.delete()
            logger.debug(f"Checkpoint {checkpoint_id} deleted from GCS")
            return True

        except Exception as e:
            logger.error(f"Failed to delete checkpoint {checkpoint_id} from GCS: {e}")
            return False
