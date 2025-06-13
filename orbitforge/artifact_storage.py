"""Artifact storage system for design records.

This module provides storage abstraction for design artifacts, supporting
both local filesystem and S3-compatible cloud storage.
"""

import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse
from loguru import logger
from datetime import datetime
import logging
from .design_record import DesignRecord, Artifact, ArtifactSource

logger = logging.getLogger(__name__)


class ArtifactStorage:
    """Manages storage and retrieval of design artifacts."""

    def __init__(self, base_uri: Optional[str] = None):
        """Initialize artifact storage.

        Args:
            base_uri: Base URI for storage (e.g., 's3://orbitforge-runs/' or 'file:///tmp/orbitforge/')
        """
        # Use current directory with outputs/artifacts as default
        default_path = Path.cwd() / "outputs" / "artifacts"
        self.base_uri = base_uri or f"file://{default_path.absolute()}"
        self.parsed_uri = urlparse(str(self.base_uri))

        if self.parsed_uri.scheme == "s3":
            self._init_s3_storage()
        elif self.parsed_uri.scheme == "file":
            self._init_local_storage()
        else:
            raise ValueError(f"Unsupported storage scheme: {self.parsed_uri.scheme}")

        logger.info(f"Artifact storage initialized: {self.base_uri}")

    def _init_s3_storage(self):
        """Initialize S3 storage backend."""
        try:
            import boto3
            from botocore.exceptions import NoCredentialsError, ClientError

            self.s3_client = boto3.client("s3")
            self.bucket_name = self.parsed_uri.netloc

            # Test connection
            try:
                self.s3_client.head_bucket(Bucket=self.bucket_name)
                logger.info(f"✓ Connected to S3 bucket: {self.bucket_name}")
            except ClientError as e:
                error_code = e.response["Error"]["Code"]
                if error_code == "404":
                    logger.warning(f"S3 bucket does not exist: {self.bucket_name}")
                else:
                    logger.error(f"S3 connection error: {e}")
                    raise

        except ImportError:
            logger.error("boto3 not installed. Install with: pip install boto3")
            raise
        except NoCredentialsError:
            logger.error("AWS credentials not configured")
            raise

    def _init_local_storage(self):
        """Initialize local filesystem storage."""
        self.local_base = Path(self.parsed_uri.path)
        self.local_base.mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Local storage initialized: {self.local_base}")

    def store_artifact(
        self, source_path: Path, design_id: str, artifact_name: str
    ) -> str:
        """Store an artifact and return its URI.

        Args:
            source_path: Local path to the artifact file
            design_id: Design ID for organizing artifacts
            artifact_name: Name of the artifact file

        Returns:
            URI of the stored artifact
        """
        if not source_path.exists():
            raise FileNotFoundError(f"Artifact file not found: {source_path}")

        # Create date-based path structure: YYYY/MM/DD/design_id/artifact_name
        now = datetime.utcnow()
        relative_path = (
            f"{now.year:04d}/{now.month:02d}/{now.day:02d}/{design_id}/{artifact_name}"
        )

        if self.parsed_uri.scheme == "s3":
            return self._store_to_s3(source_path, relative_path)
        else:
            return self._store_to_local(source_path, relative_path)

    def _store_to_s3(self, source_path: Path, relative_path: str) -> str:
        """Store artifact to S3."""
        try:
            s3_key = self.parsed_uri.path.lstrip("/") + relative_path

            self.s3_client.upload_file(str(source_path), self.bucket_name, s3_key)

            uri = f"s3://{self.bucket_name}/{s3_key}"
            logger.debug(f"✓ Stored to S3: {uri}")
            return uri

        except Exception as e:
            logger.error(f"Failed to store artifact to S3: {e}")
            raise

    def _store_to_local(self, source_path: Path, relative_path: str) -> str:
        """Store artifact to local filesystem."""
        target_path = self.local_base / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(source_path, target_path)

        uri = f"file://{target_path.absolute()}"
        logger.debug(f"✓ Stored locally: {uri}")
        return uri

    def retrieve_artifact(self, uri: str, target_path: Path) -> Path:
        """Retrieve an artifact from storage.

        Args:
            uri: URI of the artifact to retrieve
            target_path: Local path where to save the artifact

        Returns:
            Path to the retrieved artifact
        """
        parsed = urlparse(uri)

        if parsed.scheme == "s3":
            return self._retrieve_from_s3(uri, target_path)
        elif parsed.scheme == "file":
            return self._retrieve_from_local(uri, target_path)
        else:
            raise ValueError(f"Unsupported URI scheme: {parsed.scheme}")

    def _retrieve_from_s3(self, uri: str, target_path: Path) -> Path:
        """Retrieve artifact from S3."""
        parsed = urlparse(uri)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")

        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            self.s3_client.download_file(bucket, key, str(target_path))
            logger.debug(f"✓ Retrieved from S3: {uri} -> {target_path}")
            return target_path

        except Exception as e:
            logger.error(f"Failed to retrieve artifact from S3: {e}")
            raise

    def _retrieve_from_local(self, uri: str, target_path: Path) -> Path:
        """Retrieve artifact from local filesystem."""
        parsed = urlparse(uri)
        source_path = Path(parsed.path)

        if not source_path.exists():
            raise FileNotFoundError(f"Artifact not found: {source_path}")

        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        logger.debug(f"✓ Retrieved locally: {uri} -> {target_path}")
        return target_path

    def list_artifacts(self, design_id: str) -> Dict[str, str]:
        """List all artifacts for a given design ID.

        Args:
            design_id: Design ID to list artifacts for

        Returns:
            Dictionary mapping artifact names to URIs
        """
        if self.parsed_uri.scheme == "s3":
            return self._list_s3_artifacts(design_id)
        else:
            return self._list_local_artifacts(design_id)

    def _list_s3_artifacts(self, design_id: str) -> Dict[str, str]:
        """List artifacts in S3."""
        try:
            prefix = f"{self.parsed_uri.path.lstrip('/')}"

            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name, Prefix=prefix
            )

            artifacts = {}
            for obj in response.get("Contents", []):
                key = obj["Key"]
                if design_id in key:
                    artifact_name = Path(key).name
                    uri = f"s3://{self.bucket_name}/{key}"
                    artifacts[artifact_name] = uri

            return artifacts

        except Exception as e:
            logger.error(f"Failed to list S3 artifacts: {e}")
            return {}

    def _list_local_artifacts(self, design_id: str) -> Dict[str, str]:
        """List artifacts in local filesystem."""
        artifacts = {}

        # Search for design_id in directory structure
        for path in self.local_base.rglob(f"*/{design_id}/*"):
            if path.is_file():
                artifact_name = path.name
                uri = f"file://{path.absolute()}"
                artifacts[artifact_name] = uri

        return artifacts

    def fetch_artifact(self, artifact: Artifact) -> Optional[Path]:
        """Fetch an artifact from storage, returning its local path if found."""
        if artifact.source == ArtifactSource.MOCK:
            logger.info(
                f"Found artifact entry with source: mock (not physically stored)"
            )
            return None

        uri = urlparse(artifact.uri)
        if uri.scheme == "file":
            path = Path(uri.path)
            if not path.exists():
                logger.warning(
                    f"This artifact was listed in the record, but no file was stored at URI: {artifact.uri}"
                )
                return None
            return path
        elif uri.scheme == "s3":
            if not self.s3_client:
                logger.warning("S3 client not configured, cannot fetch S3 artifact")
                return None

            bucket = uri.netloc
            key = uri.path.lstrip("/")
            local_path = self.local_cache_dir / bucket / key

            if not local_path.exists():
                local_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    self.s3_client.download_file(bucket, key, str(local_path))
                except Exception as e:
                    logger.warning(f"Failed to download S3 artifact: {e}")
                    return None

            return local_path
        else:
            logger.warning(f"Unsupported URI scheme: {uri.scheme}")
            return None

    def fetch_artifacts(self, design_record: DesignRecord) -> List[Path]:
        """Fetch all artifacts for a design record."""
        paths = []
        for artifact in design_record.artifacts:
            path = self.fetch_artifact(artifact)
            if path:
                paths.append(path)
        return paths


# Global storage instance
_storage_instance: Optional[ArtifactStorage] = None


def get_storage(base_uri: Optional[str] = None) -> ArtifactStorage:
    """Get the global artifact storage instance."""
    global _storage_instance

    if _storage_instance is None or base_uri is not None:
        _storage_instance = ArtifactStorage(base_uri)

    return _storage_instance
