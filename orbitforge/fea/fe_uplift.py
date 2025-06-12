"""AWS Batch FEA uplift module for OrbitForge.

This module handles the submission and monitoring of MSC Nastran jobs
on AWS Batch for full-fidelity structural analysis.
"""

import json
import os
import time
import uuid
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import boto3
from botocore.exceptions import ClientError
from loguru import logger


class FEAUpliftError(Exception):
    """Base exception for FEA uplift operations."""

    pass


class MockFEAUpliftClient:
    """Mock client for testing AWS Batch FEA jobs."""

    def __init__(
        self,
        job_queue: str,
        job_definition: str,
        s3_bucket: str,
        max_retries: int = 3,
        aws_region: str = "us-east-1",
    ):
        """Initialize the mock FEA uplift client."""
        self.job_queue = job_queue
        self.job_definition = job_definition
        self.s3_bucket = s3_bucket
        self.max_retries = max_retries
        self.mock_jobs = {}

    def submit_batch_job(self, design_id: str, bdf_file: Path, design_dir: Path) -> str:
        """Mock submitting a batch job."""
        job_id = f"mock-job-{uuid.uuid4().hex[:8]}"
        job_name = f"fea_{design_id}_{uuid.uuid4().hex[:8]}"

        # Store job info
        self.mock_jobs[job_id] = {
            "jobId": job_id,
            "jobName": job_name,
            "jobStatus": "SUBMITTED",
            "createdAt": int(time.time()),
            "design_id": design_id,
            "bdf_file": bdf_file,
            "design_dir": design_dir,
        }

        logger.info(f"Submitted mock FEA job {job_name} with ID: {job_id}")
        return job_id

    def poll_batch_job(self, job_id: str, poll_interval: int = 1) -> Dict[str, Any]:
        """Mock polling a batch job."""
        if job_id not in self.mock_jobs:
            raise FEAUpliftError(f"Mock job {job_id} not found")

        job = self.mock_jobs[job_id]

        # Simulate job progression
        if job["jobStatus"] == "SUBMITTED":
            job["jobStatus"] = "RUNNING"
            job["startedAt"] = int(time.time())
        elif job["jobStatus"] == "RUNNING":
            job["jobStatus"] = "SUCCEEDED"
            job["stoppedAt"] = int(time.time())

            # Create mock results
            design_dir = job["design_dir"]
            results_dir = design_dir / "full_fea"
            results_dir.mkdir(exist_ok=True)

            # Copy input BDF to results dir
            shutil.copy2(job["bdf_file"], results_dir / "frame.bdf")

            # Create dummy OP2 file
            op2_file = results_dir / "frame.op2"
            op2_file.write_bytes(b"MOCK NASTRAN OUTPUT" * 100)

        return job

    def download_results(
        self, design_id: str, design_dir: Path, job_status: Dict[str, Any]
    ) -> Tuple[Path, Dict[str, Any]]:
        """Mock downloading results."""
        results_dir = design_dir / "full_fea"

        # Results should already be in place from poll_batch_job
        if not (results_dir / "frame.op2").exists():
            raise FEAUpliftError("Mock results not found")

        job_metadata = {
            "job_id": job_status["jobId"],
            "job_name": job_status["jobName"],
            "status": job_status["jobStatus"],
            "created_at": job_status.get("createdAt", 0),
            "started_at": job_status.get("startedAt", 0),
            "stopped_at": job_status.get("stoppedAt", 0),
            "exit_code": 0,
            "reason": "Mock job completed successfully",
        }

        return results_dir, job_metadata


class FEAUpliftClient:
    """Client for managing AWS Batch FEA jobs."""

    def __init__(
        self,
        job_queue: str,
        job_definition: str,
        s3_bucket: str,
        max_retries: int = 3,
        aws_region: str = "us-east-1",
    ):
        """Initialize the FEA uplift client.

        Args:
            job_queue: AWS Batch job queue name
            job_definition: AWS Batch job definition ARN
            s3_bucket: S3 bucket for storing input/output files
            max_retries: Maximum retry attempts for failed jobs
            aws_region: AWS region
        """
        self.job_queue = job_queue
        self.job_definition = job_definition
        self.s3_bucket = s3_bucket
        self.max_retries = max_retries

        # Initialize AWS clients
        self.batch_client = boto3.client("batch", region_name=aws_region)
        self.s3_client = boto3.client("s3", region_name=aws_region)

    def submit_batch_job(self, design_id: str, bdf_file: Path, design_dir: Path) -> str:
        """Submit a batch job for FEA analysis.

        Args:
            design_id: Unique identifier for the design
            bdf_file: Path to the BDF input file
            design_dir: Directory containing the design files

        Returns:
            AWS Batch job ID

        Raises:
            FEAUpliftError: If job submission fails
        """
        try:
            # Generate unique job name
            job_name = f"fea_{design_id}_{uuid.uuid4().hex[:8]}"

            # Upload BDF file to S3
            bdf_s3_key = f"inputs/{design_id}/frame.bdf"
            self._upload_file_to_s3(bdf_file, bdf_s3_key)

            # Define S3 URIs for input and output
            bdf_s3_uri = f"s3://{self.s3_bucket}/{bdf_s3_key}"
            op2_s3_uri = f"s3://{self.s3_bucket}/outputs/{design_id}/frame.op2"

            # Submit job to AWS Batch
            response = self.batch_client.submit_job(
                jobName=job_name,
                jobQueue=self.job_queue,
                jobDefinition=self.job_definition,
                containerOverrides={
                    "environment": [
                        {"name": "BDF_S3_URI", "value": bdf_s3_uri},
                        {"name": "OP2_S3_URI", "value": op2_s3_uri},
                        {"name": "DESIGN_ID", "value": design_id},
                    ]
                },
                timeout={"attemptDurationSeconds": 1800},  # 30 minutes
                retryStrategy={"attempts": self.max_retries},
            )

            job_id = response["jobId"]
            logger.info(f"Submitted FEA job {job_name} with ID: {job_id}")

            return job_id

        except ClientError as e:
            error_msg = f"Failed to submit batch job: {e}"
            logger.error(error_msg)
            raise FEAUpliftError(error_msg) from e

    def poll_batch_job(self, job_id: str, poll_interval: int = 30) -> Dict[str, Any]:
        """Poll a batch job until completion.

        Args:
            job_id: AWS Batch job ID
            poll_interval: Polling interval in seconds

        Returns:
            Job status information

        Raises:
            FEAUpliftError: If job polling fails or job fails
        """
        try:
            logger.info(f"Polling job {job_id}")

            while True:
                response = self.batch_client.describe_jobs(jobs=[job_id])

                if not response["jobs"]:
                    raise FEAUpliftError(f"Job {job_id} not found")

                job = response["jobs"][0]
                status = job["jobStatus"]

                logger.debug(f"Job {job_id} status: {status}")

                if status in ["SUCCEEDED", "FAILED"]:
                    return job
                elif status in [
                    "SUBMITTED",
                    "PENDING",
                    "RUNNABLE",
                    "STARTING",
                    "RUNNING",
                ]:
                    time.sleep(poll_interval)
                else:
                    raise FEAUpliftError(f"Unexpected job status: {status}")

        except ClientError as e:
            error_msg = f"Failed to poll batch job: {e}"
            logger.error(error_msg)
            raise FEAUpliftError(error_msg) from e

    def download_results(
        self, design_id: str, design_dir: Path, job_status: Dict[str, Any]
    ) -> Tuple[Path, Dict[str, Any]]:
        """Download results from S3 after job completion.

        Args:
            design_id: Design identifier
            design_dir: Local directory to save results
            job_status: Job status from poll_batch_job

        Returns:
            Tuple of (results_dir, job_metadata)

        Raises:
            FEAUpliftError: If download fails
        """
        try:
            # Create results directory
            results_dir = design_dir / "full_fea"
            results_dir.mkdir(exist_ok=True)

            # Download OP2 file
            op2_s3_key = f"outputs/{design_id}/frame.op2"
            op2_file = results_dir / "frame.op2"
            self._download_file_from_s3(op2_s3_key, op2_file)

            # Download BDF file (for reference)
            bdf_s3_key = f"inputs/{design_id}/frame.bdf"
            bdf_file = results_dir / "frame.bdf"
            self._download_file_from_s3(bdf_s3_key, bdf_file)

            # Create job metadata
            job_metadata = {
                "job_id": job_status["jobId"],
                "job_name": job_status["jobName"],
                "status": job_status["jobStatus"],
                "created_at": job_status.get("createdAt", 0),
                "started_at": job_status.get("startedAt", 0),
                "stopped_at": job_status.get("stoppedAt", 0),
                "exit_code": job_status.get("attempts", [{}])[0].get("exitCode"),
                "reason": job_status.get("statusReason", ""),
            }

            logger.info(f"Downloaded results for design {design_id} to {results_dir}")

            return results_dir, job_metadata

        except ClientError as e:
            error_msg = f"Failed to download results: {e}"
            logger.error(error_msg)
            raise FEAUpliftError(error_msg) from e

    def _upload_file_to_s3(self, local_file: Path, s3_key: str) -> None:
        """Upload a file to S3."""
        try:
            self.s3_client.upload_file(str(local_file), self.s3_bucket, s3_key)
            logger.debug(f"Uploaded {local_file} to s3://{self.s3_bucket}/{s3_key}")
        except ClientError as e:
            raise FEAUpliftError(f"Failed to upload {local_file}: {e}") from e

    def _download_file_from_s3(self, s3_key: str, local_file: Path) -> None:
        """Download a file from S3."""
        try:
            self.s3_client.download_file(self.s3_bucket, s3_key, str(local_file))
            logger.debug(f"Downloaded s3://{self.s3_bucket}/{s3_key} to {local_file}")
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                logger.warning(f"File not found in S3: {s3_key}")
            else:
                raise FEAUpliftError(f"Failed to download {s3_key}: {e}") from e


def get_client(mock: bool = False) -> FEAUpliftClient:
    """Get a FEA uplift client instance."""
    client_class = MockFEAUpliftClient if mock else FEAUpliftClient
    return client_class(
        job_queue=os.environ.get("AWS_BATCH_JOB_QUEUE", "orbitforge-fea-queue"),
        job_definition=os.environ.get("AWS_BATCH_JOB_DEFINITION", "orbitforge-fea-job"),
        s3_bucket=os.environ.get("AWS_S3_BUCKET", "orbitforge-fea-bucket"),
    )


def submit_batch_job(
    design_id: str, bdf_file: Path, design_dir: Path, mock: bool = False
) -> str:
    """Convenience function to submit a batch job."""
    client = get_client(mock)
    return client.submit_batch_job(design_id, bdf_file, design_dir)


def poll_batch_job(job_id: str, mock: bool = False) -> Dict[str, Any]:
    """Convenience function to poll a batch job."""
    client = get_client(mock)
    return client.poll_batch_job(job_id)


def download_results(
    design_id: str, design_dir: Path, job_status: Dict[str, Any], mock: bool = False
) -> Tuple[Path, Dict[str, Any]]:
    """Convenience function to download results."""
    client = get_client(mock)
    return client.download_results(design_id, design_dir, job_status)
