"""Integration tests for FEA uplift pipeline with AWS Batch mocking."""

import tempfile
import json
import uuid
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

from orbitforge.fea.fe_uplift import FEAUpliftClient, FEAUpliftError
from orbitforge.fea.preprocessor import convert_step_to_bdf
from orbitforge.fea.postprocessor import process_op2_results


class TestFEAUpliftClient:
    """Test FEA uplift client functionality."""

    def test_client_initialization(self):
        """Test client initialization."""
        client = FEAUpliftClient(
            job_queue="test-queue",
            job_definition="test-job-def",
            s3_bucket="test-bucket",
        )

        assert client.job_queue == "test-queue"
        assert client.job_definition == "test-job-def"
        assert client.s3_bucket == "test-bucket"
        assert client.max_retries == 3

    @patch("boto3.client")
    def test_submit_batch_job_success(self, mock_boto3_client):
        """Test successful batch job submission."""
        # Mock AWS clients
        mock_batch = Mock()
        mock_s3 = Mock()
        mock_boto3_client.side_effect = lambda service, **kwargs: {
            "batch": mock_batch,
            "s3": mock_s3,
        }[service]

        # Configure batch response
        mock_batch.submit_job.return_value = {"jobId": "test-job-123"}

        client = FEAUpliftClient(
            job_queue="test-queue",
            job_definition="test-job-def",
            s3_bucket="test-bucket",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy BDF file
            bdf_file = Path(tmpdir) / "frame.bdf"
            bdf_file.write_text("$ Test BDF file\nSOL 101\nENDDATA\n")

            design_dir = Path(tmpdir)
            design_id = "test_design"

            job_id = client.submit_batch_job(design_id, bdf_file, design_dir)

            assert job_id == "test-job-123"

            # Verify S3 upload was called
            mock_s3.upload_file.assert_called_once()

            # Verify batch job submission
            mock_batch.submit_job.assert_called_once()
            call_args = mock_batch.submit_job.call_args
            assert call_args[1]["jobQueue"] == "test-queue"
            assert call_args[1]["jobDefinition"] == "test-job-def"

    @patch("boto3.client")
    def test_submit_batch_job_failure(self, mock_boto3_client):
        """Test batch job submission failure."""
        from botocore.exceptions import ClientError

        mock_batch = Mock()
        mock_s3 = Mock()
        mock_boto3_client.side_effect = lambda service, **kwargs: {
            "batch": mock_batch,
            "s3": mock_s3,
        }[service]

        # Configure batch to raise error
        mock_batch.submit_job.side_effect = ClientError(
            {"Error": {"Code": "InvalidRequest", "Message": "Test error"}}, "SubmitJob"
        )

        client = FEAUpliftClient(
            job_queue="test-queue",
            job_definition="test-job-def",
            s3_bucket="test-bucket",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            bdf_file = Path(tmpdir) / "frame.bdf"
            bdf_file.write_text("$ Test BDF")
            design_dir = Path(tmpdir)

            with pytest.raises(FEAUpliftError):
                client.submit_batch_job("test_design", bdf_file, design_dir)

    @patch("boto3.client")
    def test_poll_batch_job_success(self, mock_boto3_client):
        """Test successful job polling."""
        mock_batch = Mock()
        mock_s3 = Mock()
        mock_boto3_client.side_effect = lambda service, **kwargs: {
            "batch": mock_batch,
            "s3": mock_s3,
        }[service]

        # Mock job status progression
        job_responses = [
            {"jobs": [{"jobId": "test-job-123", "jobStatus": "RUNNING"}]},
            {
                "jobs": [
                    {
                        "jobId": "test-job-123",
                        "jobStatus": "SUCCEEDED",
                        "jobName": "test-job",
                        "createdAt": 1234567890,
                    }
                ]
            },
        ]
        mock_batch.describe_jobs.side_effect = job_responses

        client = FEAUpliftClient(
            job_queue="test-queue",
            job_definition="test-job-def",
            s3_bucket="test-bucket",
        )

        # Mock time.sleep to speed up test
        with patch("time.sleep"):
            job_status = client.poll_batch_job("test-job-123", poll_interval=0.1)

        assert job_status["jobStatus"] == "SUCCEEDED"
        assert job_status["jobId"] == "test-job-123"

    @patch("boto3.client")
    def test_poll_batch_job_failure(self, mock_boto3_client):
        """Test job polling with failed job."""
        mock_batch = Mock()
        mock_s3 = Mock()
        mock_boto3_client.side_effect = lambda service, **kwargs: {
            "batch": mock_batch,
            "s3": mock_s3,
        }[service]

        # Mock failed job status
        mock_batch.describe_jobs.return_value = {
            "jobs": [
                {
                    "jobId": "test-job-123",
                    "jobStatus": "FAILED",
                    "statusReason": "Job failed due to test error",
                }
            ]
        }

        client = FEAUpliftClient(
            job_queue="test-queue",
            job_definition="test-job-def",
            s3_bucket="test-bucket",
        )

        job_status = client.poll_batch_job("test-job-123")

        assert job_status["jobStatus"] == "FAILED"

    @patch("boto3.client")
    def test_download_results_success(self, mock_boto3_client):
        """Test successful results download."""
        mock_batch = Mock()
        mock_s3 = Mock()
        mock_boto3_client.side_effect = lambda service, **kwargs: {
            "batch": mock_batch,
            "s3": mock_s3,
        }[service]

        client = FEAUpliftClient(
            job_queue="test-queue",
            job_definition="test-job-def",
            s3_bucket="test-bucket",
        )

        job_status = {
            "jobId": "test-job-123",
            "jobName": "test-job",
            "jobStatus": "SUCCEEDED",
            "createdAt": 1234567890,
            "startedAt": 1234567900,
            "stoppedAt": 1234567950,
            "attempts": [{"exitCode": 0}],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            design_dir = Path(tmpdir)
            design_id = "test_design"

            # Mock S3 downloads to create dummy files
            def mock_download(bucket, key, filename):
                Path(filename).write_bytes(b"dummy file content")

            mock_s3.download_file.side_effect = mock_download

            results_dir, metadata = client.download_results(
                design_id, design_dir, job_status
            )

            assert results_dir.name == "full_fea"
            assert (results_dir / "frame.op2").exists()
            assert (results_dir / "frame.bdf").exists()

            assert metadata["job_id"] == "test-job-123"
            assert metadata["status"] == "SUCCEEDED"
            assert metadata["exit_code"] == 0


class TestFEAUpliftIntegration:
    """Integration tests for the complete FEA uplift pipeline."""

    @patch("boto3.client")
    def test_complete_uplift_pipeline_success(self, mock_boto3_client):
        """Test complete uplift pipeline from STEP to results."""
        # Mock AWS clients
        mock_batch = Mock()
        mock_s3 = Mock()
        mock_boto3_client.side_effect = lambda service, **kwargs: {
            "batch": mock_batch,
            "s3": mock_s3,
        }[service]

        # Configure successful job flow
        mock_batch.submit_job.return_value = {"jobId": "test-job-123"}
        mock_batch.describe_jobs.return_value = {
            "jobs": [
                {
                    "jobId": "test-job-123",
                    "jobName": "test-job",
                    "jobStatus": "SUCCEEDED",
                    "createdAt": 1234567890,
                    "attempts": [{"exitCode": 0}],
                }
            ]
        }

        # Mock S3 download to create OP2 file
        def mock_download(bucket, key, filename):
            if key.endswith(".op2"):
                # Create dummy OP2 file
                Path(filename).write_bytes(b"dummy nastran output" * 50)
            else:
                Path(filename).write_text("$ Dummy BDF file\n")

        mock_s3.download_file.side_effect = mock_download

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Initialize client
            client = FEAUpliftClient(
                job_queue="test-queue",
                job_definition="test-job-def",
                s3_bucket="test-bucket",
            )

            # Create dummy BDF file (simulating preprocessor output)
            bdf_file = tmpdir / "frame.bdf"
            bdf_file.write_text(
                """$ Test BDF file
SOL 101
CEND
BEGIN BULK
MAT1    1       70000.0 0.33    2.7E-9
PSHELL  1       1       1.0
GRID    1               0.0     0.0     0.0
CQUAD4  1       1       1       1       1       1
ENDDATA
"""
            )

            design_id = "test_design_123"
            design_dir = tmpdir

            # Step 1: Submit job
            with patch("time.sleep"):  # Speed up polling
                job_id = client.submit_batch_job(design_id, bdf_file, design_dir)
                assert job_id == "test-job-123"

                # Step 2: Poll for completion
                job_status = client.poll_batch_job(job_id, poll_interval=0.1)
                assert job_status["jobStatus"] == "SUCCEEDED"

                # Step 3: Download results
                results_dir, metadata = client.download_results(
                    design_id, design_dir, job_status
                )

                # Step 4: Process OP2 results
                design_info = {
                    "design_id": design_id,
                    "material": "Aluminum6061",
                    "rail_thickness_mm": 1.5,
                }

                op2_file = results_dir / "frame.op2"
                summary = process_op2_results(op2_file, results_dir, design_info)

                # Verify complete pipeline
                assert summary["fea_mode"] == "full"
                assert "max_stress_fea" in summary
                assert "status_fea" in summary
                assert summary["status_fea"] in ["PASS", "FAIL"]

                # Verify files created
                assert (results_dir / "summary.json").exists()
                assert (results_dir / "stress_summary.pdf").exists() or (
                    results_dir / "stress_summary.txt"
                ).exists()

    @patch("boto3.client")
    def test_uplift_pipeline_job_failure(self, mock_boto3_client):
        """Test pipeline behavior when AWS job fails."""
        mock_batch = Mock()
        mock_s3 = Mock()
        mock_boto3_client.side_effect = lambda service, **kwargs: {
            "batch": mock_batch,
            "s3": mock_s3,
        }[service]

        # Configure job failure
        mock_batch.submit_job.return_value = {"jobId": "test-job-123"}
        mock_batch.describe_jobs.return_value = {
            "jobs": [
                {
                    "jobId": "test-job-123",
                    "jobStatus": "FAILED",
                    "statusReason": "Container failed with exit code 1",
                }
            ]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            client = FEAUpliftClient(
                job_queue="test-queue",
                job_definition="test-job-def",
                s3_bucket="test-bucket",
            )

            bdf_file = Path(tmpdir) / "frame.bdf"
            bdf_file.write_text("$ Test BDF")

            design_id = "test_design"
            design_dir = Path(tmpdir)

            job_id = client.submit_batch_job(design_id, bdf_file, design_dir)
            job_status = client.poll_batch_job(job_id)

            assert job_status["jobStatus"] == "FAILED"
            assert "Container failed" in job_status["statusReason"]


class TestCLIIntegration:
    """Test CLI integration with uplift functionality."""

    @patch.dict(
        "os.environ",
        {
            "AWS_BATCH_JOB_QUEUE": "test-queue",
            "AWS_BATCH_JOB_DEFINITION": "test-job-def",
            "AWS_S3_BUCKET": "test-bucket",
        },
    )
    @patch("boto3.client")
    @patch("orbitforge.fea.preprocessor.convert_step_to_bdf")
    def test_cli_uplift_flag_integration(self, mock_convert, mock_boto3_client):
        """Test CLI --uplift flag integration."""
        # This is a more complex integration test that would require
        # setting up the full CLI environment. For now, we test the
        # core components that the CLI would use.

        # Mock AWS clients
        mock_batch = Mock()
        mock_s3 = Mock()
        mock_boto3_client.side_effect = lambda service, **kwargs: {
            "batch": mock_batch,
            "s3": mock_s3,
        }[service]

        # Configure successful flow
        mock_batch.submit_job.return_value = {"jobId": "cli-test-job"}
        mock_batch.describe_jobs.return_value = {
            "jobs": [
                {
                    "jobId": "cli-test-job",
                    "jobName": "cli-test",
                    "jobStatus": "SUCCEEDED",
                    "createdAt": 1234567890,
                    "attempts": [{"exitCode": 0}],
                }
            ]
        }

        def mock_download(bucket, key, filename):
            Path(filename).write_bytes(b"dummy output" * 20)

        mock_s3.download_file.side_effect = mock_download

        # Mock the STEP to BDF conversion
        def mock_step_conversion(step_file, bdf_file, material_props, loads):
            bdf_file.write_text("$ Mock BDF from CLI\nSOL 101\nENDDATA\n")

        mock_convert.side_effect = mock_step_conversion

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Simulate what the CLI would do
            step_file = tmpdir / "frame.step"
            step_file.write_text("Mock STEP file content")

            design_dir = tmpdir / "design_test"
            design_dir.mkdir()

            # Material properties (from mission spec)
            material_props = {
                "elastic_modulus": 70e9,
                "poisson_ratio": 0.33,
                "density": 2700,
                "thickness": 1.5,
            }

            loads = {"accel_x": 0.0, "accel_y": 0.0, "accel_z": 9.81}

            # Simulate CLI uplift workflow
            from orbitforge.fea.fe_uplift import FEAUpliftClient
            from orbitforge.fea.postprocessor import process_op2_results

            # Convert STEP to BDF
            bdf_file = design_dir / "full_fea" / "frame.bdf"
            bdf_file.parent.mkdir(exist_ok=True)
            mock_convert(step_file, bdf_file, material_props, loads)

            # Initialize uplift client
            client = FEAUpliftClient(
                job_queue="test-queue",
                job_definition="test-job-def",
                s3_bucket="test-bucket",
            )

            # Submit and poll job
            with patch("time.sleep"):
                job_id = client.submit_batch_job("cli_test", bdf_file, design_dir)
                job_status = client.poll_batch_job(job_id, poll_interval=0.1)
                results_dir, metadata = client.download_results(
                    "cli_test", design_dir, job_status
                )

                # Process results
                design_info = {"design_id": "cli_test", "material": "Al6061"}
                op2_file = results_dir / "frame.op2"
                summary = process_op2_results(op2_file, results_dir, design_info)

                # Verify CLI-like workflow completed
                assert summary["fea_mode"] == "full"
                assert mock_convert.called
                assert mock_batch.submit_job.called
                assert (results_dir / "summary.json").exists()


class TestErrorHandling:
    """Test error handling throughout the uplift pipeline."""

    def test_missing_environment_variables(self):
        """Test behavior with missing AWS environment variables."""
        # Test that the client can be initialized with empty values
        # (validation happens at the CLI level, not in the client)
        from orbitforge.fea.fe_uplift import FEAUpliftClient
        import os

        # Save original env vars
        required_vars = [
            "AWS_BATCH_JOB_QUEUE",
            "AWS_BATCH_JOB_DEFINITION",
            "AWS_S3_BUCKET",
        ]
        saved_env = {}
        for var in required_vars:
            saved_env[var] = os.environ.pop(var, None)

        try:
            # Client should initialize with empty values (CLI validates them)
            client = FEAUpliftClient(
                job_queue=os.environ.get("AWS_BATCH_JOB_QUEUE", ""),
                job_definition=os.environ.get("AWS_BATCH_JOB_DEFINITION", ""),
                s3_bucket=os.environ.get("AWS_S3_BUCKET", ""),
            )

            # Verify empty values are set
            assert client.job_queue == ""
            assert client.job_definition == ""
            assert client.s3_bucket == ""

        finally:
            # Restore env vars
            for var, value in saved_env.items():
                if value is not None:
                    os.environ[var] = value

    @patch("boto3.client")
    def test_network_failure_resilience(self, mock_boto3_client):
        """Test resilience to network failures."""
        from botocore.exceptions import ClientError

        mock_batch = Mock()
        mock_s3 = Mock()
        mock_boto3_client.side_effect = lambda service, **kwargs: {
            "batch": mock_batch,
            "s3": mock_s3,
        }[service]

        # Simulate network error
        mock_batch.describe_jobs.side_effect = ClientError(
            {"Error": {"Code": "NetworkingError", "Message": "Connection timeout"}},
            "DescribeJobs",
        )

        client = FEAUpliftClient(
            job_queue="test-queue",
            job_definition="test-job-def",
            s3_bucket="test-bucket",
        )

        with pytest.raises(FEAUpliftError):
            client.poll_batch_job("test-job-123")


class TestRetryLogic:
    """Test retry logic for failed jobs."""

    @patch("boto3.client")
    def test_job_retry_configuration(self, mock_boto3_client):
        """Test that jobs are configured with proper retry strategy."""
        mock_batch = Mock()
        mock_s3 = Mock()
        mock_boto3_client.side_effect = lambda service, **kwargs: {
            "batch": mock_batch,
            "s3": mock_s3,
        }[service]

        mock_batch.submit_job.return_value = {"jobId": "retry-test-job"}

        client = FEAUpliftClient(
            job_queue="test-queue",
            job_definition="test-job-def",
            s3_bucket="test-bucket",
            max_retries=5,  # Custom retry count
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            bdf_file = Path(tmpdir) / "frame.bdf"
            bdf_file.write_text("$ Test BDF")
            design_dir = Path(tmpdir)

            client.submit_batch_job("retry_test", bdf_file, design_dir)

            # Verify retry strategy was set correctly
            call_args = mock_batch.submit_job.call_args
            assert call_args[1]["retryStrategy"]["attempts"] == 5
