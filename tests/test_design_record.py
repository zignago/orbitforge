"""Test suite for Design Record functionality."""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import shutil
import logging

from orbitforge.design_record import (
    DesignRecord,
    GeometryParams,
    AnalysisResults,
    ArtifactType,
    DesignStatus,
    Artifact,
    ArtifactSource,
)
from orbitforge.artifact_storage import ArtifactStorage, get_storage
from orbitforge.design_context import (
    design_record_context,
    add_build_artifacts,
    add_physics_results,
    load_mass_from_budget,
)
from orbitforge.generator.mission import MissionSpec, Material


@pytest.fixture
def sample_mission_spec():
    """Create a sample mission specification."""
    return MissionSpec(
        bus_u=3,
        payload_mass_kg=1.0,
        orbit_alt_km=550,
        mass_limit_kg=4.0,
        rail_mm=3.0,
        deck_mm=2.5,
        material=Material.AL_6061_T6,
    )


@pytest.fixture
def sample_mission_raw():
    """Create raw mission specification data."""
    return {
        "bus_u": 3,
        "payload_mass_kg": 1.0,
        "orbit_alt_km": 550,
        "mass_limit_kg": 4.0,
        "rail_mm": 3.0,
        "deck_mm": 2.5,
        "material": "Al_6061_T6",
    }


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestDesignRecord:
    """Test the DesignRecord Pydantic model."""

    def test_design_record_creation(self, sample_mission_raw):
        """Test basic design record creation."""
        geometry_params = GeometryParams(
            rail_mm=3.0, deck_mm=2.5, material="Al_6061_T6"
        )

        record = DesignRecord(
            mission_spec=sample_mission_raw, geometry_params=geometry_params
        )

        # Check that required fields are populated
        assert record.design_id is not None
        assert record.timestamp is not None
        assert record.status == DesignStatus.PENDING
        assert record.mission_spec == sample_mission_raw
        assert record.geometry_params.rail_mm == 3.0
        assert record.geometry_params.deck_mm == 2.5
        assert record.geometry_params.material == "Al_6061_T6"

    def test_design_record_validation(self, sample_mission_raw):
        """Test Pydantic validation works correctly."""
        geometry_params = GeometryParams(
            rail_mm=3.0, deck_mm=2.5, material="Al_6061_T6"
        )

        # Valid record should work
        record = DesignRecord(
            mission_spec=sample_mission_raw, geometry_params=geometry_params
        )

        # Test JSON serialization/deserialization
        json_str = record.model_dump_json()
        loaded_record = DesignRecord.model_validate_json(json_str)

        assert loaded_record.design_id == record.design_id
        assert loaded_record.mission_spec == record.mission_spec

    def test_add_artifact(self, sample_mission_raw, temp_dir):
        """Test adding artifacts to design record."""
        geometry_params = GeometryParams(
            rail_mm=3.0, deck_mm=2.5, material="Al_6061_T6"
        )

        record = DesignRecord(
            mission_spec=sample_mission_raw, geometry_params=geometry_params
        )

        # Create a test file
        test_file = temp_dir / "test.step"
        test_file.write_text("dummy STEP content")

        # Add artifact
        record.add_artifact(ArtifactType.STEP, test_file, description="Test STEP file")

        assert len(record.artifacts) == 1
        artifact = record.artifacts[0]
        assert artifact.type == ArtifactType.STEP
        assert artifact.path == str(test_file)
        assert artifact.hash_sha256 is not None
        assert artifact.size_bytes == len("dummy STEP content")

    def test_update_analysis_results(self, sample_mission_raw):
        """Test updating analysis results."""
        geometry_params = GeometryParams(
            rail_mm=3.0, deck_mm=2.5, material="Al_6061_T6"
        )

        record = DesignRecord(
            mission_spec=sample_mission_raw, geometry_params=geometry_params
        )

        # Update analysis results
        record.update_analysis_results(
            max_stress_mpa=150.5, mass_kg=2.3, custom_param="test_value"
        )

        assert record.analysis_results.max_stress_mpa == 150.5
        assert record.analysis_results.mass_kg == 2.3
        assert (
            record.analysis_results.additional_results["custom_param"] == "test_value"
        )

    def test_save_and_load_record(self, sample_mission_raw, temp_dir):
        """Test saving and loading design records."""
        geometry_params = GeometryParams(
            rail_mm=3.0, deck_mm=2.5, material="Al_6061_T6"
        )

        record = DesignRecord(
            mission_spec=sample_mission_raw, geometry_params=geometry_params
        )

        # Save record
        record_file = temp_dir / "test_record.json"
        record.save_to_file(record_file)

        assert record_file.exists()

        # Load record
        loaded_record = DesignRecord.load_from_file(record_file)

        assert loaded_record.design_id == record.design_id
        assert loaded_record.mission_spec == record.mission_spec
        assert loaded_record.geometry_params.rail_mm == record.geometry_params.rail_mm


class TestArtifactStorage:
    """Test the artifact storage system."""

    def test_local_storage_init(self, temp_dir):
        """Test local storage initialization."""
        storage_uri = f"file://{temp_dir}/artifacts"
        storage = ArtifactStorage(storage_uri)

        assert storage.base_uri == storage_uri
        assert storage.local_base == temp_dir / "artifacts"
        assert storage.local_base.exists()

    def test_store_and_retrieve_artifact(self, temp_dir):
        """Test storing and retrieving artifacts."""
        storage_uri = f"file://{temp_dir}/artifacts"
        storage = ArtifactStorage(storage_uri)

        # Create test file
        test_file = temp_dir / "test.step"
        test_file.write_text("test STEP content")

        # Store artifact
        design_id = "test-design-123"
        uri = storage.store_artifact(test_file, design_id, "test.step")

        assert uri.startswith("file://")

        # Retrieve artifact
        retrieved_file = temp_dir / "retrieved_test.step"
        result_path = storage.retrieve_artifact(uri, retrieved_file)

        assert result_path == retrieved_file
        assert retrieved_file.exists()
        assert retrieved_file.read_text() == "test STEP content"

    def test_list_artifacts(self, temp_dir):
        """Test listing artifacts for a design."""
        storage_uri = f"file://{temp_dir}/artifacts"
        storage = ArtifactStorage(storage_uri)

        # Create test files
        design_id = "test-design-456"
        test_files = ["test1.step", "test2.stl", "results.json"]

        for filename in test_files:
            test_file = temp_dir / filename
            test_file.write_text(f"content of {filename}")
            storage.store_artifact(test_file, design_id, filename)

        # List artifacts
        artifacts = storage.list_artifacts(design_id)

        assert len(artifacts) == 3
        assert all(name in artifacts for name in test_files)
        assert all(uri.startswith("file://") for uri in artifacts.values())

    def test_mock_artifact_fetch(self, caplog):
        """Test that fetching mock artifacts produces appropriate warnings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ArtifactStorage(f"file://{Path(tmpdir).absolute()}")

            mock_artifact = Artifact(
                type="step",
                uri="file:///nonexistent/path/model.step",
                source=ArtifactSource.MOCK,
                path="/nonexistent/path/model.step",  # Convert Path to string
            )

            with caplog.at_level(logging.INFO):
                result = storage.fetch_artifact(mock_artifact)
                assert result is None
                assert "Found artifact entry with source: mock" in caplog.text

    def test_missing_artifact_warning(self, caplog):
        """Test that fetching missing artifacts produces appropriate warnings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ArtifactStorage(f"file://{Path(tmpdir).absolute()}")

            real_artifact = Artifact(
                type="step",
                uri="file:///nonexistent/path/model.step",
                source=ArtifactSource.RUNTIME,
                path="/nonexistent/path/model.step",  # Convert Path to string
            )

            with caplog.at_level(logging.WARNING):
                result = storage.fetch_artifact(real_artifact)
                assert result is None
                assert (
                    "This artifact was listed in the record, but no file was stored"
                    in caplog.text
                )

    @pytest.mark.integration
    def test_full_artifact_roundtrip(self):
        """Test full roundtrip of creating and fetching real artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            storage = ArtifactStorage(f"file://{tmpdir_path}")

            # Create some test files with their corresponding ArtifactTypes
            test_files = {
                "model.step": ("dummy STEP content", ArtifactType.STEP),
                "model.stl": ("dummy STL content", ArtifactType.STL),
                "report.pdf": ("dummy PDF content", ArtifactType.REPORT),
            }

            artifacts = []
            for filename, (content, artifact_type) in test_files.items():
                file_path = tmpdir_path / filename
                file_path.write_text(content)

                artifact = Artifact(
                    type=artifact_type,
                    uri=f"file://{file_path}",
                    source=ArtifactSource.RUNTIME,
                    path=str(file_path),
                )
                artifacts.append(artifact)

            # Create test mission spec and geometry params
            mission_spec = {
                "bus_u": 3,
                "payload_mass_kg": 1.0,
                "orbit_alt_km": 550,
                "mass_limit_kg": 4.0,
            }

            geometry_params = GeometryParams(
                rail_mm=3.0, deck_mm=2.5, material="Al_6061_T6"
            )

            # Create a design record with these artifacts
            record = DesignRecord(
                design_id="test-design",
                mission_spec=mission_spec,
                geometry_params=geometry_params,
                artifacts=artifacts,
            )

            # Fetch all artifacts
            fetched_paths = storage.fetch_artifacts(record)

            # Verify all files were found
            assert len(fetched_paths) == len(test_files)
            for path in fetched_paths:
                assert path.exists()
                assert path.read_text() in [
                    content for content, _ in test_files.values()
                ]


class TestDesignContext:
    """Test the design context manager."""

    def test_design_record_context_success(
        self, sample_mission_spec, sample_mission_raw, temp_dir
    ):
        """Test successful design record context."""
        with design_record_context(
            sample_mission_spec,
            sample_mission_raw,
            temp_dir,
            {"test_override": "value"},
        ) as record:
            assert record.design_id is not None
            assert record.status == DesignStatus.PENDING
            assert record.mission_spec == sample_mission_raw
            assert record.geometry_params.additional_params["test_override"] == "value"

        # Check that record was saved
        records_dir = temp_dir / "records"
        assert records_dir.exists()

        record_files = list(records_dir.glob("*design_record.json"))
        assert len(record_files) == 1

        # Load and verify saved record
        saved_record = DesignRecord.load_from_file(record_files[0])
        assert saved_record.design_id == record.design_id
        assert saved_record.status == DesignStatus.PASS
        assert saved_record.execution_time_seconds is not None

    def test_design_record_context_failure(
        self, sample_mission_spec, sample_mission_raw, temp_dir
    ):
        """Test design record context with exception."""
        with pytest.raises(ValueError, match="Test error"):
            with design_record_context(
                sample_mission_spec, sample_mission_raw, temp_dir
            ) as record:
                assert record.design_id is not None
                raise ValueError("Test error")

        # Check that record was still saved with error status
        records_dir = temp_dir / "records"
        record_files = list(records_dir.glob("*design_record.json"))
        assert len(record_files) == 1

        saved_record = DesignRecord.load_from_file(record_files[0])
        assert saved_record.status == DesignStatus.ERROR
        assert saved_record.error_message == "Test error"

    def test_add_build_artifacts(self, sample_mission_raw, temp_dir):
        """Test adding build artifacts to design record."""
        geometry_params = GeometryParams(
            rail_mm=3.0, deck_mm=2.5, material="Al_6061_T6"
        )

        record = DesignRecord(
            mission_spec=sample_mission_raw, geometry_params=geometry_params
        )

        # Create test files
        design_dir = temp_dir / "design"
        design_dir.mkdir()

        step_file = design_dir / "frame.step"
        step_file.write_text("STEP content")

        stl_file = design_dir / "frame.stl"
        stl_file.write_text("STL content")

        mass_file = design_dir / "mass_budget.csv"
        mass_file.write_text("component,mass_kg\ntotal,2.5")

        # Add artifacts
        add_build_artifacts(record, design_dir, step_file)

        # Check artifacts were added
        artifact_types = [a.type for a in record.artifacts]
        assert ArtifactType.STEP in artifact_types
        assert ArtifactType.STL in artifact_types
        assert ArtifactType.MASS_BUDGET in artifact_types

    def test_load_mass_from_budget(self, temp_dir):
        """Test loading mass from mass budget file."""
        design_dir = temp_dir / "design"
        design_dir.mkdir()

        # Create mass budget file
        mass_file = design_dir / "mass_budget.csv"
        mass_file.write_text(
            "component,mass_kg\n" "rails,1.2\n" "deck,0.8\n" "total,2.0\n"
        )

        mass = load_mass_from_budget(design_dir)
        assert mass == 2.0

        # Test with missing file
        missing_dir = temp_dir / "missing"
        missing_mass = load_mass_from_budget(missing_dir)
        assert missing_mass is None


class TestIntegration:
    """Integration tests for the complete design record system."""

    @patch("subprocess.check_output")
    def test_git_commit_retrieval(self, mock_subprocess, sample_mission_raw):
        """Test git commit hash retrieval."""
        mock_subprocess.return_value = "abc123def456\n"

        geometry_params = GeometryParams(
            rail_mm=3.0, deck_mm=2.5, material="Al_6061_T6"
        )

        record = DesignRecord(
            mission_spec=sample_mission_raw, geometry_params=geometry_params
        )

        git_commit = record.get_git_commit()
        assert git_commit == "abc123def456"

        mock_subprocess.assert_called_once_with(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent.parent,  # instead of / "orbitforge"
            text=True,
        )

    def test_complete_workflow_simulation(
        self, sample_mission_spec, sample_mission_raw, temp_dir
    ):
        """Simulate a complete design workflow with design record."""
        cli_overrides = {
            "rail_override": 3.5,
            "deck_override": 2.8,
            "material_override": "Ti_6Al_4V",
        }

        # Create design files that would be generated
        design_dir = temp_dir / "design"
        design_dir.mkdir()

        files_to_create = [
            ("frame.step", "STEP file content"),
            ("frame.stl", "STL file content"),
            ("mass_budget.csv", "component,mass_kg\ntotal,2.1"),
            ("physics_check.json", '{"max_stress_mpa": 145.2, "status": "PASS"}'),
            ("manufacturability.json", '{"status": "PASS", "violations": []}'),
            ("report.pdf", "PDF content"),
        ]

        for filename, content in files_to_create:
            (design_dir / filename).write_text(content)

        # Run through design record context
        with design_record_context(
            sample_mission_spec, sample_mission_raw, temp_dir, cli_overrides
        ) as record:
            # Add build artifacts
            add_build_artifacts(record, design_dir, design_dir / "frame.step")

            # Load and update mass
            mass = load_mass_from_budget(design_dir)
            if mass:
                record.update_analysis_results(mass_kg=mass)

            # Simulate physics results
            mock_physics = MagicMock()
            mock_physics.max_stress_mpa = 145.2
            mock_physics.status = "PASS"
            add_physics_results(record, design_dir, mock_physics)

        # Verify final record
        records_dir = temp_dir / "records"
        record_files = list(records_dir.glob("*design_record.json"))
        assert len(record_files) == 1

        final_record = DesignRecord.load_from_file(record_files[0])
        assert final_record.status == DesignStatus.PASS
        assert final_record.analysis_results.mass_kg == 2.1
        assert final_record.analysis_results.max_stress_mpa == 145.2
        assert len(final_record.artifacts) >= 3  # At least STEP, STL, mass budget
