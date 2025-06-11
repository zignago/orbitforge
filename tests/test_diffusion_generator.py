"""Test suite for OrbitForge v0.1.3 diffusion generator.

This test suite verifies that the diffusion-based geometry generator
meets all requirements specified in the MVP v0.1.3 specification.
"""

import pytest
import torch
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

from orbitforge.generator.mission import MissionSpec, Material
from orbitforge.generator.diffusion_model import DiffusionGenerator, GeneratedMesh
from orbitforge.generator.mesh_to_solid import MeshToSolidConverter
from orbitforge.generator.diffusion_workflow import DiffusionWorkflow


class TestDiffusionModel:
    """Test the core diffusion model functionality."""

    def test_model_initialization(self):
        """Test that the diffusion model initializes correctly."""
        generator = DiffusionGenerator()
        assert generator.device is not None
        assert generator.model_path is not None

    def test_mission_spec_encoding(self):
        """Test mission specification encoding to conditioning vector."""
        spec = MissionSpec(
            bus_u=3,
            payload_mass_kg=1.0,
            orbit_alt_km=550,
            rail_mm=3.0,
            deck_mm=2.5,
            material=Material.AL_6061_T6,
        )

        generator = DiffusionGenerator()
        conditioning = generator.encode_mission_spec(spec)

        # Check that conditioning is a 6-element tensor
        assert conditioning.shape == (6,)
        assert conditioning.dtype in (torch.float32, torch.float64)
        # allow any float to above command
        # assert conditioning.dtype.is_floating_point

        # Check that values are reasonable
        assert conditioning[0] > 0  # orbit altitude (normalized)
        assert conditioning[1] > 0  # payload mass
        assert conditioning[2] > 0  # delta-V proxy
        assert conditioning[3] > 0  # rail thickness
        assert conditioning[4] > 0  # deck thickness
        assert conditioning[5] in [0.0, 1.0]  # material encoding

    def test_candidate_generation(self):
        """Test that the model generates the requested number of candidates."""
        spec = MissionSpec(
            bus_u=3,
            payload_mass_kg=1.0,
            orbit_alt_km=550,
            rail_mm=3.0,
            deck_mm=2.5,
            material=Material.AL_6061_T6,
        )

        generator = DiffusionGenerator()
        candidates = generator.generate_candidates(spec, n=5)

        # Check correct number of candidates
        assert len(candidates) == 5

        # Check each candidate has required properties
        for i, candidate in enumerate(candidates):
            assert isinstance(candidate, GeneratedMesh)
            assert candidate.vertices.shape[1] == 3  # 3D vertices
            assert candidate.faces.shape[1] == 3  # Triangular faces
            assert len(candidate.vertices) > 0
            assert len(candidate.faces) > 0
            assert isinstance(candidate.metadata, dict)
            assert "variant" in candidate.metadata
            assert candidate.metadata["variant"] == i

    def test_geometric_diversity(self):
        """Test that generated candidates have geometric diversity."""
        spec = MissionSpec(
            bus_u=3,
            payload_mass_kg=1.0,
            orbit_alt_km=550,
            rail_mm=3.0,
            deck_mm=2.5,
            material=Material.AL_6061_T6,
        )

        generator = DiffusionGenerator()
        candidates = generator.generate_candidates(spec, n=5)

        # Check mass diversity (should have >15% variation)
        masses = [c.metadata.get("estimated_mass_kg", 0) for c in candidates]
        mass_range = max(masses) - min(masses)
        mass_variation = mass_range / np.mean(masses) if np.mean(masses) > 0 else 0

        assert (
            mass_variation > 0.15
        ), f"Mass variation {mass_variation:.2f} below 15% threshold"

        # Check rail thickness diversity
        rail_thicknesses = [c.metadata.get("rail_thickness", 0) for c in candidates]
        rail_range = max(rail_thicknesses) - min(rail_thicknesses)

        assert rail_range > 0, "No variation in rail thickness"


class TestMeshToSolidConverter:
    """Test the mesh to solid conversion functionality."""

    def test_converter_initialization(self):
        """Test converter initializes correctly."""
        converter = MeshToSolidConverter()
        # Should work with or without OpenCascade
        assert converter is not None

    def test_step_conversion(self):
        """Test conversion to STEP format."""
        # Create a simple test mesh
        vertices = np.array(
            [
                [0, 0, 0],
                [10, 0, 0],
                [10, 10, 0],
                [0, 10, 0],
                [0, 0, 10],
                [10, 0, 10],
                [10, 10, 10],
                [0, 10, 10],
            ],
            dtype=float,
        )

        faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],  # Bottom
                [4, 5, 6],
                [4, 6, 7],  # Top
                [0, 1, 5],
                [0, 5, 4],  # Front
                [2, 3, 7],
                [2, 7, 6],  # Back
            ]
        )

        mesh = GeneratedMesh(vertices=vertices, faces=faces, metadata={"test": True})

        converter = MeshToSolidConverter()

        with tempfile.TemporaryDirectory() as tmpdir:
            step_path = Path(tmpdir) / "test.step"
            success = converter.convert_mesh_to_step(mesh, step_path)

            assert success, "STEP conversion failed"
            assert step_path.exists(), "STEP file not created"
            assert step_path.stat().st_size > 0, "STEP file is empty"

    def test_stl_conversion(self):
        """Test conversion to STL format."""
        # Create a simple test mesh
        vertices = np.array([[0, 0, 0], [10, 0, 0], [5, 5, 5]], dtype=float)

        faces = np.array([[0, 1, 2]])

        mesh = GeneratedMesh(vertices=vertices, faces=faces, metadata={"test": True})

        converter = MeshToSolidConverter()

        with tempfile.TemporaryDirectory() as tmpdir:
            stl_path = Path(tmpdir) / "test.stl"
            success = converter.convert_mesh_to_stl(mesh, stl_path)

            assert success, "STL conversion failed"
            assert stl_path.exists(), "STL file not created"
            assert stl_path.stat().st_size > 0, "STL file is empty"

            # Check STL format
            content = stl_path.read_text()
            assert "solid" in content.lower()
            assert "facet" in content.lower()
            assert "vertex" in content.lower()


class TestDiffusionWorkflow:
    """Test the complete diffusion workflow."""

    def test_workflow_initialization(self):
        """Test workflow initializes correctly."""
        spec = MissionSpec(
            bus_u=3,
            payload_mass_kg=1.0,
            orbit_alt_km=550,
            rail_mm=3.0,
            deck_mm=2.5,
            material=Material.AL_6061_T6,
        )

        workflow = DiffusionWorkflow(spec)
        assert workflow.spec == spec
        assert workflow.generator is not None
        assert workflow.converter is not None

    def test_complete_workflow_execution(self):
        """Test the complete workflow runs successfully."""
        spec = MissionSpec(
            bus_u=3,
            payload_mass_kg=1.0,
            orbit_alt_km=550,
            rail_mm=3.0,
            deck_mm=2.5,
            material=Material.AL_6061_T6,
        )

        workflow = DiffusionWorkflow(spec)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Run workflow with minimal validation for speed
            summary_path = workflow.run_complete_workflow(
                output_dir=output_dir,
                n_candidates=3,  # Smaller number for testing
                run_fea=True,
                run_dfam=True,
            )

            # Check summary file exists
            assert summary_path.exists()

            # Load and validate summary
            with open(summary_path) as f:
                summary = json.load(f)

            assert summary["workflow"] == "diffusion_generation"
            assert summary["total_candidates"] == 3
            assert len(summary["results"]) == 3

            # Check each result
            for result in summary["results"]:
                assert "design" in result
                assert "status" in result
                assert "mass_kg" in result
                assert result["vertices"] > 0
                assert result["faces"] > 0

                # Check files were created
                if result.get("step_file"):
                    step_path = output_dir / result["step_file"]
                    assert step_path.exists()

                if result.get("stl_file"):
                    stl_path = output_dir / result["stl_file"]
                    assert stl_path.exists()

    def test_fea_pass_rate_requirement(self):
        """Test that FEA pass rate meets ≥60% requirement."""
        spec = MissionSpec(
            bus_u=3,
            payload_mass_kg=1.0,
            orbit_alt_km=550,
            rail_mm=3.0,
            deck_mm=2.5,
            material=Material.AL_6061_T6,
        )

        workflow = DiffusionWorkflow(spec)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            summary_path = workflow.run_complete_workflow(
                output_dir=output_dir, n_candidates=5, run_fea=True, run_dfam=False
            )

            with open(summary_path) as f:
                summary = json.load(f)

            # Count FEA passes
            fea_passes = sum(
                1 for r in summary["results"] if r.get("fea_status") == "PASS"
            )
            total = len(summary["results"])
            pass_rate = fea_passes / total if total > 0 else 0

            # Note: With mock validation, this should pass
            # In production with real FEA, this verifies the 60% requirement
            assert (
                pass_rate >= 0.6
            ), f"FEA pass rate {pass_rate:.1%} below 60% requirement"

    def test_printability_requirement(self):
        """Test that all generated designs are printable (STL export succeeds)."""
        spec = MissionSpec(
            bus_u=3,
            payload_mass_kg=1.0,
            orbit_alt_km=550,
            rail_mm=3.0,
            deck_mm=2.5,
            material=Material.AL_6061_T6,
        )

        workflow = DiffusionWorkflow(spec)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            summary_path = workflow.run_complete_workflow(
                output_dir=output_dir, n_candidates=5, run_fea=False, run_dfam=True
            )

            with open(summary_path) as f:
                summary = json.load(f)

            # Check that all candidates have STL files
            stl_count = sum(
                1 for r in summary["results"] if r.get("stl_file") is not None
            )
            total = len(summary["results"])

            assert (
                stl_count == total
            ), f"Only {stl_count}/{total} designs have STL files"


class TestMVPCompliance:
    """Test compliance with MVP v0.1.3 requirements."""

    def test_cli_integration_simulation(self):
        """Simulate the CLI integration requirement."""
        # This would test: orbitforge run demo.json --generator diffusion

        spec = MissionSpec(
            bus_u=3,
            payload_mass_kg=1.0,
            orbit_alt_km=550,
            rail_mm=3.0,
            deck_mm=2.5,
            material=Material.AL_6061_T6,
        )

        # Simulate CLI workflow
        workflow = DiffusionWorkflow(spec)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # This should complete without errors
            summary_path = workflow.run_complete_workflow(
                output_dir=output_dir, n_candidates=5, run_fea=True, run_dfam=True
            )

            assert summary_path.exists()

            # Verify output structure matches CLI expectations
            with open(summary_path) as f:
                summary = json.load(f)

            assert "results" in summary
            assert len(summary["results"]) == 5

    def test_diversity_requirement(self):
        """Test that designs meet 15% variance requirement."""
        spec = MissionSpec(
            bus_u=3,
            payload_mass_kg=1.0,
            orbit_alt_km=550,
            rail_mm=3.0,
            deck_mm=2.5,
            material=Material.AL_6061_T6,
        )

        generator = DiffusionGenerator()
        candidates = generator.generate_candidates(spec, n=5)

        # Check mass variance
        masses = [c.metadata.get("estimated_mass_kg", 0) for c in candidates]
        mass_range = max(masses) - min(masses)
        mass_mean = np.mean(masses)
        mass_variance_pct = (mass_range / mass_mean) * 100 if mass_mean > 0 else 0

        assert (
            mass_variance_pct >= 15
        ), f"Mass variance {mass_variance_pct:.1f}% below 15% requirement"

        # Check rail thickness variance
        rail_thicknesses = [c.metadata.get("rail_thickness", 0) for c in candidates]
        rail_range = max(rail_thicknesses) - min(rail_thicknesses)
        rail_mean = np.mean(rail_thicknesses)
        rail_variance_pct = (rail_range / rail_mean) * 100 if rail_mean > 0 else 0

        assert (
            rail_variance_pct >= 15
        ), f"Rail variance {rail_variance_pct:.1f}% below 15% requirement"

    def test_performance_requirements(self):
        """Test MVP v0.1.3 performance requirements:
        1. Generate 3-5 valid frame designs
        2. >60% FEA pass rate
        3. 100% printability
        4. ≥15% geometric diversity
        5. CLI integration
        """
        spec = MissionSpec(
            bus_u=3,
            payload_mass_kg=1.0,
            orbit_alt_km=550,
            rail_mm=3.0,
            deck_mm=2.5,
            material=Material.AL_6061_T6,
        )

        workflow = DiffusionWorkflow(spec)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Run workflow with all validations
            summary_path = workflow.run_complete_workflow(
                output_dir=output_dir, n_candidates=5, run_fea=True, run_dfam=True
            )

            # Load and validate results
            with open(summary_path) as f:
                summary = json.load(f)

            results = summary["results"]

            # 1. Check number of valid designs (3-5 required)
            valid_designs = [r for r in results if r["status"] == "PASS"]
            assert (
                len(valid_designs) >= 3
            ), f"Only {len(valid_designs)} valid designs (need ≥3)"

            # 2. Check FEA pass rate (>60% required)
            fea_results = [r.get("fea_status", "FAIL") for r in results]
            fea_passed = sum(1 for status in fea_results if status == "PASS")
            fea_rate = (fea_passed / len(results)) * 100
            assert fea_rate > 60, f"FEA pass rate {fea_rate:.1f}% below 60% requirement"

            # 3. Check printability (100% required)
            stl_files = [r.get("stl_file") is not None for r in results]
            assert all(stl_files), "Not all designs have STL files"

            # 4. Check geometric diversity (≥15% required)
            masses = [r["mass_kg"] for r in results if r["mass_kg"] is not None]
            if len(masses) > 1:
                mass_range = max(masses) - min(masses)
                mass_variance = (mass_range / np.mean(masses)) * 100
                assert (
                    mass_variance >= 15
                ), f"Mass diversity {mass_variance:.1f}% below 15% requirement"

            # 5. Check CLI integration (files and summary exist)
            assert summary_path.exists(), "Summary file not created"
            assert "workflow" in summary, "Summary missing workflow info"
            assert summary["workflow"] == "diffusion_generation"

            # Additional performance metrics
            print("\nMVP v0.1.3 Performance Metrics:")
            print(f"✓ Valid designs: {len(valid_designs)}/5")
            print(f"✓ FEA pass rate: {fea_rate:.1f}%")
            print(f"✓ Printability: {sum(stl_files)}/5")
            print(f"✓ Mass diversity: {mass_variance:.1f}%")
            print(f"✓ Output directory: {output_dir}")


# Integration test that would be run as part of CI/CD
def test_diffusion_generator_regression():
    """Main regression test for diffusion generator feature."""

    # Create test mission spec
    spec = MissionSpec(
        bus_u=3,
        payload_mass_kg=1.0,
        orbit_alt_km=550,
        rail_mm=3.0,
        deck_mm=2.5,
        material=Material.AL_6061_T6,
    )

    # Test complete workflow
    workflow = DiffusionWorkflow(spec)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Run with validation
        summary_path = workflow.run_complete_workflow(
            output_dir=output_dir, n_candidates=5, run_fea=True, run_dfam=True
        )

        # Load results
        with open(summary_path) as f:
            summary = json.load(f)

        results = summary["results"]

        # Verify MVP requirements
        assert len(results) == 5, "Should generate 5 candidates"

        # At least 3 should pass validation (60% requirement)
        passed = [r for r in results if r["status"] == "PASS"]
        assert len(passed) >= 3, f"Only {len(passed)}/5 designs passed validation"

        # All should have CAD files
        with_files = [r for r in results if r.get("step_file") or r.get("stl_file")]
        assert len(with_files) == 5, "Not all designs have CAD files"

        # Check diversity
        masses = [r["mass_kg"] for r in results if r["mass_kg"] is not None]
        if len(masses) > 1:
            mass_range = max(masses) - min(masses)
            mass_variance = (mass_range / np.mean(masses)) * 100
            assert mass_variance >= 15, f"Mass diversity {mass_variance:.1f}% below 15%"

        print(f"✓ Diffusion generator regression test passed")
        print(f"  - Generated {len(results)} candidates")
        print(f"  - {len(passed)} passed validation")
        print(f"  - Mass range: {min(masses):.3f} - {max(masses):.3f} kg")
        print(f"  - Files created in {output_dir}")


if __name__ == "__main__":
    # Run the main regression test
    test_diffusion_generator_regression()
