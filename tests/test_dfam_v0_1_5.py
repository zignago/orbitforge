"""Tests for DfAM v0.1.5 Post-Processor implementation."""

import pytest
import tempfile
import json
from pathlib import Path
import numpy as np
import trimesh

from orbitforge.dfam import (
    DfamChecker,
    WallChecker,
    OverhangChecker,
    DrainHoleChecker,
    STLExporter,
    DfamPostProcessor,
)
from orbitforge.dfam.rules import DfamViolation
from orbitforge.generator.mission import MissionSpec, Material
from orbitforge.generator.basic_frame import build_basic_frame


class TestWallChecker:
    """Unit tests for WallChecker component."""

    def test_wall_check_thin_shell(self):
        """Test that thin shell model triggers wall thickness violation."""
        # Create a very thin box (0.5mm thick, below 0.8mm limit)
        box = trimesh.creation.box(extents=[10, 10, 0.5e-3])  # 0.5mm thick in Z

        checker = WallChecker(box, min_thickness_mm=0.8)
        violations = checker.check()

        # Should detect thin wall violation
        assert len(violations) > 0
        assert any(v.rule == "min_wall_thickness" for v in violations)
        assert all(
            v.severity == "ERROR" for v in violations if v.rule == "min_wall_thickness"
        )

    def test_wall_check_thick_part(self):
        """Test that thick part passes wall thickness check."""
        # Create a thick box (2mm thick, above 0.8mm limit)
        box = trimesh.creation.box(extents=[10e-3, 10e-3, 2e-3])  # 2mm thick

        checker = WallChecker(box, min_thickness_mm=0.8)
        violations = checker.check()

        # Should not detect wall thickness violations
        wall_violations = [v for v in violations if v.rule == "min_wall_thickness"]
        assert len(wall_violations) == 0


class TestOverhangChecker:
    """Unit tests for OverhangChecker component."""

    def test_overhang_check_steep_slope(self):
        """Test that steep sloped mesh triggers overhang violation."""
        # Create a mesh with steep overhang (60 degrees from horizontal, > 45 degree limit)
        vertices = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0.5, 0, 1],  # Creates steep overhang
                [0, 1, 0],
                [1, 1, 0],
                [0.5, 1, 1],
            ]
        )

        faces = np.array(
            [
                [0, 1, 2],  # Steep face
                [3, 5, 4],  # Steep face
                [0, 3, 4],  # Bottom face
                [0, 4, 1],  # Bottom face
                [1, 4, 5],  # Side face
                [1, 5, 2],  # Side face
                [2, 5, 3],  # Side face
                [2, 3, 0],  # Side face
            ]
        )

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.fix_normals()

        checker = OverhangChecker(mesh, max_overhang_angle=45.0)
        violations = checker.check()

        # Should detect overhang violations
        overhang_violations = [v for v in violations if v.rule == "max_overhang_angle"]
        assert len(overhang_violations) > 0
        assert all(v.severity == "WARNING" for v in overhang_violations)

    def test_overhang_check_shallow_slope(self):
        """Test that shallow slope passes overhang check."""
        # Create a mesh with shallow slope (30 degrees from horizontal, < 45 degree limit)
        vertices = np.array(
            [
                [0, 0, 0],
                [2, 0, 0],
                [1, 0, 0.5],  # Creates shallow slope
                [0, 1, 0],
                [2, 1, 0],
                [1, 1, 0.5],
            ]
        )

        faces = np.array(
            [
                [0, 1, 2],
                [3, 5, 4],
                [0, 3, 4],
                [0, 4, 1],
                [1, 4, 5],
                [1, 5, 2],
                [2, 5, 3],
                [2, 3, 0],
            ]
        )

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.fix_normals()

        checker = OverhangChecker(mesh, max_overhang_angle=45.0)
        violations = checker.check()

        # Should not detect overhang violations
        overhang_violations = [v for v in violations if v.rule == "max_overhang_angle"]
        assert len(overhang_violations) == 0


class TestDrainHoleChecker:
    """Unit tests for DrainHoleChecker component."""

    def test_drain_check_large_part(self):
        """Test that large part without outlets triggers drain hole requirement."""
        # Create a large solid box (60mm > 50mm limit)
        box = trimesh.creation.box(extents=[60e-3, 30e-3, 30e-3])

        checker = DrainHoleChecker(box, max_cavity_length_mm=50.0)
        violations = checker.check()

        # Should detect drain hole requirement
        drain_violations = [v for v in violations if v.rule == "powder_drain_holes"]
        assert len(drain_violations) > 0
        assert all(v.severity == "WARNING" for v in drain_violations)

    def test_drain_check_small_part(self):
        """Test that small part passes drain hole check."""
        # Create a small box (40mm < 50mm limit)
        box = trimesh.creation.box(extents=[40e-3, 30e-3, 30e-3])

        checker = DrainHoleChecker(box, max_cavity_length_mm=50.0)
        violations = checker.check()

        # Should not require drain holes
        drain_violations = [v for v in violations if v.rule == "powder_drain_holes"]
        assert len(drain_violations) == 0


class TestDfamChecker:
    """Integration tests for complete DfamChecker."""

    @pytest.fixture
    def sample_stl_file(self):
        """Create a sample STL file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
            # Create a simple box
            box = trimesh.creation.box(extents=[10e-3, 10e-3, 2e-3])
            box.export(f.name)
            yield Path(f.name)
            Path(f.name).unlink()  # Cleanup

    def test_run_all_checks(self, sample_stl_file):
        """Test that run_all_checks returns proper structure."""
        checker = DfamChecker(sample_stl_file)
        results = checker.run_all_checks()

        # Check result structure
        assert "status" in results
        assert "error_count" in results
        assert "warning_count" in results
        assert "wall_thickness" in results
        assert "overhang_check" in results
        assert "drain_holes" in results
        assert "violations" in results

        # Check status values
        assert results["status"] in ["PASS", "FAIL"]
        assert isinstance(results["error_count"], int)
        assert isinstance(results["warning_count"], int)
        assert results["wall_thickness"] in ["PASS", "FAIL"]
        assert results["overhang_check"] in ["PASS", "FAIL"]
        assert results["drain_holes"] in ["PASS", "FAIL"]

    def test_save_dfam_report(self, sample_stl_file):
        """Test DfAM report generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            checker = DfamChecker(sample_stl_file)
            report_path = checker.save_dfam_report(output_dir, "test_report")

            assert report_path.exists()
            assert report_path.name == "test_report.json"

            # Verify report content
            with open(report_path) as f:
                report_data = json.load(f)

            assert "status" in report_data
            assert "violations" in report_data


class TestDfamPostProcessor:
    """Integration tests for DfamPostProcessor."""

    @pytest.fixture
    def test_design_dir(self):
        """Create a test design directory with frame files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            design_dir = Path(tmpdir) / "test_design"
            design_dir.mkdir()

            # Create a sample STL file
            box = trimesh.creation.box(extents=[10e-3, 10e-3, 2e-3])
            stl_path = design_dir / "frame.stl"
            box.export(str(stl_path))

            yield design_dir

    def test_process_design(self, test_design_dir):
        """Test processing a single design."""
        processor = DfamPostProcessor()
        result = processor.process_design(test_design_dir)

        # Check result structure
        assert "dfam_status" in result
        assert "error_count" in result
        assert "warning_count" in result
        assert "wall_thickness" in result
        assert "overhang_check" in result
        assert "drain_holes" in result
        assert "stl_exported" in result
        assert "report_path" in result

        # Check that report was created
        if result["report_path"]:
            report_file = test_design_dir / result["report_path"]
            assert report_file.exists()


class TestRegressionSuite:
    """Regression tests for known-valid geometries."""

    def test_basic_frame_dfam_compliance(self):
        """Test that basic frame generator produces DfAM-compliant geometry."""
        spec = MissionSpec(
            bus_u=1,
            payload_mass_kg=0.5,
            orbit_alt_km=400,
            rail_mm=3.0,  # Above 0.8mm minimum
            deck_mm=2.5,  # Above 0.8mm minimum
            material=Material.AL_6061_T6,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Generate frame
            step_file = build_basic_frame(spec, output_dir)
            assert step_file.exists()

            # STL should also be generated
            stl_file = output_dir / "frame.stl"
            assert stl_file.exists()

            # Run DfAM checks
            processor = DfamPostProcessor()
            result = processor.process_design(output_dir)

            # Should pass DfAM checks (basic frame is designed to be manufacturable)
            assert result["dfam_status"] == "PASS"
            assert result["error_count"] == 0
            assert result["stl_exported"] == True


class TestEdgeCases:
    """Edge case tests for robustness."""

    def test_fail_all_rules(self):
        """Test model violating all 3 DfAM rules."""
        # Create problematic geometry:
        # - Very thin (violates wall thickness)
        # - Steep overhang (violates overhang angle)
        # - Large size (violates drain hole requirement)

        vertices = np.array(
            [
                [0, 0, 0],
                [60e-3, 0, 0],  # 60mm long (>50mm limit)
                [30e-3, 0, 0.1e-3],  # 0.1mm thick (< 0.8mm limit)
                [0, 10e-3, 0],
                [60e-3, 10e-3, 0],
                [30e-3, 10e-3, 0.1e-3],
            ]
        )

        # Create faces with steep overhangs
        faces = np.array(
            [
                [0, 2, 1],  # Steep overhang face
                [3, 4, 5],  # Steep overhang face
                [0, 1, 4],  # Bottom
                [0, 4, 3],  # Bottom
                [1, 2, 5],  # Side
                [1, 5, 4],  # Side
                [2, 0, 3],  # Side
                [2, 3, 5],  # Side
            ]
        )

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.fix_normals()

        # Save to temporary STL
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
            mesh.export(f.name)
            stl_path = Path(f.name)

        try:
            checker = DfamChecker(stl_path)
            results = checker.run_all_checks()

            # Should fail overall
            assert results["status"] == "FAIL"

            # Should have violations in multiple categories
            assert results["error_count"] > 0

            # Should flag multiple rules as failing
            failing_rules = 0
            if results["wall_thickness"] == "FAIL":
                failing_rules += 1
            if results["overhang_check"] == "FAIL":
                failing_rules += 1
            if results["drain_holes"] == "FAIL":
                failing_rules += 1

            assert failing_rules >= 2  # Should fail at least 2 of the 3 rules

        finally:
            stl_path.unlink()  # Cleanup

    def test_missing_stl_file(self):
        """Test handling of missing STL file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            design_dir = Path(tmpdir)

            processor = DfamPostProcessor()
            result = processor.process_design(design_dir)

            assert result["dfam_status"] == "FAIL"
            assert "STL file not found" in result["error"]
            assert result["stl_exported"] == False


class TestCLIIntegration:
    """Test CLI integration with --dfam flag."""

    def test_dfam_flag_integration(self):
        """Test that --dfam flag triggers DfAM post-processing."""
        # This would require CLI testing infrastructure
        # For now, we verify the flag exists and is properly configured
        from orbitforge.cli import app

        # Check that the dfam parameter exists in the CLI
        run_command = None
        for command in app.commands.values():
            if command.name == "run":
                run_command = command
                break

        assert run_command is not None

        # Check that dfam parameter exists
        dfam_param = None
        for param in run_command.params:
            if param.name == "dfam":
                dfam_param = param
                break

        assert dfam_param is not None
        assert dfam_param.help == "Run DfAM checks"
