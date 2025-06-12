"""Unit tests for mesh to BDF conversion functionality."""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from orbitforge.fea.preprocessor import BDFWriter, MeshGenerator, convert_step_to_bdf


class TestBDFWriter:
    """Test BDF writing functionality."""

    def test_write_bdf_basic(self):
        """Test basic BDF file writing."""
        # Create simple test mesh
        nodes = np.array(
            [
                [0.0, 0.0, 0.0],  # Node 1
                [1.0, 0.0, 0.0],  # Node 2
                [1.0, 1.0, 0.0],  # Node 3
                [0.0, 1.0, 0.0],  # Node 4
            ]
        )

        elements = np.array(
            [
                [1, 2, 3, 4],  # Element 1 (using 1-based indexing for nodes)
            ]
        )

        material_props = {
            "elastic_modulus": 70e9,  # Pa
            "poisson_ratio": 0.33,
            "density": 2700,  # kg/m3
            "thickness": 1.0,  # mm
        }

        loads = {
            "accel_x": 0.0,
            "accel_y": 0.0,
            "accel_z": 9.81,
        }

        # Write BDF file
        with tempfile.TemporaryDirectory() as tmpdir:
            bdf_file = Path(tmpdir) / "test.bdf"
            writer = BDFWriter()
            writer.write_bdf(bdf_file, nodes, elements, material_props, loads)

            # Verify file was created
            assert bdf_file.exists()

            # Read and verify contents
            with open(bdf_file) as f:
                content = f.read()

            # Check for required sections
            assert "SOL 101" in content
            assert "BEGIN BULK" in content
            assert "MAT1" in content
            assert "PSHELL" in content
            assert "GRID" in content
            assert "CQUAD4" in content
            assert "SPC1" in content
            assert "FORCE" in content
            assert "ENDDATA" in content

    def test_write_material_properties(self):
        """Test material property writing."""
        nodes = np.array([[0.0, 0.0, 0.0]])
        elements = np.array([[1, 1, 1, 1]])  # Degenerate element for test

        material_props = {
            "elastic_modulus": 200e9,  # Steel
            "poisson_ratio": 0.3,
            "density": 7850,
            "thickness": 2.0,
        }

        loads = {"accel_z": 9.81}

        with tempfile.TemporaryDirectory() as tmpdir:
            bdf_file = Path(tmpdir) / "test_material.bdf"
            writer = BDFWriter()
            writer.write_bdf(bdf_file, nodes, elements, material_props, loads)

            with open(bdf_file) as f:
                content = f.read()

            # Check material card has correct values
            assert "MAT1" in content
            # Young's modulus should be converted to N/mm2
            assert "2.00E+05" in content  # 200 GPa = 200,000 N/mm2

    def test_write_constraints(self):
        """Test constraint writing."""
        nodes = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        elements = np.array([[1, 2, 3, 4]])
        material_props = {
            "elastic_modulus": 70e9,
            "poisson_ratio": 0.33,
            "density": 2700,
            "thickness": 1.0,
        }
        loads = {"accel_z": 9.81}

        with tempfile.TemporaryDirectory() as tmpdir:
            bdf_file = Path(tmpdir) / "test_constraints.bdf"
            writer = BDFWriter()

            # Test with explicit constraints
            constraints = [1, 4]  # Fix nodes 1 and 4
            writer.write_bdf(
                bdf_file, nodes, elements, material_props, loads, constraints
            )

            with open(bdf_file) as f:
                content = f.read()

            # Check constraint card
            assert "SPC1" in content
            assert "123456" in content  # All DOFs constrained


class TestMeshGeneration:
    """Test mesh generation (requires GMSH)."""

    @pytest.mark.skipif(
        not pytest.importorskip("gmsh", minversion="4.0"), reason="GMSH not available"
    )
    def test_simple_step_mesh(self):
        """Test mesh generation from a simple geometry."""
        # This would require a real STEP file for full testing
        # For now, we'll test the interface
        mesher = MeshGenerator(mesh_size=1.0)

        # Test initialization
        assert mesher.mesh_size == 1.0

        # Test error handling for non-existent file
        with pytest.raises(RuntimeError):
            mesher.step_to_mesh(Path("nonexistent.step"))


class TestIntegration:
    """Integration tests for STEP to BDF conversion."""

    def test_convert_step_to_bdf_interface(self):
        """Test the main conversion function interface."""
        # Test with non-existent STEP file to check error handling
        with tempfile.TemporaryDirectory() as tmpdir:
            step_file = Path(tmpdir) / "test.step"
            bdf_file = Path(tmpdir) / "test.bdf"

            material_props = {
                "elastic_modulus": 70e9,
                "poisson_ratio": 0.33,
                "density": 2700,
                "thickness": 1.0,
            }

            loads = {"accel_z": 9.81}

            # Should raise error for non-existent STEP file
            with pytest.raises(RuntimeError):
                convert_step_to_bdf(step_file, bdf_file, material_props, loads)

    def test_material_property_validation(self):
        """Test validation of material properties."""
        nodes = np.array([[0.0, 0.0, 0.0]])
        elements = np.array([[1, 1, 1, 1]])
        loads = {"accel_z": 9.81}

        # Test with missing material properties
        material_props = {}  # Empty dict

        with tempfile.TemporaryDirectory() as tmpdir:
            bdf_file = Path(tmpdir) / "test.bdf"
            writer = BDFWriter()

            # Should use default values
            writer.write_bdf(bdf_file, nodes, elements, material_props, loads)

            with open(bdf_file) as f:
                content = f.read()

            # Should contain default aluminum properties
            assert "MAT1" in content


class TestLoadCases:
    """Test load case generation."""

    def test_gravity_loads(self):
        """Test gravity load generation."""
        nodes = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )
        elements = np.array([[1, 2, 2, 2]])  # Degenerate quad
        material_props = {
            "elastic_modulus": 70e9,
            "poisson_ratio": 0.33,
            "density": 2700,
            "thickness": 1.0,
        }

        loads = {
            "accel_x": 2.0,  # 2g in X
            "accel_y": 0.0,
            "accel_z": 9.81,  # 1g in Z
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            bdf_file = Path(tmpdir) / "test_loads.bdf"
            writer = BDFWriter()
            writer.write_bdf(bdf_file, nodes, elements, material_props, loads)

            with open(bdf_file) as f:
                content = f.read()

            # Check for force cards
            assert "FORCE" in content
            # Check for correct load magnitudes (2g = 19620 N/kg)
            assert "19620" in content  # 2g * 9810

    def test_no_loads(self):
        """Test case with no loads applied."""
        nodes = np.array([[0.0, 0.0, 0.0]])
        elements = np.array([[1, 1, 1, 1]])
        material_props = {
            "elastic_modulus": 70e9,
            "poisson_ratio": 0.33,
            "density": 2700,
            "thickness": 1.0,
        }

        loads = {
            "accel_x": 0.0,
            "accel_y": 0.0,
            "accel_z": 0.0,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            bdf_file = Path(tmpdir) / "test_no_loads.bdf"
            writer = BDFWriter()
            writer.write_bdf(bdf_file, nodes, elements, material_props, loads)

            # File should still be created and valid
            assert bdf_file.exists()

            with open(bdf_file) as f:
                content = f.read()

            # Should still have load section but no force cards
            assert "$ Applied loads" in content
