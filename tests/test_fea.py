"""Tests for the fast physics validation module."""

import pytest
import numpy as np
from pathlib import Path

from orbitforge.fea import (
    FastPhysicsValidator,
    MaterialProperties,
    LoadCase,
    ValidationResults,
    compute_stresses,
    solve_static,
    assemble_system,
)


# Test material properties (Al 6061-T6)
@pytest.fixture
def material_props():
    return MaterialProperties(
        name="Al_6061_T6",
        yield_strength_mpa=276,
        youngs_modulus_gpa=68.9,
        poissons_ratio=0.33,
        density_kg_m3=2700,
        cte_per_k=23.6e-6,
    )


# Test simple cube mesh
@pytest.fixture
def cube_mesh():
    # Create a simple unit cube mesh with 5 tetrahedra
    nodes = np.array(
        [
            [0, 0, 0],  # 0
            [1, 0, 0],  # 1
            [0, 1, 0],  # 2
            [0, 0, 1],  # 3
            [1, 1, 1],  # 4
        ]
    )

    elements = np.array(
        [
            [0, 1, 2, 4],  # Front tetrahedron
            [0, 2, 3, 4],  # Left tetrahedron
            [0, 1, 3, 4],  # Bottom tetrahedron
            [1, 2, 3, 4],  # Center tetrahedron
        ],
        dtype=np.int32,
    )

    return {"nodes": nodes, "elements": elements}


def test_material_properties():
    """Test material properties dataclass."""
    mat = MaterialProperties(
        name="Test",
        yield_strength_mpa=100,
        youngs_modulus_gpa=200,
        poissons_ratio=0.3,
        density_kg_m3=7800,
    )

    assert mat.name == "Test"
    assert mat.yield_strength_mpa == 100
    assert mat.youngs_modulus_gpa == 200
    assert mat.poissons_ratio == 0.3
    assert mat.density_kg_m3 == 7800
    assert mat.cte_per_k is None


def test_load_case():
    """Test load case dataclass."""
    load = LoadCase("Test", 5.0, (1.0, 0.0, 0.0))
    assert load.name == "Test"
    assert load.acceleration_g == 5.0
    assert load.direction == (1.0, 0.0, 0.0)


def test_validation_results():
    """Test validation results dataclass."""
    results = ValidationResults(
        max_stress_mpa=200,
        sigma_allow_mpa=250,
        status="PASS",
        thermal_stress_mpa=100,
        thermal_status="PASS",
    )

    assert results.max_stress_mpa == 200
    assert results.sigma_allow_mpa == 250
    assert results.status == "PASS"
    assert results.thermal_stress_mpa == 100
    assert results.thermal_status == "PASS"


def test_solver_assembly(cube_mesh, material_props):
    """Test stiffness matrix assembly."""
    K, ndof = assemble_system(
        cube_mesh["nodes"], cube_mesh["elements"], material_props.to_dict()
    )

    # Basic checks
    assert K.shape == (15, 15)  # 5 nodes * 3 DOFs
    assert ndof == 15
    assert K.nnz > 0  # Should have non-zero entries
    assert np.allclose(K.toarray(), K.toarray().T)  # Should be symmetric


def test_solver_static_solution(cube_mesh, material_props):
    """Test static solution."""
    # Setup system
    K, ndof = assemble_system(
        cube_mesh["nodes"], cube_mesh["elements"], material_props.to_dict()
    )

    # Apply unit force in z direction
    f = np.zeros(ndof)
    f[14] = 1.0  # Force on top node in z direction

    # Fix base nodes
    fixed_dofs = [0, 1, 2]  # Fix first node

    # Solve
    u = solve_static(K, ndof, f, fixed_dofs)

    # Basic checks
    assert len(u) == ndof
    assert np.all(u[fixed_dofs] == 0)  # Fixed DOFs should have zero displacement
    assert np.any(u != 0)  # Should have some non-zero displacements


def test_stress_calculation(cube_mesh, material_props):
    """Test stress calculation."""
    # Setup and solve system
    K, ndof = assemble_system(
        cube_mesh["nodes"], cube_mesh["elements"], material_props.to_dict()
    )

    f = np.zeros(ndof)
    f[14] = 1000.0  # Significant force to generate measurable stress

    fixed_dofs = [0, 1, 2]
    u = solve_static(K, ndof, f, fixed_dofs)

    # Calculate stresses
    von_mises = compute_stresses(
        cube_mesh["nodes"], cube_mesh["elements"], u, material_props.to_dict()
    )

    # Basic checks
    assert len(von_mises) == len(cube_mesh["elements"])
    assert np.all(von_mises >= 0)  # Von Mises stress should be non-negative
    assert np.any(von_mises > 0)  # Should have some non-zero stresses


def test_thermal_stress(material_props):
    """Test thermal stress calculation."""
    validator = FastPhysicsValidator(Path("dummy.step"), material_props)
    thermal_stress = validator.check_thermal_stress(-40, 60)

    # Basic checks
    assert thermal_stress > 0
    assert isinstance(thermal_stress, float)


def test_validation_status(material_props):
    """Test validation status determination."""
    # Create results that should pass
    results = ValidationResults(
        max_stress_mpa=material_props.yield_strength_mpa * 0.5,  # Well below yield
        sigma_allow_mpa=material_props.yield_strength_mpa,
        status="PASS",
        thermal_stress_mpa=material_props.yield_strength_mpa * 0.3,
        thermal_status="PASS",
    )

    assert results.status == "PASS"
    assert results.thermal_status == "PASS"

    # Create results that should fail
    results = ValidationResults(
        max_stress_mpa=material_props.yield_strength_mpa * 1.5,  # Above yield
        sigma_allow_mpa=material_props.yield_strength_mpa,
        status="FAIL",
        thermal_stress_mpa=material_props.yield_strength_mpa * 1.2,
        thermal_status="FAIL",
    )

    assert results.status == "FAIL"
    assert results.thermal_status == "FAIL"
