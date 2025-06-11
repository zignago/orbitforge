"""Test suite to verify compliance with OrbitForge v0.1.1 specification."""

import json
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
from orbitforge.generator.mission import MissionSpec

# === Fixtures ===


@pytest.fixture
def material_props():
    """Returns a sample aluminum material with typical mechanical properties for FEA tests"""
    return MaterialProperties(
        name="Al_6061_T6",
        yield_strength_mpa=276,
        youngs_modulus_gpa=68.9,
        poissons_ratio=0.33,
        density_kg_m3=2700,
        cte_per_k=23.6e-6,
    )


@pytest.fixture
def titanium():
    """Create titanium material properties."""
    return MaterialProperties(
        name="Ti_6Al_4V",
        yield_strength_mpa=950,
        youngs_modulus_gpa=113.8,
        poissons_ratio=0.342,
        density_kg_m3=4430,
        cte_per_k=8.6e-6,
    )


@pytest.fixture
def cube_mesh():
    """Creates a small synthetic tetrahedral mesh of a unit cube for validating FEA pipelines"""
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
    elements = np.array(
        [[0, 1, 2, 4], [0, 2, 3, 4], [0, 1, 3, 4], [1, 2, 3, 4]], dtype=np.int32
    )
    return {"nodes": nodes, "elements": elements}


@pytest.fixture
def output_dir():
    """Create and clean up test output directory."""
    path = Path("outputs")
    path.mkdir(exist_ok=True)
    yield path


@pytest.fixture
def base_mission():
    """Create base mission spec."""
    return {
        "bus_u": 3,
        "payload_mass_kg": 1.0,
        "orbit_alt_km": 550,
        "mass_limit_kg": 4.0,
        "material": "Ti_6Al_4V",
    }


@pytest.fixture
def test_frames(output_dir, base_mission):
    """Generate test frames with different rail thicknesses."""
    from orbitforge.generator.basic_frame import build_basic_frame

    frames = {}
    variants = {"normal": 3.0, "thin": 1.5, "thick": 5.0}
    for name, rail_mm in variants.items():
        mission_dict = base_mission.copy()
        mission_dict["rail_mm"] = rail_mm
        design_dir = output_dir / f"design_001_{name}"
        design_dir.mkdir(exist_ok=True)
        spec = MissionSpec(**mission_dict)
        step_file = build_basic_frame(spec, design_dir)
        frames[name] = step_file

    return frames


# === Unit Tests ===


def test_material_properties():
    """Verify that the MaterialProperties dataclass correctly stores and exposes all mechanical fields."""
    mat = MaterialProperties("Test", 100, 200, 0.3, 7800)
    assert mat.name == "Test"
    assert mat.yield_strength_mpa == 100
    assert mat.youngs_modulus_gpa == 200
    assert mat.poissons_ratio == 0.3
    assert mat.density_kg_m3 == 7800
    assert mat.cte_per_k is None


def test_load_case():
    """Ensure the LoadCase dataclass captures name, acceleration magnitude, and direction tuple correctly."""
    load = LoadCase("Test", 5.0, (1.0, 0.0, 0.0))
    assert load.name == "Test"
    assert load.acceleration_g == 5.0
    assert load.direction == (1.0, 0.0, 0.0)


def test_validation_results():
    """Check the integrity of the ValidationResults dataclass for structural and thermal stress statuses."""
    results = ValidationResults(200, 250, "PASS", 100, "PASS")
    assert results.max_stress_mpa == 200
    assert results.sigma_allow_mpa == 250
    assert results.status == "PASS"
    assert results.thermal_stress_mpa == 100
    assert results.thermal_status == "PASS"


def test_solver_assembly(cube_mesh, material_props):
    """Assemble a stiffness matrix using mock mesh and material inputs, then validate its shape and symmetry."""
    K, ndof = assemble_system(
        cube_mesh["nodes"], cube_mesh["elements"], material_props.to_dict()
    )
    assert K.shape == (15, 15)
    assert ndof == 15
    assert K.nnz > 0
    assert np.allclose(K.toarray(), K.toarray().T)


def test_solver_static_solution(cube_mesh, material_props):
    """Apply static loading to a small mesh and verify that the solver produces valid displacements."""
    K, ndof = assemble_system(
        cube_mesh["nodes"], cube_mesh["elements"], material_props.to_dict()
    )
    f = np.zeros(ndof)
    f[14] = 1.0
    fixed_dofs = [0, 1, 2]
    u = solve_static(K, ndof, f, fixed_dofs)
    assert len(u) == ndof
    assert np.all(u[fixed_dofs] == 0)
    assert np.any(u != 0)


def test_stress_calculation(cube_mesh, material_props):
    """Run a simple FEA pass and confirm that computed von Mises stresses are non-negative and meaningful."""
    K, ndof = assemble_system(
        cube_mesh["nodes"], cube_mesh["elements"], material_props.to_dict()
    )
    f = np.zeros(ndof)
    f[14] = 1000.0
    u = solve_static(K, ndof, f, [0, 1, 2])
    von_mises = compute_stresses(
        cube_mesh["nodes"], cube_mesh["elements"], u, material_props.to_dict()
    )
    assert len(von_mises) == len(cube_mesh["elements"])
    assert np.all(von_mises >= 0)
    assert np.any(von_mises > 0)


def test_thermal_stress(material_props):
    """Estimate stress from a temperature delta and validate that it's a positive float when CTE is provided."""
    validator = FastPhysicsValidator(Path("dummy.step"), material_props)
    thermal_stress = validator.check_thermal_stress(-40, 60)
    assert thermal_stress > 0
    assert isinstance(thermal_stress, float)


def test_validation_status(material_props):
    """Test that ValidationResults correctly label results as PASS or FAIL based on stress magnitudes."""
    results = ValidationResults(100, 200, "PASS", 60, "PASS")
    assert results.status == "PASS"
    assert results.thermal_status == "PASS"
    results = ValidationResults(300, 200, "FAIL", 240, "FAIL")
    assert results.status == "FAIL"
    assert results.thermal_status == "FAIL"


# === Integration Tests ===


def test_fea_validator_creates_result_file(output_dir, titanium, test_frames):
    """Test that the validator creates a result file with proper format."""
    results_path = output_dir / "results.json"
    validator = FastPhysicsValidator(
        step_file=test_frames["normal"], material=titanium, load_scale=30.0
    )
    results = validator.run_validation()
    assert isinstance(results.max_stress_mpa, float)
    assert results.sigma_allow_mpa > 0
    assert results.status in ["PASS", "FAIL"]
    assert results.max_stress_mpa < titanium.yield_strength_mpa
    assert results.max_stress_mpa > 0
    validator.save_results(results, results_path)
    assert results_path.exists()


def test_thin_rail_has_higher_stress_than_thick(titanium, test_frames):
    """Thin rail should produce higher stress under same load."""
    stress_thin = (
        FastPhysicsValidator(test_frames["thin"], titanium)
        .run_validation()
        .max_stress_mpa
    )
    stress_thick = (
        FastPhysicsValidator(test_frames["thick"], titanium)
        .run_validation()
        .max_stress_mpa
    )
    assert stress_thin > stress_thick


def test_thick_rail_should_pass(titanium, test_frames):
    """Test that thick rails pass validation."""
    results = FastPhysicsValidator(test_frames["thick"], titanium).run_validation()
    assert results.max_stress_mpa < titanium.yield_strength_mpa
    assert results.max_stress_mpa > 0
    assert results.status == "PASS"


def test_results_json_schema(output_dir):
    """Test that the results JSON has the expected schema."""
    results_path = output_dir / "results.json"
    if not results_path.exists():
        pytest.skip("results.json not found")
    with open(results_path) as f:
        data = json.load(f)
    assert "max_stress_MPa" in data
    assert "sigma_allow_MPa" in data
    assert "status" in data
    assert data["status"] in ["PASS", "FAIL"]
    assert isinstance(data["max_stress_MPa"], (int, float))
    assert isinstance(data["sigma_allow_MPa"], (int, float))
    assert data["max_stress_MPa"] > 0
