"""Test suite to verify compliance with OrbitForge v0.1.0 specification."""

import json
import pytest
from typer.testing import CliRunner

from orbitforge.cli import app
from orbitforge.generator.mission import MissionSpec
from orbitforge.generator.basic_frame import build_basic_frame
from orbitforge.fea import FastPhysicsValidator, MaterialProperties
from orbitforge.dfam.rules import DfamChecker


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def output_dir(tmp_path):
    """Create a temporary output directory."""
    return tmp_path / "outputs"


@pytest.fixture
def mission_specs():
    """Create test mission specs for different CubeSat sizes."""
    return {
        "3u": {
            "bus_u": 3,
            "payload_mass_kg": 1.0,
            "orbit_alt_km": 550,
            "mass_limit_kg": 4.0,
            "rail_mm": 3.0,
            "deck_mm": 2.5,
            "material": "Al_6061_T6",
        },
        "6u": {
            "bus_u": 6,
            "payload_mass_kg": 2.5,
            "orbit_alt_km": 600,
            "mass_limit_kg": 8.0,
            "rail_mm": 4.0,
            "deck_mm": 3.0,
            "material": "Ti_6Al_4V",
        },
    }


def test_cubesat_size_range(mission_specs, output_dir):
    """Test that the generator supports 3U to 6U CubeSats."""
    for size, spec in mission_specs.items():
        mission = MissionSpec(**spec)
        step_file = build_basic_frame(mission, output_dir)
        assert step_file.exists()

        # Basic size checks
        length_m = mission.bus_u * 0.1  # 1U = 10cm = 0.1m
        assert 0.3 <= round(length_m, 6) <= 0.6  # 3U to 6U range


def test_static_load_verification(mission_specs, output_dir):
    """Test static loading verification against launch vehicle environments."""
    mission = MissionSpec(**mission_specs["3u"])
    step_file = build_basic_frame(mission, output_dir)

    material = MaterialProperties(
        name=mission.material.value,
        yield_strength_mpa=mission.yield_mpa,
        youngs_modulus_gpa=mission.youngs_modulus_gpa,
        poissons_ratio=mission.poissons_ratio,
        density_kg_m3=mission.density_kg_m3,
        cte_per_k=mission.cte_per_k,
    )

    validator = FastPhysicsValidator(step_file, material)
    results = validator.run_validation()

    # Check that 6g loads are applied
    assert any(case.acceleration_g == 6.0 for case in validator.load_cases)

    # Results should include stress values
    assert hasattr(results, "max_stress_mpa")
    assert hasattr(results, "sigma_allow_mpa")
    assert results.status in ["PASS", "FAIL"]


def test_thermal_expansion_check(mission_specs, output_dir):
    """Test thermal expansion checks across standard gradient."""
    mission = MissionSpec(**mission_specs["3u"])
    step_file = build_basic_frame(mission, output_dir)

    material = MaterialProperties(
        name=mission.material.value,
        yield_strength_mpa=mission.yield_mpa,
        youngs_modulus_gpa=mission.youngs_modulus_gpa,
        poissons_ratio=mission.poissons_ratio,
        density_kg_m3=mission.density_kg_m3,
        cte_per_k=mission.cte_per_k,
    )

    validator = FastPhysicsValidator(step_file, material)
    thermal_stress = validator.check_thermal_stress(temp_cold=-40.0, temp_hot=60.0)

    # Should have reasonable thermal stress values
    assert thermal_stress > 0
    assert thermal_stress < material.yield_strength_mpa


def test_manufacturability_report(mission_specs, output_dir):
    """Test manufacturability analysis for AM."""
    mission = MissionSpec(**mission_specs["3u"])
    step_file = build_basic_frame(mission, output_dir)

    # STL file should be generated alongside STEP
    stl_file = step_file.parent / "frame.stl"
    assert stl_file.exists()

    # Run DfAM checks
    checker = DfamChecker(stl_file)
    results = checker.run_all_checks()

    # Should have basic DfAM metrics
    assert "status" in results
    assert "error_count" in results
    assert "warning_count" in results
    assert "violations" in results


def test_output_files_generation(runner, mission_specs, tmp_path):
    """Test generation of all required output files."""
    # Create mission file
    mission_file = tmp_path / "test_mission.json"
    with open(mission_file, "w") as f:
        json.dump(mission_specs["3u"], f)

    # Run with all checks enabled
    result = runner.invoke(
        app,
        [
            "run",
            str(mission_file),
            "--check",
            "--dfam",
            "--report",
            "--outdir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0

    # Find output directory
    output_dirs = list(tmp_path.glob("design_*"))
    assert len(output_dirs) == 1
    design_dir = output_dirs[0]

    # Check all required files exist
    assert (design_dir / "frame.step").exists()  # CAD skeleton
    assert (design_dir / "frame.stl").exists()  # AM-ready mesh
    assert (design_dir / "mass_budget.csv").exists()  # Mass budget
    assert (design_dir / "physics_check.json").exists()  # FEA summary
    assert (design_dir / "report.pdf").exists()  # Full report


def test_mass_budget_accuracy(mission_specs, output_dir):
    """Test mass budget calculations."""
    mission = MissionSpec(**mission_specs["3u"])
    step_file = build_basic_frame(mission, output_dir)

    # Read mass budget
    mass_file = step_file.parent / "mass_budget.csv"
    with open(mass_file) as f:
        lines = f.readlines()
        total_mass = float(lines[-1].split(",")[-1])  # Last line, last column

    # Mass should be within limits
    assert total_mass > 0
    assert total_mass <= mission.mass_limit_kg

    # For a 3U Al frame, typical mass should be 0.2-1.0 kg
    assert 0.15 <= total_mass <= 1.0


@pytest.mark.slow
def test_full_workflow_6u(runner, mission_specs, tmp_path):
    """Test complete workflow with 6U CubeSat."""
    # Create mission file
    mission_file = tmp_path / "test_6u.json"
    with open(mission_file, "w") as f:
        json.dump(mission_specs["6u"], f)

    outdir = tmp_path / "outputs"

    # Run full analysis
    result = runner.invoke(
        app,
        [
            "run",
            str(mission_file),
            "--check",
            "--dfam",
            "--report",
            "--material",
            "Ti_6Al_4V",
            "--rail",
            "4.0",
            "--deck",
            "3.0",
            "--outdir",
            str(outdir),
        ],
    )
    assert result.exit_code == 0

    # Verify outputs
    output_dirs = list(outdir.glob("design_*"))
    assert len(output_dirs) == 1
    design_dir = output_dirs[0]

    # Check all files
    required_files = [
        "frame.step",
        "frame.stl",
        "mass_budget.csv",
        "physics_check.json",
        "manufacturability.json",
        "report.pdf",
    ]
    for file in required_files:
        assert (design_dir / file).exists()

    # Load and check physics results
    with open(design_dir / "physics_check.json") as f:
        physics = json.load(f)
        assert physics["status"] in ["PASS", "FAIL"]
        assert "thermal_stress_MPa" in physics

    # Check mass budget for 6U
    with open(design_dir / "mass_budget.csv") as f:
        lines = f.readlines()
        total_mass = float(lines[-1].split(",")[-1])
        assert 0.4 <= total_mass <= 8.0  # 6U Ti frame typical range
