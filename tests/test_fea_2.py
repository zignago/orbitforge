import json
import pytest
from pathlib import Path
from orbitforge.fea import FastPhysicsValidator, MaterialProperties
from orbitforge.generator.mission import MissionSpec


@pytest.fixture
def output_dir():
    """Create and clean up test output directory."""
    path = Path("outputs")
    path.mkdir(exist_ok=True)
    yield path
    # Uncomment to clean up after tests
    # shutil.rmtree(path)


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
    variants = {
        "normal": 3.0,  # Standard rail thickness
        "thin": 1.5,  # Too thin, should fail
        "thick": 5.0,  # Extra thick, should pass
    }

    for name, rail_mm in variants.items():
        # Create variant-specific mission
        mission_dict = base_mission.copy()
        mission_dict["rail_mm"] = rail_mm

        # Create output directory
        design_dir = output_dir / f"design_001_{name}"
        design_dir.mkdir(exist_ok=True)

        # Generate frame
        spec = MissionSpec(**mission_dict)
        step_file = build_basic_frame(spec, design_dir)
        frames[name] = step_file

    return frames


def test_fea_validator_creates_result_file(output_dir, titanium, test_frames):
    """Test that the validator creates a result file with proper format."""
    results_path = output_dir / "results.json"

    validator = FastPhysicsValidator(
        step_file=test_frames["normal"],
        material=titanium,
        load_scale=30.0,  # artificially raise stress to expected test range
    )

    results = validator.run_validation()

    # Basic validation checks
    assert isinstance(results.max_stress_mpa, float)
    assert results.sigma_allow_mpa > 0
    assert results.status in ["PASS", "FAIL"]

    # Stress should be reasonable for a 6g load
    # For a 3U CubeSat with 1kg payload at 6g, we expect stresses in the 100-500 MPa range
    assert (
        results.max_stress_mpa < titanium.yield_strength_mpa
    ), f"Stress {results.max_stress_mpa} exceeds yield strength!"

    # Check that stress is at least positive
    assert results.max_stress_mpa > 0, "Stress should be positive"

    # Save results for inspection
    validator.save_results(results, results_path)
    assert results_path.exists()


def test_thin_rail_has_higher_stress_than_thick(titanium, test_frames):
    """Thin rail should produce higher stress under same load."""
    validator_thin = FastPhysicsValidator(
        step_file=test_frames["thin"], material=titanium
    )
    validator_thick = FastPhysicsValidator(
        step_file=test_frames["thick"], material=titanium
    )

    stress_thin = validator_thin.run_validation().max_stress_mpa
    stress_thick = validator_thick.run_validation().max_stress_mpa

    assert stress_thin > stress_thick, (
        "Expected thin rail to produce higher stress than thick rail, "
        f"got {stress_thin:.2f} MPa vs {stress_thick:.2f} MPa"
    )


def test_thick_rail_should_pass(titanium, test_frames):
    """Test that thick rails pass validation."""
    validator = FastPhysicsValidator(step_file=test_frames["thick"], material=titanium)
    results = validator.run_validation()

    # For thick rails, stress should be well below yield strength
    assert (
        results.max_stress_mpa < titanium.yield_strength_mpa
    ), f"Stress {results.max_stress_mpa} MPa exceeds yield strength!"
    assert (
        results.max_stress_mpa > 0
    ), "Stress should be positive, something may be wrong with loading or meshing"
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

    # Validate stress values
    assert isinstance(data["max_stress_MPa"], (int, float))
    assert isinstance(data["sigma_allow_MPa"], (int, float))
    assert data["max_stress_MPa"] > 0
