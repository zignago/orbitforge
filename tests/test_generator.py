import json
import csv
import pytest
from pathlib import Path
from pydantic import ValidationError
from orbitforge.generator.mission import MissionSpec, Material, load_materials
from orbitforge.cli import app
from orbitforge.geometry import calculate_volume_m3
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.gp import gp_Vec, gp_Trsf
from OCC.Core.TopLoc import TopLoc_Location
from typer.testing import CliRunner


@pytest.fixture
def sample_mission(tmp_path):
    """Create a sample mission JSON file."""
    mission_data = {
        "bus_u": 3,
        "payload_mass_kg": 1.0,
        "orbit_alt_km": 550,
        "mass_limit_kg": 4.0,
        "rail_mm": 3.0,
        "deck_mm": 2.5,
        "material": "Al_6061_T6",
    }
    mission_file = tmp_path / "test_mission.json"
    mission_file.write_text(json.dumps(mission_data))
    return mission_file


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


def test_mission_spec_validation():
    """Test that mission spec validation works correctly."""
    # Valid spec
    spec = MissionSpec(
        bus_u=3, payload_mass_kg=1.0, orbit_alt_km=550, mass_limit_kg=4.0
    )
    assert spec.bus_u == 3
    assert spec.material == Material.AL_6061_T6  # default material
    assert spec.rail_mm == 3.0  # default rail thickness
    assert spec.deck_mm == 2.5  # default deck thickness

    # Test lower bounds
    with pytest.raises(ValidationError) as exc_info:
        MissionSpec(bus_u=0, payload_mass_kg=1.0, orbit_alt_km=550)
    assert "greater than or equal to 1" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        MissionSpec(bus_u=3, rail_mm=-1, payload_mass_kg=1.0)
    assert "greater than 0" in str(exc_info.value)

    # Test upper bounds
    with pytest.raises(ValidationError) as exc_info:
        MissionSpec(bus_u=13, payload_mass_kg=1.0, orbit_alt_km=550)
    assert "less than or equal to 12" in str(exc_info.value)


@pytest.mark.parametrize(
    "material,expected_density",
    [(Material.AL_6061_T6, 2700), (Material.TI_6AL_4V, 4430)],
)
def test_material_density(material, expected_density):
    """Test material density calculations for all supported materials."""
    spec = MissionSpec(
        bus_u=3,
        payload_mass_kg=1.0,
        orbit_alt_km=550,
        mass_limit_kg=4.0,
        material=material,
    )
    assert spec.density_kg_m3 == expected_density


def test_reference_cube_mass():
    """Mass of a 10 × 10 × 10 cm aluminium cube should be ≈ 2.7 kg."""
    cube = BRepPrimAPI_MakeBox(0.1, 0.1, 0.1).Shape()
    vol = calculate_volume_m3(cube)  # m³

    # Expected volume is 0.001 m³ (0.1³) within 1 %
    assert vol == pytest.approx(0.001, rel=0.01)

    # Expected mass = ρ · V
    density = load_materials()["Al_6061_T6"]["density_kg_m3"]  # 2700
    expected_mass = density * vol  # ≈ 2.7 kg
    mass = expected_mass  # same calc for now
    assert mass == pytest.approx(2.7, rel=0.01)

    # Volume is translation-invariant
    tr = gp_Trsf()
    tr.SetTranslation(gp_Vec(1, 2, 3))
    moved = cube.Moved(TopLoc_Location(tr))
    assert calculate_volume_m3(moved) == pytest.approx(vol, rel=0.01)


@pytest.mark.slow
def test_generator_cli(runner, sample_mission, tmp_path):
    """Test the generator CLI."""
    with runner.isolated_filesystem(temp_dir=tmp_path) as fs:
        outdir = Path(fs) / "test_outputs"

        # Test with default parameters
        result = runner.invoke(
            app, ["run", str(sample_mission), "--outdir", str(outdir)]
        )
        assert result.exit_code == 0
        assert str(outdir) in result.stdout

        # Capture the directory just created
        design_dirs = list(outdir.glob("design_*"))
        assert len(design_dirs) == 1
        design_dir = design_dirs[0]

        # Check that STEP file exists
        step_file = design_dir / "frame.step"
        assert step_file.exists()

        # --- check baseline mass ----
        mass_file = design_dir / "mass_budget.csv"
        assert mass_file.exists()

        with open(mass_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            total_row = next((r for r in rows if r["component"] == "total"), None)
            assert total_row, "No total row in CSV"
            mass_kg = float(total_row["mass_kg"])
            assert mass_kg > 0, "Mass must be positive"
            assert mass_kg <= 4.0, "Mass exceeds limit"

        # Test with thinner walls (should reduce mass)
        # remember which design dirs already exist
        existing_dirs = set(outdir.glob("design_*"))
        result_thin = runner.invoke(
            app,
            [
                "run",
                str(sample_mission),
                "--outdir",
                str(outdir),
                "--rail",
                "2",
                "--deck",
                "1",
                "--material",
                "Al_6061_T6",
            ],
        )
        assert result_thin.exit_code == 0

        # Get mass of *new* thinner design
        new_dirs = [d for d in outdir.glob("design_*") if d not in existing_dirs]
        assert len(new_dirs) == 1, "Exactly one new design directory expected"
        thin_design = new_dirs[0]
        with open(thin_design / "mass_budget.csv") as f:
            reader = csv.DictReader(f)
            thin_mass = float(
                next(r["mass_kg"] for r in reader if r["component"] == "total")
            )

            # Require at least 0.01 % reduction.
            tol = mass_kg * 1e-4
            assert thin_mass <= mass_kg - tol, "Thinner walls should reduce mass "
            f"(baseline={mass_kg:.6f}, thin={thin_mass:.6f})"


def test_materials_command(runner):
    """Test the materials listing command."""
    # Test basic listing
    result = runner.invoke(app, ["materials"])
    assert result.exit_code == 0
    assert "Al_6061_T6" in result.stdout
    assert "Ti_6Al_4V" in result.stdout
    assert "Density (kg/m³)" in result.stdout  # Column header in table

    # Test help flag
    help_result = runner.invoke(app, ["materials", "--help"])
    assert help_result.exit_code == 0
    assert "List available materials" in help_result.stdout
