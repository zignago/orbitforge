from pathlib import Path
import csv
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox  # ✔ exists in 7.9
from OCC.Core.gp import gp_Vec, gp_Trsf
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from .mission import MissionSpec


def _translate(shape, dx, dy, dz):
    tr = gp_Trsf()
    tr.SetTranslation(gp_Vec(dx, dy, dz))
    return shape.Moved(TopLoc_Location(tr))


def calculate_volume_m3(shape) -> float:
    """Calculate volume of a shape in cubic meters."""
    props = GProp_GProps()
    brepgprop.VolumeProperties(shape, props)  # static method; no warning
    return props.Mass()  # This is actually volume since we didn't set density


def build_basic_frame(ms: MissionSpec, out_dir: Path) -> None:
    """Build a basic CubeSat frame from mission specs.

    Args:
        ms: Mission specification with dimensions and material
        out_dir: Output directory for generated files
    """
    # Convert dimensions to meters (OCC uses meters)
    u = 0.1  # 1U = 10 cm = 0.1 m
    L = ms.bus_u * u
    rail = ms.rail_mm * 1e-3  # mm to m
    deck = ms.deck_mm * 1e-3  # mm to m

    components = []

    # Four corner rails
    rail_volumes = []
    for x in (0, u - rail):
        for y in (0, u - rail):
            rail_shape = _translate(BRepPrimAPI_MakeBox(rail, rail, L).Shape(), x, y, 0)
            rail_volumes.append(calculate_volume_m3(rail_shape))
            components.append(rail_shape)

    # Two end-plates
    deck_volumes = []
    for z in (0, L - deck):
        deck_shape = _translate(BRepPrimAPI_MakeBox(u, u, deck).Shape(), 0, 0, z)
        deck_volumes.append(calculate_volume_m3(deck_shape))
        components.append(deck_shape)

    # Fuse all parts
    solid = components[0]
    for p in components[1:]:
        solid = BRepAlgoAPI_Fuse(solid, p).Shape()

    # Calculate total volume and mass
    total_volume_m3 = calculate_volume_m3(solid)
    mass_kg = total_volume_m3 * ms.density_kg_m3

    # Verify mass is within limit
    if mass_kg > ms.mass_limit_kg:
        raise ValueError(
            f"Frame mass ({mass_kg:.2f} kg) exceeds limit ({ms.mass_limit_kg} kg)"
        )

    # Write STEP file
    wr = STEPControl_Writer()
    wr.Transfer(solid, STEPControl_AsIs)
    step_file = out_dir / "frame.step"
    assert wr.Write(str(step_file)) == IFSelect_RetDone

    # Write mass budget CSV
    mass_budget = [
        ["component", "volume_m3", "mass_kg"],
        ["rails (×4)", sum(rail_volumes), sum(rail_volumes) * ms.density_kg_m3],
        ["decks (×2)", sum(deck_volumes), sum(deck_volumes) * ms.density_kg_m3],
        ["total", total_volume_m3, mass_kg],
    ]

    with open(out_dir / "mass_budget.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(mass_budget)
