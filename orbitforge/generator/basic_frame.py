from pathlib import Path
import csv
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox  # ✔ exists in 7.9
from OCC.Core.gp import gp_Vec, gp_Trsf, gp_Pnt, gp_Dir, gp_Ax1
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform

# from OCC.Core.Interface import Interface_Static_SetCVal
from OCC.Core.Interface import Interface_Static
from .mission import MissionSpec
from typing import List
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh


def _translate(shape, dx, dy, dz):
    """Translate a shape by a vector."""
    tr = gp_Trsf()
    tr.SetTranslation(gp_Vec(dx, dy, dz))
    loc = TopLoc_Location(tr)
    return shape.Moved(loc)


def calculate_volume_m3(shape) -> float:
    """Calculate volume of a shape in cubic meters."""
    props = GProp_GProps()
    brepgprop.VolumeProperties(shape, props)  # static method; no warning
    return props.Mass()  # This is actually volume since we didn't set density


def build_basic_frame(ms: MissionSpec, out_dir: Path) -> Path:
    """Build a basic CubeSat frame with rails and decks."""
    # Convert units to meters
    u = 0.1  # 1U = 100mm
    rail = ms.rail_mm / 1000
    deck = ms.deck_mm / 1000

    # Frame dimensions
    x = u - 2 * rail - 0.003  # Internal clearance
    y = x
    z = deck

    # Create components list for assembly
    components: List = []
    rail_volumes = []
    deck_volumes = []
    strut_volumes = []

    # Create rails (4x vertical beams)
    rail_shape = BRepPrimAPI_MakeBox(rail, rail, ms.bus_u * u).Shape()
    rail_positions = [
        (0, 0, 0),
        (x + rail, 0, 0),
        (0, y + rail, 0),
        (x + rail, y + rail, 0),
    ]
    for pos in rail_positions:
        translated = _translate(rail_shape, *pos)
        components.append(translated)
        rail_volumes.append(rail * rail * ms.bus_u * u)

    # Create decks (2x horizontal plates)
    deck_shape = BRepPrimAPI_MakeBox(u, u, deck).Shape()
    deck_positions = [(0, 0, 0), (0, 0, ms.bus_u * u - deck)]
    for pos in deck_positions:
        translated = _translate(deck_shape, *pos)
        components.append(translated)
        deck_volumes.append(u * u * deck)

    # Add X-brace struts for lateral stability
    strut_width = 0.75 * rail

    # Create and position struts at each level
    # Add straight, internal cross-struts at each 1U level
    for level in range(1, ms.bus_u):
        z = level * u  # Z position between decks

        # Shared sizes
        strut_height = strut_width
        strut_depth = strut_width

        # X-span (left/right face): along X axis
        strut_x = BRepPrimAPI_MakeBox(x, strut_height, strut_depth).Shape()

        # Y-span (front/back face): along Y axis
        strut_y = BRepPrimAPI_MakeBox(strut_height, x, strut_depth).Shape()

        # FRONT face (Y = rail)
        front_strut = _translate(strut_x, rail, rail, z)
        components.append(front_strut)
        strut_volumes.append(x * strut_width * strut_width)

        # BACK face (Y = back rail)
        back_strut = _translate(strut_x, rail, u - rail - strut_height, z)
        components.append(back_strut)
        strut_volumes.append(x * strut_width * strut_width)

        # LEFT face (X = rail)
        left_strut = _translate(strut_y, rail, rail, z)
        components.append(left_strut)
        strut_volumes.append(x * strut_width * strut_width)

        # RIGHT face (X = right rail)
        right_strut = _translate(strut_y, u - rail - strut_height, rail, z)
        components.append(right_strut)
        strut_volumes.append(x * strut_width * strut_width)

    # Fuse all components
    frame = components[0]
    for comp in components[1:]:
        frame = BRepAlgoAPI_Fuse(frame, comp).Shape()

    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)

    # Export STEP file
    step_file = out_dir / "frame.step"
    # Interface_Static_SetCVal("write.step.schema", "AP203")
    Interface_Static.SetCVal("write.step.schema", "AP203")
    writer = STEPControl_Writer()
    writer.Transfer(frame, STEPControl_AsIs)
    status = writer.Write(str(step_file))

    if status != IFSelect_RetDone:
        raise RuntimeError("Error writing STEP file")

    # Export STL file for 3D printing
    stl_file = out_dir / "frame.stl"

    # Create mesh for STL export
    mesh = BRepMesh_IncrementalMesh(frame, 0.1)  # 0.1mm tolerance
    mesh.Perform()
    if not mesh.IsDone():
        raise RuntimeError("Mesh generation failed")

    # Write STL file
    stl_writer = StlAPI_Writer()
    stl_writer.SetASCIIMode(True)  # Use ASCII format for better compatibility
    result = stl_writer.Write(frame, str(stl_file))
    if not result:
        raise RuntimeError("Failed to write STL file")

    # Calculate total volume and mass
    total_volume_m3 = calculate_volume_m3(frame)
    mass_kg = total_volume_m3 * ms.density_kg_m3

    # Verify mass is within limit
    if mass_kg > ms.mass_limit_kg:
        raise ValueError(
            f"Frame mass ({mass_kg:.2f} kg) exceeds limit ({ms.mass_limit_kg} kg)"
        )

    # Write mass budget CSV
    mass_budget = [
        ["component", "volume_m3", "mass_kg"],
        ["rails (×4)", sum(rail_volumes), sum(rail_volumes) * ms.density_kg_m3],
        ["decks (×2)", sum(deck_volumes), sum(deck_volumes) * ms.density_kg_m3],
        ["struts (×4)", sum(strut_volumes), sum(strut_volumes) * ms.density_kg_m3],
        ["total", total_volume_m3, mass_kg],
    ]

    with open(out_dir / "mass_budget.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(mass_budget)

    return step_file
