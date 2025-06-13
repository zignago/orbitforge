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
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_SOLID
from OCC.Core.BRep import BRep_Builder
from OCC.Core.TopoDS import TopoDS_Compound

from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Sewing
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeSolid
from OCC.Core.ShapeFix import ShapeFix_Shell
from OCC.Core.TopoDS import topods
from OCC.Core.TopAbs import TopAbs_SHELL


# from OCC.Core.Interface import Interface_Static_SetCVal
from OCC.Core.Interface import Interface_Static
from .mission import MissionSpec
from typing import List, Optional, Tuple
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from loguru import logger


def _translate(shape, dx, dy, dz):
    """Translate a shape by a vector."""
    tr = gp_Trsf()
    tr.SetTranslation(gp_Vec(dx, dy, dz))
    loc = TopLoc_Location(tr)
    return shape.Moved(loc)


def _validate_shape(shape: TopoDS_Shape, label: str = "") -> bool:
    """Validate a shape using BRepCheck."""
    analyzer = BRepCheck_Analyzer(shape)
    if not analyzer.IsValid():
        logger.error(f"Invalid shape detected{f' in {label}' if label else ''}")
        return False
    return True


def _count_solids(shape: TopoDS_Shape) -> int:
    """Count number of solids in a shape."""
    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    count = 0
    while explorer.More():
        count += 1
        explorer.Next()
    return count


def sew_and_solidify(shape: TopoDS_Shape, label: str = "") -> Optional[TopoDS_Shape]:
    """Attempt to sew and convert a compound into a solid."""
    sewing = BRepBuilderAPI_Sewing()
    sewing.Add(shape)
    sewing.Perform()

    sewn_shape = sewing.SewedShape()
    if not sewn_shape:
        logger.error(f"Sewing failed{f' for {label}' if label else ''}")
        return None

    # Extract the shell from the sewed shape
    shell_explorer = TopExp_Explorer(sewn_shape, TopAbs_SHELL)
    if not shell_explorer.More():
        logger.error(f"No shell found after sewing{f' for {label}' if label else ''}")
        return None

    shell = topods.Shell(shell_explorer.Current())

    # Fix shell before converting to solid
    shell_fixer = ShapeFix_Shell()
    shell_fixer.Init(shell)
    shell_fixer.Perform()
    fixed_shell = shell_fixer.Shell()

    solid_maker = BRepBuilderAPI_MakeSolid(fixed_shell)
    if not solid_maker.IsDone():
        logger.error(f"Failed to convert shell to solid for {label}")
        return None

    return solid_maker.Solid()


def _fuse_pair(
    shape1: TopoDS_Shape, shape2: TopoDS_Shape, label: str = ""
) -> Optional[TopoDS_Shape]:
    """Fuse two shapes with validation."""
    if not (
        _validate_shape(shape1, f"{label} - shape1")
        and _validate_shape(shape2, f"{label} - shape2")
    ):
        return None

    fuser = BRepAlgoAPI_Fuse(shape1, shape2)
    if not fuser.IsDone():
        logger.error(f"Fusion failed{f' in {label}' if label else ''}")
        return None

    result = fuser.Shape()
    if not _validate_shape(result, f"{label} - result"):
        return None

    return result


def _fuse_shapes(shapes: List[TopoDS_Shape], label: str = "") -> Optional[TopoDS_Shape]:
    """Fuse multiple shapes with validation using sequential pairwise fusion."""
    if not shapes:
        return None

    if len(shapes) == 1:
        return shapes[0]

    # Start with first shape
    result = shapes[0]

    # Sequentially fuse remaining shapes
    for i, shape in enumerate(shapes[1:], 1):
        result = _fuse_pair(result, shape, f"{label} - pair {i}")
        if result is None:
            logger.error(f"Failed to fuse pair {i} in {label}")
            return None

        # Verify intermediate result
        if not _validate_shape(result, f"{label} - intermediate {i}"):
            return None

    # Final validation
    solid_count = _count_solids(result)
    if solid_count != 1:
        logger.warning(f"{label}: fused into {solid_count} solids (expected 1?)")
    # Still return result regardless of solid count
    return result


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
    components: List[TopoDS_Shape] = []
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

    # First fuse the rails together
    rails = []
    for pos in rail_positions:
        translated = _translate(rail_shape, *pos)
        rails.append(translated)
        rail_volumes.append(calculate_volume_m3(translated))

    frame = _fuse_shapes(rails, "rail fusion")
    if frame is None:
        raise RuntimeError("Failed to fuse rails")

    # Create decks (2x horizontal plates)
    deck_shape = BRepPrimAPI_MakeBox(u, u, deck).Shape()
    deck_positions = [(0, 0, 0), (0, 0, ms.bus_u * u - deck)]

    # Fuse decks together first
    decks = []
    for pos in deck_positions:
        translated = _translate(deck_shape, *pos)
        decks.append(translated)
        deck_volumes.append(calculate_volume_m3(translated))

    deck_frame = _fuse_shapes(decks, "deck fusion")
    if deck_frame is None:
        raise RuntimeError("Failed to fuse decks")

    # Fuse rails and decks
    frame = _fuse_pair(frame, deck_frame, "rails and decks fusion")
    if frame is None:
        raise RuntimeError("Failed to fuse rails and decks")

    # Add X-brace struts for lateral stability
    strut_width = 0.75 * rail

    # Precompute static strut shapes (saves time, no shape variation by level)
    strut_height = strut_width
    strut_depth = strut_width

    strut_x = BRepPrimAPI_MakeBox(
        x, strut_height, strut_depth
    ).Shape()  # X-span (front/back)
    strut_y = BRepPrimAPI_MakeBox(
        strut_height, x, strut_depth
    ).Shape()  # Y-span (left/right)

    # Create and position struts at each level
    # Add straight, internal cross-struts at each 1U level
    for level in range(1, ms.bus_u):
        z_level = level * u

        # Create and fuse struts for this level
        level_struts = []

        # FRONT face (Y = rail)
        front_strut = _translate(strut_x, rail, rail, z_level)
        level_struts.append(front_strut)
        strut_volumes.append(calculate_volume_m3(front_strut))

        # BACK face (Y = back rail)
        back_strut = _translate(strut_x, rail, u - rail - strut_height, z_level)
        level_struts.append(back_strut)
        strut_volumes.append(calculate_volume_m3(back_strut))

        # LEFT face (X = rail)
        left_strut = _translate(strut_y, rail, rail, z_level)
        level_struts.append(left_strut)
        strut_volumes.append(calculate_volume_m3(left_strut))

        # RIGHT face (X = right rail)
        right_strut = _translate(strut_y, u - rail - strut_height, rail, z_level)
        level_struts.append(right_strut)
        strut_volumes.append(calculate_volume_m3(right_strut))

        # Fuse struts for this level
        level_frame = _fuse_shapes(level_struts, f"level {level} struts fusion")
        if level_frame is None:
            raise RuntimeError(f"Failed to fuse struts at level {level}")

        # Fuse level struts with main frame
        frame = _fuse_pair(frame, level_frame, f"level {level} fusion with frame")
        if frame is None:
            raise RuntimeError(f"Failed to fuse level {level} with frame")

    # Final validation of complete frame
    if not _validate_shape(frame, "final frame"):
        raise RuntimeError("Final frame validation failed")

    # Verify we have a single solid
    solid_count = _count_solids(frame)
    if solid_count != 1:
        logger.warning(
            f"Final frame is not a single solid ({solid_count} found), attempting to sew..."
        )
        sewed = sew_and_solidify(frame, "final frame")
        if sewed is None:
            raise RuntimeError("Failed to sew non-manifold frame into solid")
        frame = sewed  # Replace with sewed solid

    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)

    # Export STEP file
    step_file = out_dir / "frame.step"
    Interface_Static.SetCVal("write.step.schema", "AP203")
    Interface_Static.SetCVal("write.step.unit", "MM")
    Interface_Static.SetIVal("write.step.nonmanifold", 1)
    writer = STEPControl_Writer()
    writer.Transfer(frame, STEPControl_AsIs)
    status = writer.Write(str(step_file))

    if status != IFSelect_RetDone:
        raise RuntimeError("Error writing STEP file")

    # Export STL file for 3D printing
    stl_file = out_dir / "frame.stl"

    # Create mesh for STL export
    mesh = BRepMesh_IncrementalMesh(frame, 0.0001)  # 0.1 mm tolerance in meters
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
