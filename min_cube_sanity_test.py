# from orbitforge.fea import FastPhysicsValidator, MaterialProperties
# import gmsh, numpy as np, tempfile, pathlib
# from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
# from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs

# # Create temporary STEP file of solid aluminum cube
# tmp = pathlib.Path(tempfile.mkdtemp()) / 'cube.step'
# writer = STEPControl_Writer()
# writer.Transfer(BRepPrimAPI_MakeBox(0.1, 0.1, 0.1).Shape(), STEPControl_AsIs)
# writer.Write(str(tmp))

# # Define aluminum material
# mat = MaterialProperties(
#     name='Al_6061_T6', yield_strength_mpa=276,
#     youngs_modulus_gpa=68.9, poissons_ratio=0.33,
#     density_kg_m3=2700
# )

# # Run physics check
# v = FastPhysicsValidator(tmp, mat)
# print(v.run_validation())

from orbitforge.fea import FastPhysicsValidator, MaterialProperties
from orbitforge.fea import LoadCase
from pathlib import Path

# Create material
mat = MaterialProperties(
    name="Al_6061_T6",
    yield_strength_mpa=276,
    youngs_modulus_gpa=68.9,
    poissons_ratio=0.33,
    density_kg_m3=2700,
)

# Define load case
load = LoadCase(name="Axial 6g", acceleration_g=6.0, direction=[0, 0, -1])

# Instantiate validator
v = FastPhysicsValidator(step_file=Path("cube_10cm.step"), material=mat)

# Override load
results = v.run_validation(load_override=load)
print(results)
