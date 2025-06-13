# File: cube_generator.py
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from pathlib import Path


def make_cube_step(path: Path):
    shape = BRepPrimAPI_MakeBox(0.1, 0.1, 0.1).Shape()
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    writer.Write(str(path))


if __name__ == "__main__":
    make_cube_step(Path("cube_10cm.step"))
