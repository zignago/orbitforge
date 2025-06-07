from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox  # ✔ exists in 7.9
from OCC.Core.gp import gp_Vec, gp_Trsf
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone


def _translate(shape, dx, dy, dz):
    tr = gp_Trsf()
    tr.SetTranslation(gp_Vec(dx, dy, dz))
    return shape.Moved(TopLoc_Location(tr))


def build_basic_frame(ms, out_dir):
    u = 0.1  # 1 U = 10 cm
    L = ms.bus_u * u
    rail = 0.003  # 3 mm rails
    deck = 0.0025  # 2.5 mm plates

    parts = []

    # four corner rails
    for x in (0, u - rail):
        for y in (0, u - rail):
            parts.append(
                _translate(BRepPrimAPI_MakeBox(rail, rail, L).Shape(), x, y, 0)
            )

    # two end-plates
    parts.append(BRepPrimAPI_MakeBox(u, u, deck).Shape())
    parts.append(_translate(BRepPrimAPI_MakeBox(u, u, deck).Shape(), 0, 0, L - deck))

    solid = parts[0]
    for p in parts[1:]:
        solid = BRepAlgoAPI_Fuse(solid, p).Shape()

    wr = STEPControl_Writer()
    wr.Transfer(solid, STEPControl_AsIs)
    assert wr.Write(str(out_dir / "frame.step")) == IFSelect_RetDone
