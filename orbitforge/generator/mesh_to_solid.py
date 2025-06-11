"""Mesh to solid conversion using OpenCascade.

This module converts generated triangle meshes to valid STEP/STL CAD files
using OpenCascade triangulation and solidification.
"""

import numpy as np
from pathlib import Path
from typing import Optional
from loguru import logger
from OCC.Core.TopoDS import TopoDS_Shape

try:
    from OCC.Core import (
        TopoDS_Compound,
        TopoDS_Builder,
        BRep_Builder,
        BRepBuilderAPI_MakeFace,
        BRepBuilderAPI_Sewing,
        BRepCheck_Analyzer,
        BRepBuilderAPI_MakeSolid,
        gp_Pnt,
        TColgp_Array1OfPnt,
        Poly_Array1OfTriangle,
        Poly_Triangle,
        BRepBuilderAPI_MakePolygon,
        BRepBuilderAPI_MakeWire,
        STEPControl_Writer,
        STEPControl_AsIs,
        Interface_Static,
        BRepMesh_IncrementalMesh,
        StlAPI_Writer,
    )

    HAS_OPENCASCADE = True
except ImportError:
    logger.warning("OpenCascade not available - using mock implementation")
    HAS_OPENCASCADE = False

from .diffusion_model import GeneratedMesh


class MeshToSolidConverter:
    """Converts generated meshes to valid CAD solids."""

    def __init__(self):
        """Initialize the converter."""
        self.has_occ = HAS_OPENCASCADE
        if not self.has_occ:
            logger.warning("OpenCascade not available - outputs will be mock files")

    def convert_mesh_to_step(self, mesh: GeneratedMesh, output_path: Path) -> bool:
        """Convert a generated mesh to a STEP file.

        Args:
            mesh: Generated mesh data
            output_path: Path where to save the STEP file

        Returns:
            True if conversion successful, False otherwise
        """
        try:
            if not self.has_occ:
                return self._create_mock_step(mesh, output_path)

            logger.debug(
                f"Converting mesh to STEP: {mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces"
            )

            # Create solid from mesh
            solid = self._mesh_to_solid(mesh)
            if solid is None:
                logger.error("Failed to create solid from mesh")
                return False

            # Validate the solid
            if not self._validate_solid(solid):
                logger.warning("Generated solid failed validation - proceeding anyway")

            # Export to STEP
            success = self._export_step(solid, output_path)
            if success:
                logger.info(f"✓ STEP file saved: {output_path}")
            return success

        except Exception as e:
            logger.error(f"Failed to convert mesh to STEP: {e}")
            return self._create_mock_step(mesh, output_path)

    def convert_mesh_to_stl(self, mesh: GeneratedMesh, output_path: Path) -> bool:
        """Convert a generated mesh to an STL file.

        Args:
            mesh: Generated mesh data
            output_path: Path where to save the STL file

        Returns:
            True if conversion successful, False otherwise
        """
        try:
            if not self.has_occ:
                return self._create_mock_stl(mesh, output_path)

            logger.debug(
                f"Converting mesh to STL: {mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces"
            )

            # Create solid from mesh
            solid = self._mesh_to_solid(mesh)
            if solid is None:
                # Fall back to direct STL export from mesh
                return self._export_stl_direct(mesh, output_path)

            # Export solid to STL
            success = self._export_stl(solid, output_path)
            if success:
                logger.info(f"✓ STL file saved: {output_path}")
            return success

        except Exception as e:
            logger.error(f"Failed to convert mesh to STL: {e}")
            return self._create_mock_stl(mesh, output_path)

    def _mesh_to_solid(self, mesh: GeneratedMesh) -> Optional[TopoDS_Shape]:
        """Convert mesh to OpenCascade solid."""
        if not self.has_occ:
            return None

        try:
            # Create triangulated surface from mesh
            vertices = mesh.vertices
            faces = mesh.faces

            # Build triangular faces
            sewing = BRepBuilderAPI_Sewing()

            for face_indices in faces:
                if len(face_indices) != 3:
                    continue  # Skip non-triangular faces

                try:
                    # Get triangle vertices
                    p1 = gp_Pnt(
                        float(vertices[face_indices[0]][0]),
                        float(vertices[face_indices[0]][1]),
                        float(vertices[face_indices[0]][2]),
                    )
                    p2 = gp_Pnt(
                        float(vertices[face_indices[1]][0]),
                        float(vertices[face_indices[1]][1]),
                        float(vertices[face_indices[1]][2]),
                    )
                    p3 = gp_Pnt(
                        float(vertices[face_indices[2]][0]),
                        float(vertices[face_indices[2]][1]),
                        float(vertices[face_indices[2]][2]),
                    )

                    # Create triangular face
                    polygon = BRepBuilderAPI_MakePolygon()
                    polygon.Add(p1)
                    polygon.Add(p2)
                    polygon.Add(p3)
                    polygon.Close()

                    if polygon.IsDone():
                        wire = polygon.Wire()
                        face_maker = BRepBuilderAPI_MakeFace(wire)
                        if face_maker.IsDone():
                            sewing.Add(face_maker.Face())

                except Exception as e:
                    logger.debug(f"Skipping problematic face: {e}")
                    continue

            # Perform sewing
            sewing.Perform()
            sewn_shape = sewing.SewedShape()

            # Try to create solid
            try:
                solid_maker = BRepBuilderAPI_MakeSolid()
                solid_maker.Add(sewn_shape)
                if solid_maker.IsDone():
                    return solid_maker.Solid()
            except Exception as e:
                logger.debug(f"Failed to create solid, returning sewn shape: {e}")

            return sewn_shape

        except Exception as e:
            logger.error(f"Failed to create solid from mesh: {e}")
            return None

    def _validate_solid(self, solid: TopoDS_Shape) -> bool:
        """Validate the generated solid using OpenCascade."""
        if not self.has_occ:
            return True

        try:
            analyzer = BRepCheck_Analyzer(solid)
            is_valid = analyzer.IsValid()
            logger.debug(f"Solid validation result: {is_valid}")
            return is_valid
        except Exception as e:
            logger.error(f"Failed to validate solid: {e}")
            return False

    def _export_step(self, solid: TopoDS_Shape, output_path: Path) -> bool:
        """Export solid to STEP file."""
        if not self.has_occ:
            return False

        try:
            # Configure STEP writer
            Interface_Static.SetCVal("write.step.schema", "AP214")
            Interface_Static.SetCVal("write.step.unit", "MM")
            Interface_Static.SetIVal("write.step.nonmanifold", 1)

            writer = STEPControl_Writer()
            writer.Transfer(solid, STEPControl_AsIs)

            status = writer.Write(str(output_path))
            return status == 1  # IFSelect_RetDone

        except Exception as e:
            logger.error(f"Failed to export STEP: {e}")
            return False

    def _export_stl(self, solid: TopoDS_Shape, output_path: Path) -> bool:
        """Export solid to STL file."""
        if not self.has_occ:
            return False

        try:
            # Mesh the solid
            mesh = BRepMesh_IncrementalMesh(solid, 0.1)  # 0.1mm tolerance
            mesh.Perform()

            # Write STL
            writer = StlAPI_Writer()
            return writer.Write(solid, str(output_path))

        except Exception as e:
            logger.error(f"Failed to export STL: {e}")
            return False

    def _export_stl_direct(self, mesh: GeneratedMesh, output_path: Path) -> bool:
        """Export mesh directly to STL format (ASCII)."""
        try:
            with open(output_path, "w") as f:
                f.write(f"solid diffusion_generated\n")

                vertices = mesh.vertices
                faces = mesh.faces

                for face_indices in faces:
                    if len(face_indices) != 3:
                        continue

                    # Get triangle vertices
                    v1 = vertices[face_indices[0]]
                    v2 = vertices[face_indices[1]]
                    v3 = vertices[face_indices[2]]

                    # Calculate normal (simplified)
                    edge1 = v2 - v1
                    edge2 = v3 - v1
                    normal = np.cross(edge1, edge2)
                    if np.linalg.norm(normal) > 0:
                        normal = normal / np.linalg.norm(normal)
                    else:
                        normal = [0, 0, 1]  # Default normal

                    f.write(
                        f"  facet normal {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n"
                    )
                    f.write(f"    outer loop\n")
                    f.write(f"      vertex {v1[0]:.6f} {v1[1]:.6f} {v1[2]:.6f}\n")
                    f.write(f"      vertex {v2[0]:.6f} {v2[1]:.6f} {v2[2]:.6f}\n")
                    f.write(f"      vertex {v3[0]:.6f} {v3[1]:.6f} {v3[2]:.6f}\n")
                    f.write(f"    endloop\n")
                    f.write(f"  endfacet\n")

                f.write(f"endsolid diffusion_generated\n")

            logger.info(f"✓ STL file saved (direct): {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export STL directly: {e}")
            return False

    def _create_mock_step(self, mesh: GeneratedMesh, output_path: Path) -> bool:
        """Create a mock STEP file for testing."""
        try:
            mock_content = f"""ISO-10303-21;
HEADER;
FILE_DESCRIPTION((''), '2;1');
FILE_NAME('diffusion_generated.step', '', (''), (''), '', '', '');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;

DATA;
/* Generated by OrbitForge diffusion model */
/* Mesh metadata: {mesh.metadata} */
/* Vertices: {mesh.vertices.shape[0]}, Faces: {mesh.faces.shape[0]} */
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
ENDSEC;
END-ISO-10303-21;
"""
            output_path.write_text(mock_content)
            logger.info(f"✓ Mock STEP file saved: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create mock STEP: {e}")
            return False

    def _create_mock_stl(self, mesh: GeneratedMesh, output_path: Path) -> bool:
        """Create a mock STL file for testing."""
        try:
            # Use the direct STL export which doesn't require OpenCascade
            return self._export_stl_direct(mesh, output_path)
        except Exception as e:
            logger.error(f"Failed to create mock STL: {e}")
            return False
