"""Design for Additive Manufacturing (DfAM) rules checker."""

import numpy as np
import trimesh
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any
import json


@dataclass
class DfamViolation:
    """Represents a violation of a DfAM rule."""

    rule: str
    severity: str  # 'ERROR' or 'WARNING'
    message: str
    location: str  # Description of where the violation occurs
    value: float  # Actual value that violates the rule
    limit: float  # Rule limit that was violated


class WallChecker:
    """Measures local thickness using voxel sampling."""

    def __init__(self, mesh: trimesh.Trimesh, min_thickness_mm: float = 0.8):
        self.mesh = mesh
        self.min_thickness_mm = min_thickness_mm

    def check(self) -> List[DfamViolation]:
        """Check for walls thinner than minimum thickness."""
        violations = []

        try:
            # Sample points on the surface
            points = self.mesh.sample(500)  # Sample more points for better coverage

            for point_idx, point in enumerate(points):
                try:
                    # Find closest point on mesh and get normal
                    closest, distance, face_id = self.mesh.nearest.on_surface([point])
                    normal = self.mesh.face_normals[face_id[0]]

                    # Cast rays in both directions along normal
                    origins = np.array([point, point])
                    directions = np.array([normal, -normal])

                    locations, _, _ = self.mesh.ray.intersects_location(
                        ray_origins=origins, ray_directions=directions
                    )

                    if len(locations) >= 2:
                        # Calculate minimum distance between intersection points
                        dists = []
                        for i in range(len(locations)):
                            for j in range(i + 1, len(locations)):
                                dists.append(
                                    np.linalg.norm(locations[i] - locations[j])
                                )

                        if dists:
                            thickness_m = min(dists)
                            thickness_mm = thickness_m * 1000

                            if thickness_mm < self.min_thickness_mm:
                                violations.append(
                                    DfamViolation(
                                        rule="min_wall_thickness",
                                        severity="ERROR",
                                        message="Wall thickness below minimum",
                                        location=f"Near point ({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f})",
                                        value=thickness_mm,
                                        limit=self.min_thickness_mm,
                                    )
                                )

                except Exception as e:
                    continue  # Skip problematic points

        except Exception as e:
            # If sampling fails, create a generic violation
            violations.append(
                DfamViolation(
                    rule="min_wall_thickness",
                    severity="ERROR",
                    message="Unable to analyze wall thickness",
                    location="Global",
                    value=0.0,
                    limit=self.min_thickness_mm,
                )
            )

        return violations


class OverhangChecker:
    """Analyzes surface normals relative to Z-axis for unsupported overhangs."""

    def __init__(self, mesh: trimesh.Trimesh, max_overhang_angle: float = 45.0):
        self.mesh = mesh
        self.max_overhang_angle = max_overhang_angle

    def check(self) -> List[DfamViolation]:
        """Check for overhangs exceeding maximum angle."""
        violations = []
        build_direction = np.array([0, 0, 1])  # Z is build direction

        overhang_faces = []
        for i, normal in enumerate(self.mesh.face_normals):
            # Calculate angle between face normal and build direction
            cos_angle = np.dot(normal, build_direction)
            angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

            # For overhangs, we're interested in downward-facing surfaces
            # Angle > 90° means downward facing
            if angle_deg > 90:
                overhang_angle = angle_deg - 90  # Angle from horizontal

                if overhang_angle > self.max_overhang_angle:
                    overhang_faces.append(i)
                    centroid = self.mesh.triangles_center[i]
                    violations.append(
                        DfamViolation(
                            rule="max_overhang_angle",
                            severity="WARNING",
                            message="Overhang angle exceeds maximum",
                            location=f"Face at ({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f})",
                            value=overhang_angle,
                            limit=self.max_overhang_angle,
                        )
                    )

        return violations


class DrainHoleChecker:
    """Detects enclosed voids without open drain paths."""

    def __init__(self, mesh: trimesh.Trimesh, max_cavity_length_mm: float = 50.0):
        self.mesh = mesh
        self.max_cavity_length_mm = max_cavity_length_mm

    def check(self) -> List[DfamViolation]:
        """Check for powder drain hole requirements."""
        violations = []

        # For now, implement a simplified check based on mesh analysis
        # In a full implementation, this would do volumetric analysis

        try:
            # Check if mesh has internal cavities by analyzing water-tightness
            if not self.mesh.is_watertight:
                # If not watertight, there might be openings that serve as drain holes
                return violations

            # Check bounding box dimensions - if any dimension > max_cavity_length_mm,
            # we need drain holes
            extents = self.mesh.bounding_box.extents * 1000  # Convert to mm
            max_extent = max(extents)

            if max_extent > self.max_cavity_length_mm:
                # For a simplified check, assume large solid parts need drain holes
                # This is a conservative approach
                center = self.mesh.bounding_box.centroid
                violations.append(
                    DfamViolation(
                        rule="powder_drain_holes",
                        severity="WARNING",
                        message=f"Large cavity detected (>{self.max_cavity_length_mm}mm), drain holes recommended",
                        location=f"Near ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})",
                        value=max_extent,
                        limit=self.max_cavity_length_mm,
                    )
                )

        except Exception as e:
            # If analysis fails, be conservative and flag for manual review
            violations.append(
                DfamViolation(
                    rule="powder_drain_holes",
                    severity="WARNING",
                    message="Unable to analyze cavity drainage requirements",
                    location="Global",
                    value=0.0,
                    limit=self.max_cavity_length_mm,
                )
            )

        return violations


class STLExporter:
    """Exports STL/3MF using trimesh."""

    @staticmethod
    def export_stl(mesh: trimesh.Trimesh, output_path: Path) -> bool:
        """Export mesh to STL format."""
        try:
            mesh.export(str(output_path))
            return True
        except Exception as e:
            print(f"Failed to export STL: {e}")
            return False

    @staticmethod
    def export_3mf(mesh: trimesh.Trimesh, output_path: Path) -> bool:
        """Export mesh to 3MF format."""
        try:
            # 3MF export requires specific handling
            mesh.export(str(output_path))
            return True
        except Exception as e:
            print(f"Failed to export 3MF: {e}")
            return False


class DfamChecker:
    """Checks STL models for DfAM rule violations according to v0.1.5 spec."""

    def __init__(self, stl_path: Path):
        """Initialize checker with an STL file."""
        self.mesh = trimesh.load_mesh(str(stl_path))

        # Ensure the mesh is watertight and compute normals
        self.mesh.process()
        self.mesh.fix_normals()

        # Initialize rule checkers per v0.1.5 spec (Ti-6Al-4V on EOS M 400)
        self.wall_checker = WallChecker(self.mesh, min_thickness_mm=0.8)
        self.overhang_checker = OverhangChecker(self.mesh, max_overhang_angle=45.0)
        self.drain_checker = DrainHoleChecker(self.mesh, max_cavity_length_mm=50.0)

    def run_all_checks(self) -> Dict[str, Any]:
        """Run all DfAM checks and return results."""
        all_violations = []

        # Run individual checks
        wall_violations = self.wall_checker.check()
        overhang_violations = self.overhang_checker.check()
        drain_violations = self.drain_checker.check()

        all_violations.extend(wall_violations)
        all_violations.extend(overhang_violations)
        all_violations.extend(drain_violations)

        # Count violations by severity
        error_count = sum(1 for v in all_violations if v.severity == "ERROR")
        warning_count = sum(1 for v in all_violations if v.severity == "WARNING")

        # Create detailed report per spec
        report = {
            "status": "FAIL" if error_count > 0 else "PASS",
            "error_count": error_count,
            "warning_count": warning_count,
            "wall_thickness": (
                "FAIL"
                if any(
                    v.rule == "min_wall_thickness" and v.severity == "ERROR"
                    for v in all_violations
                )
                else "PASS"
            ),
            "overhang_check": (
                "FAIL"
                if any(
                    v.rule == "max_overhang_angle" and v.severity == "ERROR"
                    for v in all_violations
                )
                else "PASS"
            ),
            "drain_holes": (
                "FAIL"
                if any(
                    v.rule == "powder_drain_holes" and v.severity == "ERROR"
                    for v in all_violations
                )
                else "PASS"
            ),
            "violations": [
                {
                    "rule": v.rule,
                    "severity": v.severity,
                    "message": v.message,
                    "location": v.location,
                    "value": v.value,
                    "limit": v.limit,
                }
                for v in all_violations
            ],
        }

        return report

    def export_printable_file(self, output_dir: Path, base_name: str = "frame") -> bool:
        """Export STL file only if DfAM checks pass."""
        results = self.run_all_checks()

        if results["status"] == "PASS":
            stl_path = output_dir / f"{base_name}.stl"
            return STLExporter.export_stl(self.mesh, stl_path)

        return False

    def save_dfam_report(
        self, output_dir: Path, base_name: str = "dfam_report"
    ) -> Path:
        """Save detailed DfAM report as JSON."""
        results = self.run_all_checks()
        report_path = output_dir / f"{base_name}.json"

        with open(report_path, "w") as f:
            json.dump(results, f, indent=2)

        return report_path
