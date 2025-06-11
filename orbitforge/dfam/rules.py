"""Design for Additive Manufacturing (DfAM) rules checker."""

import numpy as np
import trimesh
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class DfamViolation:
    """Represents a violation of a DfAM rule."""

    rule: str
    severity: str  # 'ERROR' or 'WARNING'
    message: str
    location: str  # Description of where the violation occurs
    value: float  # Actual value that violates the rule
    limit: float  # Rule limit that was violated


class DfamChecker:
    """Checks STL models for DfAM rule violations."""

    def __init__(self, stl_path: Path):
        """Initialize checker with an STL file."""
        self.mesh = trimesh.load_mesh(str(stl_path))

        # Ensure the mesh is watertight and compute normals
        self.mesh.process()
        self.mesh.fix_normals()

        # Compute vertex normals by averaging face normals
        vertex_normals = np.zeros((len(self.mesh.vertices), 3))
        vertex_counts = np.zeros(len(self.mesh.vertices))

        # For each face, add its normal to all its vertices
        for face_idx, face in enumerate(self.mesh.faces):
            face_normal = self.mesh.face_normals[face_idx]
            for vertex_idx in face:
                vertex_normals[vertex_idx] += face_normal
                vertex_counts[vertex_idx] += 1

        # Average the normals and normalize
        nonzero_vertices = vertex_counts > 0
        vertex_normals[nonzero_vertices] /= vertex_counts[nonzero_vertices, np.newaxis]
        norms = np.linalg.norm(vertex_normals, axis=1)
        norms[norms == 0] = 1  # Avoid division by zero
        vertex_normals /= norms[:, np.newaxis]

        self.vertex_normals = vertex_normals

        # Default rules for metal AM (EOS M 400)
        self.rules = {
            "min_wall_thickness": 0.4,  # mm
            "max_overhang_angle": 45.0,  # degrees from vertical
            "min_hole_diameter": 0.5,  # mm
            "max_aspect_ratio": 20.0,  # length/width for thin features
        }

    def check_wall_thickness(self) -> List[DfamViolation]:
        """Check for walls thinner than minimum thickness."""
        violations = []

        try:
            points = self.mesh.sample(1000)
            print(f">>> check_wall_thickness: sampled {len(points)} points")
        except Exception as e:
            print(f">>> check_wall_thickness: mesh sampling failed: {e}")
            return violations

        for point_idx, point in enumerate(points[:50]):  # limit debug to first 50
            try:
                nearest = self.mesh.kdtree.query(point)
                closest_vertex_idx = (
                    nearest[1] if isinstance(nearest, tuple) else nearest
                )
                normal = self.vertex_normals[closest_vertex_idx]

                origins = np.vstack((point, point))
                directions = np.vstack((normal, -normal))

                locations, _, _ = self.mesh.ray.intersects_location(
                    ray_origins=origins, ray_directions=directions
                )

                if len(locations) >= 2:
                    dists = [
                        np.linalg.norm(locations[i] - locations[j])
                        for i in range(len(locations))
                        for j in range(i + 1, len(locations))
                    ]
                    thickness = max(dists)
                    thickness_mm = thickness * 1000

                    if thickness_mm < self.rules["min_wall_thickness"]:
                        violations.append(
                            DfamViolation(
                                rule="min_wall_thickness",
                                severity="ERROR",
                                message="Wall thickness below minimum",
                                location=f"Near point ({point[0]:.1f}, {point[1]:.1f}, {point[2]:.1f})",
                                value=thickness_mm,
                                limit=self.rules["min_wall_thickness"],
                            )
                        )
            except Exception as e:
                print(f">>> check_wall_thickness: error at point {point_idx}: {e}")
                continue

        print(f">>> check_wall_thickness: found {len(violations)} violations")
        return violations

    def check_overhangs(self) -> List[DfamViolation]:
        """Check for overhangs exceeding maximum angle."""
        violations = []
        build_direction = np.array([0, 0, 1])  # Assuming Z is build direction

        for i, normal in enumerate(self.mesh.face_normals):
            # Calculate angle between face normal and build direction
            cos_angle = np.dot(normal, build_direction)
            angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

            # For overhangs, we're interested in the angle from horizontal
            overhang_angle = 90 - angle_deg

            if overhang_angle > self.rules["max_overhang_angle"]:
                centroid = self.mesh.triangles_center[i]
                violations.append(
                    DfamViolation(
                        rule="max_overhang_angle",
                        severity="WARNING",
                        message="Overhang angle exceeds maximum",
                        location=f"Face at ({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f})",
                        value=overhang_angle,
                        limit=self.rules["max_overhang_angle"],
                    )
                )

        return violations

    def check_holes(self) -> List[DfamViolation]:
        """Check for holes smaller than minimum diameter."""
        violations = []

        # Ensure mesh is watertight or try to fill holes
        if not self.mesh.is_watertight:
            self.mesh.fill_holes()

        # Get boundary edges
        try:
            # Use the correct attribute for boundary edges
            boundary_edges = self.mesh.edges_boundary
        except AttributeError:
            return violations

        if len(boundary_edges) == 0:
            return violations

        if (
            not isinstance(boundary_edges, np.ndarray)
            or boundary_edges.ndim != 2
            or boundary_edges.shape[1] != 2
        ):
            print(
                "DfAM check_holes: Invalid boundary_edges shape:", boundary_edges.shape
            )
            return violations

        # Defensive copy of vertices
        try:
            vertices = np.asarray(self.mesh.vertices)
            if vertices.ndim != 2 or vertices.shape[1] != 3:
                print("DfAM check_holes: Invalid vertex shape:", vertices.shape)
                return violations

            # Check max index
            if np.max(boundary_edges) >= len(vertices):
                print(
                    "DfAM check_holes: Index out of bounds. Max index in boundary_edges:",
                    np.max(boundary_edges),
                )
                print("DfAM check_holes: Vertex array length:", len(vertices))
                return violations

            edge_vertices = vertices[boundary_edges]  # shape: (N, 2, 3)
        except Exception as e:
            print("DfAM check_holes: Exception during edge vertex indexing:", e)
            return violations

        # Group connected edges
        visited = set()
        for i, edge in enumerate(edge_vertices):
            if i in visited:
                continue
            group = [edge[0], edge[1]]
            visited.add(i)

            for j, other_edge in enumerate(edge_vertices):
                if j in visited:
                    continue
                if np.allclose(other_edge[0], group[-1]):
                    group.append(other_edge[1])
                    visited.add(j)
                elif np.allclose(other_edge[1], group[-1]):
                    group.append(other_edge[0])
                    visited.add(j)

            # Check diameter
            if len(group) >= 3:
                dists = [
                    np.linalg.norm(p1 - p2)
                    for i, p1 in enumerate(group)
                    for p2 in group[i + 1 :]
                ]
                if dists:
                    diameter_mm = max(dists) * 1000
                    if diameter_mm < self.rules["min_hole_diameter"]:
                        centroid = np.mean(group, axis=0)
                        violations.append(
                            DfamViolation(
                                rule="min_hole_diameter",
                                severity="WARNING",
                                message="Hole diameter below minimum",
                                location=f"Near ({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f})",
                                value=diameter_mm,
                                limit=self.rules["min_hole_diameter"],
                            )
                        )

        return violations

    def check_aspect_ratio(self) -> List[DfamViolation]:
        """Check for features with excessive aspect ratios."""
        violations = []

        # Get bounding box dimensions
        extents = self.mesh.bounding_box.extents
        max_ratio = max(extents) / min(extents)

        if max_ratio > self.rules["max_aspect_ratio"]:
            violations.append(
                DfamViolation(
                    rule="max_aspect_ratio",
                    severity="WARNING",
                    message="Feature aspect ratio too high",
                    location="Global",
                    value=max_ratio,
                    limit=self.rules["max_aspect_ratio"],
                )
            )

        return violations

    def run_all_checks(self) -> Dict[str, Any]:
        """Run all DfAM checks and return results."""
        all_violations = []
        all_violations.extend(self.check_wall_thickness())
        all_violations.extend(self.check_overhangs())
        all_violations.extend(self.check_holes())
        all_violations.extend(self.check_aspect_ratio())

        # Count violations by severity
        error_count = sum(1 for v in all_violations if v.severity == "ERROR")
        warning_count = sum(1 for v in all_violations if v.severity == "WARNING")

        return {
            "status": "FAIL" if error_count > 0 else "PASS",
            "error_count": error_count,
            "warning_count": warning_count,
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
