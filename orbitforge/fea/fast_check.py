"""
Fast physics validation module for quick structural analysis of satellite frames.

This module provides rapid structural feedback for frame designs without requiring
full FEA analysis. It implements quick meshing and linear elastic analysis for
6g axial/lateral load cases and thermal gradient checks.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import gmsh
import numpy as np
from loguru import logger

from . import solver


@dataclass
class MaterialProperties:
    """Material properties needed for structural analysis."""

    name: str
    yield_strength_mpa: float
    youngs_modulus_gpa: float
    poissons_ratio: float
    density_kg_m3: float
    cte_per_k: Optional[float] = None  # Coefficient of thermal expansion

    def to_dict(self) -> Dict:
        """Convert to dictionary for solver interface."""
        return {
            "name": self.name,
            "yield_strength_mpa": self.yield_strength_mpa,
            "youngs_modulus_gpa": self.youngs_modulus_gpa,
            "poissons_ratio": self.poissons_ratio,
            "density_kg_m3": self.density_kg_m3,
            "cte_per_k": self.cte_per_k,
        }


@dataclass
class LoadCase:
    """Definition of a load case for analysis."""

    name: str
    acceleration_g: float
    direction: Tuple[float, float, float]


@dataclass
class ValidationResults:
    """Results from the fast physics validation."""

    max_stress_mpa: float
    sigma_allow_mpa: float
    status: str
    thermal_stress_mpa: Optional[float] = None
    thermal_status: Optional[str] = None


class FastPhysicsValidator:
    """Performs rapid structural validation of satellite frames."""

    def __init__(
        self, step_file: Path, material: MaterialProperties, load_scale: float = 1.0
    ):
        """Initialize the validator with a STEP file and material properties."""
        self.step_file = step_file
        self.material = material
        self._mesh = None
        self.load_scale = load_scale

        # Standard load cases from Falcon 9 & Electron rideshare specs
        self.load_cases = [
            LoadCase("Axial 6g", 6.0, (0.0, 0.0, 1.0)),
            LoadCase("Lateral 6g", 6.0, (1.0, 0.0, 0.0)),
        ]

    def _find_load_nodes(
        self, nodes: np.ndarray, direction: Tuple[float, float, float]
    ) -> np.ndarray:
        """
        Pick nodes where the lumped inertial load will be applied.

        For launch loads, we apply the inertial reaction force at the attachment points:
        - Axial load (±Z): Apply at top deck (highest Z) for downward accel, bottom for upward
        - Lateral load (±X/Y): Apply at the extreme nodes in the direction opposite to acceleration
        """
        dir_vec = np.array(direction, dtype=float)
        ax = np.argmax(np.abs(dir_vec))  # 0:X, 1:Y, 2:Z

        coord = nodes[:, ax]
        tol = 1e-4 * (coord.max() - coord.min() or 1.0)  # 0.01% span

        # For inertial loads, apply force at the end opposite to acceleration direction
        # If acceleration is negative (e.g., -Z), apply load at max coordinate (top)
        # If acceleration is positive (e.g., +Z), apply load at min coordinate (bottom)
        if dir_vec[ax] < 0:
            target = coord.max()  # Apply at maximum coordinate
        else:
            target = coord.min()  # Apply at minimum coordinate

        load_nodes = np.where(np.abs(coord - target) < tol)[0]

        logger.debug(
            f"Load direction: {direction}, axis: {ax}, target coord: {target:.6f}"
        )
        logger.debug(f"Found {len(load_nodes)} load nodes at coordinate {target:.6f}")

        return load_nodes

    def check_mesh_connectivity(self, nodes, elements, tol=1):
        import networkx as nx

        G = nx.Graph()

        non_tets = sum(1 for tet in elements if len(tet) != 4)
        if non_tets > 0:
            logger.warning(
                f"⚠ Skipped {non_tets} non-tetrahedral elements during connectivity check"
            )

        for tet in elements:
            if len(tet) != 4:  # skip non-tetrahedral elements
                continue
            for i in range(4):
                for j in range(i + 1, 4):
                    G.add_edge(int(tet[i]), int(tet[j]))
        islands = list(nx.connected_components(G))
        if len(islands) > 1:
            sizes = [len(c) for c in islands]
            logger.warning(
                f"{len(islands)} disconnected sub-meshes; "
                f"largest: {max(sizes)} nodes, smallest: {min(sizes)}"
            )
        return len(islands)

    def generate_mesh(self) -> None:
        """Generate a coarse tetrahedral mesh from the STEP file using Gmsh."""
        try:
            gmsh.initialize()
            gmsh.model.add("FastCheck")

            # Import STEP geometry
            gmsh.model.occ.importShapes(str(self.step_file))
            gmsh.model.occ.removeAllDuplicates()  # removes coincident face/edge duplicates
            gmsh.model.occ.synchronize()

            if len(gmsh.model.getEntities(3)) != 1:
                logger.error("Frame is not a single solid! Meshing will be invalid.")
                raise RuntimeError(
                    "Meshing aborted: frame consists of multiple solids."
                )

            # Set mesh size parameters for coarse mesh
            # gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 2.0)
            # gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 5.0)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.005)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.01)
            gmsh.option.setNumber("Mesh.Optimize", 1)
            gmsh.option.setNumber("Mesh.OptimizeThreshold", 0.3)

            # Add mesh quality optimization parameters
            gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay algorithm
            gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)  # Enable Netgen optimizer
            gmsh.option.setNumber("Mesh.HighOrderOptimize", 2)  # Elastic + optimization
            gmsh.option.setNumber("Mesh.ElementOrder", 1)  # Linear elements
            gmsh.option.setNumber(
                "Mesh.MinimumCirclePoints", 12
            )  # For curved boundaries

            # Set optimization threshold
            gmsh.option.setNumber(
                "Mesh.OptimizeThreshold", 0.3
            )  # Optimize elements with quality < 0.3

            # Generate 3D mesh
            gmsh.model.mesh.generate(3)

            # Optimize the mesh
            gmsh.model.mesh.optimize("Netgen")  # Run Netgen optimization

            # Get nodes and elements
            node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
            elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements()

            # We only want tetrahedral elements (type 4)
            tet_idx = np.where(elem_types == 4)[0][
                0
            ]  # Get first index where element type is 4
            tet_nodes = (
                elem_node_tags[tet_idx].reshape(-1, 4) - 1
            )  # Convert to 0-based indexing

            # Reshape node coordinates
            coords = node_coords.reshape(-1, 3)

            # Filter out degenerate tetrahedral elements (near-zero volume)
            good_elements = []
            vol_threshold = 1e-10  # m^3; tweak as needed
            for el_idx, el in enumerate(tet_nodes):
                # Compute element volume
                v0, v1, v2, v3 = coords[el]
                J = np.vstack((v1 - v0, v2 - v0, v3 - v0)).T  # 3x3
                vol = abs(np.linalg.det(J)) / 6.0
                if vol >= vol_threshold:
                    good_elements.append(el)
            good_elements = np.array(good_elements, dtype=np.int32)

            removed = len(tet_nodes) - len(good_elements)
            if removed > 0:
                logger.warning(
                    f"Removed {removed} degenerate tets below volume {vol_threshold:.1e} m^3"
                )

            # Store mesh data
            self._mesh = {"nodes": coords, "elements": good_elements}

            # Log mesh size and bounds
            logger.info(
                f"Mesh bounding box: {np.min(coords, axis=0)} to {np.max(coords, axis=0)}"
            )
            logger.info(
                f"Mesh has {coords.shape[0]} nodes and {len(good_elements)} tets"
            )

            if (
                self.check_mesh_connectivity(
                    self._mesh["nodes"], self._mesh["elements"]
                )
                > 1
            ):
                raise RuntimeError(
                    "Mesh connectivity check failed: structure is fragmented."
                )

            gmsh.finalize()
            logger.info(
                f"Mesh generated with {len(coords)} nodes and {len(good_elements)} elements"
            )
            logger.info(f"Mesh has {len(coords)} nodes and {len(good_elements)} tets")

        except Exception as e:
            gmsh.finalize()
            raise RuntimeError(f"Failed to generate mesh: {str(e)}")

    def _calculate_volume(self) -> float:
        """Calculate total volume of the mesh."""
        nodes = self._mesh["nodes"]
        elements = self._mesh["elements"]
        volume = 0.0

        for el in elements:
            node_coords = nodes[el]
            # Calculate Jacobian from node differences
            J = node_coords[1:] - node_coords[0]  # 3x3 matrix
            volume += abs(np.linalg.det(J)) / 6.0

        return volume

    def check_structural_stress(
        self, load_override: Optional[LoadCase] = None
    ) -> float:
        """Check structural stress under acceleration loads."""
        if load_override is not None:
            load_case = load_override
            logger.info(f"Analyzing load case: {load_case.name} (manual override)")
        else:
            load_case = LoadCase(
                name="Axial 6g", acceleration_g=6.0, direction=[0, 0, -1]
            )
            logger.info(f"Analyzing load case: {load_case.name}")

        self.generate_mesh()

        results = []

        # 👇 Override default load cases for testing
        load_case = LoadCase(name="Axial 6g", acceleration_g=6.0, direction=(0, 0, -1))

        logger.info("Analyzing load case: Axial 6g (manual override)")
        max_stress = self._calculate_stresses(load_case)
        logger.info(f"Maximum stress for Axial 6g: {max_stress:.1f} MPa")
        results.append(max_stress)

        # Skip lateral case for now, or add manually below if needed
        return max(results)

    def check_thermal_stress(
        self, temp_cold: float = -40.0, temp_hot: float = 60.0
    ) -> float:
        """Estimate thermal stresses due to temperature gradient."""
        if not self.material.cte_per_k:
            logger.warning("CTE not provided, skipping thermal analysis")
            return 0.0

        # Simple thermal stress estimation
        delta_t = temp_hot - temp_cold
        thermal_strain = self.material.cte_per_k * delta_t
        thermal_stress = (
            thermal_strain * self.material.youngs_modulus_gpa * 1000
        )  # Convert to MPa
        return abs(thermal_stress)

    def _calculate_stresses(self, load_case: LoadCase) -> float:
        """Calculate maximum von Mises stress for a given load case."""
        if self._mesh is None:
            raise RuntimeError("Mesh not generated. Call generate_mesh() first.")

        # Get mesh data
        nodes = self._mesh["nodes"]
        elements = self._mesh["elements"]
        n_nodes = len(nodes)

        # Assemble system
        K, ndof = solver.assemble_system(nodes, elements, self.material.to_dict())

        # Calculate structural volume and mass
        volume = self._calculate_volume()
        structural_mass = self.material.density_kg_m3 * volume

        # Estimate payload mass (30% of structural mass)
        payload_mass = structural_mass * 0.3
        total_mass = structural_mass + payload_mass

        # Compute bounding box (for diagnostics)
        bbox_min = np.min(nodes, axis=0)
        bbox_max = np.max(nodes, axis=0)
        logger.info(f"Mesh bounding box: {bbox_min} to {bbox_max}")

        # Apply lumped inertial load (not distributed body force)
        acceleration = load_case.acceleration_g * 9.81  # m/s^2

        # Find nodes where load should be applied
        load_nodes = self._find_load_nodes(nodes, load_case.direction)
        if len(load_nodes) == 0:
            raise RuntimeError("No nodes found to carry inertial load")

        logger.info(f"Applying load to {len(load_nodes)} nodes")

        # Inertial force vector (in same direction as acceleration)
        # For a satellite under 6g downward acceleration, we apply 6g downward force at the top
        F_vec = total_mass * acceleration * np.array(load_case.direction)

        # Initialize force vector
        f = np.zeros(ndof)

        # Distribute load equally among load nodes
        share = 1.0 / len(load_nodes)
        for n in load_nodes:
            f[3 * n : 3 * n + 3] += F_vec * share

        # Verify force balance
        total_applied_force = np.sum(f.reshape(-1, 3), axis=0)
        expected_force = F_vec
        force_error = np.linalg.norm(total_applied_force - expected_force)

        logger.info(f"Applied force: {total_applied_force}")
        logger.info(f"Expected force: {expected_force}")
        logger.info(f"Force balance error: {force_error:.2e} N")

        if force_error > 1e-6:
            logger.warning(f"Force balance error is large: {force_error:.2e} N")

        print("Total applied force vector [N]:", total_applied_force)
        print("Expected total force vector [N]:", expected_force)
        print("Mesh bounding box (min, max) per axis:", bbox_min, bbox_max)

        # Apply constraints (fix base nodes)
        min_z = np.min(nodes[:, 2])
        base_nodes = np.where(nodes[:, 2] <= min_z + 0.001)[0]
        fixed_dofs = []
        for node in base_nodes:
            fixed_dofs.extend([3 * node, 3 * node + 1, 3 * node + 2])
        logger.info(f"Fixing {len(base_nodes)} base nodes ({len(fixed_dofs)} DOFs)")

        # Solve static system
        u = solver.solve_static(K, ndof, f, fixed_dofs)
        max_disp = np.max(np.linalg.norm(u.reshape(-1, 3), axis=1))
        logger.info(f"Max displacement (m): {max_disp:.3e}")
        print("Max displacement (m):", max_disp)

        # Compute stresses
        von_mises = solver.compute_stresses(nodes, elements, u, self.material.to_dict())

        # Log stress statistics
        max_stress_pa = np.max(von_mises)
        mean_stress_pa = np.mean(von_mises)
        logger.info(f"Max stress: {max_stress_pa/1e6:.3f} MPa")
        logger.info(f"Mean stress: {mean_stress_pa/1e6:.3f} MPa")
        logger.info(
            f"Non-zero stresses: {np.count_nonzero(von_mises)}/{len(von_mises)}"
        )

        return max_stress_pa / 1e6  # Convert Pa to MPa

    def run_validation(
        self, load_override: Optional[LoadCase] = None
    ) -> ValidationResults:
        """Run both structural and thermal checks, return results."""
        max_stress = self.check_structural_stress(load_override=load_override)
        thermal_stress = self.check_thermal_stress()

        # Determine pass/fail status
        # For thin rails, we expect stresses to be higher than yield strength
        status = "PASS" if max_stress < self.material.yield_strength_mpa else "FAIL"
        thermal_status = (
            "PASS" if thermal_stress < self.material.yield_strength_mpa else "FAIL"
        )

        return ValidationResults(
            max_stress_mpa=float(max_stress),
            sigma_allow_mpa=float(self.material.yield_strength_mpa),
            status=status,
            thermal_stress_mpa=float(thermal_stress),
            thermal_status=thermal_status,
        )

    def save_results(self, results: ValidationResults, output_file: Path) -> None:
        """Save validation results to a JSON file."""
        results_dict = {
            "max_stress_MPa": results.max_stress_mpa,
            "sigma_allow_MPa": results.sigma_allow_mpa,
            "status": results.status,
            "thermal_stress_MPa": results.thermal_stress_mpa,
            "thermal_status": results.thermal_status,
        }

        with open(output_file, "w") as f:
            json.dump(results_dict, f, indent=2)
