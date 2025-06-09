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

    def generate_mesh(self) -> None:
        """Generate a coarse tetrahedral mesh from the STEP file using Gmsh."""
        try:
            gmsh.initialize()
            gmsh.model.add("FastCheck")

            # Import STEP geometry
            gmsh.model.occ.importShapes(str(self.step_file))
            gmsh.model.occ.synchronize()

            # Set mesh size parameters for coarse mesh
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 2.0)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 5.0)

            # Generate 3D mesh
            gmsh.model.mesh.generate(3)

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

            # Store mesh data
            self._mesh = {"nodes": coords, "elements": tet_nodes}

            gmsh.finalize()
            logger.info(
                f"Mesh generated with {len(coords)} nodes and {len(tet_nodes)} elements"
            )

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

    def check_structural_stress(self) -> float:
        """Calculate maximum stress across all load cases."""
        if self._mesh is None:
            self.generate_mesh()

        max_stress = 0.0
        for load_case in self.load_cases:
            logger.info(f"Analyzing load case: {load_case.name}")
            stress = self._calculate_stresses(load_case)
            max_stress = max(max_stress, stress)
            logger.info(f"Maximum stress for {load_case.name}: {stress:.1f} MPa")

        return max_stress

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

        # Create load vector (F = ma)
        mass_per_node = self.material.density_kg_m3 * self._calculate_volume() / n_nodes
        acceleration = load_case.acceleration_g * 9.81  # Convert to m/s^2

        f = np.zeros(ndof)
        for i in range(n_nodes):
            f[3 * i : 3 * i + 3] = (
                mass_per_node
                * acceleration
                * self.load_scale
                * np.array(load_case.direction)
            )

        # Apply constraints (fix base nodes)
        min_z = np.min(nodes[:, 2])
        base_nodes = np.where(nodes[:, 2] <= min_z + 0.001)[0]
        fixed_dofs = []
        for node in base_nodes:
            fixed_dofs.extend([3 * node, 3 * node + 1, 3 * node + 2])

        # Solve system
        u = solver.solve_static(K, ndof, f, fixed_dofs)

        # Calculate stresses
        von_mises = solver.compute_stresses(nodes, elements, u, self.material.to_dict())

        return np.max(von_mises) / 1e6  # Convert Pa to MPa

    def run_validation(self) -> ValidationResults:
        """Run both structural and thermal checks, return results."""
        max_stress = self.check_structural_stress()
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
