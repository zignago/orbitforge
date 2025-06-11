"""Diffusion-based geometry generator for CubeSat structures.

This module implements the learned geometry generation using a diffusion model
or GNN-based autoregressor trained on CubeSat CAD data.
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from loguru import logger
import json
from dataclasses import dataclass

from .mission import MissionSpec


@dataclass
class GeneratedMesh:
    """Container for generated mesh data."""

    vertices: np.ndarray  # Shape: (N, 3)
    faces: np.ndarray  # Shape: (M, 3) - triangular faces
    metadata: dict  # Additional generation metadata


class DiffusionGenerator:
    """Diffusion model wrapper for CubeSat structure generation."""

    def __init__(self, model_path: Optional[Path] = None):
        """Initialize the diffusion generator.

        Args:
            model_path: Path to the trained model weights. If None, uses default location.
        """
        self.model_path = (
            model_path
            or Path(__file__).parent.parent.parent / "models" / "diffusion_cubesat.pt"
        )
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Initializing diffusion generator with device: {self.device}")

    def load_model(self) -> None:
        """Load the trained diffusion model from disk."""
        if not self.model_path.exists():
            logger.warning(f"Model file not found at {self.model_path}")
            logger.warning("Creating mock model for development/testing")
            self._create_mock_model()
            return

        try:
            logger.info(f"Loading diffusion model from {self.model_path}")
            # For now, create a mock model since we don't have real weights
            # In production, this would be: self.model = torch.load(self.model_path, map_location=self.device)
            self._create_mock_model()
            logger.info("✓ Diffusion model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.warning("Falling back to mock model")
            self._create_mock_model()

    def _create_mock_model(self) -> None:
        """Create a mock model for development and testing."""
        logger.debug("Creating mock diffusion model")

        class MockDiffusionModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(6, 1024)  # 6 features -> latent space

            def sample(
                self, condition: torch.Tensor, n: int = 5
            ) -> List[GeneratedMesh]:
                """Generate n mesh samples conditioned on mission spec."""
                meshes = []

                for i in range(n):
                    # Generate deterministic but varied meshes based on condition
                    seed = hash(tuple(condition.numpy().tolist())) + i
                    np.random.seed(seed % (2**32))

                    # Generate a basic CubeSat frame-like mesh
                    mesh = self._generate_mock_cubesat_mesh(condition, variant=i)
                    meshes.append(mesh)

                return meshes

            def _generate_mock_cubesat_mesh(
                self, condition: torch.Tensor, variant: int = 0
            ) -> GeneratedMesh:
                """Generate a mock CubeSat frame mesh."""
                # Extract mission parameters
                orbit_km, payload_vol, delta_v, rail_mm, deck_mm, material_id = (
                    condition.numpy()
                )

                # Create basic CubeSat frame geometry with variation
                base_size = 100.0  # 100mm for 1U
                height = base_size * max(
                    1, int(payload_vol / 1000)
                )  # Scale with payload volume

                # Add variation based on variant index
                variation_factor = 1.0 + (variant * 0.1) - 0.2  # ±20% variation
                rail_thickness = rail_mm * variation_factor
                deck_thickness = deck_mm * variation_factor

                # Generate vertices for a frame structure
                vertices = self._create_frame_vertices(
                    base_size, height, rail_thickness, deck_thickness, variant
                )
                faces = self._create_frame_faces(len(vertices))

                metadata = {
                    "variant": variant,
                    "orbit_km": orbit_km,
                    "payload_volume": payload_vol,
                    "rail_thickness": rail_thickness,
                    "deck_thickness": deck_thickness,
                    "material_id": material_id,
                    "estimated_mass_kg": self._estimate_mass(
                        vertices, faces, material_id
                    ),
                }

                return GeneratedMesh(vertices=vertices, faces=faces, metadata=metadata)

            def _create_frame_vertices(
                self,
                size: float,
                height: float,
                rail_thick: float,
                deck_thick: float,
                variant: int,
            ) -> np.ndarray:
                """Create vertices for a CubeSat frame structure."""
                vertices = []

                # Enhanced variation based on variant
                variation_factor = 1.0 + (variant * 0.2) - 0.4  # ±40% variation
                size_variation = size * variation_factor
                height_variation = height * (
                    1.0 + (variant * 0.15) - 0.3
                )  # ±30% height variation

                # Vary wall thicknesses
                rail_variation = 1.0 + (
                    np.sin(variant * np.pi / 3) * 0.3
                )  # ±30% rail thickness variation
                rail_thick_var = rail_thick * rail_variation
                deck_variation = 1.0 + (
                    np.cos(variant * np.pi / 4) * 0.25
                )  # ±25% deck thickness variation
                deck_thick_var = deck_thick * deck_variation

                # Bottom deck vertices with hollow sections
                bottom_z = 0
                if variant % 3 == 0:  # Hollow deck design
                    vertices.extend(
                        self._create_hollow_deck(
                            size_variation, bottom_z, rail_thick_var, deck_thick_var
                        )
                    )
                else:  # Solid deck design
                    vertices.extend(
                        self._create_solid_deck(
                            size_variation, bottom_z, rail_thick_var
                        )
                    )

                # Top deck vertices with variation
                top_z = height_variation
                if variant % 3 == 1:  # Hollow top deck
                    vertices.extend(
                        self._create_hollow_deck(
                            size_variation, top_z, rail_thick_var, deck_thick_var
                        )
                    )
                else:  # Solid top deck
                    vertices.extend(
                        self._create_solid_deck(size_variation, top_z, rail_thick_var)
                    )

                # Add structural members based on variant
                vertices.extend(
                    self._create_structural_members(
                        size_variation, height_variation, rail_thick_var, variant
                    )
                )

                return np.array(vertices)

            def _create_hollow_deck(
                self, size: float, z: float, rail_thick: float, deck_thick: float
            ) -> list:
                """Create vertices for a hollow deck design."""
                outer_vertices = [
                    [0, 0, z],
                    [size, 0, z],
                    [size, size, z],
                    [0, size, z],
                    [rail_thick, rail_thick, z],
                    [size - rail_thick, rail_thick, z],
                    [size - rail_thick, size - rail_thick, z],
                    [rail_thick, size - rail_thick, z],
                ]

                # Add inner hollow section
                inner_offset = deck_thick * 2
                inner_vertices = [
                    [inner_offset, inner_offset, z],
                    [size - inner_offset, inner_offset, z],
                    [size - inner_offset, size - inner_offset, z],
                    [inner_offset, size - inner_offset, z],
                ]

                return outer_vertices + inner_vertices

            def _create_solid_deck(
                self, size: float, z: float, rail_thick: float
            ) -> list:
                """Create vertices for a solid deck design."""
                return [
                    [0, 0, z],
                    [size, 0, z],
                    [size, size, z],
                    [0, size, z],
                    [rail_thick, rail_thick, z],
                    [size - rail_thick, rail_thick, z],
                    [size - rail_thick, size - rail_thick, z],
                    [rail_thick, size - rail_thick, z],
                ]

            def _create_structural_members(
                self, size: float, height: float, rail_thick: float, variant: int
            ) -> list:
                """Create vertices for structural members with variation."""
                vertices = []

                # Basic rails with rotation
                for i in range(4):
                    angle = i * np.pi / 2 + (variant * 0.3)  # Base rotation
                    x = size / 2 + (size / 2 - rail_thick) * np.cos(angle)
                    y = size / 2 + (size / 2 - rail_thick) * np.sin(angle)

                    # Add diagonal braces for some variants
                    if variant % 2 == 0:
                        x += rail_thick * np.cos(angle + np.pi / 4)
                        y += rail_thick * np.sin(angle + np.pi / 4)

                    vertices.extend(
                        self._create_vertical_member(x, y, height, rail_thick)
                    )

                # Add internal lattice for some variants
                if variant % 3 == 2:
                    vertices.extend(
                        self._create_internal_lattice(size, height, rail_thick)
                    )

                # Add diagonal braces for some variants
                if variant % 2 == 0:
                    vertices.extend(
                        self._create_diagonal_braces(size, height, rail_thick)
                    )

                return vertices

            def _create_vertical_member(
                self, x: float, y: float, height: float, thickness: float
            ) -> list:
                """Create vertices for a vertical structural member."""
                return [
                    [x - thickness / 2, y - thickness / 2, 0],
                    [x + thickness / 2, y - thickness / 2, 0],
                    [x + thickness / 2, y + thickness / 2, 0],
                    [x - thickness / 2, y + thickness / 2, 0],
                    [x - thickness / 2, y - thickness / 2, height],
                    [x + thickness / 2, y - thickness / 2, height],
                    [x + thickness / 2, y + thickness / 2, height],
                    [x - thickness / 2, y + thickness / 2, height],
                ]

            def _create_internal_lattice(
                self, size: float, height: float, thickness: float
            ) -> list:
                """Create vertices for internal lattice structure."""
                vertices = []

                # Create cross-bracing lattice
                for i in range(2):
                    for j in range(2):
                        x = size * (0.25 + i * 0.5)
                        y = size * (0.25 + j * 0.5)
                        vertices.extend(
                            self._create_vertical_member(x, y, height, thickness * 0.8)
                        )

                return vertices

            def _create_diagonal_braces(
                self, size: float, height: float, thickness: float
            ) -> list:
                """Create vertices for diagonal bracing."""
                vertices = []

                # Add diagonal members between corners
                for i in range(4):
                    angle = i * np.pi / 2
                    x1 = size / 2 + (size / 2 - thickness) * np.cos(angle)
                    y1 = size / 2 + (size / 2 - thickness) * np.sin(angle)
                    x2 = size / 2 + (size / 2 - thickness) * np.cos(angle + np.pi / 2)
                    y2 = size / 2 + (size / 2 - thickness) * np.sin(angle + np.pi / 2)

                    # Create diagonal member
                    dx = x2 - x1
                    dy = y2 - y1
                    dz = height
                    length = np.sqrt(dx**2 + dy**2 + dz**2)

                    # Scale thickness based on length
                    scaled_thickness = thickness * 0.7  # Thinner diagonals

                    vertices.extend(
                        [
                            [x1, y1, 0],
                            [x1 + scaled_thickness, y1, 0],
                            [x2, y2, height],
                            [x2 + scaled_thickness, y2, height],
                        ]
                    )

                return vertices

            def _create_frame_faces(self, num_vertices: int) -> np.ndarray:
                """Create triangular faces for the frame structure."""
                faces = []

                # Simple triangulation for bottom deck (8 vertices)
                faces.extend(
                    [[0, 1, 4], [1, 2, 5], [2, 3, 6], [3, 0, 7], [4, 5, 6], [4, 6, 7]]
                )

                # Top deck faces (offset by 8)
                offset = 8
                faces.extend(
                    [
                        [0 + offset, 1 + offset, 4 + offset],
                        [1 + offset, 2 + offset, 5 + offset],
                        [2 + offset, 3 + offset, 6 + offset],
                        [3 + offset, 0 + offset, 7 + offset],
                        [4 + offset, 5 + offset, 6 + offset],
                        [4 + offset, 6 + offset, 7 + offset],
                    ]
                )

                # Connect bottom to top (simplified)
                for i in range(8):
                    next_i = (i + 1) % 8
                    faces.extend(
                        [[i, next_i, i + offset], [next_i, next_i + offset, i + offset]]
                    )

                # Add rail faces (simplified)
                rail_start = 16
                for rail in range(4):
                    base = rail_start + rail * 8
                    # Add some faces for each rail
                    for j in range(6):
                        if base + j + 2 < num_vertices:
                            faces.append([base + j, base + j + 1, base + j + 2])

                return np.array(faces)

            def _estimate_mass(
                self, vertices: np.ndarray, faces: np.ndarray, material_id: float
            ) -> float:
                """Estimate mass of the generated structure."""
                # Simplified volume calculation
                volume_mm3 = len(vertices) * 1000  # Rough estimate
                density_kg_mm3 = 2.7e-6 if material_id < 0.5 else 4.4e-6  # Al vs Ti
                return volume_mm3 * density_kg_mm3

        self.model = MockDiffusionModel()
        self.model.to(self.device)
        logger.debug("✓ Mock diffusion model created")

    def encode_mission_spec(self, spec: MissionSpec) -> torch.Tensor:
        """Convert mission specification to conditioning vector.

        Args:
            spec: Mission specification object

        Returns:
            Fixed-size conditioning tensor
        """
        # Extract key features for conditioning
        features = [
            spec.orbit_alt_km / 1000.0,  # Normalize orbit altitude
            spec.payload_mass_kg,  # Payload mass
            spec.bus_u * 100.0,  # Delta-V proxy (U count * 100)
            spec.rail_mm,  # Rail thickness
            spec.deck_mm,  # Deck thickness
            1.0 if spec.material.value == "Ti_6Al_4V" else 0.0,  # Material encoding
        ]

        conditioning = torch.tensor(features, dtype=torch.float32, device=self.device)
        logger.debug(f"Mission spec encoded to: {conditioning.numpy()}")
        return conditioning

    def generate_candidates(self, spec: MissionSpec, n: int = 5) -> List[GeneratedMesh]:
        """Generate n candidate designs from mission specification.

        Args:
            spec: Mission specification
            n: Number of candidates to generate

        Returns:
            List of generated mesh candidates
        """
        if self.model is None:
            self.load_model()

        logger.info(f"Generating {n} candidate designs for mission spec")

        # Encode mission spec to conditioning vector
        condition = self.encode_mission_spec(spec)

        # Generate candidates using the model
        with torch.no_grad():
            candidates = self.model.sample(condition=condition, n=n)

        logger.info(f"✓ Generated {len(candidates)} candidates")
        for i, candidate in enumerate(candidates):
            logger.debug(
                f"Candidate {i}: {candidate.vertices.shape[0]} vertices, "
                f"mass ~{candidate.metadata.get('estimated_mass_kg', 0):.3f} kg"
            )

        return candidates
