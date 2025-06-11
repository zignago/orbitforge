"""
Multi-design generator for OrbitForge v0.1.2.

This module implements parameter jittering and multi-design generation capabilities
to allow users to explore multiple structural frame candidates per mission input.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from loguru import logger
import uuid

from .mission import MissionSpec, Material
from .basic_frame import build_basic_frame
from ..fea import FastPhysicsValidator, MaterialProperties


@dataclass
class DesignVariant:
    """Represents a single design variant with its parameters and results."""

    design_id: str
    rail_mm: float
    deck_mm: float
    material: Material
    mass_kg: Optional[float] = None
    max_stress_mpa: Optional[float] = None
    status: Optional[str] = None
    step_file: Optional[Path] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        # Convert Path to string for JSON serialization
        if self.step_file:
            result["step_file"] = str(self.step_file)
        return result


class ParameterJitter:
    """Handles bounded parameter jittering with reproducible randomness."""

    def __init__(self, seed: int = 42):
        """Initialize with a random seed for reproducibility."""
        self.rng = random.Random(seed)
        logger.info(f"Initialized parameter jitter with seed: {seed}")

    def jitter_rail_thickness(
        self, base_value: float, variation_mm: float = 0.5
    ) -> float:
        """Apply random jitter to rail thickness within bounds."""
        jitter = self.rng.uniform(-variation_mm, variation_mm)
        result = max(1.0, base_value + jitter)  # Minimum 1mm thickness
        logger.debug(
            f"Rail thickness: {base_value:.2f} → {result:.2f} mm (jitter: {jitter:+.2f})"
        )
        return result

    def jitter_deck_thickness(
        self, base_value: float, variation_mm: float = 0.3
    ) -> float:
        """Apply random jitter to deck thickness within bounds."""
        jitter = self.rng.uniform(-variation_mm, variation_mm)
        result = max(1.0, base_value + jitter)  # Minimum 1mm thickness
        logger.debug(
            f"Deck thickness: {base_value:.2f} → {result:.2f} mm (jitter: {jitter:+.2f})"
        )
        return result

    def jitter_material(
        self, base_material: Material, change_probability: float = 0.3
    ) -> Material:
        """Occasionally change material with given probability."""
        if self.rng.random() < change_probability:
            # Switch between available materials
            materials = list(Material)
            other_materials = [m for m in materials if m != base_material]
            if other_materials:
                result = self.rng.choice(other_materials)
                logger.debug(f"Material: {base_material.value} → {result.value}")
                return result
        return base_material


class MultiDesignGenerator:
    """Generates multiple design variants with parameter jittering."""

    def __init__(self, base_spec: MissionSpec, num_variants: int, seed: int = 42):
        """Initialize the multi-design generator."""
        self.base_spec = base_spec
        self.num_variants = min(max(1, num_variants), 10)  # Clamp between 1-10
        self.jitter = ParameterJitter(seed)
        self.variants: List[DesignVariant] = []

        logger.info(
            f"Initialized multi-design generator for {self.num_variants} variants"
        )

    def generate_variants(self) -> List[DesignVariant]:
        """Generate parameter variants based on the base specification."""
        variants = []

        for i in range(self.num_variants):
            design_id = f"design_{i+1:03d}"

            # Apply parameter jittering
            rail_mm = self.jitter.jitter_rail_thickness(self.base_spec.rail_mm)
            deck_mm = self.jitter.jitter_deck_thickness(self.base_spec.deck_mm)
            material = self.jitter.jitter_material(self.base_spec.material)

            variant = DesignVariant(
                design_id=design_id, rail_mm=rail_mm, deck_mm=deck_mm, material=material
            )
            variants.append(variant)

            logger.info(
                f"Generated variant {design_id}: rail={rail_mm:.2f}mm, deck={deck_mm:.2f}mm, material={material.value}"
            )

        self.variants = variants
        return variants

    def build_designs(self, output_dir: Path) -> List[DesignVariant]:
        """Build all design variants and store their geometry files."""
        if not self.variants:
            self.generate_variants()

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        built_variants = []

        for variant in self.variants:
            try:
                # Create variant-specific output directory
                variant_dir = output_dir / variant.design_id
                variant_dir.mkdir(parents=True, exist_ok=True)

                # Create modified mission spec for this variant
                variant_spec = MissionSpec(
                    bus_u=self.base_spec.bus_u,
                    payload_mass_kg=self.base_spec.payload_mass_kg,
                    orbit_alt_km=self.base_spec.orbit_alt_km,
                    mass_limit_kg=self.base_spec.mass_limit_kg,
                    rail_mm=variant.rail_mm,
                    deck_mm=variant.deck_mm,
                    material=variant.material,
                )

                # Build the frame
                logger.info(f"Building frame for {variant.design_id}...")
                step_file = build_basic_frame(variant_spec, variant_dir)
                variant.step_file = step_file

                # Extract mass from the mass budget file
                mass_budget_file = variant_dir / "mass_budget.csv"
                if mass_budget_file.exists():
                    # Read the last line (total) from mass budget
                    with open(mass_budget_file, "r") as f:
                        lines = f.readlines()
                        if len(lines) > 1:
                            total_line = lines[-1].strip().split(",")
                            if len(total_line) >= 3:
                                variant.mass_kg = float(total_line[2])

                logger.info(
                    f"✓ Built {variant.design_id} (mass: {variant.mass_kg:.3f} kg)"
                )
                built_variants.append(variant)

            except Exception as e:
                logger.error(f"Failed to build {variant.design_id}: {str(e)}")
                # Still add the variant but mark it as failed
                variant.status = "BUILD_FAILED"
                built_variants.append(variant)

        return built_variants

    def run_batch_fea(self, variants: List[DesignVariant]) -> List[DesignVariant]:
        """Run FEA validation on all variants."""
        logger.info("Running batch FEA validation...")

        for variant in variants:
            if not variant.step_file or variant.status == "BUILD_FAILED":
                continue

            try:
                # Create mission spec for this variant to get material properties
                variant_spec = MissionSpec(
                    bus_u=self.base_spec.bus_u,
                    payload_mass_kg=self.base_spec.payload_mass_kg,
                    orbit_alt_km=self.base_spec.orbit_alt_km,
                    mass_limit_kg=self.base_spec.mass_limit_kg,
                    rail_mm=variant.rail_mm,
                    deck_mm=variant.deck_mm,
                    material=variant.material,
                )

                # Create material properties for validator
                material_props = MaterialProperties(
                    name=variant.material.value,
                    yield_strength_mpa=variant_spec.yield_mpa,
                    youngs_modulus_gpa=variant_spec.youngs_modulus_gpa,
                    poissons_ratio=variant_spec.poissons_ratio,
                    density_kg_m3=variant_spec.density_kg_m3,
                    cte_per_k=variant_spec.cte_per_k,
                )

                # Run validation
                validator = FastPhysicsValidator(variant.step_file, material_props)
                results = validator.run_validation()

                # Store results
                variant.max_stress_mpa = results.max_stress_mpa
                variant.status = results.status

                # Save detailed results to file
                results_file = variant.step_file.parent / "results.json"
                validator.save_results(results, results_file)

                logger.info(
                    f"✓ FEA completed for {variant.design_id}: {results.status} (σ_max = {results.max_stress_mpa:.1f} MPa)"
                )

            except Exception as e:
                logger.error(f"FEA failed for {variant.design_id}: {str(e)}")
                variant.status = "FEA_FAILED"

        return variants

    def rank_designs(self, variants: List[DesignVariant]) -> List[DesignVariant]:
        """Rank designs by status (PASS first) then by mass (ascending)."""

        def sort_key(variant):
            # Primary sort: status (PASS comes first)
            status_priority = 0 if variant.status == "PASS" else 1
            # Secondary sort: mass (lower is better, handle None values)
            mass = variant.mass_kg if variant.mass_kg is not None else float("inf")
            return (status_priority, mass)

        ranked = sorted(variants, key=sort_key)
        logger.info(f"Ranked {len(ranked)} designs")
        return ranked

    def generate_summary(self, variants: List[DesignVariant], output_dir: Path) -> Path:
        """Generate summary.json with metadata for all designs."""
        summary_data = []

        for variant in variants:
            summary_data.append(
                {
                    "design": variant.design_id,
                    "rail_mm": variant.rail_mm,
                    "deck_mm": variant.deck_mm,
                    "material": variant.material.value,
                    "mass_kg": variant.mass_kg,
                    "max_stress_MPa": variant.max_stress_mpa,
                    "status": variant.status or "UNKNOWN",
                    "step_file": str(variant.step_file) if variant.step_file else None,
                }
            )

        summary_file = output_dir / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary_data, f, indent=2)

        logger.info(f"Generated summary at {summary_file}")
        return summary_file

    def run_complete_workflow(self, output_dir: Path, run_fea: bool = True) -> Path:
        """Run the complete multi-design workflow."""
        logger.info("Starting multi-design generation workflow...")

        # Generate parameter variants
        variants = self.generate_variants()

        # Build all designs
        variants = self.build_designs(output_dir)

        # Run FEA if requested
        if run_fea:
            variants = self.run_batch_fea(variants)

        # Rank designs
        variants = self.rank_designs(variants)

        # Generate summary
        summary_file = self.generate_summary(variants, output_dir)

        logger.info(
            f"Multi-design workflow completed. Summary available at {summary_file}"
        )
        return summary_file
