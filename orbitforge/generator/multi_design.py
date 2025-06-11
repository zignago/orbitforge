"""
Multi-design generator for OrbitForge v0.1.2.

This module implements parameter jittering and multi-design generation capabilities
to allow users to explore multiple structural frame candidates per mission input.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, asdict
from loguru import logger
import uuid
import functools
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from .mission import MissionSpec, Material, load_materials
from .basic_frame import build_basic_frame
from ..fea import FastPhysicsValidator, MaterialProperties
from ..config.design_config import config


class DesignCache:
    """Cache for design results to avoid redundant computations."""

    def __init__(self, max_size_mb: int = config.resources.MAX_CACHE_SIZE_MB):
        self._cache: Dict[str, Tuple[float, Any]] = {}  # (timestamp, data)
        self._max_size_mb = max_size_mb
        self._cleanup_threshold = 0.9  # Cleanup when 90% full

    def _make_key(self, **kwargs) -> str:
        """Create a cache key from parameters."""
        # Round float values to 3 decimal places for consistent keys
        formatted_kwargs = {}
        for k, v in sorted(kwargs.items()):
            if isinstance(v, float):
                formatted_kwargs[k] = f"{v:.3f}"
            else:
                formatted_kwargs[k] = str(v)
        return "|".join(f"{k}={v}" for k, v in sorted(formatted_kwargs.items()))

    def _estimate_size_mb(self) -> float:
        """Estimate current cache size in MB."""
        import sys

        return sys.getsizeof(self._cache) / (1024 * 1024)

    def _cleanup_old_entries(self):
        """Remove old entries if cache is too large."""
        if self._estimate_size_mb() < self._max_size_mb * self._cleanup_threshold:
            return

        current_time = time.time()
        expiry_time = current_time - (config.resources.CACHE_EXPIRY_HOURS * 3600)

        self._cache = {
            k: (t, d) for k, (t, d) in self._cache.items() if t > expiry_time
        }

    def get(self, **kwargs) -> Optional[Any]:
        """Get cached result if available."""
        key = self._make_key(**kwargs)
        if key in self._cache:
            timestamp, data = self._cache[key]
            if time.time() - timestamp < config.resources.CACHE_EXPIRY_HOURS * 3600:
                return data
            del self._cache[key]
        return None

    def put(self, data: Any, **kwargs):
        """Cache result with parameters as key."""
        self._cleanup_old_entries()
        key = self._make_key(**kwargs)
        self._cache[key] = (time.time(), data)


# Global cache instance
_design_cache = DesignCache()


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
        self.seed = seed
        self.rng = random.Random(seed)
        self._used_combinations: Set[str] = set()
        logger.info(f"Initialized parameter jitter with seed: {seed}")

    def reset(self):
        """Reset the random number generator to initial state."""
        self.rng = random.Random(self.seed)
        self._used_combinations.clear()

    def _is_unique_combination(
        self, rail: float, deck: float, material: Material
    ) -> bool:
        """Check if parameter combination is unique."""
        key = f"{rail:.3f}|{deck:.3f}|{material.value}"
        if key in self._used_combinations:
            return False
        self._used_combinations.add(key)
        return True

    def jitter_rail_thickness(
        self, base_value: float, variation_mm: float = config.params.RAIL_VARIATION_MM
    ) -> float:
        """Apply random jitter to rail thickness within bounds."""
        jitter = self.rng.uniform(-variation_mm, variation_mm)
        result = round(
            max(
                config.params.MIN_RAIL_THICKNESS,
                min(config.params.MAX_RAIL_THICKNESS, base_value + jitter),
            ),
            3,  # Round to 3 decimal places for consistency
        )
        logger.debug(
            f"Rail thickness: {base_value:.2f} → {result:.2f} mm (jitter: {jitter:+.2f})"
        )
        return result

    def jitter_deck_thickness(
        self, base_value: float, variation_mm: float = config.params.DECK_VARIATION_MM
    ) -> float:
        """Apply random jitter to deck thickness within bounds."""
        jitter = self.rng.uniform(-variation_mm, variation_mm)
        result = round(
            max(
                config.params.MIN_DECK_THICKNESS,
                min(config.params.MAX_DECK_THICKNESS, base_value + jitter),
            ),
            3,  # Round to 3 decimal places for consistency
        )
        logger.debug(
            f"Deck thickness: {base_value:.2f} → {result:.2f} mm (jitter: {jitter:+.2f})"
        )
        return result

    def jitter_material(
        self,
        base_material: Material,
        change_probability: float = config.params.MATERIAL_CHANGE_PROBABILITY,
    ) -> Material:
        """Occasionally change material with given probability."""
        if self.rng.random() < change_probability:
            # Switch between available materials
            materials = sorted(list(Material))  # Sort for consistency
            other_materials = [m for m in materials if m != base_material]
            if other_materials:
                result = self.rng.choice(other_materials)
                logger.debug(f"Material: {base_material.value} → {result.value}")
                return result
        return base_material

    def generate_unique_variant(
        self,
        base_rail: float,
        base_deck: float,
        base_material: Material,
        max_attempts: int = 10,
    ) -> Tuple[float, float, Material]:
        """Generate unique parameter combination with retry logic."""
        for _ in range(max_attempts):
            rail = self.jitter_rail_thickness(base_rail)
            deck = self.jitter_deck_thickness(base_deck)
            material = self.jitter_material(base_material)

            if self._is_unique_combination(rail, deck, material):
                return rail, deck, material

        # If max attempts reached, make larger variations
        logger.warning("Max attempts reached, increasing variation ranges")
        rail = self.jitter_rail_thickness(
            base_rail, variation_mm=config.params.RAIL_VARIATION_MM * 2
        )
        deck = self.jitter_deck_thickness(
            base_deck, variation_mm=config.params.DECK_VARIATION_MM * 2
        )
        material = self.jitter_material(base_material, change_probability=0.5)
        return rail, deck, material


def run_fea_process(variant_dict: Dict) -> Dict:
    """Run FEA validation in a separate process."""
    try:
        # Reconstruct objects from dict
        variant = DesignVariant(**variant_dict)
        if not variant.step_file or variant.status == "BUILD_FAILED":
            return variant_dict

        # Create mission spec
        spec = MissionSpec(
            bus_u=3,  # Default values since we only need material properties
            payload_mass_kg=1.0,
            orbit_alt_km=550,
            mass_limit_kg=4.0,
            material=variant.material,
        )

        # Create material properties
        material_props = MaterialProperties(
            name=variant.material.value,
            yield_strength_mpa=spec.yield_mpa,
            youngs_modulus_gpa=spec.youngs_modulus_gpa,
            poissons_ratio=spec.poissons_ratio,
            density_kg_m3=spec.density_kg_m3,
            cte_per_k=spec.cte_per_k,
        )

        # Run validation
        validator = FastPhysicsValidator(Path(variant.step_file), material_props)
        results = validator.run_validation()

        # Update variant with results
        variant_dict["max_stress_mpa"] = results.max_stress_mpa
        variant_dict["status"] = results.status

        # Save results
        results_file = Path(variant.step_file).parent / "results.json"
        validator.save_results(results, results_file)

        logger.info(
            f"✓ FEA completed for {variant.design_id}: {results.status} "
            f"(σ_max = {results.max_stress_mpa:.1f} MPa)"
        )

    except Exception as e:
        logger.error(f"FEA failed for {variant_dict['design_id']}: {str(e)}")
        variant_dict["status"] = "FEA_FAILED"

    return variant_dict


class MultiDesignGenerator:
    """Generates multiple design variants with parameter jittering."""

    def __init__(self, base_spec: MissionSpec, num_variants: int, seed: int = 42):
        """Initialize the multi-design generator."""
        self.base_spec = base_spec
        self.num_variants = min(max(1, num_variants), 10)  # Clamp between 1-10
        self.jitter = ParameterJitter(seed)
        self.variants: List[DesignVariant] = []
        self._executor = None
        self._process_executor = None

        logger.info(
            f"Initialized multi-design generator for {self.num_variants} variants"
        )

    def _init_executors(self):
        """Initialize thread and process executors if not already initialized."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=config.resources.MAX_PARALLEL_FEA
            )
        if self._process_executor is None:
            self._process_executor = ProcessPoolExecutor(
                max_workers=config.resources.MAX_PARALLEL_FEA
            )

    def _cleanup_executors(self):
        """Clean up executors properly."""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
        if self._process_executor:
            self._process_executor.shutdown(wait=True)
            self._process_executor = None

    def generate_variants(self) -> List[DesignVariant]:
        """Generate parameter variants based on the base specification."""
        variants = []

        # Reset jitter to ensure consistent results
        self.jitter.reset()

        for i in range(self.num_variants):
            design_id = f"design_{i+1:03d}"

            # Generate unique parameter combination
            rail_mm, deck_mm, material = self.jitter.generate_unique_variant(
                self.base_spec.rail_mm, self.base_spec.deck_mm, self.base_spec.material
            )

            variant = DesignVariant(
                design_id=design_id, rail_mm=rail_mm, deck_mm=deck_mm, material=material
            )
            variants.append(variant)

            logger.info(
                f"Generated variant {design_id}: rail={rail_mm:.2f}mm, deck={deck_mm:.2f}mm, material={material.value}"
            )

        self.variants = variants
        return variants

    def _build_single_design(
        self, variant: DesignVariant, output_dir: Path
    ) -> DesignVariant:
        """Build a single design variant."""
        try:
            # Check cache first
            cached_result = _design_cache.get(
                rail_mm=variant.rail_mm,
                deck_mm=variant.deck_mm,
                material=variant.material.value,
            )
            if cached_result:
                logger.info(f"Using cached result for {variant.design_id}")
                return cached_result

            # Create variant-specific output directory
            variant_dir = output_dir / variant.design_id
            try:
                variant_dir.mkdir(parents=True, exist_ok=True)
            except (PermissionError, OSError) as e:
                logger.error(
                    f"Failed to create directory for {variant.design_id}: {str(e)}"
                )
                variant.status = "BUILD_FAILED"
                return variant

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
            try:
                step_file = build_basic_frame(variant_spec, variant_dir)
                variant.step_file = step_file
            except (PermissionError, OSError) as e:
                logger.error(f"Failed to build frame for {variant.design_id}: {str(e)}")
                variant.status = "BUILD_FAILED"
                return variant

            # Extract mass from the mass budget file
            mass_budget_file = variant_dir / "mass_budget.csv"
            if mass_budget_file.exists():
                try:
                    # Read the last line (total) from mass budget
                    with open(mass_budget_file, "r") as f:
                        lines = f.readlines()
                        if len(lines) > 1:
                            total_line = lines[-1].strip().split(",")
                            if len(total_line) >= 3:
                                variant.mass_kg = float(total_line[2])
                except (PermissionError, OSError) as e:
                    logger.warning(
                        f"Failed to read mass budget for {variant.design_id}: {str(e)}"
                    )

            logger.info(f"✓ Built {variant.design_id} (mass: {variant.mass_kg:.3f} kg)")

            # Cache the result
            _design_cache.put(
                variant,
                rail_mm=variant.rail_mm,
                deck_mm=variant.deck_mm,
                material=variant.material.value,
            )

            return variant

        except Exception as e:
            logger.error(f"Failed to build {variant.design_id}: {str(e)}")
            variant.status = "BUILD_FAILED"
            return variant

    def build_designs(self, output_dir: Path) -> List[DesignVariant]:
        """Build all design variants and store their geometry files."""
        if not self.variants:
            self.generate_variants()

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            self._init_executors()
            # Build designs in parallel
            futures = []
            for variant in self.variants:
                future = self._executor.submit(
                    self._build_single_design, variant, output_dir
                )
                futures.append(future)

            # Collect results
            built_variants = []
            for future in as_completed(futures):
                try:
                    variant = future.result()
                    built_variants.append(variant)
                except Exception as e:
                    logger.error(f"Parallel build failed: {str(e)}")

            return built_variants
        except Exception as e:
            logger.error(f"Build designs failed: {str(e)}")
            raise
        finally:
            self._cleanup_executors()

    def run_batch_fea(self, variants: List[DesignVariant]) -> List[DesignVariant]:
        """Run FEA validation on all variants using multiprocessing."""
        logger.info("Running batch FEA validation...")

        try:
            self._init_executors()
            # Convert variants to dictionaries for multiprocessing
            variant_dicts = [v.to_dict() for v in variants]

            # Submit FEA jobs to process pool
            futures = []
            for variant_dict in variant_dicts:
                future = self._process_executor.submit(run_fea_process, variant_dict)
                futures.append(future)

            # Collect results
            validated_variants = []
            for future in as_completed(futures):
                try:
                    variant_dict = future.result()
                    # Reconstruct DesignVariant from dict
                    variant = DesignVariant(
                        design_id=variant_dict["design_id"],
                        rail_mm=variant_dict["rail_mm"],
                        deck_mm=variant_dict["deck_mm"],
                        material=Material[
                            variant_dict["material"].upper().replace("-", "_")
                        ],
                        mass_kg=variant_dict.get("mass_kg"),
                        max_stress_mpa=variant_dict.get("max_stress_mpa"),
                        status=variant_dict.get("status"),
                        step_file=(
                            Path(variant_dict["step_file"])
                            if variant_dict.get("step_file")
                            else None
                        ),
                    )
                    validated_variants.append(variant)
                except Exception as e:
                    logger.error(f"Failed to process FEA result: {str(e)}")

            return validated_variants

        finally:
            self._cleanup_executors()

    def rank_designs(self, variants: List[DesignVariant]) -> List[DesignVariant]:
        """Rank designs by multiple criteria."""
        from .mission import load_materials

        # Load materials data once
        materials_data = load_materials()

        def sort_key(variant):
            # Primary sort: status (PASS first)
            status_priority = 0 if variant.status == "PASS" else 1

            # Secondary sort: mass relative to limit
            mass_ratio = float("inf")
            if variant.mass_kg is not None and self.base_spec.mass_limit_kg > 0:
                mass_ratio = variant.mass_kg / self.base_spec.mass_limit_kg

            # Tertiary sort: stress relative to yield
            stress_ratio = float("inf")
            if variant.max_stress_mpa is not None and variant.material:
                # Get yield stress directly from materials data
                yield_stress = materials_data[variant.material]["yield_mpa"]
                stress_ratio = variant.max_stress_mpa / yield_stress

            return (status_priority, mass_ratio, stress_ratio)

        ranked = sorted(variants, key=sort_key)
        logger.info(f"Ranked {len(ranked)} designs")
        return ranked

    def generate_summary(self, variants: List[DesignVariant], output_dir: Path) -> Path:
        """Generate detailed summary with validation metrics."""
        from .mission import load_materials

        # Load materials data once
        materials_data = load_materials()
        summary_data = []

        for variant in variants:
            # Calculate additional metrics
            mass_margin = None
            safety_factor = None

            if variant.mass_kg is not None and self.base_spec.mass_limit_kg > 0:
                mass_margin = 1 - (variant.mass_kg / self.base_spec.mass_limit_kg)

            if variant.max_stress_mpa is not None and variant.material:
                yield_stress = materials_data[variant.material]["yield_mpa"]
                safety_factor = yield_stress / variant.max_stress_mpa

            summary_data.append(
                {
                    "design": variant.design_id,
                    "rail_mm": variant.rail_mm,
                    "deck_mm": variant.deck_mm,
                    "material": variant.material.value,
                    "mass_kg": variant.mass_kg,
                    "mass_margin": mass_margin,
                    "max_stress_MPa": variant.max_stress_mpa,
                    "safety_factor": safety_factor,
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
        """Run the complete multi-design workflow with parallel processing."""
        logger.info("Starting multi-design generation workflow...")

        try:
            # Generate parameter variants
            variants = self.generate_variants()

            # Build all designs in parallel (using threads)
            variants = self.build_designs(output_dir)

            # Run FEA if requested (using processes)
            if run_fea:
                variants = self.run_batch_fea(variants)

            # Rank designs
            variants = self.rank_designs(variants)

            # Generate summary
            summary_file = self.generate_summary(variants, output_dir)

            # Generate reports for each variant
            for variant in variants:
                if variant.step_file and variant.step_file.parent.exists():
                    design_dir = variant.step_file.parent

                    # Load or create physics results
                    physics_results = {}
                    physics_file = design_dir / "physics_check.json"
                    if physics_file.exists():
                        with open(physics_file) as f:
                            physics_results = json.load(f)

                    # Load or create DfAM results
                    dfam_results = {"status": "NOT RUN", "violations": []}
                    dfam_file = design_dir / "manufacturability.json"
                    if dfam_file.exists():
                        with open(dfam_file) as f:
                            dfam_results = json.load(f)

                    # Create mission spec for this variant
                    variant_spec = MissionSpec(
                        bus_u=self.base_spec.bus_u,
                        payload_mass_kg=self.base_spec.payload_mass_kg,
                        orbit_alt_km=self.base_spec.orbit_alt_km,
                        mass_limit_kg=self.base_spec.mass_limit_kg,
                        rail_mm=variant.rail_mm,
                        deck_mm=variant.deck_mm,
                        material=variant.material,
                    )

                    # Generate report
                    from ..reporting.pdf_report import generate_report

                    generate_report(
                        design_dir=design_dir,
                        mission_spec=variant_spec.model_dump(),
                        physics_results=physics_results,
                        dfam_results=dfam_results,
                    )

            logger.info(
                f"Multi-design workflow completed. Summary available at {summary_file}"
            )
            return summary_file

        except Exception as e:
            logger.error(f"Multi-design workflow failed: {str(e)}")
            raise

        finally:
            self._cleanup_executors()
