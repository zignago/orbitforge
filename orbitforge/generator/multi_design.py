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
from ..dfam import DfamPostProcessor
from ..config.design_config import config


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
    dfam_status: Optional[str] = None  # Added for DfAM v0.1.5
    dfam_error_count: Optional[int] = None
    dfam_warning_count: Optional[int] = None
    _timestamp: Optional[float] = None  # Added for cache expiry

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        # Convert Path to string for JSON serialization
        if self.step_file:
            result["step_file"] = str(self.step_file)
        if "_timestamp" in result:
            del result["_timestamp"]  # Don't include in serialization
        return result


class DesignCache:
    """Cache for design results to avoid redundant computations."""

    def __init__(self, max_size_mb: int = config.resources.MAX_CACHE_SIZE_MB):
        self._cache: Dict[str, Tuple[float, Any]] = {}  # (timestamp, data)
        self._max_size_mb = max_size_mb
        self._cleanup_threshold = 0.9  # Cleanup when 90% full
        self._seed_cache: Dict[int, List[DesignVariant]] = {}  # Cache by seed

    def _make_key(self, **kwargs) -> str:
        """Create a cache key from parameters."""
        # Round float values to 6 decimal places for better precision
        formatted_kwargs = {}
        for k, v in sorted(kwargs.items()):
            if isinstance(v, float):
                formatted_kwargs[k] = f"{v:.6f}"
            elif isinstance(v, Material):
                formatted_kwargs[k] = v.value
            else:
                formatted_kwargs[k] = str(v)
        return "|".join(f"{k}={v}" for k, v in sorted(formatted_kwargs.items()))

    def _estimate_size_mb(self) -> float:
        """Estimate current cache size in MB."""
        import sys

        total_size = sys.getsizeof(self._cache) + sys.getsizeof(self._seed_cache)
        return total_size / (1024 * 1024)

    def _cleanup_old_entries(self):
        """Remove old entries if cache is too large."""
        if self._estimate_size_mb() < self._max_size_mb * self._cleanup_threshold:
            return

        current_time = time.time()
        expiry_time = current_time - (config.resources.CACHE_EXPIRY_HOURS * 3600)

        # Clean parameter cache
        self._cache = {
            k: (t, d) for k, (t, d) in self._cache.items() if t > expiry_time
        }

        # Clean seed cache older than 1 hour
        seed_expiry = current_time - 3600  # 1 hour
        self._seed_cache = {
            k: v
            for k, v in self._seed_cache.items()
            if v and getattr(v[0], "_timestamp", current_time) > seed_expiry
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

    def get_by_seed(self, seed: int) -> Optional[List[DesignVariant]]:
        """Get cached variants for a specific seed."""
        return self._seed_cache.get(seed)

    def put_by_seed(self, seed: int, variants: List[DesignVariant]):
        """Cache variants for a specific seed."""
        # Add timestamp to variants for expiry
        current_time = time.time()
        for variant in variants:
            variant._timestamp = current_time
        self._seed_cache[seed] = variants


# Global cache instance
_design_cache = DesignCache()


class ParameterJitter:
    """Handles bounded parameter jittering with reproducible randomness."""

    def __init__(self, seed: int = 42):
        """Initialize with a random seed for reproducibility."""
        self.seed = seed
        self.rng = random.Random(seed)
        self._used_combinations: Set[str] = set()
        self._jitter_sequence = []  # Store sequence of jitters
        self._sequence_index = 0
        logger.info(f"Initialized parameter jitter with seed: {seed}")

    def reset(self):
        """Reset the random number generator to initial state."""
        self.rng = random.Random(self.seed)
        self._used_combinations.clear()
        self._jitter_sequence = []
        self._sequence_index = 0

    def _get_next_jitter(self, min_val: float, max_val: float) -> float:
        """Get next jitter value, either from sequence or generate new."""
        if self._sequence_index < len(self._jitter_sequence):
            # Replay from sequence
            jitter = self._jitter_sequence[self._sequence_index]
            self._sequence_index += 1
            return jitter

        # Generate new jitter
        jitter = self.rng.uniform(min_val, max_val)
        self._jitter_sequence.append(jitter)
        self._sequence_index += 1
        return jitter

    def _is_unique_combination(
        self, rail: float, deck: float, material: Material
    ) -> bool:
        """Check if parameter combination is unique."""
        key = f"{rail:.6f}|{deck:.6f}|{material.value}"  # Increased precision
        if key in self._used_combinations:
            return False
        self._used_combinations.add(key)
        return True

    def jitter_rail_thickness(
        self, base_value: float, variation_mm: float = config.params.RAIL_VARIATION_MM
    ) -> float:
        """Apply random jitter to rail thickness within bounds."""
        jitter = self._get_next_jitter(-variation_mm, variation_mm)
        result = round(
            max(
                config.params.MIN_RAIL_THICKNESS,
                min(config.params.MAX_RAIL_THICKNESS, base_value + jitter),
            ),
            6,  # Increased precision
        )
        logger.debug(
            f"Rail thickness: {base_value:.3f} → {result:.3f} mm (jitter: {jitter:+.3f})"
        )
        return result

    def jitter_deck_thickness(
        self, base_value: float, variation_mm: float = config.params.DECK_VARIATION_MM
    ) -> float:
        """Apply random jitter to deck thickness within bounds."""
        jitter = self._get_next_jitter(-variation_mm, variation_mm)
        result = round(
            max(
                config.params.MIN_DECK_THICKNESS,
                min(config.params.MAX_DECK_THICKNESS, base_value + jitter),
            ),
            6,  # Increased precision
        )
        logger.debug(
            f"Deck thickness: {base_value:.3f} → {result:.3f} mm (jitter: {jitter:+.3f})"
        )
        return result

    def jitter_material(
        self,
        base_material: Material,
        change_probability: float = config.params.MATERIAL_CHANGE_PROBABILITY,
    ) -> Material:
        """Occasionally change material with given probability."""
        change = self._get_next_jitter(0, 1) < change_probability
        if change:
            # Switch between available materials
            materials = sorted(list(Material))  # Sort for consistency
            other_materials = [m for m in materials if m != base_material]
            if other_materials:
                # Use deterministic selection based on sequence
                index = int(self._get_next_jitter(0, len(other_materials) - 0.001))
                result = other_materials[index]
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
        self.seed = seed
        self.jitter = ParameterJitter(seed)
        self.variants: List[DesignVariant] = []
        self._executor = None
        self._process_executor = None

        logger.info(
            f"Initialized multi-design generator for {self.num_variants} variants (seed: {seed})"
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
        # Check seed cache first
        cached_variants = _design_cache.get_by_seed(self.seed)
        if cached_variants and len(cached_variants) == self.num_variants:
            logger.info(f"Using cached variants for seed {self.seed}")
            self.variants = cached_variants
            return cached_variants

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
                design_id=design_id,
                rail_mm=rail_mm,
                deck_mm=deck_mm,
                material=material,
                _timestamp=time.time(),  # Add timestamp when creating
            )
            variants.append(variant)

            logger.info(
                f"Generated variant {design_id}: rail={rail_mm:.6f}mm, deck={deck_mm:.6f}mm, material={material.value}"
            )

        self.variants = variants
        # Cache variants by seed
        _design_cache.put_by_seed(self.seed, variants)
        return variants

    def _build_single_design(
        self, variant: DesignVariant, output_dir: Path
    ) -> DesignVariant:
        """Build a single design variant."""
        try:
            # Check cache first with more precise parameters
            cached_result = _design_cache.get(
                rail_mm=variant.rail_mm,
                deck_mm=variant.deck_mm,
                material=variant.material,
                design_id=variant.design_id,  # Include design_id for better uniqueness
                seed=self.seed,  # Include seed for reproducibility
            )
            if cached_result:
                logger.info(f"Using cached result for {variant.design_id}")
                # Preserve the original timestamp
                cached_result._timestamp = variant._timestamp
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

            logger.info(f"✓ Built {variant.design_id} (mass: {variant.mass_kg:.6f} kg)")

            # Cache the result with more precise parameters
            _design_cache.put(
                variant,
                rail_mm=variant.rail_mm,
                deck_mm=variant.deck_mm,
                material=variant.material,
                design_id=variant.design_id,
                seed=self.seed,
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
            # Build designs sequentially to maintain order
            built_variants = []
            for variant in self.variants:
                try:
                    built_variant = self._build_single_design(variant, output_dir)
                    built_variants.append(built_variant)
                except Exception as e:
                    logger.error(f"Build failed for {variant.design_id}: {str(e)}")
                    variant.status = "BUILD_FAILED"
                    built_variants.append(variant)

            # Cache the built variants
            _design_cache.put_by_seed(self.seed, built_variants)
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

    def run_batch_dfam(
        self, variants: List[DesignVariant], output_dir: Path
    ) -> List[DesignVariant]:
        """Run DfAM post-processing on all variants."""
        logger.info("Running batch DfAM post-processing...")

        processor = DfamPostProcessor()

        for variant in variants:
            if variant.step_file and variant.step_file.parent.exists():
                design_dir = variant.step_file.parent

                try:
                    dfam_result = processor.process_design(
                        design_dir, variant.step_file, base_name="frame"
                    )

                    # Update variant with DfAM results
                    variant.dfam_status = dfam_result["dfam_status"]
                    variant.dfam_error_count = dfam_result.get("error_count", 0)
                    variant.dfam_warning_count = dfam_result.get("warning_count", 0)

                    logger.info(
                        f"DfAM processed {variant.design_id}: {variant.dfam_status}"
                    )

                except Exception as e:
                    logger.error(f"DfAM processing failed for {variant.design_id}: {e}")
                    variant.dfam_status = "ERROR"
                    variant.dfam_error_count = 1
                    variant.dfam_warning_count = 0
            else:
                logger.warning(f"No STEP file for DfAM processing: {variant.design_id}")
                variant.dfam_status = "NOT_RUN"
                variant.dfam_error_count = 0
                variant.dfam_warning_count = 0

        return variants

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
                    "dfam_status": variant.dfam_status or "NOT_RUN",
                    "dfam_error_count": variant.dfam_error_count or 0,
                    "dfam_warning_count": variant.dfam_warning_count or 0,
                }
            )

        summary_file = output_dir / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary_data, f, indent=2)

        logger.info(f"Generated summary at {summary_file}")
        return summary_file

    def run_complete_workflow(
        self, output_dir: Path, run_fea: bool = True, run_dfam: bool = False
    ) -> Path:
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

            # Run DfAM post-processing if requested
            if run_dfam:
                variants = self.run_batch_dfam(variants, output_dir)

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
