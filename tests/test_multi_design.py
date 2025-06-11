"""
Tests for multi-design generation functionality (v0.1.2).

This module tests parameter jittering, multi-design generation,
batch FEA processing, and summary generation capabilities.
"""

import json
import tempfile
from pathlib import Path
import pytest
import time
from concurrent.futures import ThreadPoolExecutor

from orbitforge.generator.mission import MissionSpec, Material
from orbitforge.generator.multi_design import (
    ParameterJitter,
    MultiDesignGenerator,
    DesignVariant,
    DesignCache,
    _design_cache,
)
from orbitforge.config.design_config import config, DesignConfig


class TestDesignCache:
    """Test the design result caching system."""

    def test_cache_storage_and_retrieval(self):
        """Test basic cache operations."""
        cache = DesignCache(max_size_mb=100)

        # Store test data
        test_data = {"test": "data"}
        cache.put(test_data, param1="value1", param2=42)

        # Retrieve test data
        result = cache.get(param1="value1", param2=42)
        assert result == test_data

        # Test cache miss
        miss = cache.get(param1="wrong", param2=42)
        assert miss is None

    def test_cache_expiry(self):
        """Test that cached items expire correctly."""
        cache = DesignCache(max_size_mb=100)
        test_data = {"test": "data"}

        # Override cache expiry for testing
        original_expiry = config.resources.CACHE_EXPIRY_HOURS
        config.resources.CACHE_EXPIRY_HOURS = 0.0001  # Very short expiry

        try:
            cache.put(test_data, key="test")
            time.sleep(0.5)  # Wait for expiry
            result = cache.get(key="test")
            assert result is None
        finally:
            config.resources.CACHE_EXPIRY_HOURS = original_expiry

    def test_cache_size_limit(self):
        """Test that cache respects size limits."""
        cache = DesignCache(max_size_mb=1)

        # Add many items to trigger cleanup
        for i in range(1000):
            cache.put({"large": "x" * 1000}, key=f"test_{i}")

        # Check that cache size is managed
        assert cache._estimate_size_mb() <= 1


class TestParameterJitter:
    """Test parameter jittering with bounded and deterministic randomness."""

    def test_rail_thickness_jittering_bounds(self):
        """Test that rail thickness jittering stays within reasonable bounds."""
        jitter = ParameterJitter(seed=42)
        base_value = 3.0

        # Test multiple jitters to ensure they're bounded
        for _ in range(100):
            result = jitter.jitter_rail_thickness(base_value)
            assert (
                config.params.MIN_RAIL_THICKNESS
                <= result
                <= config.params.MAX_RAIL_THICKNESS
            )
            assert abs(result - base_value) <= config.params.RAIL_VARIATION_MM

    def test_deck_thickness_jittering_bounds(self):
        """Test that deck thickness jittering stays within reasonable bounds."""
        jitter = ParameterJitter(seed=42)
        base_value = 2.5

        for _ in range(100):
            result = jitter.jitter_deck_thickness(base_value)
            assert (
                config.params.MIN_DECK_THICKNESS
                <= result
                <= config.params.MAX_DECK_THICKNESS
            )
            assert abs(result - base_value) <= config.params.DECK_VARIATION_MM

    def test_material_jittering_probability(self):
        """Test that material changes happen with expected probability."""
        jitter = ParameterJitter(seed=42)
        base_material = Material.AL_6061_T6

        # Test with 0% probability - should never change
        for _ in range(50):
            result = jitter.jitter_material(base_material, change_probability=0.0)
            assert result == base_material

        # Test with 100% probability - should always change (if alternatives exist)
        changes = 0
        for _ in range(50):
            result = jitter.jitter_material(base_material, change_probability=1.0)
            if result != base_material:
                changes += 1
        assert changes > 40

    def test_unique_variant_generation(self):
        """Test that generate_unique_variant produces unique combinations."""
        jitter = ParameterJitter(seed=42)
        base_rail = 3.0
        base_deck = 2.5
        base_material = Material.AL_6061_T6

        # Generate multiple variants
        variants = set()
        for _ in range(5):
            rail, deck, material = jitter.generate_unique_variant(
                base_rail, base_deck, base_material
            )
            variant_key = f"{rail:.3f}|{deck:.3f}|{material.value}"
            variants.add(variant_key)

        # Check that all variants are unique
        assert len(variants) == 5

    def test_reproducibility_with_seed(self):
        """Test that jittering is reproducible with the same seed."""
        base_rail = 3.0
        base_deck = 2.5
        base_material = Material.AL_6061_T6

        # Generate sequence with seed 42
        jitter1 = ParameterJitter(seed=42)
        seq1_rail = [jitter1.jitter_rail_thickness(base_rail) for _ in range(10)]
        seq1_deck = [jitter1.jitter_deck_thickness(base_deck) for _ in range(10)]
        seq1_mat = [jitter1.jitter_material(base_material) for _ in range(10)]

        # Generate same sequence with same seed
        jitter2 = ParameterJitter(seed=42)
        seq2_rail = [jitter2.jitter_rail_thickness(base_rail) for _ in range(10)]
        seq2_deck = [jitter2.jitter_deck_thickness(base_deck) for _ in range(10)]
        seq2_mat = [jitter2.jitter_material(base_material) for _ in range(10)]

        # Should be identical
        assert seq1_rail == seq2_rail
        assert seq1_deck == seq2_deck
        assert seq1_mat == seq2_mat


class TestMultiDesignGenerator:
    """Test multi-design generation and workflow."""

    @pytest.fixture
    def base_spec(self):
        """Create a base mission specification for testing."""
        return MissionSpec(
            bus_u=3,
            payload_mass_kg=1.0,
            orbit_alt_km=550,
            mass_limit_kg=4.0,
            rail_mm=3.0,
            deck_mm=2.5,
            material=Material.AL_6061_T6,
        )

    def test_variant_generation_count(self, base_spec):
        """Test that correct number of variants is generated."""
        generator = MultiDesignGenerator(base_spec, num_variants=5, seed=42)
        variants = generator.generate_variants()

        assert len(variants) == 5
        assert all(isinstance(v, DesignVariant) for v in variants)
        assert all(v.design_id.startswith("design_") for v in variants)

    def test_variant_generation_clamps_bounds(self, base_spec):
        """Test that variant count is clamped to 1-10 range."""
        # Test lower bound
        generator = MultiDesignGenerator(base_spec, num_variants=-5, seed=42)
        assert generator.num_variants == 1

        # Test upper bound
        generator = MultiDesignGenerator(base_spec, num_variants=20, seed=42)
        assert generator.num_variants == 10

    def test_variant_parameters_differ(self, base_spec):
        """Test that generated variants have different parameters."""
        generator = MultiDesignGenerator(base_spec, num_variants=5, seed=42)
        variants = generator.generate_variants()

        # Check that parameters differ between variants
        rail_values = [v.rail_mm for v in variants]
        deck_values = [v.deck_mm for v in variants]
        material_values = [v.material for v in variants]

        # Should have some variation
        assert len(set(rail_values)) > 1
        assert len(set(deck_values)) > 1
        assert len(set(material_values)) >= 1

    def test_parallel_build_and_fea(self, base_spec):
        """Test parallel processing of builds and FEA."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create generator with multiple variants
            generator = MultiDesignGenerator(base_spec, num_variants=3, seed=42)

            # Time parallel execution
            start_time = time.time()
            variants = generator.build_designs(output_dir)
            build_time = time.time() - start_time

            # Verify parallel execution
            assert len(variants) == 3
            assert all(
                v.step_file is not None for v in variants if v.status != "BUILD_FAILED"
            )

            # Run FEA in parallel
            start_time = time.time()
            validated = generator.run_batch_fea(variants)
            fea_time = time.time() - start_time

            # Verify FEA results
            assert len(validated) == 3
            assert all(v.status is not None for v in validated)

    def test_caching_effectiveness(self, base_spec):
        """Test that caching reduces computation time."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # First run without cache
            _design_cache._cache.clear()
            generator1 = MultiDesignGenerator(base_spec, num_variants=2, seed=42)
            start_time = time.time()
            variants1 = generator1.build_designs(output_dir)
            first_run_time = time.time() - start_time

            # Second run with cache
            generator2 = MultiDesignGenerator(base_spec, num_variants=2, seed=42)
            start_time = time.time()
            variants2 = generator2.build_designs(output_dir)
            second_run_time = time.time() - start_time

            # Second run should be significantly faster due to caching
            # Allow for some timing variance but ensure it's faster
            assert (
                second_run_time < first_run_time * 0.9
            )  # Second run should be at least 10% faster

            # Results should be identical
            assert len(variants1) == len(variants2)
            for v1, v2 in zip(variants1, variants2):
                assert (
                    abs(v1.rail_mm - v2.rail_mm) < 0.001
                )  # Allow small floating point differences
                assert abs(v1.deck_mm - v2.deck_mm) < 0.001
                assert v1.material == v2.material

    def test_design_ranking_with_metrics(self, base_spec):
        """Test that designs are ranked correctly using multiple metrics."""
        # Create mock variants with different metrics
        variants = [
            DesignVariant(
                "design_001",
                3.0,
                2.5,
                Material.AL_6061_T6,
                mass_kg=2.5,
                max_stress_mpa=150.0,
                status="PASS",
            ),
            DesignVariant(
                "design_002",
                3.1,
                2.4,
                Material.AL_6061_T6,
                mass_kg=2.2,
                max_stress_mpa=180.0,
                status="PASS",
            ),
            DesignVariant(
                "design_003",
                3.2,
                2.6,
                Material.AL_6061_T6,
                mass_kg=2.8,
                max_stress_mpa=120.0,
                status="PASS",
            ),
            DesignVariant(
                "design_004",
                2.9,
                2.3,
                Material.AL_6061_T6,
                mass_kg=3.0,
                max_stress_mpa=200.0,
                status="FAIL",
            ),
        ]

        generator = MultiDesignGenerator(base_spec, num_variants=4, seed=42)
        ranked = generator.rank_designs(variants)

        # Verify ranking order
        assert ranked[0].status == "PASS"
        assert ranked[0].mass_kg == 2.2  # Lowest mass among PASS
        assert ranked[-1].status == "FAIL"

    def test_summary_generation_metrics(self, base_spec):
        """Test that summary includes all validation metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create mock variants with metrics
            variants = [
                DesignVariant(
                    "design_001",
                    3.0,
                    2.5,
                    Material.AL_6061_T6,
                    mass_kg=2.5,
                    max_stress_mpa=150.0,
                    status="PASS",
                ),
                DesignVariant(
                    "design_002",
                    3.1,
                    2.4,
                    Material.AL_6061_T6,
                    mass_kg=2.8,
                    max_stress_mpa=180.0,
                    status="PASS",
                ),
            ]

            generator = MultiDesignGenerator(base_spec, num_variants=2, seed=42)
            summary_file = generator.generate_summary(variants, output_dir)

            # Check summary content
            with open(summary_file) as f:
                summary_data = json.load(f)

            assert len(summary_data) == 2
            for design in summary_data:
                assert "mass_margin" in design
                assert "safety_factor" in design
                assert isinstance(design["mass_margin"], (float, type(None)))
                assert isinstance(design["safety_factor"], (float, type(None)))


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def base_spec(self):
        """Create a base mission specification for testing."""
        return MissionSpec(
            bus_u=3,
            payload_mass_kg=1.0,
            orbit_alt_km=550,
            mass_limit_kg=4.0,
            rail_mm=3.0,
            deck_mm=2.5,
            material=Material.AL_6061_T6,
        )

    def test_all_variants_fail_scenario(self):
        """Test handling when all variants fail validation."""
        base_spec = MissionSpec(
            bus_u=6,
            payload_mass_kg=1.0,
            orbit_alt_km=550,
            mass_limit_kg=0.5,  # Extremely low mass limit
            rail_mm=0.5,  # Very thin rails
            deck_mm=0.3,  # Very thin decks
            material=Material.AL_6061_T6,
        )

        generator = MultiDesignGenerator(base_spec, num_variants=3, seed=42)
        variants = generator.generate_variants()

        # Should still generate variants even if they might fail
        assert len(variants) == 3
        assert all(v.design_id.startswith("design_") for v in variants)

    def test_parallel_processing_error_handling(self, base_spec):
        """Test that parallel processing handles errors gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create a generator that will cause some failures
            generator = MultiDesignGenerator(base_spec, num_variants=5, seed=42)

            try:
                # Create a subdirectory that will be read-only
                test_dir = output_dir / "test_dir"
                test_dir.mkdir()
                test_dir.chmod(0o444)  # Read-only

                # Should handle failures without crashing
                variants = generator.build_designs(test_dir)
                assert len(variants) > 0
                assert any(v.status == "BUILD_FAILED" for v in variants)

            finally:
                # Restore permissions and clean up
                for path in output_dir.rglob("*"):
                    try:
                        path.chmod(0o777)
                    except Exception:
                        pass
                generator._cleanup_executors()

    def test_resource_cleanup(self, base_spec):
        """Test that resources are properly cleaned up."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            generator = MultiDesignGenerator(base_spec, num_variants=3, seed=42)

            # Run workflow and check executor shutdown
            generator.run_complete_workflow(output_dir, run_fea=True)

            # Verify executors are shut down
            assert generator._executor is None
            assert generator._process_executor is None

    def test_cache_cleanup(self):
        """Test that cache cleanup works correctly."""
        cache = DesignCache(max_size_mb=1)

        # Fill cache beyond threshold
        large_data = {"data": "x" * 1000000}  # ~1MB of data
        for i in range(10):
            cache.put(large_data, key=f"test_{i}")

        # Force cleanup
        cache._cleanup_old_entries()

        # Verify cache size is reduced
        assert cache._estimate_size_mb() <= 1


class TestConfiguration:
    """Test configuration management."""

    def test_config_serialization(self):
        """Test configuration serialization and deserialization."""
        original_config = DesignConfig()
        config_dict = original_config.to_dict()

        # Modify some values
        config_dict["parameter_ranges"]["RAIL_VARIATION_MM"] = 0.8
        config_dict["resource_limits"]["MAX_PARALLEL_FEA"] = 8

        # Create new config from dict
        new_config = DesignConfig.from_dict(config_dict)

        assert new_config.params.RAIL_VARIATION_MM == 0.8
        assert new_config.resources.MAX_PARALLEL_FEA == 8

    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid parallel FEA count
        config.resources.MAX_PARALLEL_FEA = -1
        with pytest.raises(ValueError):
            config.validate()

        # Reset to valid value
        config.resources.MAX_PARALLEL_FEA = 4

        # Test invalid cache size
        config.resources.MAX_CACHE_SIZE_MB = 0
        with pytest.raises(ValueError):
            config.validate()

        # Reset to valid value
        config.resources.MAX_CACHE_SIZE_MB = 1000
