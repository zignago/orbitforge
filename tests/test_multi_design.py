"""
Tests for multi-design generation functionality (v0.1.2).

This module tests parameter jittering, multi-design generation,
batch FEA processing, and summary generation capabilities.
"""

import json
import tempfile
from pathlib import Path
import pytest

from orbitforge.generator.mission import MissionSpec, Material
from orbitforge.generator.multi_design import (
    ParameterJitter,
    MultiDesignGenerator,
    DesignVariant,
)


class TestParameterJitter:
    """Test parameter jittering with bounded and deterministic randomness."""

    def test_rail_thickness_jittering_bounds(self):
        """Test that rail thickness jittering stays within reasonable bounds."""
        jitter = ParameterJitter(seed=42)
        base_value = 3.0

        # Test multiple jitters to ensure they're bounded
        for _ in range(100):
            result = jitter.jitter_rail_thickness(base_value, variation_mm=0.5)
            # Should be within ±0.5mm of base value and at least 1.0mm
            assert result >= 1.0
            assert 2.5 <= result <= 3.5

    def test_deck_thickness_jittering_bounds(self):
        """Test that deck thickness jittering stays within reasonable bounds."""
        jitter = ParameterJitter(seed=42)
        base_value = 2.5

        for _ in range(100):
            result = jitter.jitter_deck_thickness(base_value, variation_mm=0.3)
            # Should be within ±0.3mm of base value and at least 1.0mm
            assert result >= 1.0
            assert 2.2 <= result <= 2.8

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
        # Should have significant number of changes
        assert changes > 40

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

        # Check that at least some parameters differ between variants
        rail_values = [v.rail_mm for v in variants]
        deck_values = [v.deck_mm for v in variants]

        # Should have some variation (not all identical)
        assert len(set(rail_values)) > 1 or len(set(deck_values)) > 1

    def test_build_designs_creates_directories(self, base_spec):
        """Test that build_designs creates proper directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            generator = MultiDesignGenerator(base_spec, num_variants=3, seed=42)
            variants = generator.build_designs(output_dir)

            # Check that directories were created
            for variant in variants:
                variant_dir = output_dir / variant.design_id
                assert variant_dir.exists()
                assert variant_dir.is_dir()

                # Check for expected files (if build succeeded)
                if variant.status != "BUILD_FAILED":
                    assert (variant_dir / "frame.step").exists()
                    assert (variant_dir / "frame.stl").exists()
                    assert (variant_dir / "mass_budget.csv").exists()

    def test_design_ranking(self, base_spec):
        """Test that designs are ranked correctly by status and mass."""
        # Create mock variants with different statuses and masses
        variants = [
            DesignVariant(
                "design_001", 3.0, 2.5, Material.AL_6061_T6, mass_kg=2.5, status="FAIL"
            ),
            DesignVariant(
                "design_002", 3.1, 2.4, Material.AL_6061_T6, mass_kg=2.2, status="PASS"
            ),
            DesignVariant(
                "design_003", 3.2, 2.6, Material.AL_6061_T6, mass_kg=2.8, status="PASS"
            ),
            DesignVariant(
                "design_004",
                2.9,
                2.3,
                Material.AL_6061_T6,
                mass_kg=3.0,
                status="FEA_FAILED",
            ),
        ]

        generator = MultiDesignGenerator(base_spec, num_variants=4, seed=42)
        ranked = generator.rank_designs(variants)

        # PASS should come first, then sorted by mass
        assert ranked[0].status == "PASS" and ranked[0].mass_kg == 2.2
        assert ranked[1].status == "PASS" and ranked[1].mass_kg == 2.8
        # Failed designs should come after
        assert ranked[2].status in ["FAIL", "FEA_FAILED"]
        assert ranked[3].status in ["FAIL", "FEA_FAILED"]

    def test_summary_generation_format(self, base_spec):
        """Test that summary.json is generated with correct format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create mock variants
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
                    Material.TI_6AL_4V,
                    mass_kg=2.8,
                    max_stress_mpa=180.0,
                    status="PASS",
                ),
            ]

            generator = MultiDesignGenerator(base_spec, num_variants=2, seed=42)
            summary_file = generator.generate_summary(variants, output_dir)

            # Check that file was created
            assert summary_file.exists()

            # Check format
            with open(summary_file) as f:
                summary_data = json.load(f)

            assert len(summary_data) == 2

            # Check required fields
            for design in summary_data:
                required_fields = [
                    "design",
                    "rail_mm",
                    "deck_mm",
                    "material",
                    "mass_kg",
                    "max_stress_MPa",
                    "status",
                ]
                for field in required_fields:
                    assert field in design

                # Check types
                assert isinstance(design["design"], str)
                assert isinstance(design["rail_mm"], float)
                assert isinstance(design["deck_mm"], float)
                assert isinstance(design["material"], str)
                assert design["material"] in ["Al_6061_T6", "Ti_6Al_4V"]


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_all_variants_fail_scenario(self):
        """Test handling when all variants fail validation."""
        # This would be tested with a mission spec that causes all designs to fail
        # For example, very restrictive mass limits or extreme parameters
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

    def test_single_variant_mode(self):
        """Test that single variant (multi=1) works correctly."""
        base_spec = MissionSpec(
            bus_u=3,
            payload_mass_kg=1.0,
            orbit_alt_km=550,
            mass_limit_kg=4.0,
            rail_mm=3.0,
            deck_mm=2.5,
            material=Material.AL_6061_T6,
        )

        generator = MultiDesignGenerator(base_spec, num_variants=1, seed=42)
        variants = generator.generate_variants()

        assert len(variants) == 1
        assert variants[0].design_id == "design_001"


class TestIntegration:
    """Integration tests for complete workflow."""

    def test_complete_workflow_without_fea(self):
        """Test complete workflow without FEA (geometry generation only)."""
        base_spec = MissionSpec(
            bus_u=3,
            payload_mass_kg=1.0,
            orbit_alt_km=550,
            mass_limit_kg=4.0,
            rail_mm=3.0,
            deck_mm=2.5,
            material=Material.AL_6061_T6,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            generator = MultiDesignGenerator(base_spec, num_variants=3, seed=42)
            summary_file = generator.run_complete_workflow(output_dir, run_fea=False)

            # Check that summary was created
            assert summary_file.exists()

            # Check that design directories exist
            for i in range(1, 4):
                design_dir = output_dir / f"design_{i:03d}"
                assert design_dir.exists()

    def test_deterministic_output_with_seed(self):
        """Test that using the same seed produces identical results."""
        base_spec = MissionSpec(
            bus_u=3,
            payload_mass_kg=1.0,
            orbit_alt_km=550,
            mass_limit_kg=4.0,
            rail_mm=3.0,
            deck_mm=2.5,
            material=Material.AL_6061_T6,
        )

        # Generate first set
        gen1 = MultiDesignGenerator(base_spec, num_variants=3, seed=42)
        variants1 = gen1.generate_variants()

        # Generate second set with same seed
        gen2 = MultiDesignGenerator(base_spec, num_variants=3, seed=42)
        variants2 = gen2.generate_variants()

        # Should be identical
        for v1, v2 in zip(variants1, variants2):
            assert v1.rail_mm == v2.rail_mm
            assert v1.deck_mm == v2.deck_mm
            assert v1.material == v2.material
