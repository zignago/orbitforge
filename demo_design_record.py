#!/usr/bin/env python3
"""Demonstration script for OrbitForge Design Record functionality.

This script demonstrates the complete Design Record system for tracking
geometry/analysis pipeline runs.
"""

import json
from pathlib import Path
from orbitforge.generator.mission import MissionSpec, Material
from orbitforge.design_record import DesignRecord, GeometryParams, ArtifactType
from orbitforge.artifact_storage import get_storage
from orbitforge.design_context import design_record_context


def main():
    """Run a complete demonstration of the Design Record system."""
    print("=" * 60)
    print("OrbitForge Design Record System Demonstration")
    print("=" * 60)

    # Create a demo mission specification
    demo_spec = MissionSpec(
        bus_u=3,
        payload_mass_kg=1.2,
        orbit_alt_km=650,
        rail_mm=3.5,
        deck_mm=2.8,
        material=Material.AL_6061_T6,
    )

    demo_spec_raw = {
        "bus_u": 3,
        "payload_mass_kg": 1.2,
        "orbit_alt_km": 650,
        "rail_mm": 3.5,
        "deck_mm": 2.8,
        "material": "Al_6061_T6",
    }

    print(f"\nMission Specification:")
    print(f"  Bus size: {demo_spec.bus_u}U")
    print(f"  Payload mass: {demo_spec.payload_mass_kg} kg")
    print(f"  Orbit altitude: {demo_spec.orbit_alt_km} km")
    print(f"  Rail thickness: {demo_spec.rail_mm} mm")
    print(f"  Deck thickness: {demo_spec.deck_mm} mm")
    print(f"  Material: {demo_spec.material}")

    # Set up output directory
    output_dir = Path("outputs/design_record_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory: {output_dir}")

    # CLI overrides
    cli_overrides = {"rail_override": 3.5, "material_override": "Al_6061_T6"}

    print("\nInitializing Design Record system...")

    # Demonstrate the design record context manager
    with design_record_context(
        demo_spec, demo_spec_raw, output_dir, cli_overrides
    ) as record:
        print(f"Created Design Record: {record.design_id}")
        print(f"Timestamp: {record.timestamp}")
        print(f"Git commit: {record.git_commit}")

        # Simulate design generation
        design_dir = output_dir / f"design_{record.design_id[:8]}"
        design_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nSimulating design generation in: {design_dir}")

        # Create simulated output files
        artifacts_to_create = [
            ("frame.step", "# Simulated STEP file content\nSTEP-File"),
            ("frame.stl", "# Simulated STL file content\nsolid frame"),
            ("mass_budget.csv", "component,mass_kg\nrails,1.2\ndeck,0.8\ntotal,2.0"),
            (
                "physics_check.json",
                json.dumps(
                    {
                        "max_stress_mpa": 142.3,
                        "thermal_stress_mpa": 15.2,
                        "status": "PASS",
                        "factor_of_safety": 1.94,
                    },
                    indent=2,
                ),
            ),
            (
                "manufacturability.json",
                json.dumps(
                    {"status": "PASS", "violations": [], "score": 0.85}, indent=2
                ),
            ),
            ("report.pdf", "# Simulated PDF report content"),
        ]

        for filename, content in artifacts_to_create:
            file_path = design_dir / filename
            file_path.write_text(content)
            print(f"  Created: {filename}")

        # Add artifacts to design record
        print("\nAdding artifacts to Design Record...")

        # Add each artifact with proper types
        step_file = design_dir / "frame.step"
        record.add_artifact(
            ArtifactType.STEP, step_file, description="Generated STEP CAD file"
        )

        stl_file = design_dir / "frame.stl"
        record.add_artifact(
            ArtifactType.STL, stl_file, description="Generated STL mesh for 3D printing"
        )

        mass_file = design_dir / "mass_budget.csv"
        record.add_artifact(
            ArtifactType.MASS_BUDGET, mass_file, description="Component mass breakdown"
        )

        physics_file = design_dir / "physics_check.json"
        record.add_artifact(
            ArtifactType.PHYSICS_CHECK,
            physics_file,
            description="FEA validation results",
        )

        dfam_file = design_dir / "manufacturability.json"
        record.add_artifact(
            ArtifactType.DFAM_CHECK,
            dfam_file,
            description="Design for Additive Manufacturing analysis",
        )

        report_file = design_dir / "report.pdf"
        record.add_artifact(
            ArtifactType.REPORT, report_file, description="Comprehensive design report"
        )

        print(f"  Added {len(record.artifacts)} artifacts")

        # Update analysis results
        print("\nUpdating analysis results...")
        record.update_analysis_results(
            max_stress_mpa=142.3,
            thermal_stress_mpa=15.2,
            factor_of_safety=1.94,
            mass_kg=2.0,
            manufacturability_score=0.85,
        )

        print("  ✓ Structural analysis results")
        print("  ✓ Thermal analysis results")
        print("  ✓ Mass budget")
        print("  ✓ Manufacturability score")

        # Add some warnings
        record.add_warning("Rail thickness near minimum recommended value")
        record.add_warning("Consider thermal expansion analysis for high-orbit mission")

        print(f"  Added {len(record.warnings)} warnings")

    print("\n" + "=" * 60)
    print("Design Record System Demonstration Complete")
    print("=" * 60)

    # Show final results
    records_dir = output_dir / "records"
    record_files = list(records_dir.glob("*design_record.json"))

    if record_files:
        record_file = record_files[0]
        saved_record = DesignRecord.load_from_file(record_file)

        print(f"\nFinal Design Record Summary:")
        print(f"  Design ID: {saved_record.design_id}")
        print(f"  Status: {saved_record.status}")
        print(f"  Execution time: {saved_record.execution_time_seconds:.2f} seconds")
        print(f"  Artifacts: {len(saved_record.artifacts)}")
        print(f"  Mass: {saved_record.analysis_results.mass_kg} kg")
        print(f"  Max stress: {saved_record.analysis_results.max_stress_mpa} MPa")
        print(f"  Warnings: {len(saved_record.warnings)}")

        print(f"\nArtifacts generated:")
        for artifact in saved_record.artifacts:
            print(f"  - {artifact.type}: {Path(artifact.path).name}")
            print(f"    Size: {artifact.size_bytes} bytes")
            print(f"    Hash: {artifact.hash_sha256[:16]}...")

        print(f"\nDesign Record saved to: {record_file}")

    # Demonstrate artifact storage
    print("\n" + "-" * 40)
    print("Artifact Storage Demonstration")
    print("-" * 40)

    storage = get_storage()
    print(f"Storage URI: {storage.base_uri}")

    if record_files:
        design_id = saved_record.design_id
        artifacts = storage.list_artifacts(design_id)
        print(f"\nArtifacts in storage for design {design_id[:8]}:")
        for name, uri in artifacts.items():
            print(f"  - {name}: {uri}")

    print("\n" + "=" * 60)
    print("CLI Usage Examples:")
    print("=" * 60)
    print("To run with design record logging:")
    print("  orbitforge run missions/demo_3u.json -o outputs/demo")
    print()
    print("To list recent designs:")
    print("  orbitforge list-designs")
    print()
    print("To fetch artifacts for a design:")
    print(
        f"  orbitforge fetch {saved_record.design_id[:8] if record_files else '<design-id>'}"
    )
    print()
    print("Design record will be saved to:")
    print("  outputs/demo/records/<design-id>_design_record.json")
    print()
    print("All sections will be populated with:")
    print("  ✓ Mission specification (verbatim)")
    print("  ✓ Geometry parameters (for replay)")
    print("  ✓ Generated artifacts (with URIs)")
    print("  ✓ Analysis results (FEA, thermal, DfAM)")
    print("  ✓ Git commit hash")
    print("  ✓ Execution metadata")
    print("=" * 60)


if __name__ == "__main__":
    main()
