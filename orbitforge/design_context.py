"""Design Record context manager for automatic logging.

This module provides a context manager that wraps the entire run_mission()
command to automatically build, populate, and save design records.
"""

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Any, Optional, Generator
from loguru import logger

from .design_record import DesignRecord, GeometryParams, DesignStatus, ArtifactType
from .artifact_storage import get_storage
from .generator.mission import MissionSpec


@contextmanager
def design_record_context(
    mission_spec: MissionSpec,
    mission_spec_raw: Dict[str, Any],
    output_dir: Path,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> Generator[DesignRecord, None, None]:
    """Context manager for automatic design record creation and logging.

    Args:
        mission_spec: Parsed and validated mission specification
        mission_spec_raw: Raw mission specification from JSON file
        output_dir: Output directory for the design run
        cli_overrides: CLI parameter overrides applied

    Yields:
        DesignRecord: The design record instance for populating during run
    """
    start_time = time.time()
    storage = get_storage()

    # Create geometry parameters from mission spec and CLI overrides
    geometry_params = GeometryParams(
        rail_mm=mission_spec.rail_mm,
        deck_mm=mission_spec.deck_mm,
        material=mission_spec.material.value,
        additional_params=cli_overrides or {},
    )

    # Initialize design record
    record = DesignRecord(
        mission_spec=mission_spec_raw, geometry_params=geometry_params
    )

    # Set git commit separately to avoid validation issues
    record.git_commit = record.get_git_commit()

    logger.info(f"Started design run {record.design_id}")

    try:
        yield record

        # If we get here, the run completed successfully
        record.set_status(DesignStatus.PASS)
        logger.info(f"✓ Design run {record.design_id} completed successfully")

    except Exception as e:
        # Handle any errors that occurred during the run
        error_msg = str(e)
        logger.error(f"✗ Design run {record.design_id} failed: {error_msg}")
        record.set_status(DesignStatus.ERROR, error_msg)

        # Re-raise the exception after logging
        raise

    finally:
        # Always save the record, regardless of success/failure
        end_time = time.time()
        record.execution_time_seconds = end_time - start_time

        # Ensure records directory exists
        records_dir = output_dir / "records"
        records_dir.mkdir(parents=True, exist_ok=True)

        # Save the design record
        record_file = records_dir / f"{record.design_id}_design_record.json"
        record.save_to_file(record_file)

        logger.info(f"Design record saved: {record_file}")


def add_build_artifacts(
    record: DesignRecord, design_dir: Path, step_file: Optional[Path] = None
) -> None:
    """Add build artifacts to the design record.

    Args:
        record: Design record to update
        design_dir: Directory containing design outputs
        step_file: Path to STEP file if available
    """
    storage = get_storage()

    # Add STEP file
    if step_file and step_file.exists():
        uri = storage.store_artifact(step_file, record.design_id, step_file.name)
        record.add_artifact(
            ArtifactType.STEP, step_file, description="Generated STEP CAD file", uri=uri
        )

    # Add STL file
    stl_file = design_dir / "frame.stl"
    if stl_file.exists():
        uri = storage.store_artifact(stl_file, record.design_id, stl_file.name)
        record.add_artifact(
            ArtifactType.STL,
            stl_file,
            description="Generated STL file for 3D printing",
            uri=uri,
        )

    # Add mass budget
    mass_file = design_dir / "mass_budget.csv"
    if mass_file.exists():
        uri = storage.store_artifact(mass_file, record.design_id, mass_file.name)
        record.add_artifact(
            ArtifactType.MASS_BUDGET,
            mass_file,
            description="Mass budget breakdown",
            uri=uri,
        )


def add_physics_results(
    record: DesignRecord, design_dir: Path, physics_results: Any
) -> None:
    """Add physics validation results to the design record.

    Args:
        record: Design record to update
        design_dir: Directory containing design outputs
        physics_results: Physics validation results object
    """
    storage = get_storage()

    # Add physics check file
    physics_file = design_dir / "physics_check.json"
    if physics_file.exists():
        uri = storage.store_artifact(physics_file, record.design_id, physics_file.name)
        record.add_artifact(
            ArtifactType.PHYSICS_CHECK,
            physics_file,
            description="FEA validation results",
            uri=uri,
        )

    # Update analysis results
    if physics_results:
        if hasattr(physics_results, "max_stress_mpa"):
            record.update_analysis_results(
                max_stress_mpa=physics_results.max_stress_mpa
            )
        if (
            hasattr(physics_results, "thermal_stress_mpa")
            and physics_results.thermal_stress_mpa
        ):
            record.update_analysis_results(
                thermal_stress_mpa=physics_results.thermal_stress_mpa
            )
        if hasattr(physics_results, "status"):
            # Update overall status if physics failed
            if physics_results.status == "FAIL":
                record.set_status(DesignStatus.FAIL, "Physics validation failed")

        # Handle dictionary results (for loaded JSON)
        if isinstance(physics_results, dict):
            record.update_analysis_results(
                **{
                    k: v
                    for k, v in physics_results.items()
                    if k
                    in ["max_stress_mpa", "thermal_stress_mpa", "max_deflection_mm"]
                }
            )


def add_dfam_results(record: DesignRecord, design_dir: Path, dfam_results: Any) -> None:
    """Add DfAM validation results to the design record.

    Args:
        record: Design record to update
        design_dir: Directory containing design outputs
        dfam_results: DfAM validation results
    """
    storage = get_storage()

    # Add DfAM check file
    dfam_file = design_dir / "manufacturability.json"
    if dfam_file.exists():
        uri = storage.store_artifact(dfam_file, record.design_id, dfam_file.name)
        record.add_artifact(
            ArtifactType.DFAM_CHECK,
            dfam_file,
            description="Design for Additive Manufacturing check results",
            uri=uri,
        )

    # Update analysis results
    if dfam_results:
        if isinstance(dfam_results, dict):
            violations = dfam_results.get("violations", [])
            record.update_analysis_results(dfam_violations=violations)

            # Update overall status if DfAM failed
            if dfam_results.get("status") == "FAIL":
                record.set_status(DesignStatus.FAIL, "DfAM validation failed")


def add_report_artifacts(record: DesignRecord, design_dir: Path) -> None:
    """Add report artifacts to the design record.

    Args:
        record: Design record to update
        design_dir: Directory containing design outputs
    """
    storage = get_storage()

    # Add PDF report
    report_file = design_dir / "report.pdf"
    if report_file.exists():
        uri = storage.store_artifact(report_file, record.design_id, report_file.name)
        record.add_artifact(
            ArtifactType.REPORT,
            report_file,
            description="Comprehensive design report",
            uri=uri,
        )


def load_mass_from_budget(design_dir: Path) -> Optional[float]:
    """Load mass from mass budget file.

    Args:
        design_dir: Directory containing mass budget

    Returns:
        Total mass in kg if available
    """
    mass_file = design_dir / "mass_budget.csv"
    if not mass_file.exists():
        return None

    try:
        import csv

        with open(mass_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("component") == "total":
                    return float(row.get("mass_kg", 0))
        return None
    except Exception as e:
        logger.warning(f"Could not parse mass budget: {e}")
        return None
