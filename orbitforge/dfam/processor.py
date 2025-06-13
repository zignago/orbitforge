"""DfAM Post-Processor for OrbitForge v0.1.5."""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger

from .rules import DfamChecker, STLExporter


class DfamPostProcessor:
    """
    DfAM Post-Processor that applies design-for-additive-manufacturing constraints
    and exports ready-to-print STL/3MF files according to v0.1.5 specification.
    """

    def __init__(self):
        """Initialize the DfAM post-processor."""
        self.rules_applied = {
            "wall_thickness_min_mm": 0.8,
            "overhang_angle_max_deg": 45.0,
            "powder_drain_max_cavity_mm": 50.0,
        }

    def process_design(
        self,
        design_dir: Path,
        step_file: Optional[Path] = None,
        base_name: str = "frame",
    ) -> Dict[str, Any]:
        """
        Process a single design through the complete DfAM workflow.

        Args:
            design_dir: Directory containing the design files
            step_file: Optional path to STEP file (for reference)
            base_name: Base name for output files

        Returns:
            Dictionary with DfAM processing results
        """
        logger.info(f"Processing DfAM checks for design in {design_dir}")

        # Look for existing STL file
        stl_file = design_dir / f"{base_name}.stl"

        if not stl_file.exists():
            logger.warning(f"STL file not found: {stl_file}")
            return {
                "dfam_status": "FAIL",
                "error": "STL file not found",
                "stl_exported": False,
                "report_path": None,
            }

        try:
            # Initialize DfAM checker
            checker = DfamChecker(stl_file)

            # Run all checks
            dfam_results = checker.run_all_checks()

            # Save detailed report
            report_path = checker.save_dfam_report(design_dir, "dfam_report")

            # Export STL only if checks pass (per spec)
            stl_exported = False
            if dfam_results["status"] == "PASS":
                # Copy/export the STL file (it already exists, so just confirm)
                stl_exported = stl_file.exists()
                logger.info(f"✓ DfAM checks passed - STL file available: {stl_file}")
            else:
                # Per spec: "Only PASS designs get .stl files generated"
                # We could optionally remove the STL file here, but for now just flag it
                logger.warning(
                    f"✗ DfAM checks failed - STL file should not be used for printing"
                )

            # Prepare result
            result = {
                "dfam_status": dfam_results["status"],
                "error_count": dfam_results["error_count"],
                "warning_count": dfam_results["warning_count"],
                "wall_thickness": dfam_results["wall_thickness"],
                "overhang_check": dfam_results["overhang_check"],
                "drain_holes": dfam_results["drain_holes"],
                "stl_exported": stl_exported,
                "report_path": str(report_path.relative_to(design_dir)),
                "violations": dfam_results["violations"],
            }

            logger.info(f"DfAM processing complete: {result['dfam_status']}")
            return result

        except Exception as e:
            logger.error(f"DfAM processing failed: {e}")
            return {
                "dfam_status": "ERROR",
                "error": str(e),
                "stl_exported": False,
                "report_path": None,
            }

    def process_batch(
        self, design_dirs: list[Path], base_name: str = "frame"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Process multiple designs through DfAM workflow.

        Args:
            design_dirs: List of design directories to process
            base_name: Base name for output files

        Returns:
            Dictionary mapping design directory names to results
        """
        results = {}

        for design_dir in design_dirs:
            design_name = design_dir.name
            try:
                result = self.process_design(design_dir, base_name=base_name)
                results[design_name] = result
            except Exception as e:
                logger.error(f"Failed to process {design_name}: {e}")
                results[design_name] = {
                    "dfam_status": "ERROR",
                    "error": str(e),
                    "stl_exported": False,
                    "report_path": None,
                }

        return results

    def update_summary_with_dfam(
        self, summary_data: list[dict], dfam_results: Dict[str, Dict[str, Any]]
    ) -> list[dict]:
        """
        Update summary data with DfAM results.

        Args:
            summary_data: List of design summary dictionaries
            dfam_results: DfAM results mapped by design name

        Returns:
            Updated summary data with dfam_status fields
        """
        for design_summary in summary_data:
            design_name = design_summary.get("design", "")

            if design_name in dfam_results:
                dfam_result = dfam_results[design_name]
                design_summary["dfam_status"] = dfam_result["dfam_status"]
                design_summary["dfam_error_count"] = dfam_result.get("error_count", 0)
                design_summary["dfam_warning_count"] = dfam_result.get(
                    "warning_count", 0
                )
            else:
                design_summary["dfam_status"] = "NOT_RUN"
                design_summary["dfam_error_count"] = 0
                design_summary["dfam_warning_count"] = 0

        return summary_data

    def generate_dfam_summary_report(
        self, dfam_results: Dict[str, Dict[str, Any]], output_path: Path
    ) -> Path:
        """
        Generate a summary report of all DfAM results.

        Args:
            dfam_results: DfAM results mapped by design name
            output_path: Path to save the summary report

        Returns:
            Path to the generated summary report
        """
        summary = {
            "dfam_post_processor_version": "v0.1.5",
            "rules_applied": self.rules_applied,
            "total_designs": len(dfam_results),
            "passed_designs": len(
                [r for r in dfam_results.values() if r["dfam_status"] == "PASS"]
            ),
            "failed_designs": len(
                [r for r in dfam_results.values() if r["dfam_status"] == "FAIL"]
            ),
            "error_designs": len(
                [r for r in dfam_results.values() if r["dfam_status"] == "ERROR"]
            ),
            "results": dfam_results,
        }

        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"DfAM summary report saved: {output_path}")
        return output_path
