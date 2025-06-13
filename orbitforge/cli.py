"""Command-line interface for OrbitForge."""

import uuid
from pathlib import Path
import typer
from typing import Optional
from rich.console import Console
from rich.table import Table
from loguru import logger
import sys
import json
from .generator.mission import MissionSpec, Material, load_materials
from .fea import FastPhysicsValidator, MaterialProperties
from .dfam.rules import DfamChecker
from .reporting.pdf_report import generate_report
from dataclasses import asdict, is_dataclass
from rich import print

# Import design record components
from .design_record import DesignRecord
from .artifact_storage import get_storage
from .design_context import (
    design_record_context,
    add_build_artifacts,
    add_physics_results,
    add_dfam_results,
    add_report_artifacts,
    load_mass_from_budget,
)

# Configure logger
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | {level} | {message}",
    level="INFO",
)

app = typer.Typer(
    help="OrbitForge: Generate CubeSat structures from mission specs",
    rich_markup_mode="rich",
)
console = Console()


def get_status(obj):
    if hasattr(obj, "status"):
        return obj.status
    elif isinstance(obj, dict):
        return obj.get("status")
    return None


@app.command(name="run")
def run_mission(
    mission_json: Path = typer.Argument(
        ..., help="Path to mission JSON spec file", exists=True
    ),
    outdir: Path = typer.Option(
        "outputs", "--outdir", "-o", help="Output directory for generated files"
    ),
    rail: Optional[float] = typer.Option(
        None, "--rail", help="Rail thickness in mm (overrides spec)"
    ),
    deck: Optional[float] = typer.Option(
        None, "--deck", help="Deck thickness in mm (overrides spec)"
    ),
    material: Optional[Material] = typer.Option(
        None, "--material", "-m", help="Material selection (overrides spec)"
    ),
    check: bool = typer.Option(False, "--check", help="Run fast physics validation"),
    dfam: bool = typer.Option(False, "--dfam", help="Run DfAM checks"),
    report: bool = typer.Option(False, "--report", help="Generate PDF report"),
    multi: int = typer.Option(
        1, "--multi", help="Generate multiple design variants (1-10)"
    ),
    seed: int = typer.Option(
        42, "--seed", help="Random seed for reproducible parameter jittering"
    ),
    generator: Optional[str] = typer.Option(
        None, "--generator", help="Generator type: 'diffusion' for ML-based generation"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
):
    """Generate a CubeSat structure from a mission specification."""
    from .generator.basic_frame import build_basic_frame
    from .generator.multi_design import MultiDesignGenerator
    from .generator.diffusion_workflow import DiffusionWorkflow

    # Set log level based on verbose flag
    if verbose:
        logger.remove()
        logger.add(
            sys.stderr,
            format="<green>{time:HH:mm:ss}</green> | {level} | {message}",
            level="DEBUG",
        )

    # Validate multi parameter
    if multi < 1 or multi > 10:
        console.print("[red]Error:[/] --multi must be between 1 and 10")
        raise typer.Exit(1)

    # Load and validate spec
    try:
        with open(mission_json) as fp:
            spec = MissionSpec.model_validate_json(fp.read())
    except Exception as e:
        console.print(f"[red]Error:[/] {str(e)}")
        raise typer.Exit(1)

    # Apply CLI overrides
    if rail is not None:
        spec.rail_mm = rail
    if deck is not None:
        spec.deck_mm = deck
    if material is not None:
        spec.material = material

    # Create output directory
    outdir.mkdir(parents=True, exist_ok=True)

    # Check if diffusion generator is requested
    if generator == "diffusion":
        console.print(f"Using diffusion generator for {multi} candidates...")

        # Run diffusion workflow
        workflow = DiffusionWorkflow(spec)
        summary_file = workflow.run_complete_workflow(
            outdir,
            n_candidates=max(multi, 5),  # Minimum 5 for diffusion
            run_fea=check,
            run_dfam=dfam,
        )

        # Load and display results
        with open(summary_file) as f:
            summary_data = json.load(f)["results"]

        # Display summary table
        table = Table(
            title=f"Diffusion-Generated Designs ({len(summary_data)} candidates)"
        )
        table.add_column("Design", style="cyan")
        table.add_column("Mass (kg)", justify="right", style="green")
        table.add_column("Max Stress (MPa)", justify="right", style="yellow")
        table.add_column("Status", justify="center")
        table.add_column("Files", justify="left")

        for result in summary_data:
            design = result["design"]
            mass = f"{result['mass_kg']:.3f}" if "mass_kg" in result else "N/A"
            stress = (
                f"{result.get('max_stress_MPa', 0):.1f}"
                if "max_stress_MPa" in result
                else "N/A"
            )
            status = result.get("status", "UNKNOWN")
            files = []
            if result.get("step_file"):
                files.append("STEP")
            if result.get("stl_file"):
                files.append("STL")
            files_str = ", ".join(files)

            status_style = {"PASS": "green", "FAIL": "red", "UNKNOWN": "yellow"}.get(
                status, "white"
            )

            table.add_row(
                design,
                mass,
                stress,
                f"[{status_style}]{status}[/{status_style}]",
                files_str,
            )

        console.print(table)
        console.print(
            f"\n✓ Generated {len(summary_data)} designs using diffusion model in [blue]{outdir}[/]"
        )
        console.print(f"Summary available at [blue]{summary_file}[/]")

        # Check if any designs passed (for exit code)
        passed_designs = [d for d in summary_data if d["status"] == "PASS"]
        if check and not passed_designs:
            console.print("[red]No designs passed validation[/]")
            raise typer.Exit(1)

        return

    # Check if multi-design generation is requested
    elif multi > 1:
        console.print(f"Generating {multi} design variants...")

        # Run multi-design workflow
        generator = MultiDesignGenerator(spec, multi, seed)
        summary_file = generator.run_complete_workflow(outdir, run_fea=check)

        # Load and display results
        with open(summary_file) as f:
            summary_data = json.load(f)

        # Display summary table
        table = Table(title=f"Design Variants Summary ({len(summary_data)} designs)")
        table.add_column("Design")
        table.add_column("Mass (kg)")
        table.add_column("Max Stress (MPa)")
        table.add_column("Status")
        table.add_column("Rail (mm)")
        table.add_column("Deck (mm)")
        table.add_column("Material")

        for design in summary_data:
            mass_str = f"{design['mass_kg']:.3f}" if design["mass_kg"] else "N/A"
            stress_str = (
                f"{design['max_stress_MPa']:.1f}" if design["max_stress_MPa"] else "N/A"
            )
            status = design["status"] or "UNKNOWN"
            status_color = (
                "green"
                if status == "PASS"
                else (
                    "red"
                    if status in ["FAIL", "FEA_FAILED", "BUILD_FAILED"]
                    else "yellow"
                )
            )

            table.add_row(
                design["design"],
                mass_str,
                stress_str,
                f"[{status_color}]{status}[/]",
                f"{design['rail_mm']:.2f}",
                f"{design['deck_mm']:.2f}",
                design["material"],
            )

        console.print(table)
        console.print(
            f"\n✓ Generated {len(summary_data)} variants in [blue]{outdir}[/]"
        )
        console.print(f"Summary available at [blue]{summary_file}[/]")

        # Check if any designs passed (for exit code)
        passed_designs = [d for d in summary_data if d["status"] == "PASS"]
        if check and not passed_designs:
            console.print("[red]No designs passed validation[/]")
            raise typer.Exit(1)

        return

    else:
        # Single design mode with design record logging
        # Prepare CLI overrides for design record
        cli_overrides = {}
        if rail is not None:
            cli_overrides["rail_override"] = rail
        if deck is not None:
            cli_overrides["deck_override"] = deck
        if material is not None:
            cli_overrides["material_override"] = material.value

        # Load raw mission spec for design record
        with open(mission_json) as fp:
            mission_spec_raw = json.load(fp)

        # Use design record context manager
        with design_record_context(
            spec, mission_spec_raw, outdir, cli_overrides
        ) as design_record:
            design_dir = outdir / f"design_{design_record.design_id[:8]}"
            design_dir.mkdir(parents=True, exist_ok=True)

            # Generate frame
            try:
                console.print("Building frame...")
                step_file = build_basic_frame(spec, design_dir)
                console.print(f"✓ Generated frame at [blue]{design_dir}[/]")

                # Add build artifacts to design record
                add_build_artifacts(design_record, design_dir, step_file)

                # Load mass from budget and update design record
                mass_kg = load_mass_from_budget(design_dir)
                if mass_kg:
                    design_record.update_analysis_results(mass_kg=mass_kg)

                # Store results for report
                physics_results = None
                dfam_results = None

                # Run physics validation if requested
                if check:
                    console.print("\nRunning fast physics validation...")
                    try:
                        if verbose:
                            console.print("\nMaterial properties from spec:")
                            console.print(f"Material: {spec.material}")
                            console.print(f"Yield strength: {spec.yield_mpa} MPa")
                            console.print(
                                f"Young's modulus: {spec.youngs_modulus_gpa} GPa"
                            )
                            console.print(f"Poisson's ratio: {spec.poissons_ratio}")
                            console.print(f"Density: {spec.density_kg_m3} kg/m³")
                            console.print(f"CTE: {spec.cte_per_k} /K")

                        # Create material properties for validator
                        material_props = MaterialProperties(
                            name=spec.material.value,
                            yield_strength_mpa=spec.yield_mpa,
                            youngs_modulus_gpa=spec.youngs_modulus_gpa,
                            poissons_ratio=spec.poissons_ratio,
                            density_kg_m3=spec.density_kg_m3,
                            cte_per_k=spec.cte_per_k,
                        )

                        # Run validation
                        validator = FastPhysicsValidator(step_file, material_props)
                        physics_results = validator.run_validation()

                        # Save results
                        results_file = design_dir / "physics_check.json"
                        validator.save_results(physics_results, results_file)

                        # Print results
                        console.print("\n[bold]Physics Validation Results:[/]")
                        console.print(
                            f"Maximum Stress: {physics_results.max_stress_mpa:.1f} MPa"
                        )
                        console.print(
                            f"Allowable Stress: {physics_results.sigma_allow_mpa:.1f} MPa"
                        )
                        console.print(
                            f"Status: [{'green' if physics_results.status == 'PASS' else 'red'}]{physics_results.status}[/]"
                        )

                        if physics_results.thermal_stress_mpa is not None:
                            console.print("\n[bold]Thermal Analysis:[/]")
                            console.print(
                                f"Thermal Stress: {physics_results.thermal_stress_mpa:.1f} MPa"
                            )
                            console.print(
                                f"Status: [{'green' if physics_results.thermal_status == 'PASS' else 'red'}]{physics_results.thermal_status}[/]"
                            )

                            if check and physics_results:
                                if (
                                    physics_results.status != "PASS"
                                    or getattr(
                                        physics_results, "thermal_status", "PASS"
                                    )
                                    == "FAIL"
                                ):
                                    raise typer.Exit(1)

                        console.print(
                            f"\nDetailed results saved to [blue]{results_file}[/]"
                        )

                        # Add physics results to design record
                        add_physics_results(design_record, design_dir, physics_results)

                    except Exception as e:
                        console.print(
                            f"\n[red]Error during physics validation:[/] {str(e)}"
                        )
                        if verbose:
                            import traceback

                            console.print("\nFull traceback:")
                            console.print(traceback.format_exc())
                        raise typer.Exit(1)

                # Run DfAM checks if requested
                if dfam:
                    console.print("\nRunning DfAM checks...")
                    try:
                        checker = DfamChecker(design_dir / "frame.stl")
                        dfam_results = checker.run_all_checks()

                        # Save results
                        dfam_file = design_dir / "manufacturability.json"
                        with open(dfam_file, "w") as f:
                            json.dump(dfam_results, f, indent=2)

                        # Print summary
                        console.print("\n[bold]DfAM Check Results:[/]")
                        console.print(
                            f"Status: [{'green' if dfam_results['status'] == 'PASS' else 'red'}]{dfam_results['status']}[/]"
                        )
                        console.print(f"Errors: {dfam_results['error_count']}")
                        console.print(f"Warnings: {dfam_results['warning_count']}")

                        if dfam_results["violations"]:
                            console.print("\nViolations:")
                            for v in dfam_results["violations"]:
                                color = "red" if v["severity"] == "ERROR" else "yellow"
                                console.print(
                                    f"[{color}]{v['severity']}[/] {v['rule']}: {v['message']} "
                                    f"(value: {v['value']:.2f}, limit: {v['limit']:.2f})"
                                )

                        console.print(
                            f"\nDetailed results saved to [blue]{dfam_file}[/]"
                        )

                        # Add DfAM results to design record
                        add_dfam_results(design_record, design_dir, dfam_results)

                    except Exception as e:
                        console.print(f"\n[red]Error during DfAM checks:[/] {str(e)}")
                        if verbose:
                            import traceback

                            console.print("\nFull traceback:")
                            console.print(traceback.format_exc())
                        raise typer.Exit(1)

                # Generate PDF report if requested
                if report:
                    console.print("\nGenerating PDF report...")
                    try:
                        # Load results if not already in memory
                        if (
                            physics_results is None
                            and (design_dir / "physics_check.json").exists()
                        ):
                            with open(design_dir / "physics_check.json") as f:
                                physics_results = json.load(f)

                        if (
                            dfam_results is None
                            and (design_dir / "manufacturability.json").exists()
                        ):
                            with open(design_dir / "manufacturability.json") as f:
                                dfam_results = json.load(f)

                        # Ensure both are dictionaries
                        if is_dataclass(physics_results):
                            physics_results = asdict(physics_results)
                        if is_dataclass(dfam_results):
                            dfam_results = asdict(dfam_results)

                        # Generate report
                        pdf_file = generate_report(
                            design_dir=design_dir,
                            mission_spec=spec.model_dump(),
                            physics_results=physics_results or {},
                            dfam_results=dfam_results
                            or {"status": "NOT RUN", "violations": []},
                        )

                        console.print(f"\nReport generated at [blue]{pdf_file}[/]")

                        # Add report artifacts to design record
                        add_report_artifacts(design_record, design_dir)

                    except Exception as e:
                        console.print(f"\n[red]Error generating report:[/] {str(e)}")
                        if verbose:
                            import traceback

                            console.print("\nFull traceback:")
                            console.print(traceback.format_exc())
                        raise typer.Exit(1)

                # Exit with status code if validation failed
                if check and physics_results and get_status(physics_results) != "PASS":
                    raise typer.Exit(1)
                if dfam and dfam_results and get_status(dfam_results) != "PASS":
                    raise typer.Exit(1)

            except ValueError as e:
                console.print(f"[red]Error:[/] {e}")
                raise typer.Exit(1)


@app.command()
def materials():
    """List available materials and their properties."""
    materials = load_materials()

    table = Table(title="Available Materials")
    table.add_column("Material")
    table.add_column("Density (kg/m³)")
    table.add_column("Yield (MPa)")
    table.add_column("Young's (GPa)")
    table.add_column("Description")

    for mat_id, props in materials.items():
        table.add_row(
            mat_id,
            str(props["density_kg_m3"]),
            str(props["yield_mpa"]),
            str(props["youngs_modulus_gpa"]),
            props.get("description", ""),
        )

    console.print(table)


@app.command(name="fetch")
def fetch_design(
    design_id: str = typer.Argument(..., help="Design ID to fetch artifacts for"),
    output_dir: Path = typer.Option(
        ".", "--output", "-o", help="Directory to save fetched artifacts"
    ),
    storage_uri: Optional[str] = typer.Option(
        None, "--storage", help="Override storage URI (e.g., s3://bucket/path)"
    ),
):
    """Fetch all artifacts for a design by its ID."""
    try:
        # Initialize storage
        if storage_uri:
            storage = get_storage(storage_uri)
        else:
            storage = get_storage()

        console.print(f"Fetching artifacts for design: [cyan]{design_id}[/]")

        # Create output directory for this design
        design_output_dir = output_dir / f"design_{design_id}"
        design_output_dir.mkdir(parents=True, exist_ok=True)

        # List available artifacts
        artifacts = storage.list_artifacts(design_id)

        if not artifacts:
            console.print(f"[yellow]No artifacts found for design ID: {design_id}[/]")
            console.print("Available options:")
            console.print("1. Check that the design ID is correct")
            console.print("2. Verify storage configuration")
            console.print("3. Ensure artifacts were stored during the original run")
            return

        console.print(f"Found {len(artifacts)} artifacts:")

        # Create table for artifacts
        table = Table(title=f"Artifacts for Design {design_id}")
        table.add_column("Artifact", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Local Path", style="green")

        # Fetch each artifact
        success_count = 0
        for artifact_name, uri in artifacts.items():
            try:
                local_path = design_output_dir / artifact_name
                storage.retrieve_artifact(uri, local_path)
                table.add_row(artifact_name, "[green]✓ Downloaded[/]", str(local_path))
                success_count += 1
            except Exception as e:
                table.add_row(artifact_name, f"[red]✗ Failed[/]", f"Error: {str(e)}")
                logger.error(f"Failed to fetch {artifact_name}: {e}")

        console.print(table)

        # Look for design record
        record_files = list(design_output_dir.glob(f"*{design_id}*design_record.json"))
        if record_files:
            record_file = record_files[0]
            try:
                record = DesignRecord.load_from_file(record_file)
                console.print(f"\n[bold]Design Record Summary:[/]")
                console.print(f"Design ID: [cyan]{record.design_id}[/]")
                console.print(f"Timestamp: {record.timestamp}")
                console.print(
                    f"Status: [{'green' if record.status == 'PASS' else 'red'}]{record.status}[/]"
                )
                console.print(f"Material: {record.geometry_params.material}")
                console.print(f"Rail thickness: {record.geometry_params.rail_mm} mm")
                console.print(f"Deck thickness: {record.geometry_params.deck_mm} mm")

                if record.analysis_results.mass_kg:
                    console.print(f"Mass: {record.analysis_results.mass_kg:.3f} kg")
                if record.analysis_results.max_stress_mpa:
                    console.print(
                        f"Max stress: {record.analysis_results.max_stress_mpa:.1f} MPa"
                    )

            except Exception as e:
                logger.warning(f"Could not parse design record: {e}")

        console.print(
            f"\n✓ Successfully fetched {success_count}/{len(artifacts)} artifacts"
        )
        console.print(f"Artifacts saved to: [blue]{design_output_dir}[/]")

    except Exception as e:
        console.print(f"[red]Error fetching design:[/] {str(e)}")
        raise typer.Exit(1)


@app.command(name="list-designs")
def list_designs(
    limit: int = typer.Option(
        10, "--limit", "-n", help="Maximum number of designs to show"
    ),
    storage_uri: Optional[str] = typer.Option(
        None, "--storage", help="Override storage URI"
    ),
):
    """List recent design records."""
    try:
        # Look for design records in the local outputs directory
        outputs_dir = Path("outputs")
        record_files = []

        if outputs_dir.exists():
            # Search for design record files
            record_files = list(outputs_dir.rglob("*design_record.json"))
            record_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        if not record_files:
            console.print("[yellow]No design records found in outputs directory[/]")
            return

        # Create table for designs
        table = Table(title=f"Recent Designs (showing {min(limit, len(record_files))})")
        table.add_column("Design ID", style="cyan")
        table.add_column("Timestamp", style="white")
        table.add_column("Status", justify="center")
        table.add_column("Material", style="green")
        table.add_column("Bus Size", justify="center")
        table.add_column("Mass (kg)", justify="right", style="yellow")

        # Load and display records
        for record_file in record_files[:limit]:
            try:
                record = DesignRecord.load_from_file(record_file)

                # Format timestamp
                timestamp = (
                    record.timestamp.split("T")[0]
                    if "T" in record.timestamp
                    else record.timestamp
                )

                # Get bus size from mission spec
                bus_size = f"{record.mission_spec.get('bus_u', 'N/A')}U"

                # Format mass
                mass_str = (
                    f"{record.analysis_results.mass_kg:.3f}"
                    if record.analysis_results.mass_kg
                    else "N/A"
                )

                # Status color
                status_color = {
                    "PASS": "green",
                    "FAIL": "red",
                    "ERROR": "red",
                    "PENDING": "yellow",
                }.get(record.status, "white")

                table.add_row(
                    record.design_id[:8],  # Short ID
                    timestamp,
                    f"[{status_color}]{record.status}[/{status_color}]",
                    record.geometry_params.material,
                    bus_size,
                    mass_str,
                )

            except Exception as e:
                logger.warning(f"Could not parse {record_file}: {e}")

        console.print(table)
        console.print(f"\nFound {len(record_files)} total design records")
        console.print("Use [cyan]orbitforge fetch <design-id>[/] to retrieve artifacts")

    except Exception as e:
        console.print(f"[red]Error listing designs:[/] {str(e)}")
        raise typer.Exit(1)
