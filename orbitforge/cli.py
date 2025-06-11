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
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
):
    """Generate a CubeSat structure from a mission specification."""
    from .generator.basic_frame import build_basic_frame

    # Set log level based on verbose flag
    if verbose:
        logger.remove()
        logger.add(
            sys.stderr,
            format="<green>{time:HH:mm:ss}</green> | {level} | {message}",
            level="DEBUG",
        )

    # Load and validate spec
    with open(mission_json) as fp:
        spec = MissionSpec.model_validate_json(fp.read())

    # Apply CLI overrides
    if rail is not None:
        spec.rail_mm = rail
    if deck is not None:
        spec.deck_mm = deck
    if material is not None:
        spec.material = material

    # Create output directory
    design_dir = outdir / f"design_{uuid.uuid4().hex[:4]}"
    design_dir.mkdir(parents=True, exist_ok=True)

    # Generate frame
    try:
        console.print("Building frame...")
        step_file = build_basic_frame(spec, design_dir)
        console.print(f"✓ Generated frame at [blue]{design_dir}[/]")

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
                    console.print(f"Young's modulus: {spec.youngs_modulus_gpa} GPa")
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

                console.print(f"\nDetailed results saved to [blue]{results_file}[/]")

            except Exception as e:
                console.print(f"\n[red]Error during physics validation:[/] {str(e)}")
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

                console.print(f"\nDetailed results saved to [blue]{dfam_file}[/]")

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

            except Exception as e:
                console.print(f"\n[red]Error generating report:[/] {str(e)}")
                if verbose:
                    import traceback

                    console.print("\nFull traceback:")
                    console.print(traceback.format_exc())
                raise typer.Exit(1)

        # Exit with status code if validation failed
        if check and physics_results and physics_results["status"] != "PASS":
            raise typer.Exit(1)
        if dfam and dfam_results and dfam_results["status"] != "PASS":
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
