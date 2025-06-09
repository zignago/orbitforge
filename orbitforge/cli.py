"""Command-line interface for OrbitForge."""

import uuid
from pathlib import Path
import typer
from typing import Optional
from rich.console import Console
from rich.table import Table
from loguru import logger
import sys
from .generator.mission import MissionSpec, Material, load_materials
from .fea import FastPhysicsValidator, MaterialProperties

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
        print("Building frame...")  # Debug print
        step_file = build_basic_frame(spec, design_dir)
        console.print(f"✓ Generated frame at [blue]{design_dir}[/]")

        # Run physics validation if requested
        if check:
            print("Starting validation...")  # Debug print
            console.print("\nRunning fast physics validation...")
            try:
                if verbose:
                    print("Verbose mode enabled")  # Debug print
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
                results = validator.validate()

                # Save results
                results_file = design_dir / "physics_check.json"
                validator.save_results(results, results_file)

                # Print results
                console.print("\n[bold]Physics Validation Results:[/]")
                console.print(f"Maximum Stress: {results.max_stress_mpa:.1f} MPa")
                console.print(f"Allowable Stress: {results.sigma_allow_mpa:.1f} MPa")
                console.print(
                    f"Status: [{'green' if results.status == 'PASS' else 'red'}]{results.status}[/]"
                )

                if results.thermal_stress_mpa is not None:
                    console.print("\n[bold]Thermal Analysis:[/]")
                    console.print(
                        f"Thermal Stress: {results.thermal_stress_mpa:.1f} MPa"
                    )
                    console.print(
                        f"Status: [{'green' if results.thermal_status == 'PASS' else 'red'}]{results.thermal_status}[/]"
                    )

                console.print(f"\nDetailed results saved to [blue]{results_file}[/]")

                # Exit with status code if validation failed
                if results.status != "PASS":
                    raise typer.Exit(1)

            except Exception as e:
                console.print(f"\n[red]Error during physics validation:[/] {str(e)}")
                if verbose:
                    import traceback

                    console.print("\nFull traceback:")
                    console.print(traceback.format_exc())
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
