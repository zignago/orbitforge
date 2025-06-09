import uuid
from pathlib import Path
import typer
from typing import Optional
from rich.console import Console
from rich.table import Table
from .mission import MissionSpec, Material, load_materials


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
):
    """Generate a CubeSat structure from a mission specification."""
    from .basic_frame import build_basic_frame

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
        build_basic_frame(spec, design_dir)
        console.print(f"✓ Generated frame at [blue]{design_dir}[/]")
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
    table.add_column("Description")

    for mat_id, props in materials.items():
        table.add_row(mat_id, str(props["density_kg_m3"]), props.get("description", ""))

    console.print(table)


if __name__ == "__main__":
    app()
