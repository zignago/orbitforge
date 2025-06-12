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
    uplift: bool = typer.Option(
        False, "--uplift", "-u", help="Run full-fidelity FEA on AWS Batch"
    ),
    mock_batch: bool = typer.Option(
        False, "--mock-batch", help="Use mock AWS Batch for testing"
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
        summary_file = generator.run_complete_workflow(
            outdir, run_fea=check, run_uplift=uplift, mock_batch=mock_batch
        )

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
        table.add_column("FEA Mode", justify="center")

        for design in summary_data:
            mass_str = f"{design['mass_kg']:.3f}" if design["mass_kg"] else "N/A"

            # Handle stress value display
            stress_fea = design.get("max_stress_fea")
            stress_mpa = design.get("max_stress_MPa")

            if stress_fea is not None:
                stress_str = f"{stress_fea:.1f}"
            elif stress_mpa is not None:
                stress_str = f"{stress_mpa:.1f}"
            else:
                stress_str = "N/A"

            status = design.get("status_fea") or design.get("status") or "UNKNOWN"
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
                design["design_id"],
                mass_str,
                stress_str,
                f"[{status_color}]{status}[/]",
                f"{design['rail_mm']:.2f}",
                f"{design['deck_mm']:.2f}",
                design["material"],
                f"[blue]{design.get('fea_mode', 'fast')}[/]",
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
        # Single design mode (original behavior)
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

                        if check and physics_results:
                            if (
                                physics_results.status != "PASS"
                                or getattr(physics_results, "thermal_status", "PASS")
                                == "FAIL"
                            ):
                                raise typer.Exit(1)

                    console.print(
                        f"\nDetailed results saved to [blue]{results_file}[/]"
                    )

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

                    console.print(f"\nDetailed results saved to [blue]{dfam_file}[/]")

                except Exception as e:
                    console.print(f"\n[red]Error during DfAM checks:[/] {str(e)}")
                    if verbose:
                        import traceback

                        console.print("\nFull traceback:")
                        console.print(traceback.format_exc())
                    raise typer.Exit(1)

            # Run full-fidelity FEA uplift if requested
            if uplift:
                console.print("\nRunning full-fidelity FEA uplift on AWS Batch...")
                try:
                    from .fea.fe_uplift import FEAUpliftClient
                    from .fea.preprocessor import convert_step_to_bdf
                    from .fea.postprocessor import process_op2_results
                    import os

                    # Check AWS credentials
                    required_env_vars = [
                        "AWS_BATCH_JOB_QUEUE",
                        "AWS_BATCH_JOB_DEFINITION",
                        "AWS_S3_BUCKET",
                    ]
                    missing_vars = [
                        var for var in required_env_vars if not os.environ.get(var)
                    ]
                    if missing_vars and not mock_batch:
                        console.print(
                            f"[red]Error:[/] Missing required environment variables: {', '.join(missing_vars)}"
                        )
                        console.print(
                            "Please set AWS Batch configuration environment variables."
                        )
                        raise typer.Exit(1)

                    # Prepare material properties
                    material_props = {
                        "elastic_modulus": spec.youngs_modulus_gpa
                        * 1e9,  # Convert GPa to Pa
                        "poisson_ratio": spec.poissons_ratio,
                        "density": spec.density_kg_m3,
                        "thickness": spec.rail_mm,  # Use rail thickness as shell thickness
                    }

                    # Prepare loads (basic gravity load)
                    loads = {
                        "accel_x": 0.0,
                        "accel_y": 0.0,
                        "accel_z": 9.81,  # 1g gravity
                    }

                    # Generate design ID
                    design_id = f"design_{uuid.uuid4().hex[:8]}"

                    # Convert STEP to BDF
                    bdf_file = design_dir / "full_fea" / "frame.bdf"
                    bdf_file.parent.mkdir(exist_ok=True)

                    console.print("Converting STEP to Nastran BDF format...")
                    convert_step_to_bdf(step_file, bdf_file, material_props, loads)

                    # Submit job
                    console.print("Submitting FEA job...")
                    if mock_batch:
                        console.print("[yellow]Using mock AWS Batch for testing[/]")
                    job_id = submit_batch_job(
                        design_id, bdf_file, design_dir, mock=mock_batch
                    )
                    console.print(f"Job submitted with ID: {job_id}")

                    # Poll for completion
                    console.print("Waiting for job completion...")
                    job_status = poll_batch_job(job_id, mock=mock_batch)

                    if job_status["jobStatus"] == "SUCCEEDED":
                        console.print("[green]FEA job completed successfully![/]")

                        # Download results
                        console.print("Downloading results...")
                        results_dir, job_metadata = download_results(
                            design_id, design_dir, job_status, mock=mock_batch
                        )

                        # Process OP2 results
                        console.print("Processing analysis results...")
                        op2_file = results_dir / "frame.op2"

                        design_info = {
                            "design_id": design_id,
                            "material": spec.material.value,
                            "rail_thickness_mm": spec.rail_mm,
                            "deck_thickness_mm": spec.deck_mm,
                        }

                        fea_summary = process_op2_results(
                            op2_file,
                            results_dir,
                            design_info,
                            material_yield_strength=spec.yield_mpa,
                        )

                        # Update main summary with FEA results
                        summary_file = design_dir / "summary.json"
                        if summary_file.exists():
                            with open(summary_file, "r") as f:
                                summary = json.load(f)
                        else:
                            summary = {}

                        summary.update(fea_summary)

                        with open(summary_file, "w") as f:
                            json.dump(summary, f, indent=2)

                        # Print results
                        console.print("\n[bold]Full FEA Results:[/]")
                        console.print(
                            f"Max Stress: {fea_summary['max_stress_fea']:.1f} MPa"
                        )
                        console.print(
                            f"Safety Factor: {fea_summary['safety_factor']:.2f}"
                        )
                        console.print(
                            f"Status: [{'green' if fea_summary['status_fea'] == 'PASS' else 'red'}]{fea_summary['status_fea']}[/]"
                        )
                        console.print(f"Results saved to [blue]{results_dir}[/]")

                        # Exit with error if FEA failed
                        if fea_summary["status_fea"] != "PASS":
                            raise typer.Exit(1)

                    else:
                        console.print(
                            f"[red]FEA job failed with status: {job_status['jobStatus']}[/]"
                        )
                        console.print(
                            f"Reason: {job_status.get('statusReason', 'Unknown')}"
                        )
                        raise typer.Exit(1)

                except Exception as e:
                    console.print(f"\n[red]Error during FEA uplift:[/] {str(e)}")
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
