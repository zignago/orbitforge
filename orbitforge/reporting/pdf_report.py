"""PDF report generator for OrbitForge."""

import csv
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from collections import defaultdict

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart


def create_stress_chart(results: Dict[str, Any]) -> Drawing:
    """Create a bar chart comparing stress values."""
    drawing = Drawing(400, 200)

    bc = VerticalBarChart()
    bc.x = 50
    bc.y = 50
    bc.height = 125
    bc.width = 300

    # Get values with defaults
    max_stress = (
        results.get("max_stress_mpa", 0) or results.get("max_stress_MPa", 0) or 0
    )
    allow_stress = (
        results.get("sigma_allow_mpa", 0) or results.get("sigma_allow_MPa", 0) or 0
    )
    status = results.get("status", "UNKNOWN")

    bc.data = [[max_stress], [allow_stress]]
    bc.categoryAxis.categoryNames = ["Max Stress", "Allowable"]
    bc.bars[0].fillColor = colors.red if status == "FAIL" else colors.green
    bc.bars[1].fillColor = colors.blue

    drawing.add(bc)
    return drawing


def generate_report(
    design_dir: Path,
    mission_spec: Dict[str, Any],
    physics_results: Dict[str, Any],
    dfam_results: Dict[str, Any],
) -> Path:
    """Generate a PDF report summarizing the design and analysis results."""

    # Setup document
    pdf_file = design_dir / "report.pdf"
    doc = SimpleDocTemplate(
        str(pdf_file),
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72,
    )

    # Styles
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    heading_style = styles["Heading1"]
    normal_style = styles["Normal"]

    # Build content
    content = []

    # Title
    content.append(Paragraph(f"OrbitForge Design Report", title_style))
    content.append(
        Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", normal_style
        )
    )
    content.append(Spacer(1, 0.2 * inch))

    # Mission Specs
    content.append(Paragraph("Mission Specifications", heading_style))
    spec_data = [
        ["Parameter", "Value"],
        ["Bus Size", f"{mission_spec['bus_u']}U"],
        ["Payload Mass", f"{mission_spec['payload_mass_kg']:.2f} kg"],
        ["Orbit Altitude", f"{mission_spec['orbit_alt_km']} km"],
        ["Material", mission_spec["material"].split(".")[-1]],
    ]
    spec_table = Table(spec_data)
    spec_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 14),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("TEXTCOLOR", (0, 1), (-1, -1), colors.black),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 1), (-1, -1), 12),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
    )
    content.append(spec_table)
    content.append(Spacer(1, 0.2 * inch))

    # Mass Budget
    content.append(Paragraph("Mass Budget", heading_style))
    with open(design_dir / "mass_budget.csv") as f:
        reader = csv.reader(f)
        mass_data = list(reader)
    mass_table = Table(mass_data)
    mass_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    )
    content.append(mass_table)
    content.append(Spacer(1, 0.2 * inch))

    # Structural Analysis
    content.append(Paragraph("Structural Analysis", heading_style))

    # Handle missing or empty physics results
    if physics_results and "status" in physics_results:
        stress_status = physics_results["status"]
        stress_color = colors.red if stress_status == "FAIL" else colors.green
        content.append(
            Paragraph(
                f"Status: <font color={stress_color}>{stress_status}</font>",
                normal_style,
            )
        )
        content.append(create_stress_chart(physics_results))

        if physics_results.get("thermal_stress_MPa"):
            content.append(
                Paragraph(
                    f"Thermal Stress: {physics_results['thermal_stress_MPa']:.1f} MPa",
                    normal_style,
                )
            )
            content.append(
                Paragraph(
                    f"Thermal Status: {physics_results['thermal_status']}", normal_style
                )
            )
    else:
        content.append(
            Paragraph("No structural analysis results available", normal_style)
        )
    content.append(Spacer(1, 0.2 * inch))

    # DfAM Analysis
    content.append(Paragraph("Manufacturability Analysis", heading_style))
    if dfam_results and "status" in dfam_results:
        dfam_status = dfam_results["status"]
        dfam_color = colors.red if dfam_status == "FAIL" else colors.green
        content.append(
            Paragraph(
                f"Status: <font color={dfam_color}>{dfam_status}</font>", normal_style
            )
        )
        content.append(
            Paragraph(
                f"Errors: {dfam_results.get('error_count', 0)}, Warnings: {dfam_results.get('warning_count', 0)}",
                normal_style,
            )
        )

        if dfam_results.get("violations"):
            # Group violations by (rule, severity, message, value, limit)
            grouped = defaultdict(int)
            for v in dfam_results["violations"]:
                key = (
                    v["rule"],
                    v["severity"],
                    v["message"],
                    round(v["value"], 2),
                    round(v["limit"], 2),
                )
                grouped[key] += 1

            # Table header
            violation_data = [
                ["Rule", "Severity", "Message", "Value", "Limit", "Count"]
            ]

            # Add each grouped entry
            for (rule, severity, message, value, limit), count in grouped.items():
                violation_data.append(
                    [
                        rule,
                        severity,
                        message,
                        f"{value:.2f}",
                        f"{limit:.2f}",
                        f"×{count}",
                    ]
                )

            violation_table = Table(violation_data)
            violation_table.setStyle(
                TableStyle(
                    [
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ]
                )
            )
            content.append(violation_table)
    else:
        content.append(
            Paragraph("No manufacturability analysis results available", normal_style)
        )

    # Build PDF
    doc.build(content)
    return pdf_file
