"""Unit tests for OP2 parsing functionality."""

import tempfile
import json
from pathlib import Path
import numpy as np
import pytest

from orbitforge.fea.postprocessor import OP2Parser, ReportGenerator, process_op2_results


class TestOP2Parser:
    """Test OP2 parsing functionality."""

    def test_parser_initialization(self):
        """Test OP2 parser initialization."""
        parser = OP2Parser()
        assert parser.stress_data == {}
        assert parser.displacement_data == {}
        assert parser.element_forces == {}

    def test_parse_nonexistent_file(self):
        """Test parsing of non-existent OP2 file."""
        parser = OP2Parser()
        with pytest.raises(RuntimeError, match="OP2 file not found"):
            parser.parse_op2_file(Path("nonexistent.op2"))

    def test_parse_dummy_results(self):
        """Test generation of dummy results."""
        parser = OP2Parser()

        # Create empty file to trigger dummy results
        with tempfile.TemporaryDirectory() as tmpdir:
            op2_file = Path(tmpdir) / "test.op2"
            op2_file.write_bytes(
                b"dummy binary data" * 10
            )  # Create file with some content

            results = parser.parse_op2_file(op2_file)

            # Check structure
            assert "max_stress_MPa" in results
            assert "max_displacement_mm" in results
            assert "elements_analyzed" in results
            assert "nodes_analyzed" in results
            assert "parse_status" in results
            assert "warnings" in results

            # Check types and ranges
            assert isinstance(results["max_stress_MPa"], float)
            assert isinstance(results["max_displacement_mm"], float)
            assert isinstance(results["elements_analyzed"], int)
            assert isinstance(results["nodes_analyzed"], int)

            # Check reasonable ranges for dummy data
            assert 0 < results["max_stress_MPa"] < 1000  # Reasonable stress range
            assert 0 < results["max_displacement_mm"] < 10  # Reasonable displacement
            assert results["elements_analyzed"] > 0
            assert results["nodes_analyzed"] > 0

    def test_generate_dummy_results_reproducible(self):
        """Test that dummy results are reproducible."""
        parser = OP2Parser()

        results1 = parser._generate_dummy_results()
        results2 = parser._generate_dummy_results()

        # Should be identical due to fixed random seed
        assert results1["max_stress_MPa"] == results2["max_stress_MPa"]
        assert results1["max_displacement_mm"] == results2["max_displacement_mm"]
        assert results1["elements_analyzed"] == results2["elements_analyzed"]
        assert results1["nodes_analyzed"] == results2["nodes_analyzed"]


class TestReportGenerator:
    """Test PDF and text report generation."""

    def test_report_generator_initialization(self):
        """Test report generator initialization."""
        generator = ReportGenerator()
        assert generator is not None

    def test_generate_text_report(self):
        """Test text report generation as fallback."""
        generator = ReportGenerator()

        analysis_results = {
            "max_stress_MPa": 125.5,
            "max_displacement_mm": 0.15,
            "elements_analyzed": 500,
            "nodes_analyzed": 750,
            "warnings": ["Test warning"],
        }

        design_info = {
            "design_id": "test_001",
            "material": "Aluminum6061",
            "rail_thickness_mm": 1.5,
            "yield_strength_MPa": 275.0,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            report_file = Path(tmpdir) / "test_report.txt"

            generator._generate_text_report(report_file, analysis_results, design_info)

            assert report_file.exists()

            content = report_file.read_text()

            # Check content structure
            assert "OrbitForge FEA Analysis Report" in content
            assert "Design Information:" in content
            assert "Analysis Results:" in content
            assert "Assessment:" in content
            assert "Max Stress: 125.5 MPa" in content
            assert "Max Displacement: 0.150 mm" in content
            assert "Elements Analyzed: 500" in content
            assert "Nodes Analyzed: 750" in content
            assert "Safety Factor:" in content
            assert "Warnings:" in content
            assert "Test warning" in content

    def test_report_status_assessment(self):
        """Test status assessment in reports."""
        generator = ReportGenerator()

        # Test PASS case
        analysis_results = {
            "max_stress_MPa": 100.0,  # Low stress
            "max_displacement_mm": 0.1,
            "elements_analyzed": 100,
            "nodes_analyzed": 150,
            "warnings": [],
        }

        design_info = {
            "design_id": "test_pass",
            "material": "Aluminum6061",
            "yield_strength_MPa": 275.0,  # High yield strength
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            report_file = Path(tmpdir) / "test_pass.txt"
            generator._generate_text_report(report_file, analysis_results, design_info)

            content = report_file.read_text()
            assert "Status: PASS" in content
            assert "Safety Factor: 2.75" in content

        # Test FAIL case
        analysis_results["max_stress_MPa"] = 300.0  # High stress

        with tempfile.TemporaryDirectory() as tmpdir:
            report_file = Path(tmpdir) / "test_fail.txt"
            generator._generate_text_report(report_file, analysis_results, design_info)

            content = report_file.read_text()
            assert "Status: FAIL" in content
            assert "Safety Factor: 0.92" in content

    @pytest.mark.skipif(
        not pytest.importorskip("reportlab"), reason="ReportLab not available"
    )
    def test_generate_pdf_report(self):
        """Test PDF report generation (if ReportLab is available)."""
        generator = ReportGenerator()

        analysis_results = {
            "max_stress_MPa": 150.0,
            "max_displacement_mm": 0.2,
            "elements_analyzed": 300,
            "nodes_analyzed": 450,
            "warnings": ["Sample warning"],
        }

        design_info = {
            "design_id": "test_pdf",
            "material": "Aluminum6061",
            "rail_thickness_mm": 2.0,
            "yield_strength_MPa": 275.0,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_file = Path(tmpdir) / "test_report.pdf"

            generator.generate_pdf_report(pdf_file, analysis_results, design_info)

            # Check file was created
            assert pdf_file.exists()
            assert pdf_file.stat().st_size > 0  # Non-empty file


class TestProcessOP2Results:
    """Test the main OP2 processing function."""

    def test_process_op2_results_complete(self):
        """Test complete OP2 processing workflow."""
        design_info = {
            "design_id": "test_design",
            "material": "Aluminum6061",
            "rail_thickness_mm": 1.5,
            "deck_thickness_mm": 2.0,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create dummy OP2 file
            op2_file = tmpdir / "frame.op2"
            op2_file.write_bytes(b"dummy nastran output" * 20)

            # Process results
            summary = process_op2_results(
                op2_file, tmpdir, design_info, material_yield_strength=275.0
            )

            # Check summary structure
            assert "fea_mode" in summary
            assert summary["fea_mode"] == "full"
            assert "max_stress_fea" in summary
            assert "status_fea" in summary
            assert "safety_factor" in summary
            assert "max_displacement_mm" in summary
            assert "elements_analyzed" in summary
            assert "nodes_analyzed" in summary
            assert "analysis_warnings" in summary
            assert "reports_generated" in summary

            # Check files were created
            assert (tmpdir / "summary.json").exists()
            # Either PDF or text report should exist
            assert (tmpdir / "stress_summary.pdf").exists() or (
                tmpdir / "stress_summary.txt"
            ).exists()

    def test_process_op2_results_pass_fail(self):
        """Test pass/fail determination based on safety factor."""
        design_info = {"design_id": "test", "material": "Al"}

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create dummy OP2 file
            op2_file = tmpdir / "frame.op2"
            op2_file.write_bytes(b"dummy" * 50)

            # Test with high yield strength (should pass)
            summary = process_op2_results(
                op2_file,
                tmpdir,
                design_info,
                material_yield_strength=1000.0,  # Very high yield strength
            )

            assert summary["status_fea"] == "PASS"
            assert summary["safety_factor"] > 1.2

    def test_process_op2_results_error_handling(self):
        """Test error handling in OP2 processing."""
        design_info = {"design_id": "test", "material": "Al"}

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Non-existent OP2 file
            op2_file = tmpdir / "nonexistent.op2"

            with pytest.raises(RuntimeError):
                process_op2_results(op2_file, tmpdir, design_info)


class TestStressAnalysis:
    """Test stress analysis and safety factor calculations."""

    def test_safety_factor_calculation(self):
        """Test safety factor calculations."""
        # Manual calculation test
        yield_strength = 275.0  # MPa
        max_stress = 137.5  # MPa
        expected_safety_factor = yield_strength / max_stress  # = 2.0

        design_info = {"yield_strength_MPa": yield_strength}

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create mock OP2 file
            op2_file = tmpdir / "test.op2"
            op2_file.write_bytes(b"test" * 100)

            # Mock the parser to return specific stress value
            from orbitforge.fea.postprocessor import OP2Parser

            original_generate_dummy = OP2Parser._generate_dummy_results

            def mock_dummy_results(self):
                return {
                    "max_stress_MPa": max_stress,
                    "max_displacement_mm": 0.1,
                    "elements_analyzed": 100,
                    "nodes_analyzed": 150,
                    "mean_stress_MPa": max_stress * 0.5,
                    "stress_std_MPa": max_stress * 0.1,
                }

            OP2Parser._generate_dummy_results = mock_dummy_results

            try:
                summary = process_op2_results(
                    op2_file,
                    tmpdir,
                    {"design_id": "test"},
                    material_yield_strength=yield_strength,
                )

                assert abs(summary["safety_factor"] - expected_safety_factor) < 0.01
                assert summary["status_fea"] == "PASS"  # SF = 2.0 > 1.2

            finally:
                # Restore original method
                OP2Parser._generate_dummy_results = original_generate_dummy

    def test_boundary_safety_factors(self):
        """Test boundary conditions for safety factor assessment."""
        test_cases = [
            (275.0, 229.0, "PASS"),  # SF = 1.201 (just above boundary)
            (275.0, 230.0, "FAIL"),  # SF = 1.196 (just below boundary)
            (275.0, 137.5, "PASS"),  # SF = 2.0 (well above boundary)
            (275.0, 275.0, "FAIL"),  # SF = 1.0 (exactly at yield)
            (275.0, 300.0, "FAIL"),  # SF = 0.92 (above yield)
        ]

        for yield_strength, max_stress, expected_status in test_cases:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)

                op2_file = tmpdir / "test.op2"
                op2_file.write_bytes(b"test" * 50)

                # Mock specific stress value
                from orbitforge.fea.postprocessor import OP2Parser

                original_generate_dummy = OP2Parser._generate_dummy_results

                def mock_specific_stress(self):
                    return {
                        "max_stress_MPa": max_stress,
                        "max_displacement_mm": 0.1,
                        "elements_analyzed": 100,
                        "nodes_analyzed": 150,
                        "mean_stress_MPa": max_stress * 0.5,
                        "stress_std_MPa": max_stress * 0.1,
                    }

                OP2Parser._generate_dummy_results = mock_specific_stress

                try:
                    summary = process_op2_results(
                        op2_file,
                        tmpdir,
                        {"design_id": f"test_{max_stress}"},
                        material_yield_strength=yield_strength,
                    )

                    assert (
                        summary["status_fea"] == expected_status
                    ), f"Expected {expected_status} for SF={yield_strength/max_stress:.3f}"

                finally:
                    OP2Parser._generate_dummy_results = original_generate_dummy
