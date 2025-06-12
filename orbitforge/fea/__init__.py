"""
Fast physics validation package for Orbitforge.

This package provides rapid structural feedback for satellite frame designs
without requiring full FEA analysis.
"""

from .fast_check import (
    FastPhysicsValidator,
    MaterialProperties,
    LoadCase,
    ValidationResults,
)
from .solver import compute_stresses, solve_static, assemble_system
from .fe_uplift import FEAUpliftClient, FEAUpliftError
from .preprocessor import convert_step_to_bdf, MeshGenerator, BDFWriter
from .postprocessor import process_op2_results, OP2Parser, ReportGenerator

__all__ = [
    "FastPhysicsValidator",
    "MaterialProperties",
    "LoadCase",
    "ValidationResults",
    "compute_stresses",
    "solve_static",
    "assemble_system",
    "FEAUpliftClient",
    "FEAUpliftError",
    "convert_step_to_bdf",
    "MeshGenerator",
    "BDFWriter",
    "process_op2_results",
    "OP2Parser",
    "ReportGenerator",
]
