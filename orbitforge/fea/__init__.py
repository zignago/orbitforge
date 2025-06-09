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

__all__ = [
    "FastPhysicsValidator",
    "MaterialProperties",
    "LoadCase",
    "ValidationResults",
    "compute_stresses",
    "solve_static",
    "assemble_system",
]
