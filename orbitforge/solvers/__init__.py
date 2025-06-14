"""Solver interfaces for OrbitForge subsystem analysis.

This module provides a unified interface for all subsystem solvers
to support AI-driven design optimization.
"""

from .protocol import BudgetSolver
from .orbit_solver import OrbitSolver
from .propulsion_solver import PropulsionSolver
from .power_solver import PowerSolver
from .rf_solver import RFSolver
from .thermal_solver import ThermalSolver
from .solver_registry import SolverRegistry, get_solver_registry

__all__ = [
    "BudgetSolver",
    "OrbitSolver",
    "PropulsionSolver",
    "PowerSolver",
    "RFSolver",
    "ThermalSolver",
    "SolverRegistry",
    "get_solver_registry",
]
