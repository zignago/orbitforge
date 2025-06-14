"""Solver registry for managing and orchestrating all subsystem solvers."""

import asyncio
from typing import Dict, Any, List, Optional, Type
from loguru import logger

from .protocol import BudgetSolver
from .orbit_solver import OrbitSolver
from .propulsion_solver import PropulsionSolver
from .power_solver import PowerSolver
from .rf_solver import RFSolver
from .thermal_solver import ThermalSolver
from ..design_record import DesignRecord


class SolverRegistry:
    """Registry for managing all subsystem solvers.

    Provides unified interface for running single or multiple solvers
    with support for async execution and result aggregation.
    """

    def __init__(self):
        self._solvers: Dict[str, BudgetSolver] = {}
        self._register_default_solvers()

    def _register_default_solvers(self):
        """Register all default solvers."""
        self.register_solver("orbit", OrbitSolver())
        self.register_solver("propulsion", PropulsionSolver())
        self.register_solver("power", PowerSolver())
        self.register_solver("rf", RFSolver())
        self.register_solver("thermal", ThermalSolver())

    def register_solver(self, name: str, solver: BudgetSolver):
        """Register a solver with the registry."""
        self._solvers[name] = solver
        logger.debug(
            f"Registered solver: {name} ({solver.solver_name} v{solver.solver_version})"
        )

    def get_solver(self, name: str) -> Optional[BudgetSolver]:
        """Get a solver by name."""
        return self._solvers.get(name)

    def list_solvers(self) -> List[str]:
        """List all registered solver names."""
        return list(self._solvers.keys())

    def evaluate_single(
        self, solver_name: str, design: DesignRecord, **kwargs
    ) -> Dict[str, Any]:
        """Evaluate a single solver."""
        solver = self.get_solver(solver_name)
        if not solver:
            raise ValueError(f"Solver '{solver_name}' not found")

        try:
            result = solver.evaluate(design, **kwargs)
            return {
                "solver": solver_name,
                "solver_version": solver.solver_version,
                "result": result,
            }
        except Exception as e:
            logger.error(f"Solver {solver_name} failed: {e}")
            return {
                "solver": solver_name,
                "solver_version": getattr(solver, "solver_version", "unknown"),
                "result": {
                    "status": "ERROR",
                    "margin": 0.0,
                    "warnings": [f"Solver execution failed: {str(e)}"],
                    "details": {},
                },
            }

    def evaluate_all(
        self,
        design: DesignRecord,
        solver_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Evaluate all registered solvers."""
        if solver_configs is None:
            solver_configs = {}

        results = {}
        overall_status = "PASS"
        overall_margin = 1.0
        all_warnings = []

        for solver_name in self._solvers.keys():
            solver_kwargs = solver_configs.get(solver_name, {})
            result = self.evaluate_single(solver_name, design, **solver_kwargs)
            results[solver_name] = result

            # Aggregate overall status
            solver_result = result["result"]
            if solver_result["status"] == "FAIL":
                overall_status = "FAIL"
                overall_margin = 0.0
            elif solver_result["status"] == "ERROR" and overall_status != "FAIL":
                overall_status = "ERROR"
                overall_margin = 0.0
            elif solver_result["status"] == "WARNING" and overall_status == "PASS":
                overall_status = "WARNING"

            # Aggregate margin (minimum of all solvers)
            if overall_status != "FAIL" and overall_status != "ERROR":
                overall_margin = min(overall_margin, solver_result.get("margin", 1.0))

            # Collect warnings
            all_warnings.extend(solver_result.get("warnings", []))

        return {
            "overall_status": overall_status,
            "overall_margin": overall_margin,
            "warnings": all_warnings,
            "solver_results": results,
            "design_id": design.design_id,
            "timestamp": design.timestamp,
        }

    async def evaluate_all_async(
        self,
        design: DesignRecord,
        solver_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Evaluate all solvers asynchronously for faster execution."""
        if solver_configs is None:
            solver_configs = {}

        # Create async tasks for all solvers
        tasks = []
        solver_names = []

        for solver_name, solver in self._solvers.items():
            solver_kwargs = solver_configs.get(solver_name, {})
            task = asyncio.create_task(
                self._evaluate_solver_async(solver_name, solver, design, solver_kwargs)
            )
            tasks.append(task)
            solver_names.append(solver_name)

        # Wait for all tasks to complete
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        results = {}
        overall_status = "PASS"
        overall_margin = 1.0
        all_warnings = []

        for solver_name, result in zip(solver_names, results_list):
            if isinstance(result, Exception):
                # Handle exceptions
                logger.error(f"Async solver {solver_name} failed: {result}")
                results[solver_name] = {
                    "solver": solver_name,
                    "solver_version": "unknown",
                    "result": {
                        "status": "ERROR",
                        "margin": 0.0,
                        "warnings": [f"Async execution failed: {str(result)}"],
                        "details": {},
                    },
                }
            else:
                results[solver_name] = result

            # Aggregate overall status
            solver_result = results[solver_name]["result"]
            if solver_result["status"] == "FAIL":
                overall_status = "FAIL"
                overall_margin = 0.0
            elif solver_result["status"] == "ERROR" and overall_status != "FAIL":
                overall_status = "ERROR"
                overall_margin = 0.0
            elif solver_result["status"] == "WARNING" and overall_status == "PASS":
                overall_status = "WARNING"

            # Aggregate margin
            if overall_status != "FAIL" and overall_status != "ERROR":
                overall_margin = min(overall_margin, solver_result.get("margin", 1.0))

            # Collect warnings
            all_warnings.extend(solver_result.get("warnings", []))

        return {
            "overall_status": overall_status,
            "overall_margin": overall_margin,
            "warnings": all_warnings,
            "solver_results": results,
            "design_id": design.design_id,
            "timestamp": design.timestamp,
            "execution_mode": "async",
        }

    async def _evaluate_solver_async(
        self,
        solver_name: str,
        solver: BudgetSolver,
        design: DesignRecord,
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Evaluate a single solver asynchronously."""
        try:
            result = await solver.evaluate_async(design, **kwargs)
            return {
                "solver": solver_name,
                "solver_version": solver.solver_version,
                "result": result,
            }
        except Exception as e:
            logger.error(f"Async solver {solver_name} failed: {e}")
            return {
                "solver": solver_name,
                "solver_version": getattr(solver, "solver_version", "unknown"),
                "result": {
                    "status": "ERROR",
                    "margin": 0.0,
                    "warnings": [f"Async execution failed: {str(e)}"],
                    "details": {},
                },
            }

    def get_solver_info(self) -> Dict[str, Dict[str, str]]:
        """Get information about all registered solvers."""
        info = {}
        for name, solver in self._solvers.items():
            info[name] = {
                "name": solver.solver_name,
                "version": solver.solver_version,
                "class": solver.__class__.__name__,
            }
        return info


# Global registry instance
_registry_instance: Optional[SolverRegistry] = None


def get_solver_registry() -> SolverRegistry:
    """Get the global solver registry instance."""
    global _registry_instance

    if _registry_instance is None:
        _registry_instance = SolverRegistry()

    return _registry_instance
