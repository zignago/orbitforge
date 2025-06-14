"""Protocol definition for all subsystem solvers."""

from typing import Protocol, Dict, Any
from ..design_record import DesignRecord


class BudgetSolver(Protocol):
    """Protocol for all subsystem budget solvers.

    This provides a uniform interface for AI agents to evaluate
    candidate designs across all subsystems.
    """

    def evaluate(self, design: DesignRecord, **kwargs) -> Dict[str, Any]:
        """Evaluate a design for this subsystem.

        Args:
            design: Complete design record with geometry and mission spec
            **kwargs: Solver-specific parameters and overrides

        Returns:
            Dictionary with solver results including:
            - status: "PASS" | "FAIL" | "WARNING"
            - margin: Safety margin (positive = good)
            - details: Solver-specific metrics
            - warnings: List of warning messages
        """
        ...

    async def evaluate_async(self, design: DesignRecord, **kwargs) -> Dict[str, Any]:
        """Async version of evaluate for concurrent execution."""
        ...

    @property
    def solver_name(self) -> str:
        """Human-readable name of this solver."""
        ...

    @property
    def solver_version(self) -> str:
        """Version string for reproducibility."""
        ...
