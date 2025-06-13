"""Design Record schema for tracking geometry/analysis pipeline runs.

This module provides the canonical Design Record structure that links inputs,
intermediate artifacts, solver settings, and results for each run.
"""

import uuid
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field
from loguru import logger


class ArtifactType(str, Enum):
    """Types of artifacts generated during pipeline runs."""

    STEP = "step"
    STL = "stl"
    MESH = "mesh"
    RESULTS = "results"
    REPORT = "report"
    MASS_BUDGET = "mass_budget"
    PHYSICS_CHECK = "physics_check"
    DFAM_CHECK = "dfam_check"


class DesignStatus(str, Enum):
    """Status of design record."""

    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"
    PENDING = "PENDING"


class ArtifactSource(str, Enum):
    MOCK = "mock"
    RUNTIME = "runtime"


class Artifact(BaseModel):
    """Represents a single artifact file."""

    type: ArtifactType
    path: str
    uri: Optional[str] = None
    hash_sha256: Optional[str] = None
    size_bytes: Optional[int] = None
    description: Optional[str] = None
    source: ArtifactSource = Field(
        default=ArtifactSource.RUNTIME,
        description="Source of the artifact - mock or runtime",
    )
    metadata: Optional[Dict[str, Any]] = None


class GeometryParams(BaseModel):
    """All stochastic/variable geometry parameters for replay."""

    rail_mm: float
    deck_mm: float
    material: str
    design_seed: Optional[int] = None
    # Add more parameters as the generator evolves
    additional_params: Dict[str, Any] = Field(default_factory=dict)


class AnalysisResults(BaseModel):
    """Comprehensive analysis results."""

    # Structural results
    max_stress_mpa: Optional[float] = None
    max_deflection_mm: Optional[float] = None
    factor_of_safety: Optional[float] = None
    mass_kg: Optional[float] = None

    # Thermal results
    thermal_stress_mpa: Optional[float] = None
    thermal_expansion_mm: Optional[float] = None
    max_temperature_k: Optional[float] = None

    # Manufacturing results
    manufacturability_score: Optional[float] = None
    dfam_violations: List[Dict[str, Any]] = Field(default_factory=list)

    # Additional results
    additional_results: Dict[str, Any] = Field(default_factory=dict)


class DesignRecord(BaseModel):
    """Canonical Design Record schema for tracking all pipeline runs."""

    # Core identification
    design_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    git_commit: Optional[str] = None

    # Input specification
    mission_spec: Dict[str, Any] = Field(
        description="Verbatim JSON spec the user supplied"
    )

    # Geometry parameters (for replay)
    geometry_params: GeometryParams

    # Generated artifacts
    artifacts: List[Artifact] = Field(default_factory=list)

    # Analysis results
    analysis_results: AnalysisResults = Field(default_factory=AnalysisResults)

    # Overall status
    status: DesignStatus = DesignStatus.PENDING

    # Execution metadata
    execution_time_seconds: Optional[float] = None
    error_message: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)

    def get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).parent.parent,
                text=True,
            ).strip()
            return commit
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"Could not determine git commit: {e}")
            return None

    def add_artifact(
        self,
        artifact_type: ArtifactType,
        path: Path,
        description: Optional[str] = None,
        uri: Optional[str] = None,
    ) -> None:
        """Add an artifact to the record."""
        import hashlib

        artifact_path = str(path)
        hash_sha256 = None
        size_bytes = None

        if path.exists():
            # Calculate file hash
            hasher = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            hash_sha256 = hasher.hexdigest()
            size_bytes = path.stat().st_size

        artifact = Artifact(
            type=artifact_type,
            path=artifact_path,
            uri=uri,
            hash_sha256=hash_sha256,
            size_bytes=size_bytes,
            description=description,
        )

        self.artifacts.append(artifact)
        logger.debug(f"Added artifact: {artifact_type} -> {path}")

    def update_analysis_results(self, **kwargs) -> None:
        """Update analysis results with new data."""
        for key, value in kwargs.items():
            if hasattr(self.analysis_results, key):
                setattr(self.analysis_results, key, value)
            else:
                self.analysis_results.additional_results[key] = value

    def set_status(
        self, status: DesignStatus, error_message: Optional[str] = None
    ) -> None:
        """Set the overall status of the design record."""
        self.status = status
        if error_message:
            self.error_message = error_message

    def add_warning(self, warning: str) -> None:
        """Add a warning message to the record."""
        self.warnings.append(warning)
        logger.warning(f"Design {self.design_id}: {warning}")

    def save_to_file(self, output_path: Path) -> None:
        """Save the design record to a JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write(self.model_dump_json(indent=2))

        logger.info(f"Design record saved to {output_path}")

    @classmethod
    def load_from_file(cls, file_path: Path) -> "DesignRecord":
        """Load a design record from a JSON file."""
        with open(file_path, "r") as f:
            data = f.read()
        return cls.model_validate_json(data)
