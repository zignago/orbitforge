"""
Configuration settings for design generation and validation.

This module contains all configurable parameters for the multi-design generator,
including parameter ranges, validation thresholds, and resource limits.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ParameterRanges:
    """Parameter variation ranges for design generation."""

    # Thickness variations (mm)
    MIN_RAIL_THICKNESS: float = 1.0
    MAX_RAIL_THICKNESS: float = 5.0
    RAIL_VARIATION_MM: float = 0.5

    MIN_DECK_THICKNESS: float = 1.0
    MAX_DECK_THICKNESS: float = 4.0
    DECK_VARIATION_MM: float = 0.3

    # Material selection
    MATERIAL_CHANGE_PROBABILITY: float = 0.3

    def __post_init__(self):
        """Validate parameter ranges."""
        if self.MIN_RAIL_THICKNESS <= 0:
            raise ValueError("MIN_RAIL_THICKNESS must be positive")
        if self.MAX_RAIL_THICKNESS <= self.MIN_RAIL_THICKNESS:
            raise ValueError(
                "MAX_RAIL_THICKNESS must be greater than MIN_RAIL_THICKNESS"
            )
        if self.RAIL_VARIATION_MM <= 0:
            raise ValueError("RAIL_VARIATION_MM must be positive")

        if self.MIN_DECK_THICKNESS <= 0:
            raise ValueError("MIN_DECK_THICKNESS must be positive")
        if self.MAX_DECK_THICKNESS <= self.MIN_DECK_THICKNESS:
            raise ValueError(
                "MAX_DECK_THICKNESS must be greater than MIN_DECK_THICKNESS"
            )
        if self.DECK_VARIATION_MM <= 0:
            raise ValueError("DECK_VARIATION_MM must be positive")

        if not 0 <= self.MATERIAL_CHANGE_PROBABILITY <= 1:
            raise ValueError("MATERIAL_CHANGE_PROBABILITY must be between 0 and 1")


@dataclass
class ResourceLimits:
    """Resource limits for parallel processing and caching."""

    MAX_PARALLEL_FEA: int = 4  # Maximum parallel FEA processes
    MAX_CACHE_SIZE_MB: int = 1000  # Maximum cache size in MB
    CACHE_EXPIRY_HOURS: int = 24  # Cache expiry time in hours

    def __post_init__(self):
        """Validate resource limits."""
        if self.MAX_PARALLEL_FEA <= 0:
            raise ValueError("MAX_PARALLEL_FEA must be positive")
        if self.MAX_CACHE_SIZE_MB <= 0:
            raise ValueError("MAX_CACHE_SIZE_MB must be positive")
        if self.CACHE_EXPIRY_HOURS <= 0:
            raise ValueError("CACHE_EXPIRY_HOURS must be positive")


@dataclass
class ValidationThresholds:
    """Thresholds for design validation."""

    MAX_MASS_MARGIN: float = 0.95  # Maximum allowed mass as fraction of limit
    MIN_SAFETY_FACTOR: float = 1.5  # Minimum safety factor for stress
    MAX_DEFLECTION_MM: float = 0.5  # Maximum allowed deflection

    def __post_init__(self):
        """Validate thresholds."""
        if not 0 < self.MAX_MASS_MARGIN <= 1:
            raise ValueError("MAX_MASS_MARGIN must be between 0 and 1")
        if self.MIN_SAFETY_FACTOR <= 1:
            raise ValueError("MIN_SAFETY_FACTOR must be greater than 1")
        if self.MAX_DEFLECTION_MM <= 0:
            raise ValueError("MAX_DEFLECTION_MM must be positive")


class DesignConfig:
    """Global configuration for design generation."""

    def __init__(self):
        self.params = ParameterRanges()
        self.resources = ResourceLimits()
        self.thresholds = ValidationThresholds()

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "parameter_ranges": self.params.__dict__,
            "resource_limits": self.resources.__dict__,
            "validation_thresholds": self.thresholds.__dict__,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "DesignConfig":
        """Create configuration from dictionary."""
        instance = cls()

        # Update and validate parameter ranges
        if "parameter_ranges" in config_dict:
            new_params = ParameterRanges()
            for k, v in config_dict["parameter_ranges"].items():
                if hasattr(new_params, k):
                    setattr(new_params, k, v)
            new_params.__post_init__()  # Validate
            instance.params = new_params

        # Update and validate resource limits
        if "resource_limits" in config_dict:
            new_resources = ResourceLimits()
            for k, v in config_dict["resource_limits"].items():
                if hasattr(new_resources, k):
                    setattr(new_resources, k, v)
            new_resources.__post_init__()  # Validate
            instance.resources = new_resources

        # Update and validate thresholds
        if "validation_thresholds" in config_dict:
            new_thresholds = ValidationThresholds()
            for k, v in config_dict["validation_thresholds"].items():
                if hasattr(new_thresholds, k):
                    setattr(new_thresholds, k, v)
            new_thresholds.__post_init__()  # Validate
            instance.thresholds = new_thresholds

        return instance

    def validate(self):
        """Validate entire configuration."""
        self.params.__post_init__()
        self.resources.__post_init__()
        self.thresholds.__post_init__()


# Default configuration instance
config = DesignConfig()
config.validate()  # Ensure default configuration is valid
