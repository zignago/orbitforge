from pydantic import BaseModel, Field
from enum import Enum
from pathlib import Path
import yaml
from typing import Dict, Any
from loguru import logger


class Material(str, Enum):
    AL_6061_T6 = "Al_6061_T6"  # Standard aluminum alloy
    TI_6AL_4V = "Ti_6Al_4V"  # Titanium alloy for AM


# Load material properties from YAML
def load_materials():
    materials_file = Path(__file__).parent / "materials.yaml"
    if not materials_file.exists():
        # Create default materials file if it doesn't exist
        materials = {
            "Al_6061_T6": {
                "density_kg_m3": 2700,
                "yield_mpa": 276,
                "youngs_modulus_gpa": 68.9,
                "poissons_ratio": 0.33,
                "cte_per_k": 23.6e-6,
            },
            "Ti_6Al_4V": {
                "density_kg_m3": 4430,
                "yield_mpa": 880,
                "youngs_modulus_gpa": 113.8,
                "poissons_ratio": 0.342,
                "cte_per_k": 8.6e-6,
            },
        }
        materials_file.write_text(yaml.safe_dump(materials))

    logger.debug(f"Loading materials from {materials_file}")
    materials = yaml.safe_load(materials_file.read_text())
    logger.debug(f"Loaded materials: {materials}")
    return materials


class MissionSpec(BaseModel):
    # Mission parameters
    bus_u: int = Field(ge=1, le=12, description="CubeSat units (1U = 10×10×10 cm)")
    payload_mass_kg: float = Field(gt=0, description="Mass of the payload in kg")
    orbit_alt_km: float = Field(gt=0, description="Orbital altitude in km")
    mass_limit_kg: float = Field(12.0, gt=0, description="Maximum allowed mass in kg")

    # Structural parameters (all in mm)
    rail_mm: float = Field(
        3.0, gt=0, description="Thickness of the CubeSat rails in mm"
    )
    deck_mm: float = Field(2.5, gt=0, description="Thickness of top/bottom decks in mm")
    material: Material = Field(Material.AL_6061_T6, description="Material selection")

    # Material properties
    @property
    def material_properties(self) -> Dict[str, Any]:
        """Return all material properties."""
        logger.debug(f"Getting properties for material: {self.material}")
        props = load_materials()[self.material]
        logger.debug(f"Material properties: {props}")
        return props

    @property
    def density_kg_m3(self) -> float:
        """Return material density in kg/m³."""
        return self.material_properties["density_kg_m3"]

    @property
    def density_kg_mm3(self) -> float:
        """Return material density in kg/mm³ for convenience."""
        return self.density_kg_m3 * 1e-9  # Convert from kg/m³ to kg/mm³

    @property
    def yield_mpa(self) -> float:
        """Return material yield strength in MPa."""
        return self.material_properties["yield_mpa"]

    @property
    def youngs_modulus_gpa(self) -> float:
        """Return Young's modulus in GPa."""
        return self.material_properties["youngs_modulus_gpa"]

    @property
    def poissons_ratio(self) -> float:
        """Return Poisson's ratio."""
        return self.material_properties["poissons_ratio"]

    @property
    def cte_per_k(self) -> float:
        """Return coefficient of thermal expansion in 1/K."""
        return self.material_properties["cte_per_k"]
