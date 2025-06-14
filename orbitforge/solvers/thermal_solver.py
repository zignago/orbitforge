"""Thermal analysis solver for temperature predictions."""

import asyncio
import math
from typing import Dict, Any, List, Tuple
from loguru import logger

from ..design_record import DesignRecord


class ThermalSolver:
    """Thermal analysis solver for CubeSat temperature predictions.

    Provides simplified thermal modeling for component temperature
    estimation and thermal balance analysis.
    """

    def __init__(self):
        self._solver_name = "ThermalSolver"
        self._solver_version = "1.0.0"

        # Thermal constants
        self.STEFAN_BOLTZMANN = 5.67e-8  # W/m²/K⁴
        self.SOLAR_CONSTANT = 1361  # W/m²
        self.SPACE_TEMP_K = 4  # Deep space background temperature

        # Material properties (simplified)
        self.materials = {
            "Al_6061_T6": {
                "thermal_conductivity": 167,  # W/m/K
                "specific_heat": 896,  # J/kg/K
                "density": 2700,  # kg/m³
                "emissivity": 0.05,  # Bare aluminum
                "absorptivity": 0.15,
            },
            "Ti_6Al_4V": {
                "thermal_conductivity": 7.2,
                "specific_heat": 526,
                "density": 4430,
                "emissivity": 0.35,
                "absorptivity": 0.45,
            },
            "Carbon_Fiber": {
                "thermal_conductivity": 1.0,
                "specific_heat": 800,
                "density": 1600,
                "emissivity": 0.85,
                "absorptivity": 0.90,
            },
        }

        # Component thermal characteristics
        self.components = {
            "electronics": {
                "power_density_w_per_kg": 50,
                "max_temp_c": 85,
                "min_temp_c": -40,
            },
            "battery": {
                "power_density_w_per_kg": 5,
                "max_temp_c": 60,
                "min_temp_c": -20,
            },
            "solar_panels": {
                "power_density_w_per_kg": 0,
                "max_temp_c": 120,
                "min_temp_c": -150,
            },
            "structure": {
                "power_density_w_per_kg": 0,
                "max_temp_c": 150,
                "min_temp_c": -180,
            },
        }

    def evaluate(self, design: DesignRecord, **kwargs) -> Dict[str, Any]:
        """Evaluate thermal performance of the design."""
        try:
            mission_spec = design.mission_spec
            geometry_params = design.geometry_params

            orbit_alt_km = mission_spec.get("orbit_alt_km", 550)
            bus_u = mission_spec.get("bus_u", 3)
            material = geometry_params.material

            # Extract thermal parameters
            internal_power_w = kwargs.get(
                "internal_power_w", 3.5
            )  # Electronics + payload
            eclipse_fraction = kwargs.get("eclipse_fraction", 0.35)

            # Calculate thermal environment
            thermal_env = self._analyze_thermal_environment(
                orbit_alt_km, eclipse_fraction, kwargs
            )

            # Calculate heat loads
            heat_loads = self._analyze_heat_loads(
                bus_u, material, internal_power_w, thermal_env, kwargs
            )

            # Calculate component temperatures
            temp_analysis = self._analyze_component_temperatures(
                heat_loads, material, bus_u, kwargs
            )

            # Determine thermal status
            status = "PASS"
            warnings = []
            margin = 1.0

            # Check temperature limits
            for component, temps in temp_analysis["component_temps"].items():
                comp_specs = self.components.get(
                    component, self.components["electronics"]
                )

                if temps["max_temp_c"] > comp_specs["max_temp_c"]:
                    status = "FAIL"
                    warnings.append(
                        f"{component} overheating: {temps['max_temp_c']:.1f}°C > {comp_specs['max_temp_c']}°C"
                    )
                    margin = 0.0
                elif temps["max_temp_c"] > comp_specs["max_temp_c"] - 10:
                    status = "WARNING"
                    warnings.append(
                        f"{component} near temperature limit: {temps['max_temp_c']:.1f}°C"
                    )
                    margin = min(
                        margin, (comp_specs["max_temp_c"] - temps["max_temp_c"]) / 10
                    )

                if temps["min_temp_c"] < comp_specs["min_temp_c"]:
                    status = "FAIL"
                    warnings.append(
                        f"{component} too cold: {temps['min_temp_c']:.1f}°C < {comp_specs['min_temp_c']}°C"
                    )
                    margin = 0.0
                elif temps["min_temp_c"] < comp_specs["min_temp_c"] + 10:
                    status = "WARNING"
                    warnings.append(
                        f"{component} near cold limit: {temps['min_temp_c']:.1f}°C"
                    )
                    margin = min(
                        margin, (temps["min_temp_c"] - comp_specs["min_temp_c"]) / 10
                    )

            return {
                "status": status,
                "margin": margin,
                "warnings": warnings,
                "details": {
                    "thermal_environment": thermal_env,
                    "heat_loads": heat_loads,
                    "temperature_analysis": temp_analysis,
                    "max_component_temp_c": temp_analysis["max_temp_c"],
                    "min_component_temp_c": temp_analysis["min_temp_c"],
                    "thermal_gradient_c": temp_analysis["thermal_gradient_c"],
                },
            }

        except Exception as e:
            logger.error(f"ThermalSolver evaluation failed: {e}")
            return {
                "status": "ERROR",
                "margin": 0.0,
                "warnings": [f"Solver error: {str(e)}"],
                "details": {},
            }

    async def evaluate_async(self, design: DesignRecord, **kwargs) -> Dict[str, Any]:
        """Async version of evaluate."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.evaluate, design, **kwargs
        )

    def _analyze_thermal_environment(
        self, orbit_alt_km: float, eclipse_fraction: float, kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze the thermal environment in orbit."""
        # Solar flux (simplified - no orbital eccentricity)
        solar_flux = self.SOLAR_CONSTANT

        # Earth infrared radiation (simplified)
        earth_temp_k = kwargs.get(
            "earth_avg_temp_k", 255
        )  # Earth's effective temperature
        earth_radius_km = 6371
        orbit_radius_km = earth_radius_km + orbit_alt_km

        # View factor to Earth
        earth_view_factor = 0.5 * (
            1 - math.sqrt(1 - (earth_radius_km / orbit_radius_km) ** 2)
        )

        # Earth IR flux
        earth_ir_flux = self.STEFAN_BOLTZMANN * earth_temp_k**4 * earth_view_factor

        # Albedo flux (reflected sunlight from Earth)
        earth_albedo = kwargs.get("earth_albedo", 0.3)
        albedo_flux = solar_flux * earth_albedo * earth_view_factor

        return {
            "solar_flux_w_per_m2": solar_flux,
            "earth_ir_flux_w_per_m2": earth_ir_flux,
            "albedo_flux_w_per_m2": albedo_flux,
            "eclipse_fraction": eclipse_fraction,
            "earth_view_factor": earth_view_factor,
            "orbit_alt_km": orbit_alt_km,
        }

    def _analyze_heat_loads(
        self,
        bus_u: int,
        material: str,
        internal_power_w: float,
        thermal_env: Dict[str, Any],
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Analyze heat loads on the CubeSat."""
        # Get material properties
        mat_props = self.materials.get(material, self.materials["Al_6061_T6"])

        # Calculate surface area (simplified cube)
        cube_side_m = 0.1 * bus_u  # 10cm per U
        surface_area_m2 = 6 * cube_side_m**2  # 6 faces

        # Solar heating (when in sunlight)
        solar_area_m2 = cube_side_m**2  # One face facing sun
        solar_heating_w = (
            thermal_env["solar_flux_w_per_m2"]
            * solar_area_m2
            * mat_props["absorptivity"]
        )

        # Earth IR heating
        earth_ir_heating_w = (
            thermal_env["earth_ir_flux_w_per_m2"]
            * surface_area_m2
            * mat_props["absorptivity"]
        )

        # Albedo heating
        albedo_heating_w = (
            thermal_env["albedo_flux_w_per_m2"]
            * surface_area_m2
            * mat_props["absorptivity"]
        )

        # Internal heat generation
        internal_heating_w = internal_power_w

        # Radiative cooling to space
        avg_temp_k = kwargs.get("estimated_avg_temp_k", 273)  # Initial estimate
        radiative_cooling_w = (
            self.STEFAN_BOLTZMANN
            * mat_props["emissivity"]
            * surface_area_m2
            * avg_temp_k**4
        )

        # Heat loads during sunlight and eclipse
        sunlight_heat_in_w = (
            solar_heating_w + earth_ir_heating_w + albedo_heating_w + internal_heating_w
        )
        eclipse_heat_in_w = earth_ir_heating_w + albedo_heating_w + internal_heating_w

        return {
            "surface_area_m2": surface_area_m2,
            "solar_heating_w": solar_heating_w,
            "earth_ir_heating_w": earth_ir_heating_w,
            "albedo_heating_w": albedo_heating_w,
            "internal_heating_w": internal_heating_w,
            "radiative_cooling_w": radiative_cooling_w,
            "sunlight_heat_in_w": sunlight_heat_in_w,
            "eclipse_heat_in_w": eclipse_heat_in_w,
            "material": material,
        }

    def _analyze_component_temperatures(
        self,
        heat_loads: Dict[str, Any],
        material: str,
        bus_u: int,
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Analyze temperatures of individual components."""
        # Get material properties
        mat_props = self.materials.get(material, self.materials["Al_6061_T6"])

        # Calculate component masses (simplified)
        total_mass_kg = bus_u * 1.3  # Typical mass per U
        mass_fractions = {
            "electronics": 0.3,
            "battery": 0.3,
            "solar_panels": 0.2,
            "structure": 0.2,
        }

        # Calculate heat distribution
        internal_power = heat_loads["internal_heating_w"]
        component_temps = {}
        max_temp_c = -float("inf")
        min_temp_c = float("inf")

        for component, fraction in mass_fractions.items():
            mass = total_mass_kg * fraction
            power_density = self.components[component]["power_density_w_per_kg"]

            # Component heat generation
            component_power = mass * power_density
            if component == "electronics":
                # Electronics get most of the internal power
                component_power += internal_power * 0.8
            elif component == "battery":
                # Battery gets some waste heat
                component_power += internal_power * 0.1

            # Calculate equilibrium temperature
            surface_area = heat_loads["surface_area_m2"] * fraction
            total_heat_in = (
                component_power
                + heat_loads["solar_heating_w"] * fraction
                + heat_loads["earth_ir_heating_w"] * fraction
                + heat_loads["albedo_heating_w"] * fraction
            )

            # Hot case (full sun)
            hot_temp_k = self._solve_equilibrium_temp(
                total_heat_in, surface_area, mat_props["emissivity"]
            )

            # Cold case (eclipse)
            cold_heat_in = component_power + heat_loads["earth_ir_heating_w"] * fraction
            cold_temp_k = self._solve_equilibrium_temp(
                cold_heat_in, surface_area, mat_props["emissivity"]
            )

            # Convert to Celsius
            hot_temp_c = hot_temp_k - 273.15
            cold_temp_c = cold_temp_k - 273.15

            component_temps[component] = {
                "max_temp_c": hot_temp_c,
                "min_temp_c": cold_temp_c,
                "avg_temp_c": (hot_temp_c + cold_temp_c) / 2,
                "power_w": component_power,
            }

            max_temp_c = max(max_temp_c, hot_temp_c)
            min_temp_c = min(min_temp_c, cold_temp_c)

        thermal_gradient_c = max_temp_c - min_temp_c

        return {
            "component_temps": component_temps,
            "max_temp_c": max_temp_c,
            "min_temp_c": min_temp_c,
            "thermal_gradient_c": thermal_gradient_c,
            "material": material,
        }

    def _solve_equilibrium_temp(
        self, heat_in_w: float, surface_area_m2: float, emissivity: float
    ) -> float:
        """Solve for equilibrium temperature using Stefan-Boltzmann law."""
        # Heat_in = Heat_out
        # Heat_in = σ * ε * A * T⁴
        # T = (Heat_in / (σ * ε * A))^(1/4)

        if heat_in_w <= 0:
            return self.SPACE_TEMP_K

        temp_k = (
            heat_in_w / (self.STEFAN_BOLTZMANN * emissivity * surface_area_m2)
        ) ** (1 / 4)

        # Ensure reasonable bounds
        return max(self.SPACE_TEMP_K, min(temp_k, 400))  # 4K to 400K

    @property
    def solver_name(self) -> str:
        return self._solver_name

    @property
    def solver_version(self) -> str:
        return self._solver_version
