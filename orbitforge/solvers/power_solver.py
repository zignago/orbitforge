"""Power budget solver for solar generation and battery sizing."""

import asyncio
import math
from typing import Dict, Any, List, Tuple
from loguru import logger

from ..design_record import DesignRecord


class PowerSolver:
    """Power budget solver for CubeSat electrical systems.

    Calculates solar power generation, battery requirements,
    and power margin analysis.
    """

    def __init__(self):
        self._solver_name = "PowerSolver"
        self._solver_version = "1.0.0"

        # Solar panel specifications (per U)
        self.solar_panel_specs = {
            "efficiency": 0.30,  # 30% efficient cells
            "area_per_u_m2": 0.01,  # 100 cm² per U face
            "degradation_per_year": 0.025,  # 2.5% per year
            "temperature_coefficient": -0.004,  # per °C
            "packing_factor": 0.85,  # 85% of face covered
        }

        # Battery specifications
        self.battery_specs = {
            "energy_density_wh_per_kg": 150,  # Li-ion typical
            "depth_of_discharge": 0.8,  # 80% DOD
            "charge_efficiency": 0.95,  # 95% charge efficiency
            "discharge_efficiency": 0.95,  # 95% discharge efficiency
            "cycle_life": 3000,  # Cycles at 80% DOD
            "temperature_derating": 0.7,  # Cold space operation
        }

        # Solar constants
        self.SOLAR_CONSTANT = 1361  # W/m² at 1 AU
        self.EARTH_ALBEDO = 0.3

    def evaluate(self, design: DesignRecord, **kwargs) -> Dict[str, Any]:
        """Evaluate power generation and storage requirements."""
        try:
            mission_spec = design.mission_spec
            bus_u = mission_spec.get("bus_u", 3)
            orbit_alt_km = mission_spec.get("orbit_alt_km", 550)

            # Extract power requirements
            payload_power_w = kwargs.get("payload_power_w", 2.0)
            bus_power_w = kwargs.get("bus_power_w", 1.5)
            peak_power_w = kwargs.get("peak_power_w", payload_power_w + bus_power_w)
            duty_cycle = kwargs.get("duty_cycle", 0.3)  # 30% active time

            # Mission parameters
            mission_duration_years = kwargs.get("mission_duration_years", 2)
            eclipse_fraction = kwargs.get("eclipse_fraction", 0.35)  # From orbit solver

            # Calculate solar power generation
            solar_analysis = self._analyze_solar_generation(
                bus_u, orbit_alt_km, mission_duration_years, kwargs
            )

            # Calculate power consumption
            consumption_analysis = self._analyze_power_consumption(
                payload_power_w, bus_power_w, peak_power_w, duty_cycle
            )

            # Calculate battery requirements
            battery_analysis = self._analyze_battery_requirements(
                consumption_analysis["avg_power_w"],
                eclipse_fraction,
                orbit_alt_km,
                kwargs,
            )

            # Calculate power margin
            power_margin = self._calculate_power_margin(
                solar_analysis, consumption_analysis, battery_analysis
            )

            # Determine status
            status = "PASS"
            warnings = []
            margin = power_margin["margin_ratio"]

            if power_margin["margin_ratio"] < 0:
                status = "FAIL"
                warnings.append(
                    f"Insufficient power: {power_margin['deficit_w']:.1f}W deficit"
                )
                margin = 0.0
            elif power_margin["margin_ratio"] < 0.2:
                status = "WARNING"
                warnings.append(
                    f"Low power margin: {power_margin['margin_ratio']*100:.1f}%"
                )

            if (
                battery_analysis["mass_kg"]
                > mission_spec.get("mass_limit_kg", 4.0) * 0.3
            ):
                status = "WARNING"
                warnings.append("Battery mass exceeds 30% of total mass budget")
                margin = min(margin, 0.5)

            return {
                "status": status,
                "margin": margin,
                "warnings": warnings,
                "details": {
                    "solar_analysis": solar_analysis,
                    "consumption_analysis": consumption_analysis,
                    "battery_analysis": battery_analysis,
                    "power_margin": power_margin,
                    "avg_power_margin_w": power_margin["margin_w"],
                    "eclipse_duration_min": battery_analysis["eclipse_duration_min"],
                },
            }

        except Exception as e:
            logger.error(f"PowerSolver evaluation failed: {e}")
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

    def _analyze_solar_generation(
        self,
        bus_u: int,
        orbit_alt_km: float,
        mission_duration_years: float,
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Analyze solar power generation capability."""
        # Calculate available solar panel area
        deployable_panels = kwargs.get("deployable_panels", False)
        if deployable_panels:
            # Deployable panels can use more area (2 side panels)
            panel_area_m2 = (
                bus_u * self.solar_panel_specs["area_per_u_m2"] * 3
            )  # 1 face + 2 deployable
        else:
            # Body-mounted panels only
            panel_area_m2 = (
                bus_u * self.solar_panel_specs["area_per_u_m2"] * 1.5
            )  # Average sun-facing area

        panel_area_m2 *= self.solar_panel_specs["packing_factor"]

        # Calculate solar flux at orbit
        solar_flux = self.SOLAR_CONSTANT  # Assume LEO, no distance correction needed

        # Account for orbital geometry (more realistic)
        geometric_efficiency = kwargs.get(
            "geometric_efficiency", 0.5
        )  # 50% avg due to angles and eclipse

        # Calculate power generation
        initial_power_w = (
            panel_area_m2
            * solar_flux
            * self.solar_panel_specs["efficiency"]
            * geometric_efficiency
        )

        # Account for degradation over mission life
        degradation_factor = (
            1 - self.solar_panel_specs["degradation_per_year"]
        ) ** mission_duration_years
        avg_power_w = initial_power_w * degradation_factor

        # Account for temperature effects (simplified)
        temp_derating = kwargs.get(
            "temperature_derating", 0.9
        )  # 10% loss due to temperature
        avg_power_w *= temp_derating

        return {
            "panel_area_m2": panel_area_m2,
            "initial_power_w": initial_power_w,
            "avg_power_w": avg_power_w,
            "degradation_factor": degradation_factor,
            "geometric_efficiency": geometric_efficiency,
            "deployable_panels": deployable_panels,
            "bus_u": bus_u,
        }

    def _analyze_power_consumption(
        self,
        payload_power_w: float,
        bus_power_w: float,
        peak_power_w: float,
        duty_cycle: float,
    ) -> Dict[str, Any]:
        """Analyze power consumption patterns."""
        # Calculate average power consumption
        avg_payload_power = payload_power_w * duty_cycle
        avg_bus_power = bus_power_w  # Bus systems run continuously
        avg_total_power = avg_payload_power + avg_bus_power

        # Peak power during active operations
        peak_total_power = peak_power_w

        return {
            "payload_power_w": payload_power_w,
            "bus_power_w": bus_power_w,
            "avg_payload_power_w": avg_payload_power,
            "avg_bus_power_w": avg_bus_power,
            "avg_power_w": avg_total_power,
            "peak_power_w": peak_total_power,
            "duty_cycle": duty_cycle,
        }

    def _analyze_battery_requirements(
        self,
        avg_power_w: float,
        eclipse_fraction: float,
        orbit_alt_km: float,
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Analyze battery sizing requirements."""
        # Calculate orbital period and eclipse duration
        earth_radius = 6371  # km
        orbit_radius = earth_radius + orbit_alt_km
        orbital_period_min = 2 * math.pi * math.sqrt(orbit_radius**3 / 398600.4418) / 60
        eclipse_duration_min = orbital_period_min * eclipse_fraction

        # Energy required during eclipse
        eclipse_energy_wh = avg_power_w * eclipse_duration_min / 60

        # Add safety margin and account for battery limitations
        safety_margin = kwargs.get("battery_safety_margin", 1.3)  # 30% margin
        required_energy_wh = eclipse_energy_wh * safety_margin

        # Account for depth of discharge and efficiency
        battery_capacity_wh = required_energy_wh / (
            self.battery_specs["depth_of_discharge"]
            * self.battery_specs["discharge_efficiency"]
            * self.battery_specs["temperature_derating"]
        )

        # Calculate battery mass
        battery_mass_kg = (
            battery_capacity_wh / self.battery_specs["energy_density_wh_per_kg"]
        )

        # Check cycle life
        cycles_per_day = 24 * 60 / orbital_period_min
        mission_duration_years = kwargs.get("mission_duration_years", 2)
        total_cycles = cycles_per_day * 365 * mission_duration_years
        cycle_life_margin = self.battery_specs["cycle_life"] / total_cycles

        return {
            "required_energy_wh": required_energy_wh,
            "battery_capacity_wh": battery_capacity_wh,
            "mass_kg": battery_mass_kg,
            "eclipse_duration_min": eclipse_duration_min,
            "orbital_period_min": orbital_period_min,
            "cycles_per_day": cycles_per_day,
            "total_cycles": total_cycles,
            "cycle_life_margin": cycle_life_margin,
            "safety_margin": safety_margin,
        }

    def _calculate_power_margin(
        self,
        solar_analysis: Dict[str, Any],
        consumption_analysis: Dict[str, Any],
        battery_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate overall power margin."""
        # Available power during sunlight
        available_power_w = solar_analysis["avg_power_w"]

        # Required power (average)
        required_power_w = consumption_analysis["avg_power_w"]

        # Power margin
        margin_w = available_power_w - required_power_w
        margin_ratio = (
            margin_w / required_power_w if required_power_w > 0 else float("inf")
        )

        # Check if we can charge batteries during sunlight
        charging_power_w = margin_w * self.battery_specs["charge_efficiency"]
        charging_time_available_min = battery_analysis["orbital_period_min"] * (
            1 - 0.35
        )  # Non-eclipse time

        can_charge_batteries = (
            charging_power_w * charging_time_available_min / 60
            >= battery_analysis["required_energy_wh"]
        )

        return {
            "available_power_w": available_power_w,
            "required_power_w": required_power_w,
            "margin_w": margin_w,
            "margin_ratio": margin_ratio,
            "deficit_w": abs(margin_w) if margin_w < 0 else 0,
            "can_charge_batteries": can_charge_batteries,
            "charging_power_w": charging_power_w,
        }

    @property
    def solver_name(self) -> str:
        return self._solver_name

    @property
    def solver_version(self) -> str:
        return self._solver_version
