"""Propulsion and launch feasibility solver."""

import asyncio
import math
from typing import Dict, Any, List, Tuple, Optional
from loguru import logger

from ..design_record import DesignRecord


class PropulsionSolver:
    """Propulsion solver for delta-V calculations and launch vehicle selection.

    Provides analysis of propulsion requirements and launch feasibility
    for CubeSat missions.
    """

    def __init__(self):
        self._solver_name = "PropulsionSolver"
        self._solver_version = "1.0.0"

        # Launch vehicle performance data (simplified)
        self.launch_vehicles = {
            "Falcon 9 Rideshare": {
                "max_mass_kg": 200,  # Per rideshare slot
                "cost_per_kg": 5000,  # USD/kg
                "target_orbits": ["SSO", "LEO"],
                "altitude_range_km": (400, 800),
            },
            "Electron": {
                "max_mass_kg": 300,
                "cost_per_kg": 15000,
                "target_orbits": ["SSO", "LEO", "Polar"],
                "altitude_range_km": (300, 1200),
            },
            "PSLV": {
                "max_mass_kg": 100,  # CubeSat capacity
                "cost_per_kg": 8000,
                "target_orbits": ["SSO", "LEO"],
                "altitude_range_km": (400, 900),
            },
        }

        # Propulsion system options for CubeSats
        self.propulsion_systems = {
            "None": {"delta_v_ms": 0, "mass_fraction": 0, "cost_usd": 0},
            "Cold Gas": {"delta_v_ms": 50, "mass_fraction": 0.15, "cost_usd": 15000},
            "Monoprop": {"delta_v_ms": 200, "mass_fraction": 0.25, "cost_usd": 35000},
            "Electric": {"delta_v_ms": 500, "mass_fraction": 0.20, "cost_usd": 50000},
        }

    def evaluate(self, design: DesignRecord, **kwargs) -> Dict[str, Any]:
        """Evaluate propulsion requirements and launch feasibility."""
        try:
            mission_spec = design.mission_spec
            mass_kg = mission_spec.get("mass_limit_kg", 4.0)
            orbit_alt_km = mission_spec.get("orbit_alt_km", 550)

            # Extract mission requirements
            required_delta_v = kwargs.get("required_delta_v_ms", 0)  # m/s
            mission_type = kwargs.get("mission_type", "LEO")
            target_orbit = kwargs.get("target_orbit", "SSO")

            # Calculate launch requirements
            launch_analysis = self._analyze_launch_options(
                mass_kg, orbit_alt_km, target_orbit
            )

            # Calculate propulsion requirements
            propulsion_analysis = self._analyze_propulsion_needs(
                mass_kg, required_delta_v, kwargs.get("propulsion_type", "auto")
            )

            # Calculate orbital maneuver delta-V requirements
            maneuver_analysis = self._analyze_orbital_maneuvers(orbit_alt_km, kwargs)

            # Determine overall status
            status = "PASS"
            warnings = []
            margin = 1.0

            # Check launch feasibility
            if not launch_analysis["feasible_vehicles"]:
                status = "FAIL"
                warnings.append("No suitable launch vehicle found")
                margin = 0.0
            elif launch_analysis["cost_usd"] > kwargs.get("max_launch_cost", 100000):
                status = "WARNING"
                warnings.append(
                    f"Launch cost exceeds budget: ${launch_analysis['cost_usd']:,.0f}"
                )
                margin = min(
                    margin,
                    kwargs.get("max_launch_cost", 100000) / launch_analysis["cost_usd"],
                )

            # Check propulsion capability
            if required_delta_v > propulsion_analysis["available_delta_v"]:
                if required_delta_v > 0:
                    status = "FAIL"
                    warnings.append(
                        f"Insufficient delta-V: need {required_delta_v}m/s, have {propulsion_analysis['available_delta_v']}m/s"
                    )
                    margin = 0.0

            return {
                "status": status,
                "margin": margin,
                "warnings": warnings,
                "details": {
                    "launch_analysis": launch_analysis,
                    "propulsion_analysis": propulsion_analysis,
                    "maneuver_analysis": maneuver_analysis,
                    "total_delta_v_budget_ms": maneuver_analysis["total_delta_v_ms"],
                    "recommended_launch_vehicle": launch_analysis[
                        "recommended_vehicle"
                    ],
                    "recommended_propulsion": propulsion_analysis["recommended_system"],
                },
            }

        except Exception as e:
            logger.error(f"PropulsionSolver evaluation failed: {e}")
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

    def _analyze_launch_options(
        self, mass_kg: float, orbit_alt_km: float, target_orbit: str
    ) -> Dict[str, Any]:
        """Analyze available launch vehicle options."""
        feasible_vehicles = []

        for vehicle_name, specs in self.launch_vehicles.items():
            if (
                mass_kg <= specs["max_mass_kg"]
                and target_orbit in specs["target_orbits"]
                and specs["altitude_range_km"][0]
                <= orbit_alt_km
                <= specs["altitude_range_km"][1]
            ):

                cost = mass_kg * specs["cost_per_kg"]
                feasible_vehicles.append(
                    {
                        "name": vehicle_name,
                        "cost_usd": cost,
                        "mass_margin_kg": specs["max_mass_kg"] - mass_kg,
                        "specs": specs,
                    }
                )

        # Sort by cost
        feasible_vehicles.sort(key=lambda x: x["cost_usd"])

        recommended_vehicle = (
            feasible_vehicles[0]["name"] if feasible_vehicles else None
        )
        total_cost = (
            feasible_vehicles[0]["cost_usd"] if feasible_vehicles else float("inf")
        )

        return {
            "feasible_vehicles": feasible_vehicles,
            "recommended_vehicle": recommended_vehicle,
            "cost_usd": total_cost,
            "mass_kg": mass_kg,
            "target_orbit": target_orbit,
            "orbit_alt_km": orbit_alt_km,
        }

    def _analyze_propulsion_needs(
        self, mass_kg: float, required_delta_v: float, propulsion_type: str
    ) -> Dict[str, Any]:
        """Analyze propulsion system requirements."""
        if propulsion_type == "auto":
            # Auto-select based on delta-V requirements
            if required_delta_v == 0:
                selected_system = "None"
            elif required_delta_v <= 50:
                selected_system = "Cold Gas"
            elif required_delta_v <= 200:
                selected_system = "Monoprop"
            else:
                selected_system = "Electric"
        else:
            selected_system = propulsion_type

        if selected_system not in self.propulsion_systems:
            selected_system = "None"

        system_specs = self.propulsion_systems[selected_system]

        # Calculate mass budget impact
        propulsion_mass_kg = mass_kg * system_specs["mass_fraction"]
        remaining_mass_kg = mass_kg - propulsion_mass_kg

        return {
            "recommended_system": selected_system,
            "available_delta_v": system_specs["delta_v_ms"],
            "required_delta_v": required_delta_v,
            "propulsion_mass_kg": propulsion_mass_kg,
            "remaining_mass_kg": remaining_mass_kg,
            "cost_usd": system_specs["cost_usd"],
            "mass_fraction": system_specs["mass_fraction"],
        }

    def _analyze_orbital_maneuvers(
        self, orbit_alt_km: float, kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze typical orbital maneuver requirements."""
        # Standard CubeSat maneuvers
        maneuvers = {
            "orbit_maintenance": kwargs.get("maintenance_delta_v", 10),  # m/s per year
            "attitude_control": kwargs.get("attitude_delta_v", 5),  # m/s per year
            "deorbit": kwargs.get(
                "deorbit_delta_v", self._calc_deorbit_delta_v(orbit_alt_km)
            ),
            "formation_flying": kwargs.get("formation_delta_v", 0),  # m/s
            "rendezvous": kwargs.get("rendezvous_delta_v", 0),  # m/s
        }

        mission_duration_years = kwargs.get("mission_duration_years", 2)

        # Calculate total delta-V budget
        total_delta_v = (
            maneuvers["orbit_maintenance"] * mission_duration_years
            + maneuvers["attitude_control"] * mission_duration_years
            + maneuvers["deorbit"]
            + maneuvers["formation_flying"]
            + maneuvers["rendezvous"]
        )

        return {
            "maneuvers": maneuvers,
            "mission_duration_years": mission_duration_years,
            "total_delta_v_ms": total_delta_v,
            "annual_delta_v_ms": (
                maneuvers["orbit_maintenance"] + maneuvers["attitude_control"]
            ),
        }

    def _calc_deorbit_delta_v(self, orbit_alt_km: float) -> float:
        """Calculate delta-V required for deorbit maneuver."""
        # Simplified calculation based on altitude
        if orbit_alt_km < 400:
            return 5  # Natural decay sufficient
        elif orbit_alt_km < 600:
            return 20 + (orbit_alt_km - 400) * 0.1
        else:
            return 40 + (orbit_alt_km - 600) * 0.05

    @property
    def solver_name(self) -> str:
        return self._solver_name

    @property
    def solver_version(self) -> str:
        return self._solver_version
