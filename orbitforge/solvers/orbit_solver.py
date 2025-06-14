"""Orbital mechanics solver for coverage and propagation analysis."""

import asyncio
import math
from typing import Dict, Any, List, Tuple
from loguru import logger

from ..design_record import DesignRecord


class OrbitSolver:
    """Orbital mechanics solver using simplified analytical models.

    Provides fast orbital coverage and propagation calculations
    suitable for early design iterations.
    """

    def __init__(self):
        self._solver_name = "OrbitSolver"
        self._solver_version = "1.0.0"

        # Earth constants
        self.EARTH_RADIUS_KM = 6371.0
        self.EARTH_MU = 398600.4418  # km³/s²
        self.EARTH_J2 = 1.08263e-3
        self.EARTH_ROTATION_RATE = 7.2921159e-5  # rad/s

    def evaluate(self, design: DesignRecord, **kwargs) -> Dict[str, Any]:
        """Evaluate orbital coverage and mechanics for the design."""
        try:
            mission_spec = design.mission_spec
            orbit_alt_km = mission_spec.get("orbit_alt_km", 550)

            # Extract optional parameters
            inclination_deg = kwargs.get("inclination_deg", 97.4)  # Sun-sync default
            lat_band_deg = kwargs.get("lat_band_deg", 60)  # Coverage latitude band
            revisit_hours = kwargs.get("revisit_hours", 24)  # Required revisit time

            # Calculate orbital parameters
            orbit_radius = self.EARTH_RADIUS_KM + orbit_alt_km
            orbital_period = self._calc_orbital_period(orbit_radius)
            orbital_velocity = self._calc_orbital_velocity(orbit_radius)

            # Calculate coverage metrics
            coverage_fraction = self._calc_fractional_coverage(
                orbit_alt_km, inclination_deg, lat_band_deg, revisit_hours
            )

            # Calculate eclipse fraction for power analysis
            eclipse_fraction = self._calc_eclipse_fraction(orbit_radius)

            # Assess orbital decay
            decay_rate_km_per_day = self._calc_decay_rate(
                orbit_alt_km, mission_spec.get("mass_limit_kg", 4.0)
            )
            mission_duration_years = kwargs.get("mission_duration_years", 2)
            total_decay_km = decay_rate_km_per_day * 365 * mission_duration_years

            # Determine status
            status = "PASS"
            warnings = []
            margin = 1.0

            if coverage_fraction < 0.8:
                status = "WARNING"
                warnings.append(f"Low coverage fraction: {coverage_fraction:.2f}")
                margin = min(margin, coverage_fraction / 0.8)

            if orbit_alt_km - total_decay_km < 300:
                status = "FAIL"
                warnings.append(
                    f"Orbit will decay below 300km in {mission_duration_years} years"
                )
                margin = 0.0
            elif orbit_alt_km - total_decay_km < 400:
                status = "WARNING"
                warnings.append(
                    f"Significant orbital decay expected: {total_decay_km:.1f}km"
                )
                margin = min(margin, (orbit_alt_km - total_decay_km - 300) / 100)

            return {
                "status": status,
                "margin": margin,
                "warnings": warnings,
                "details": {
                    "orbital_period_min": orbital_period / 60,
                    "orbital_velocity_km_s": orbital_velocity,
                    "coverage_fraction": coverage_fraction,
                    "eclipse_fraction": eclipse_fraction,
                    "decay_rate_km_per_day": decay_rate_km_per_day,
                    "total_decay_km": total_decay_km,
                    "final_altitude_km": orbit_alt_km - total_decay_km,
                    "inclination_deg": inclination_deg,
                    "lat_band_deg": lat_band_deg,
                    "revisit_hours": revisit_hours,
                },
            }

        except Exception as e:
            logger.error(f"OrbitSolver evaluation failed: {e}")
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

    def _calc_orbital_period(self, orbit_radius_km: float) -> float:
        """Calculate orbital period in seconds."""
        return 2 * math.pi * math.sqrt(orbit_radius_km**3 / self.EARTH_MU)

    def _calc_orbital_velocity(self, orbit_radius_km: float) -> float:
        """Calculate orbital velocity in km/s."""
        return math.sqrt(self.EARTH_MU / orbit_radius_km)

    def _calc_fractional_coverage(
        self, alt_km: float, inc_deg: float, lat_band_deg: float, revisit_hours: float
    ) -> float:
        """Calculate fractional coverage of latitude band."""
        # Simplified coverage model based on orbital geometry
        orbit_radius = self.EARTH_RADIUS_KM + alt_km

        # Ground track width (simplified)
        ground_track_width_deg = 2 * math.degrees(
            math.asin(self.EARTH_RADIUS_KM / orbit_radius)
        )

        # Number of orbits in revisit period
        orbital_period_hours = self._calc_orbital_period(orbit_radius) / 3600
        orbits_per_revisit = revisit_hours / orbital_period_hours

        # Coverage efficiency based on inclination and latitude band
        inc_rad = math.radians(inc_deg)
        lat_band_rad = math.radians(lat_band_deg)

        if inc_deg < lat_band_deg:
            # Cannot cover full latitude band
            coverage_efficiency = inc_deg / lat_band_deg
        else:
            coverage_efficiency = 1.0

        # Simplified coverage calculation
        coverage_per_orbit = ground_track_width_deg / 360 * coverage_efficiency
        total_coverage = min(
            1.0, coverage_per_orbit * orbits_per_revisit * 0.8
        )  # 80% efficiency

        return total_coverage

    def _calc_eclipse_fraction(self, orbit_radius_km: float) -> float:
        """Calculate fraction of orbit in Earth's shadow."""
        # Simplified eclipse calculation
        beta_angle = math.asin(self.EARTH_RADIUS_KM / orbit_radius_km)
        eclipse_angle = 2 * beta_angle
        eclipse_fraction = eclipse_angle / (2 * math.pi)

        return min(0.4, eclipse_fraction)  # Cap at 40% for typical orbits

    def _calc_decay_rate(self, alt_km: float, mass_kg: float) -> float:
        """Calculate orbital decay rate in km/day."""
        # Simplified atmospheric drag model
        if alt_km > 800:
            return 0.001  # Negligible decay above 800km
        elif alt_km > 600:
            return 0.01 * (800 - alt_km) / 200  # Linear interpolation
        elif alt_km > 400:
            return 0.01 + 0.05 * (600 - alt_km) / 200
        else:
            return 0.1 + 0.5 * (400 - alt_km) / 100  # Rapid decay below 400km

    @property
    def solver_name(self) -> str:
        return self._solver_name

    @property
    def solver_version(self) -> str:
        return self._solver_version
