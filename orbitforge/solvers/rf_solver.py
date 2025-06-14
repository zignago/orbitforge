"""RF link budget solver for communication analysis."""

import asyncio
import math
from typing import Dict, Any, List, Tuple
from loguru import logger

from ..design_record import DesignRecord


class RFSolver:
    """RF link budget solver for CubeSat communication systems.

    Calculates link budgets, data rates, and communication feasibility
    for various ground station and relay scenarios.
    """

    def __init__(self):
        self._solver_name = "RFSolver"
        self._solver_version = "1.0.0"

        # Frequency band specifications
        self.frequency_bands = {
            "UHF": {"freq_mhz": 435, "path_loss_factor": 1.0, "rain_loss_db": 0.5},
            "S-band": {"freq_mhz": 2400, "path_loss_factor": 1.0, "rain_loss_db": 2.0},
            "X-band": {"freq_mhz": 8400, "path_loss_factor": 1.0, "rain_loss_db": 5.0},
            "Ka-band": {
                "freq_mhz": 32000,
                "path_loss_factor": 1.0,
                "rain_loss_db": 10.0,
            },
        }

        # Ground station types
        self.ground_stations = {
            "Amateur": {
                "antenna_gain_dbi": 15,
                "noise_temp_k": 150,
                "availability": 0.7,
            },
            "University": {
                "antenna_gain_dbi": 25,
                "noise_temp_k": 100,
                "availability": 0.8,
            },
            "Commercial": {
                "antenna_gain_dbi": 35,
                "noise_temp_k": 50,
                "availability": 0.95,
            },
            "Deep Space": {
                "antenna_gain_dbi": 60,
                "noise_temp_k": 20,
                "availability": 0.99,
            },
        }

        # Constants
        self.BOLTZMANN_CONSTANT = 1.38e-23  # J/K
        self.SPEED_OF_LIGHT = 3e8  # m/s

    def evaluate(self, design: DesignRecord, **kwargs) -> Dict[str, Any]:
        """Evaluate RF link budget and communication capability."""
        try:
            mission_spec = design.mission_spec
            orbit_alt_km = mission_spec.get("orbit_alt_km", 550)

            # Extract communication requirements
            frequency_band = kwargs.get("frequency_band", "UHF")
            data_rate_kbps = kwargs.get("data_rate_kbps", 9.6)
            tx_power_w = kwargs.get("tx_power_w", 1.0)
            tx_antenna_gain_dbi = kwargs.get("tx_antenna_gain_dbi", 2.0)
            ground_station_type = kwargs.get("ground_station_type", "University")

            # Mission parameters
            elevation_angle_deg = kwargs.get("min_elevation_deg", 10)
            required_ebno_db = kwargs.get("required_ebno_db", 12)  # For BER 1e-5

            # Calculate link budget
            link_analysis = self._analyze_link_budget(
                orbit_alt_km,
                frequency_band,
                tx_power_w,
                tx_antenna_gain_dbi,
                ground_station_type,
                elevation_angle_deg,
                kwargs,
            )

            # Calculate data throughput
            throughput_analysis = self._analyze_data_throughput(
                link_analysis, data_rate_kbps, orbit_alt_km, kwargs
            )

            # Calculate communication windows
            comm_windows = self._analyze_communication_windows(
                orbit_alt_km, elevation_angle_deg, kwargs
            )

            # Determine status
            status = "PASS"
            warnings = []
            margin = link_analysis["link_margin_db"] / required_ebno_db

            if link_analysis["ebno_db"] < required_ebno_db:
                status = "FAIL"
                warnings.append(
                    f"Insufficient link margin: {link_analysis['link_margin_db']:.1f}dB"
                )
                margin = 0.0
            elif link_analysis["link_margin_db"] < 3:
                status = "WARNING"
                warnings.append(
                    f"Low link margin: {link_analysis['link_margin_db']:.1f}dB"
                )

            if throughput_analysis["daily_data_mb"] < kwargs.get(
                "min_daily_data_mb", 10
            ):
                status = "WARNING"
                warnings.append("Low daily data throughput")
                margin = min(margin, 0.5)

            return {
                "status": status,
                "margin": margin,
                "warnings": warnings,
                "details": {
                    "link_analysis": link_analysis,
                    "throughput_analysis": throughput_analysis,
                    "comm_windows": comm_windows,
                    "ebno_db": link_analysis["ebno_db"],
                    "link_margin_db": link_analysis["link_margin_db"],
                    "daily_data_mb": throughput_analysis["daily_data_mb"],
                },
            }

        except Exception as e:
            logger.error(f"RFSolver evaluation failed: {e}")
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

    def _analyze_link_budget(
        self,
        orbit_alt_km: float,
        frequency_band: str,
        tx_power_w: float,
        tx_antenna_gain_dbi: float,
        ground_station_type: str,
        elevation_angle_deg: float,
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Analyze RF link budget."""
        # Get frequency band and ground station specs
        band_specs = self.frequency_bands.get(
            frequency_band, self.frequency_bands["UHF"]
        )
        gs_specs = self.ground_stations.get(
            ground_station_type, self.ground_stations["University"]
        )

        freq_hz = band_specs["freq_mhz"] * 1e6
        wavelength_m = self.SPEED_OF_LIGHT / freq_hz

        # Calculate slant range
        earth_radius_km = 6371
        slant_range_km = self._calc_slant_range(orbit_alt_km, elevation_angle_deg)

        # Link budget calculation
        # EIRP (dBW)
        tx_power_dbw = 10 * math.log10(tx_power_w)
        eirp_dbw = tx_power_dbw + tx_antenna_gain_dbi

        # Free space path loss (dB)
        fspl_db = (
            20 * math.log10(slant_range_km * 1000) + 20 * math.log10(freq_hz) - 147.55
        )

        # Atmospheric losses
        atmospheric_loss_db = kwargs.get("atmospheric_loss_db", 0.5)
        rain_loss_db = band_specs["rain_loss_db"] * kwargs.get("rain_factor", 1.0)

        # Received power (dBW)
        rx_power_dbw = (
            eirp_dbw
            - fspl_db
            - atmospheric_loss_db
            - rain_loss_db
            + gs_specs["antenna_gain_dbi"]
        )

        # Noise power (dBW)
        noise_power_dbw = (
            10 * math.log10(self.BOLTZMANN_CONSTANT)
            + 10 * math.log10(gs_specs["noise_temp_k"])
            + 10 * math.log10(kwargs.get("bandwidth_hz", 12500))
        )

        # C/N ratio
        cn_db = rx_power_dbw - noise_power_dbw

        # Eb/No (assuming BPSK modulation)
        data_rate_bps = kwargs.get("data_rate_kbps", 9.6) * 1000
        ebno_db = cn_db - 10 * math.log10(data_rate_bps)

        # Link margin
        required_ebno_db = kwargs.get("required_ebno_db", 12)
        link_margin_db = ebno_db - required_ebno_db

        return {
            "frequency_band": frequency_band,
            "freq_mhz": band_specs["freq_mhz"],
            "slant_range_km": slant_range_km,
            "eirp_dbw": eirp_dbw,
            "fspl_db": fspl_db,
            "rx_power_dbw": rx_power_dbw,
            "noise_power_dbw": noise_power_dbw,
            "cn_db": cn_db,
            "ebno_db": ebno_db,
            "link_margin_db": link_margin_db,
            "ground_station_type": ground_station_type,
            "elevation_angle_deg": elevation_angle_deg,
        }

    def _analyze_data_throughput(
        self,
        link_analysis: Dict[str, Any],
        data_rate_kbps: float,
        orbit_alt_km: float,
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Analyze data throughput capability."""
        # Calculate pass duration
        pass_duration_min = self._calc_pass_duration(
            orbit_alt_km, link_analysis["elevation_angle_deg"]
        )

        # Number of passes per day
        orbital_period_min = (
            2 * math.pi * math.sqrt((6371 + orbit_alt_km) ** 3 / 398600.4418) / 60
        )
        passes_per_day = (24 * 60 / orbital_period_min) * kwargs.get(
            "ground_station_visibility", 0.15
        )

        # Data per pass
        data_per_pass_mb = (data_rate_kbps * pass_duration_min * 60) / (
            8 * 1024
        )  # Convert to MB

        # Daily data throughput
        daily_data_mb = data_per_pass_mb * passes_per_day

        # Account for protocol overhead and retransmissions
        efficiency_factor = kwargs.get("protocol_efficiency", 0.8)
        effective_daily_data_mb = daily_data_mb * efficiency_factor

        return {
            "data_rate_kbps": data_rate_kbps,
            "pass_duration_min": pass_duration_min,
            "passes_per_day": passes_per_day,
            "data_per_pass_mb": data_per_pass_mb,
            "daily_data_mb": daily_data_mb,
            "effective_daily_data_mb": effective_daily_data_mb,
            "protocol_efficiency": efficiency_factor,
        }

    def _analyze_communication_windows(
        self, orbit_alt_km: float, elevation_angle_deg: float, kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze communication window characteristics."""
        # Calculate orbital parameters
        orbital_period_min = (
            2 * math.pi * math.sqrt((6371 + orbit_alt_km) ** 3 / 398600.4418) / 60
        )

        # Ground station coverage
        coverage_angle_deg = 2 * math.degrees(math.acos(6371 / (6371 + orbit_alt_km)))

        # Pass characteristics
        max_pass_duration_min = self._calc_pass_duration(
            orbit_alt_km, 0
        )  # Overhead pass
        avg_pass_duration_min = self._calc_pass_duration(
            orbit_alt_km, elevation_angle_deg
        )

        # Revisit time
        revisit_hours = orbital_period_min / 60 * kwargs.get("revisit_orbits", 15)

        return {
            "orbital_period_min": orbital_period_min,
            "coverage_angle_deg": coverage_angle_deg,
            "max_pass_duration_min": max_pass_duration_min,
            "avg_pass_duration_min": avg_pass_duration_min,
            "revisit_hours": revisit_hours,
            "elevation_angle_deg": elevation_angle_deg,
        }

    def _calc_slant_range(
        self, orbit_alt_km: float, elevation_angle_deg: float
    ) -> float:
        """Calculate slant range to satellite."""
        earth_radius_km = 6371
        orbit_radius_km = earth_radius_km + orbit_alt_km
        elevation_rad = math.radians(elevation_angle_deg)

        # Law of cosines
        slant_range_km = math.sqrt(
            earth_radius_km**2
            + orbit_radius_km**2
            - 2
            * earth_radius_km
            * orbit_radius_km
            * math.cos(math.pi / 2 + elevation_rad)
        )

        return slant_range_km

    def _calc_pass_duration(
        self, orbit_alt_km: float, elevation_angle_deg: float
    ) -> float:
        """Calculate pass duration in minutes."""
        earth_radius_km = 6371
        orbit_radius_km = earth_radius_km + orbit_alt_km

        # Half-angle of visibility cone
        if elevation_angle_deg == 0:
            half_angle_rad = math.acos(earth_radius_km / orbit_radius_km)
        else:
            # Approximate for non-zero elevation
            half_angle_rad = math.acos(earth_radius_km / orbit_radius_km) * (
                1 - elevation_angle_deg / 90
            )

        # Orbital angular velocity
        orbital_period_sec = 2 * math.pi * math.sqrt(orbit_radius_km**3 / 398600.4418)
        angular_velocity_rad_per_sec = 2 * math.pi / orbital_period_sec

        # Pass duration
        pass_duration_sec = 2 * half_angle_rad / angular_velocity_rad_per_sec

        return pass_duration_sec / 60  # Convert to minutes

    @property
    def solver_name(self) -> str:
        return self._solver_name

    @property
    def solver_version(self) -> str:
        return self._solver_version
