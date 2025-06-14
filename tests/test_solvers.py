"""Test suite for the solver system."""

import pytest
import asyncio
from typing import Dict, Any

from orbitforge.design_record import DesignRecord, GeometryParams
from orbitforge.solvers import (
    OrbitSolver,
    PropulsionSolver,
    PowerSolver,
    RFSolver,
    ThermalSolver,
    SolverRegistry,
    get_solver_registry,
)


@pytest.fixture
def sample_design_record():
    """Create a sample design record for testing."""
    mission_spec = {
        "bus_u": 3,
        "payload_mass_kg": 1.0,
        "orbit_alt_km": 550,
        "mass_limit_kg": 4.0,
        "rail_mm": 3.0,
        "deck_mm": 2.5,
        "material": "Al_6061_T6",
    }

    geometry_params = GeometryParams(rail_mm=3.0, deck_mm=2.5, material="Al_6061_T6")

    return DesignRecord(
        design_id="test-design-123",
        mission_spec=mission_spec,
        geometry_params=geometry_params,
    )


class TestOrbitSolver:
    """Test the orbital mechanics solver."""

    def test_basic_evaluation(self, sample_design_record):
        """Test basic orbit solver evaluation."""
        solver = OrbitSolver()
        result = solver.evaluate(sample_design_record)

        assert result["status"] in ["PASS", "WARNING", "FAIL"]
        assert "margin" in result
        assert "details" in result
        assert "orbital_period_min" in result["details"]
        assert "coverage_fraction" in result["details"]

    def test_low_altitude_warning(self, sample_design_record):
        """Test that low altitude generates warnings."""
        # Modify mission spec for low altitude
        sample_design_record.mission_spec["orbit_alt_km"] = 350

        solver = OrbitSolver()
        result = solver.evaluate(sample_design_record, mission_duration_years=5)

        # Should warn about orbital decay
        assert result["status"] in ["WARNING", "FAIL"]
        assert any("decay" in warning.lower() for warning in result["warnings"])

    def test_coverage_calculation(self, sample_design_record):
        """Test coverage fraction calculation."""
        solver = OrbitSolver()
        result = solver.evaluate(
            sample_design_record,
            inclination_deg=98,  # Sun-synchronous
            lat_band_deg=60,
            revisit_hours=12,
        )

        coverage = result["details"]["coverage_fraction"]
        assert 0 <= coverage <= 1
        assert coverage > 0.5  # Should have reasonable coverage

    @pytest.mark.asyncio
    async def test_async_evaluation(self, sample_design_record):
        """Test async evaluation."""
        solver = OrbitSolver()
        result = await solver.evaluate_async(sample_design_record)

        assert result["status"] in ["PASS", "WARNING", "FAIL"]
        assert "details" in result


class TestPropulsionSolver:
    """Test the propulsion and launch solver."""

    def test_basic_evaluation(self, sample_design_record):
        """Test basic propulsion solver evaluation."""
        solver = PropulsionSolver()
        result = solver.evaluate(sample_design_record)

        assert result["status"] in ["PASS", "WARNING", "FAIL"]
        assert "details" in result
        assert "launch_analysis" in result["details"]
        assert "propulsion_analysis" in result["details"]

    def test_launch_vehicle_selection(self, sample_design_record):
        """Test launch vehicle selection logic."""
        solver = PropulsionSolver()
        result = solver.evaluate(
            sample_design_record, target_orbit="SSO", max_launch_cost=50000
        )

        launch_analysis = result["details"]["launch_analysis"]
        assert "recommended_vehicle" in launch_analysis
        assert "feasible_vehicles" in launch_analysis

        # Should find at least one feasible vehicle for 4kg CubeSat
        assert len(launch_analysis["feasible_vehicles"]) > 0

    def test_propulsion_system_selection(self, sample_design_record):
        """Test propulsion system auto-selection."""
        solver = PropulsionSolver()

        # Test with no delta-V requirement
        result = solver.evaluate(sample_design_record, required_delta_v_ms=0)
        prop_analysis = result["details"]["propulsion_analysis"]
        assert prop_analysis["recommended_system"] == "None"

        # Test with moderate delta-V requirement
        result = solver.evaluate(sample_design_record, required_delta_v_ms=100)
        prop_analysis = result["details"]["propulsion_analysis"]
        assert prop_analysis["recommended_system"] in ["Cold Gas", "Monoprop"]

    def test_delta_v_budget(self, sample_design_record):
        """Test delta-V budget calculation."""
        solver = PropulsionSolver()
        result = solver.evaluate(
            sample_design_record, mission_duration_years=3, formation_delta_v=50
        )

        maneuver_analysis = result["details"]["maneuver_analysis"]
        assert "total_delta_v_ms" in maneuver_analysis
        assert maneuver_analysis["total_delta_v_ms"] > 0


class TestPowerSolver:
    """Test the power budget solver."""

    def test_basic_evaluation(self, sample_design_record):
        """Test basic power solver evaluation."""
        solver = PowerSolver()
        result = solver.evaluate(sample_design_record)

        assert result["status"] in ["PASS", "WARNING", "FAIL"]
        assert "details" in result
        assert "solar_analysis" in result["details"]
        assert "battery_analysis" in result["details"]
        assert "power_margin" in result["details"]

    def test_solar_power_generation(self, sample_design_record):
        """Test solar power generation calculation."""
        solver = PowerSolver()
        result = solver.evaluate(
            sample_design_record, deployable_panels=True, geometric_efficiency=0.8
        )

        solar_analysis = result["details"]["solar_analysis"]
        assert solar_analysis["avg_power_w"] > 0
        assert solar_analysis["deployable_panels"] == True

        # Deployable panels should generate more power
        result_body_mounted = solver.evaluate(
            sample_design_record, deployable_panels=False
        )
        solar_body = result_body_mounted["details"]["solar_analysis"]
        assert solar_analysis["avg_power_w"] > solar_body["avg_power_w"]

    def test_battery_sizing(self, sample_design_record):
        """Test battery sizing calculation."""
        solver = PowerSolver()
        result = solver.evaluate(
            sample_design_record,
            payload_power_w=5.0,  # High power payload
            eclipse_fraction=0.4,
        )

        battery_analysis = result["details"]["battery_analysis"]
        assert battery_analysis["battery_capacity_wh"] > 0
        assert battery_analysis["mass_kg"] > 0
        assert battery_analysis["eclipse_duration_min"] > 0

    def test_power_margin_calculation(self, sample_design_record):
        """Test power margin calculation."""
        solver = PowerSolver()

        # Test with low power requirements (should pass)
        result = solver.evaluate(
            sample_design_record, payload_power_w=1.0, bus_power_w=0.5
        )
        assert result["details"]["power_margin"]["margin_w"] > 0

        # Test with high power requirements (should fail)
        result = solver.evaluate(
            sample_design_record,
            payload_power_w=20.0,  # Very high power
            bus_power_w=5.0,
        )
        assert result["details"]["power_margin"]["margin_w"] < 0


class TestRFSolver:
    """Test the RF link budget solver."""

    def test_basic_evaluation(self, sample_design_record):
        """Test basic RF solver evaluation."""
        solver = RFSolver()
        result = solver.evaluate(sample_design_record)

        assert result["status"] in ["PASS", "WARNING", "FAIL"]
        assert "details" in result
        assert "link_analysis" in result["details"]
        assert "throughput_analysis" in result["details"]

    def test_frequency_bands(self, sample_design_record):
        """Test different frequency bands."""
        solver = RFSolver()

        bands = ["UHF", "S-band", "X-band"]
        for band in bands:
            result = solver.evaluate(
                sample_design_record, frequency_band=band, tx_power_w=2.0
            )

            link_analysis = result["details"]["link_analysis"]
            assert link_analysis["frequency_band"] == band
            assert "ebno_db" in link_analysis
            assert "link_margin_db" in link_analysis

    def test_ground_station_types(self, sample_design_record):
        """Test different ground station types."""
        solver = RFSolver()

        stations = ["Amateur", "University", "Commercial"]
        for station in stations:
            result = solver.evaluate(sample_design_record, ground_station_type=station)

            link_analysis = result["details"]["link_analysis"]
            assert link_analysis["ground_station_type"] == station

    def test_data_throughput(self, sample_design_record):
        """Test data throughput calculation."""
        solver = RFSolver()
        result = solver.evaluate(
            sample_design_record, data_rate_kbps=19.2, min_elevation_deg=5
        )

        throughput = result["details"]["throughput_analysis"]
        assert throughput["daily_data_mb"] > 0
        assert throughput["passes_per_day"] > 0
        assert throughput["pass_duration_min"] > 0


class TestThermalSolver:
    """Test the thermal analysis solver."""

    def test_basic_evaluation(self, sample_design_record):
        """Test basic thermal solver evaluation."""
        solver = ThermalSolver()
        result = solver.evaluate(sample_design_record)

        assert result["status"] in ["PASS", "WARNING", "FAIL"]
        assert "details" in result
        assert "thermal_environment" in result["details"]
        assert "temperature_analysis" in result["details"]

    def test_material_effects(self, sample_design_record):
        """Test different materials."""
        solver = ThermalSolver()

        materials = ["Al_6061_T6", "Ti_6Al_4V", "Carbon_Fiber"]
        for material in materials:
            sample_design_record.geometry_params.material = material
            result = solver.evaluate(sample_design_record)

            temp_analysis = result["details"]["temperature_analysis"]
            assert "max_temp_c" in temp_analysis
            assert "min_temp_c" in temp_analysis
            assert temp_analysis["max_temp_c"] > temp_analysis["min_temp_c"]

    def test_component_temperatures(self, sample_design_record):
        """Test component temperature analysis."""
        solver = ThermalSolver()
        result = solver.evaluate(sample_design_record, internal_power_w=5.0)

        temp_analysis = result["details"]["temperature_analysis"]
        component_temps = temp_analysis["component_temps"]

        # Check that all expected components are analyzed
        expected_components = ["electronics", "battery", "solar_panels", "structure"]
        for comp in expected_components:
            assert comp in component_temps
            assert "max_temp_c" in component_temps[comp]
            assert "min_temp_c" in component_temps[comp]

    def test_thermal_limits(self, sample_design_record):
        """Test thermal limit checking."""
        solver = ThermalSolver()

        # Test with very high internal power (should cause overheating)
        result = solver.evaluate(
            sample_design_record, internal_power_w=50.0  # Very high power
        )

        # Should generate warnings or failures for overheating
        if result["status"] in ["WARNING", "FAIL"]:
            assert any("temp" in warning.lower() for warning in result["warnings"])


class TestSolverRegistry:
    """Test the solver registry system."""

    def test_registry_initialization(self):
        """Test that registry initializes with all solvers."""
        registry = SolverRegistry()
        solvers = registry.list_solvers()

        expected_solvers = ["orbit", "propulsion", "power", "rf", "thermal"]
        for solver_name in expected_solvers:
            assert solver_name in solvers

    def test_single_solver_evaluation(self, sample_design_record):
        """Test evaluating a single solver through registry."""
        registry = SolverRegistry()
        result = registry.evaluate_single("orbit", sample_design_record)

        assert "solver" in result
        assert "solver_version" in result
        assert "result" in result
        assert result["solver"] == "orbit"

    def test_all_solvers_evaluation(self, sample_design_record):
        """Test evaluating all solvers."""
        registry = SolverRegistry()
        result = registry.evaluate_all(sample_design_record)

        assert "overall_status" in result
        assert "overall_margin" in result
        assert "solver_results" in result

        # Check that all solvers were evaluated
        solver_results = result["solver_results"]
        expected_solvers = ["orbit", "propulsion", "power", "rf", "thermal"]
        for solver_name in expected_solvers:
            assert solver_name in solver_results

    @pytest.mark.asyncio
    async def test_async_evaluation(self, sample_design_record):
        """Test async evaluation of all solvers."""
        registry = SolverRegistry()
        result = await registry.evaluate_all_async(sample_design_record)

        assert "overall_status" in result
        assert "execution_mode" in result
        assert result["execution_mode"] == "async"

        # Should have results from all solvers
        solver_results = result["solver_results"]
        expected_solvers = ["orbit", "propulsion", "power", "rf", "thermal"]
        for solver_name in expected_solvers:
            assert solver_name in solver_results

    def test_solver_configurations(self, sample_design_record):
        """Test solver evaluation with custom configurations."""
        registry = SolverRegistry()

        solver_configs = {
            "orbit": {"inclination_deg": 98, "mission_duration_years": 3},
            "power": {"payload_power_w": 3.0, "deployable_panels": True},
            "rf": {"frequency_band": "S-band", "data_rate_kbps": 38.4},
        }

        result = registry.evaluate_all(sample_design_record, solver_configs)

        # Check that configurations were applied
        orbit_result = result["solver_results"]["orbit"]["result"]
        assert orbit_result["details"]["inclination_deg"] == 98

        power_result = result["solver_results"]["power"]["result"]
        assert power_result["details"]["solar_analysis"]["deployable_panels"] == True

        rf_result = result["solver_results"]["rf"]["result"]
        assert rf_result["details"]["link_analysis"]["frequency_band"] == "S-band"

    def test_global_registry(self):
        """Test global registry singleton."""
        registry1 = get_solver_registry()
        registry2 = get_solver_registry()

        assert registry1 is registry2  # Should be the same instance
        assert len(registry1.list_solvers()) > 0


class TestGoldenCases:
    """Golden case tests based on historical CubeSat missions."""

    def test_cubesat_3u_leo_mission(self):
        """Test against a typical 3U LEO CubeSat mission."""
        mission_spec = {
            "bus_u": 3,
            "payload_mass_kg": 1.5,
            "orbit_alt_km": 550,
            "mass_limit_kg": 4.0,
        }

        geometry_params = GeometryParams(
            rail_mm=3.0, deck_mm=2.5, material="Al_6061_T6"
        )

        design = DesignRecord(
            mission_spec=mission_spec, geometry_params=geometry_params
        )

        registry = SolverRegistry()

        # Configuration based on typical 3U CubeSat
        solver_configs = {
            "orbit": {
                "inclination_deg": 97.4,  # Sun-synchronous
                "mission_duration_years": 2,
            },
            "power": {
                "payload_power_w": 2.0,
                "bus_power_w": 1.5,
                "deployable_panels": False,
            },
            "rf": {
                "frequency_band": "UHF",
                "data_rate_kbps": 1.2,
                "tx_power_w": 4.0,
                "tx_antenna_gain_dbi": 8.0,
                "rx_antenna_gain_dbi": 35.0,  # Optional if not using ground_station_type
                "ground_station_type": "University",
                "min_elevation_deg": 10,
            },
            "thermal": {
                "internal_power_w": 3.5,
                "radiator_area_m2": 0.02,  # Small dedicated radiator area
                "eclipse_fraction": 0.3,  # Typical eclipse fraction for LEO
            },
            "propulsion": {
                "required_delta_v_ms": 0,  # No propulsion
                "target_orbit": "SSO",
            },
        }

        result = registry.evaluate_all(design, solver_configs)

        # DEBUG ================================
        thermal_result = result["solver_results"]["thermal"]["result"]
        import pprint

        pprint.pprint(thermal_result["details"])
        # ======================================

        # Verify reasonable results for a typical 3U CubeSat
        assert result["overall_status"] in ["PASS", "WARNING"]

        # Check specific solver results
        orbit_result = result["solver_results"]["orbit"]["result"]
        assert orbit_result["details"]["orbital_period_min"] > 90  # Typical LEO period
        assert orbit_result["details"]["orbital_period_min"] < 120

        power_result = result["solver_results"]["power"]["result"]
        assert (
            power_result["details"]["solar_analysis"]["avg_power_w"] > 3
        )  # Should generate enough power

        rf_result = result["solver_results"]["rf"]["result"]
        import pprint

        pprint.pprint(rf_result["details"])
        assert (
            rf_result["details"]["link_analysis"]["ebno_db"] > 10
        )  # Should have decent link

        thermal_result = result["solver_results"]["thermal"]["result"]

        # Check each component against its specific temperature limits
        component_temps = thermal_result["details"]["temperature_analysis"][
            "component_temps"
        ]

        # Electronics should be within operational limits
        assert component_temps["electronics"]["max_temp_c"] < 85  # Max electronics temp
        assert (
            component_temps["electronics"]["min_temp_c"] > -40
        )  # Min electronics temp

        # Battery should be within operational limits
        assert component_temps["battery"]["max_temp_c"] < 60  # Max battery temp
        assert component_temps["battery"]["min_temp_c"] > -20  # Min battery temp

        # Solar panels can handle higher temperatures
        assert (
            component_temps["solar_panels"]["max_temp_c"] < 120
        )  # Max solar panel temp
        assert (
            component_temps["solar_panels"]["min_temp_c"] > -150
        )  # Min solar panel temp

        # Structure can handle the widest temperature range
        assert component_temps["structure"]["max_temp_c"] < 150  # Max structure temp
        assert component_temps["structure"]["min_temp_c"] > -180  # Min structure temp

    def test_high_power_6u_mission(self):
        """Test a high-power 6U mission scenario."""
        mission_spec = {
            "bus_u": 6,
            "payload_mass_kg": 3.0,
            "orbit_alt_km": 600,
            "mass_limit_kg": 12.0,
        }

        geometry_params = GeometryParams(
            rail_mm=4.0, deck_mm=3.0, material="Al_6061_T6"
        )

        design = DesignRecord(
            mission_spec=mission_spec, geometry_params=geometry_params
        )

        registry = SolverRegistry()

        solver_configs = {
            "power": {
                "payload_power_w": 8.0,  # High power payload
                "bus_power_w": 2.0,
                "deployable_panels": True,
            },
            "rf": {
                "frequency_band": "S-band",
                "data_rate_kbps": 100,  # High data rate
                "tx_power_w": 5.0,
            },
            "thermal": {"internal_power_w": 10.0},  # High internal power
        }

        result = registry.evaluate_all(design, solver_configs)

        # Should handle high power requirements
        power_result = result["solver_results"]["power"]["result"]
        assert power_result["status"] in [
            "PASS",
            "WARNING",
        ]  # Should be feasible with deployable panels

        # Thermal should be more challenging
        thermal_result = result["solver_results"]["thermal"]["result"]
        # May have warnings due to high power dissipation
