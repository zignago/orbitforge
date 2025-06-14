#!/usr/bin/env python3
"""Demo script for the OrbitForge solver system."""

import asyncio
from orbitforge.design_record import DesignRecord, GeometryParams
from orbitforge.solvers import get_solver_registry


def main():
    """Demonstrate the solver system."""
    print("=" * 60)
    print("OrbitForge Solver System Demo")
    print("=" * 60)

    # Create a sample design
    mission_spec = {
        "bus_u": 3,
        "payload_mass_kg": 1.5,
        "orbit_alt_km": 550,
        "mass_limit_kg": 4.0,
    }

    geometry_params = GeometryParams(rail_mm=3.0, deck_mm=2.5, material="Al_6061_T6")

    design = DesignRecord(mission_spec=mission_spec, geometry_params=geometry_params)

    print(f"Design ID: {design.design_id}")
    print(
        f"Mission: {mission_spec['bus_u']}U CubeSat at {mission_spec['orbit_alt_km']}km"
    )
    print()

    # Get solver registry
    registry = get_solver_registry()
    print(f"Available solvers: {', '.join(registry.list_solvers())}")
    print()

    # Define solver configurations based on typical 3U CubeSat
    solver_configs = {
        "orbit": {
            "inclination_deg": 97.4,  # Sun-synchronous
            "mission_duration_years": 2,
        },
        "power": {
            "payload_power_w": 2.0,
            "bus_power_w": 1.5,
            "deployable_panels": True,  # Added deployable panels for better power
        },
        "rf": {
            "frequency_band": "UHF",
            "data_rate_kbps": 9.6,
            "tx_power_w": 1.0,
            "ground_station_type": "University",
        },
        "thermal": {"internal_power_w": 3.5},  # Total internal power dissipation
        "propulsion": {
            "required_delta_v_ms": 0,  # No propulsion
            "target_orbit": "SSO",
        },
    }

    # Test individual solvers
    print("Testing Individual Solvers:")
    print("-" * 40)

    for solver_name in ["orbit", "power", "rf", "thermal"]:
        try:
            # Pass solver configs as kwargs
            config = solver_configs.get(solver_name, {})
            result = registry.evaluate_single(solver_name, design, **config)
            solver_result = result["result"]
            print(f"{solver_name.upper()} Solver:")
            print(f"  Status: {solver_result['status']}")
            print(f"  Margin: {solver_result['margin']:.2f}")
            if solver_result.get("warnings"):
                for w in solver_result.get("warnings", []):
                    print(f"    - {w}")
            print()
        except Exception as e:
            print(f"{solver_name.upper()} Solver: ERROR - {e}")
            print()

    # Test all solvers together
    print("Testing All Solvers Together:")
    print("-" * 40)

    try:
        result = registry.evaluate_all(design, solver_configs)
        print(f"Overall Status: {result['overall_status']}")
        print(f"Overall Margin: {result['overall_margin']:.2f}")
        for w in result.get("warnings", []):
            print(f"    - {w}")
        print()

        print("Individual Results:")
        for solver_name, solver_result in result["solver_results"].items():
            status = solver_result["result"]["status"]
            margin = solver_result["result"]["margin"]
            print(f"  {solver_name}: {status} (margin: {margin:.2f})")

    except Exception as e:
        print(f"All solvers evaluation failed: {e}")

    print()
    print("=" * 60)
    print("Demo Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
