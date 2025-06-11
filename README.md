# OrbitForge

Generate flight-ready CubeSat structures from mission specifications.

## Features

- Complete structural skeleton for 3U to 6U CubeSats
- Static loading verification against Falcon 9 and Electron environments
- Thermal expansion checks across standard gradient (-40°C to +60°C)
- Manufacturability analysis for additive manufacturing
- Comprehensive PDF reports with mass budget, FEA results, and DfAM checks

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/orbitforge.git
cd orbitforge

# Install in development mode
pip install -e ".[dev]"
```

## Quick Start

1. Create a mission spec (or use a sample from `missions/`):

```json
{
    "bus_u": 3,
    "payload_mass_kg": 1.0,
    "orbit_alt_km": 550,
    "mass_limit_kg": 4.0,
    "rail_mm": 3.0,
    "deck_mm": 2.5,
    "material": "Al_6061_T6"
}
```

2. Run OrbitForge:

```bash
# Basic frame generation
orbitforge run missions/demo_3u.json

# With physics validation
orbitforge run missions/demo_3u.json --check

# With DfAM checks
orbitforge run missions/demo_3u.json --dfam

# Generate full report
orbitforge run missions/demo_3u.json --check --dfam --report
```

3. Check outputs in the `outputs/design_*` directory:
   - `frame.step` - CAD model
   - `frame.stl` - AM-ready mesh
   - `mass_budget.csv` - Component-wise mass breakdown
   - `physics_check.json` - FEA results
   - `manufacturability.json` - DfAM analysis
   - `report.pdf` - Comprehensive design report

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests (excluding slow tests)
pytest -v -m "not slow"

# Run all tests
pytest -v

# Run linters
black .
flake8 .
mypy orbitforge
```

## License

MIT License - see LICENSE file for details.
