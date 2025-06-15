# OrbitForge

**Generate flight-ready CubeSat structures from mission specifications.**

OrbitForge helps small satellite teams accelerate design iteration by generating and validating CubeSat frames from a simple mission JSON spec. With CLI-based workflows, built-in physics checks, and manufacturability analysis, it's built to help you get to orbit faster.

## Features

- Rule-based and ML-based geometry generation
- Static load and thermal expansion validation (fast FEA)
- Manufacturability checks for additive manufacturing (DfAM)
- PDF reports with frame drawings, FEA results, DfAM findings
- Multi-design generation with parametric jittering
- Artifact logging to local or S3-compatible backends

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/orbitforge.git
cd orbitforge

# Install in development mode
mamba install -c ".[dev]"
```

## Command-Line Interface
OrbitForge includes a CLI built with [Typer](https://typer.tiangolo.com). After install, invoke it as:

```bash
orbitforge <command> [options]
```
Get help at any time:
```bash
orbitforge --help
orbitforge <command> --help
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

# Generate multiple design variants (NEW v0.1.2)
orbitforge run missions/demo_3u.json --multi 5 --check

# With DfAM checks and full pdf report
orbitforge run missions/demo_3u.json --check --dfam --report
```

3. Check outputs in the `outputs/` directory:
   - Single design mode: `design_*` folder with frame files
   - Multi-design mode: `design_001/`, `design_002/`, etc. folders
   - `summary.json` - Comparison table of all variants (multi-design mode)

### Multi-Design Generation (v0.1.2)

OrbitForge can generate multiple structural variants with parameter jittering to explore the design space:

```bash
# Generate 5 design variants with reproducible randomness
orbitforge run missions/demo_3u.json --multi 5 --seed 42 --check
```

**Features:**
- Parameter jittering: Random variations in rail thickness, deck thickness, and material
- Batch FEA processing: Automatic validation of all variants
- Ranking: Designs sorted by PASS/FAIL status and mass
- Summary table: Interactive comparison of all designs

**Sample Output:**
```
┏━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Design    ┃ Mass (kg) ┃ Max Stress (MPa) ┃ Status ┃ Rail (mm)  ┃ Deck (mm)  ┃ Material    ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ design_002│ 2.845     │ 145.2            │ PASS   │ 2.78       │ 2.43       │ Al_6061_T6  │
│ design_001│ 3.120     │ 156.7            │ PASS   │ 3.15       │ 2.55       │ Al_6061_T6  │
│ design_003│ 3.456     │ 198.4            │ PASS   │ 3.42       │ 2.71       │ Ti_6Al_4V   │
└───────────┴───────────┴──────────────────┴────────┴────────────┴────────────┴─────────────┘
```

**File Structure (Multi-Design):**
```
outputs/
├── design_001/
│   ├── frame.step       # CAD model
│   ├── frame.stl        # AM-ready mesh
│   ├── mass_budget.csv  # Mass breakdown
│   └── results.json     # FEA results
├── design_002/
│   └── ...
├── design_003/
│   └── ...
└── summary.json         # Comparison table
```

## CLI Commands
### `orbitforge run`
Generate CubeSat structures and run validation/reporting.
| Option                      | Description                             |
| --------------------------- | --------------------------------------- |
| `mission_json` (positional) | Path to mission spec JSON               |
| `-o, --outdir`              | Output directory (default: `./outputs`) |
| `--rail`, `--deck`          | Override rail/deck thickness            |
| `-m, --material`            | Override material (e.g. `Al_6061_T6`)   |
| `--check`                   | Run static + thermal physics checks     |
| `--dfam`                    | Run DfAM manufacturability checks       |
| `--report`                  | Generate PDF report                     |
| `--multi <int>`             | Create multiple design variants         |
| `--seed <int>`              | RNG seed for reproducibility            |
| `--generator diffusion`     | Use ML-based generation                 |
| `-v, --verbose`             | Show debug logs                         |

Each run generates a DesignRecord with CLI inputs, hashes, and validation outcomes.

### `orbitforge materials`
- Print a table of available materials, including:
- Material ID
- Density
- Yield Strength
- Young’s Modulus
- Description

### `orbitforge fetch`
Download all artifacts for a previous design run.
| Option                   | Description                            |
| ------------------------ | -------------------------------------- |
| `design_id` (positional) | 8–32 char design hash                  |
| `-o, --output`           | Destination directory                  |
| `--storage`              | Override backend (e.g. `s3://bucket/`) |

### `orbitforge list-designs`
List recent design records in local storage.
| Option        | Description                             |
| ------------- | --------------------------------------- |
| `-n, --limit` | Number of records to show (default: 10) |

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests (excluding slow tests)
pytest -v -m "not slow"

# Run all tests including multi-design tests
pytest -v

# Run linters
black .
flake8 .
mypy orbitforge
```

## License

MIT License - see LICENSE file for details.
