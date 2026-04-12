# biolab-runners

## Project Purpose

Standalone, modular Python library containing two computational biology runners extracted from the [OralBiome-AMP](https://github.com/Lambda-Biolab/OralBiome-AMP) pipeline:

1. **Boltz2Runner** — Runs Boltz-2 structure predictions for peptide-protein complexes
2. **OpenMMRunner** — Runs OpenMM molecular dynamics simulations with multi-stage equilibration

The runners are designed for researchers who want to use these tools in their own pipelines without importing the full OralBiome-AMP codebase.

## Architecture

```
biolab_runners/
├── boltz2/
│   ├── config.py     # Boltz2Config, ConfidenceScores, PredictionResult, QualityGate
│   ├── runner.py     # Boltz2Runner class + apply_quality_gate()
│   └── utils.py      # YAML writer, output parser, availability check
└── openmm/
    ├── config.py     # OpenMMConfig, SimulationResult, EquilibrationStage
    ├── runner.py     # OpenMMRunner class (build, equilibrate, produce)
    └── utils.py      # Output verification, checkpoint loading, availability check
```

## How Boltz2Runner Works

**Input:** Receptor sequence + peptide sequence (strings), optional pocket constraints
**Output:** `PredictionResult` with structure path, confidence scores, quality gate

1. Writes a Boltz-2 v2 YAML input file (multi-chain, optional MSA paths, optional pocket constraints)
2. Invokes `boltz predict` CLI via subprocess with steering potentials enabled
3. Parses output: PDB structure + confidence JSON (pTM, ipTM, pLDDT, binding affinity)
4. Applies quality gate: PASS / CONDITIONAL / FAIL based on thresholds
5. Caches receptor MSA CSV for reuse across predictions against the same target

**Key config:** `Boltz2Config(accelerator, num_workers, recycling_steps, use_potentials, timeout_seconds)`

## How OpenMMRunner Works

**Input:** `OpenMMConfig` with receptor/peptide PDB paths, simulation parameters
**Output:** `SimulationResult` with trajectory DCD, energy CSV, state XML

1. Builds solvated system: PDBFixer → Modeller → addSolvent (dodecahedral box, 140mM NaCl)
2. Energy minimization (1000 iterations)
3. 3-stage equilibration:
   - NVT 100ps with strong backbone restraints (k=1000 kJ/mol/nm²)
   - NPT 100ps with reduced restraints (k=100)
   - NPT 200ps with gradual ramp (100→0) + 100ps unrestrained
4. Production NPT with periodic checkpointing
5. Early abort at 5ns/10ns if peptide dissociates (PBC-corrected RMSD)
6. SIGTERM handler for clean shutdown on cloud preemption

**Key config:** `OpenMMConfig(receptor_pdb, peptide_pdb, production_ns, temperature_k, protein_ff, openmm_platform)`

## Dependency Notes

### Boltz-2
- Requires `boltz` CLI on PATH: `pip install boltz>=2.2`
- GPU with 24 GB VRAM recommended (RTX 4090)
- Optional: `cuequivariance-torch` for custom CUDA kernels (set `no_kernels=False`)
- ColabFold MSA server used by default (`use_msa_server=True`)

### OpenMM
- Best installed via conda: `conda install -c conda-forge openmm pdbfixer`
- pip install works but only provides OpenCL platform (no CUDA)
- GPU required for reasonable performance (~290 ns/day on RTX 4090)
- Force fields: CHARMM36m bundled with OpenMM, TIP3P water model

## Linting Standards

- **ruff**: `line-length = 100`, `target-version = "py311"`, Google-style docstrings
- **pyright**: `typeCheckingMode = "basic"`, `reportMissingImports = false` (openmm may not be installed)
- **pytest**: All tests in `tests/`, use mocking to avoid needing actual binaries

Run checks:
```bash
ruff check biolab_runners/ tests/
ruff format --check biolab_runners/ tests/
pyright biolab_runners/
pytest tests/ -v
```

## How to Add a New Runner

1. Create `biolab_runners/new_runner/` with `__init__.py`, `config.py`, `runner.py`, `utils.py`
2. Define a config dataclass in `config.py` with all parameters
3. Define a result dataclass for structured output
4. Implement the runner class with:
   - `__init__(self, config)` — store config
   - `run()` (or `predict()`) — main entry point
   - `dry_run` parameter — validate without executing
   - Idempotency — skip if output already exists
   - Logging via `logging.getLogger(__name__)`
5. Add tests in `tests/test_new_runner.py` using mocks
6. Add optional extras in `pyproject.toml`
7. Export from `__init__.py`

## Common Gotchas

- **Boltz-2 pLDDT rescaling:** Boltz-2 v2 reports `complex_plddt` on 0-1 scale, but quality gates use 0-100. The parser auto-detects and rescales.
- **Boltz-2 steering potentials:** Without `--use_potentials`, Boltz-2 produces physically impossible structures (atoms overlapping at <2.0A) in 30-60% of predictions. Always enabled by default.
- **OpenMM platform:** pip-installed OpenMM only has OpenCL (not CUDA). conda-installed OpenMM has both. Set `openmm_platform="CUDA"` if using conda.
- **OpenMM resume:** Re-solvating on resume produces different water molecule counts (non-deterministic), causing atom count mismatch with state.xml. Always load the original topology.pdb on resume.
- **Restraint force on resume:** state.xml records the "k" parameter from equilibration restraints. The restraint force must be added to the system (with k=0) even on resume, or loadState() fails.
- **PBC correction:** Without minimum image convention, peptide RMSD can be ~100A due to periodic boundary crossings. All RMSD checks in the runner apply PBC correction.
- **Early abort thresholds:** Per-target iRMSD thresholds are expert-calibrated. Unknown targets use 3.5A default. The abort threshold is 2x the iRMSD threshold.
