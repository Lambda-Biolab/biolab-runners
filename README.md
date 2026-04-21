# biolab-runners

[![CodeFactor](https://www.codefactor.io/repository/github/lambda-biolab/biolab-runners/badge)](https://www.codefactor.io/repository/github/lambda-biolab/biolab-runners)
[![CodeQL](https://github.com/Lambda-Biolab/biolab-runners/actions/workflows/codeql.yml/badge.svg)](https://github.com/Lambda-Biolab/biolab-runners/actions/workflows/codeql.yml)
[![Dependabot Updates](https://github.com/Lambda-Biolab/biolab-runners/actions/workflows/dependabot/dependabot-updates/badge.svg)](https://github.com/Lambda-Biolab/biolab-runners/actions/workflows/dependabot/dependabot-updates)

Standalone, modular Python runners for **Boltz-2 structure prediction** and **OpenMM molecular dynamics simulations**.

Extracted from the [OralBiome-AMP](https://github.com/Lambda-Biolab/OralBiome-AMP) pipeline for use in other research projects.

## Features

- **Boltz2Runner** — Local GPU structure prediction with quality gating, MSA caching, pocket constraints, and dry-run mode
- **OpenMMRunner** — Full MD pipeline: system building, 3-stage equilibration, production NPT, checkpointing, early abort, SIGTERM handling
- Config-driven with dataclasses (no magic strings)
- Structured result objects (not raw dicts)
- Full type annotations (pyright-clean)
- Python logging (no print statements)
- Dry-run mode for both runners

## Installation

```bash
# Core (no heavy dependencies)
pip install biolab-runners

# With Boltz-2 support
pip install biolab-runners[boltz2]

# With OpenMM support (conda recommended for GPU)
pip install biolab-runners[openmm]

# Everything
pip install biolab-runners[all]
```

For OpenMM with CUDA support, use conda:

```bash
conda install -c conda-forge openmm pdbfixer
pip install biolab-runners
```

## Quick Start

### Boltz-2 Structure Prediction

```python
from pathlib import Path
from biolab_runners.boltz2 import Boltz2Runner, Boltz2Config

# Configure
config = Boltz2Config(
    accelerator="gpu",
    recycling_steps=3,
    use_potentials=True,  # Steering potentials — substantially reduces clashes
)

# Run prediction
runner = Boltz2Runner(config)
result = runner.predict_complex(
    receptor_sequence="MVKLTAEG...",
    peptide_sequence="RWKLFKKIEK",
    name="demo_complex",
    output_dir=Path("results/predictions"),
)

# Check results
print(f"Quality: {result.quality_gate}")       # PASS / CONDITIONAL / FAIL
print(f"ipTM: {result.confidence.iptm:.3f}")   # Interface confidence
print(f"pTM: {result.confidence.ptm:.3f}")     # Overall fold quality
print(f"dG: {result.confidence.binding_affinity} kcal/mol")  # Binding affinity
print(f"Structure: {result.structure_path}")    # Path to PDB file
```

### Boltz-2 with Pocket Constraints

```python
result = runner.predict_complex(
    receptor_sequence="MVKLTAEG...",
    peptide_sequence="RWKLFKKIEK",
    name="constrained_pred",
    pocket_contacts=[("A", 123), ("A", 125), ("A", 156)],
)
```

### Boltz-2 Dry Run

```python
result = runner.predict_complex(
    receptor_sequence="MVKLTAEG...",
    peptide_sequence="RWKLFKKIEK",
    dry_run=True,  # Validates inputs, logs command, no GPU needed
)
```

### OpenMM MD Simulation

```python
from biolab_runners.openmm import OpenMMRunner, OpenMMConfig

# Configure simulation
config = OpenMMConfig(
    receptor_pdb="receptor.pdb",
    peptide_pdb="peptide.pdb",
    output_dir="results/md/demo",
    target="demo",
    peptide_id="PEP001",
    production_ns=100.0,            # 100 ns production run
    temperature_k=310.0,            # 37 C (body temperature)
    protein_ff="charmm36m",         # Force field
    openmm_platform="OpenCL",       # GPU platform
    target_irmsd_threshold_a=3.5,   # Early-abort reference (per-system)
)

# Run simulation
runner = OpenMMRunner(config)
result = runner.run()

print(f"Trajectory: {result.trajectory_path}")
print(f"Total: {result.total_ns} ns in {result.elapsed_seconds:.0f}s")
print(f"Performance: {result.ns_per_day:.0f} ns/day")
print(f"Early abort: {result.early_abort} ({result.abort_reason})")
```

### OpenMM Buffer Presets

`OpenMMConfig` ships with preset classmethods for common biological environments. Presets set ionic concentrations, pH, and temperature; all other fields can still be passed as keyword overrides.

```python
# Saliva-like (140 mM NaCl + 1.4 mM CaCl2 + 0.5 mM KH2PO4, pH 6.2, 310 K)
config = OpenMMConfig.saliva(
    receptor_pdb="receptor.pdb",
    peptide_pdb="peptide.pdb",
    output_dir="results/md/oral",
    production_ns=100.0,
)

# Physiological / PBS-like (150 mM NaCl, pH 7.4, 310 K)
config = OpenMMConfig.physiological(receptor_pdb=..., peptide_pdb=..., output_dir=...)

# Gastric fluid (150 mM NaCl, pH 2.0, 310 K)
config = OpenMMConfig.gastric(receptor_pdb=..., peptide_pdb=..., output_dir=...)

# Small-intestinal fluid (150 mM NaCl, pH 6.8, 310 K)
config = OpenMMConfig.intestinal(receptor_pdb=..., peptide_pdb=..., output_dir=...)
```

Caller keywords always win over preset values, so you can mix and match:

```python
# Physiological buffer but with a custom temperature
config = OpenMMConfig.physiological(
    receptor_pdb="rec.pdb",
    peptide_pdb="pep.pdb",
    output_dir="out/",
    temperature_k=300.0,
)
```

For environments not covered by a preset, instantiate `OpenMMConfig` directly and set `nacl_mol`, `cacl2_mol`, `kh2po4_mol`, `protonation_ph`, and `temperature_k` explicitly.

Note: very low pH (e.g. gastric) affects protonation of His/Asp/Glu/N-termini. Verify that the selected protein force field handles the target regime.

### OpenMM Dry Run

```python
result = runner.run(dry_run=True)  # Validates config, no GPU needed
```

## Quality Gates (Boltz-2)

Predictions are automatically classified:

| Gate | Criteria |
|------|----------|
| **PASS** | ipTM >= 0.7, pLDDT >= 70, no clashes |
| **CONDITIONAL** | ipTM 0.5-0.7, or pLDDT 50-70, or 1-3 mild clashes |
| **FAIL** | ipTM < 0.5, or pLDDT < 50, or severe clashes, or > 3 clashes |

## Early Abort (OpenMM)

The MD runner checks peptide stability at 5 ns and 10 ns:

- **5 ns check:** Peptide Cα RMSD vs post-equilibration reference. Abort if it exceeds `2 × config.target_irmsd_threshold_a`.
- **10 ns check:** RMSD slope between 5–10 ns. Abort if > 0.05 Å/ns (drift).

`target_irmsd_threshold_a` defaults to 3.5 Å, which is a reasonable mid-range value for peptide-protein complexes. Tighter binders (small pockets, rigid peptides) justify lower values; floppier binders justify higher values. Set it per system rather than relying on the default — binding-site geometry varies and there is no universal threshold.

## Equilibration Protocol

3-stage protocol for peptide-protein complexes:

1. **NVT 100 ps** — Strong backbone restraints (k=1000 kJ/mol/nm²)
2. **NPT 100 ps** — Reduced restraints (k=100 kJ/mol/nm²)
3. **NPT 200 ps** — Gradual ramp (100→0) + 100 ps unrestrained

Solvation: dodecahedral box with TIP3P water. Ionic conditions are configurable via `OpenMMConfig` fields or the buffer presets (`physiological`, `saliva`, `gastric`, `intestinal`). Defaults are physiological PBS-like (150 mM NaCl, pH 7.4, 310 K).

## Development

```bash
# Install dev dependencies
pip install -e ".[all]"
pip install ruff pyright pytest pytest-cov

# Lint
ruff check biolab_runners/ tests/
ruff format --check biolab_runners/ tests/

# Type check
pyright biolab_runners/

# Test
pytest tests/ -v
```

## License

MIT
