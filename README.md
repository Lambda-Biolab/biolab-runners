# biolab-runners

[![Version](https://img.shields.io/badge/version-0.4.0-8A2BE2)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache_2.0-blue)](LICENSE)
[![Tests](https://github.com/Lambda-Biolab/biolab-runners/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Lambda-Biolab/biolab-runners/actions/workflows/ci.yml)
[![Dependabot Updates](https://github.com/Lambda-Biolab/biolab-runners/actions/workflows/dependabot/dependabot-updates/badge.svg?branch=main)](https://github.com/Lambda-Biolab/biolab-runners/actions/workflows/dependabot/dependabot-updates)
Standalone, modular Python runners for **Boltz-2** structure prediction, **OpenMM** molecular dynamics, **RFdiffusion** backbone generation, **ProteinMPNN** sequence design, **Rosetta** InterfaceAnalyzer, **GROMACS** integration, and **gmx_MMPBSA** per-residue decomposition.

Extracted from the [OralBiome-AMP](https://github.com/Lambda-Biolab/OralBiome-AMP) pipeline for use in other research projects.

## Features

- **Boltz2Runner** — Local GPU structure prediction with quality gating, MSA caching, pocket constraints, and dry-run mode
- **OpenMMRunner** — Full MD pipeline: system building, 3-stage equilibration, production NPT, checkpointing, early abort, SIGTERM handling. `OpenMMConfig.from_md_spec()` is the canonical construction path going forward (slice 12 — projects `bioml_tools.md.system_spec.MDSpec` onto the matching `OpenMMConfig` slot, with fail-closed allowlist for engine-specific overlays)
- **GmxMMPBSARunner** (slice 14) — Optional gmx_MMPBSA integration for per-residue MM/PBSA decomposition. Gracefully degrades to `status="unsupported"` when the binary is missing
- **RFdiffusionRunner** — Subprocess wrapper for RFdiffusion backbone generation (in-package `rfdiffusion` console script; `RFDIFFUSION_HOME` → upstream clone): target-conditioned peptide binder design (`target_pdb` → stock `inference.input_pdb` + chain-referencing `contigs`, byte-for-byte) or generic unconditional generation. Truthful topology modes — `inference.cyclic`/`cyc_chains` are emitted only for the head-to-tail variants; disulfide pairs are recorded as downstream closure intent (RFdiffusion cannot encode them)
- **ProteinMPNNRunner** — Subprocess wrapper for fixed-backbone sequence design (CLI translation from biolab-runners contract to upstream `protein_mpnn_run.py`)
- **RosettaRunner** — Subprocess wrapper for Rosetta InterfaceAnalyzer (license-gated; skips when the license is absent)
- **GROMACSRunner** — Subprocess wrapper for `.xvg` parsing integration
- **GROMACS protocol** — Checkpoint-resumable `pdb2gmx → box → solvate → ions → minimize → NVT → NPT → production` pipeline
  (`GromacsProtocolRunner`). Supports a **prebuilt mode**: when
  `GromacsProtocolConfig.prebuilt_topology` and
  `prebuilt_coordinates` are both set (e.g. from `PeptidePrepRunner`),
  the topology stage is skipped and the caller-supplied `.top`/`.gro`
  are staged into the canonical `topol.top`/`processed.gro` names;
  source digests are recorded in the stage manifest so a changed
  prebuilt source invalidates the cached downstream stages.
- **PeptidePrepRunner** — Local peptide preparation: threads a designed
  sequence onto a backbone PDB, applies optional D-substitution /
  head-to-tail / disulfide closure (via engine-neutral callback
  Protocols), minimises with a backbone restraint, and exports a
  parity-checked `prepared.pdb` + GROMACS `prepared.top`/`prepared.gro`
  of the *same* OpenMM system. Idempotent via a digest-bound manifest:
  the science config digest binds sequence/topology/force-field/
  physics/callback-identities (D-substitution configs require explicit
  `coordinate_transformer_identity` / `chirality_validator_identity`),
  while execution controls (`force`, `dry_run`, paths) are excluded so
  a force rebuild is reusable and dry-runs never poison production
  artifacts. Optional deps (OpenMM, PDBFixer, ParmEd) are
  lazy-imported; the runner fails closed with a structured result when
  an optional dependency is missing. See
  `biolab_runners.peptide_prep` for the full contract.
- Config-driven with dataclasses (no magic strings)
- Structured result objects (not raw dicts)
- Full type annotations (pyright-clean)
- Python logging (no print statements)
- Dry-run mode for all runners
- **Scientific-validation integration suite** — 8 tests gated by `@pytest.mark.integration` driving parsers and the OpenMM runner on real reference inputs (barnase chain A, ProteinMPNN FASTA, GROMACS energy.xvg). End-to-end CUDA smoke is opt-in via `BIOLAB_RUN_HEAVY_CUDA_TESTS=1`.

### Runner CLI contracts

The `Boltz2Runner`, `OpenMMRunner`, `RFdiffusionRunner`, and
`ProteinMPNNRunner` are pure-Python wrappers around their upstream
binaries. The runner-level CLI contract is what the biolab-runners
binary expects:

| Runner | Required upstream binary | CLI contract |
|---|---|---|
| `Boltz2Runner` | `boltz` (Conda) | upstream CLI passes through directly |
| `OpenMMRunner` | `python -m openmm` (Python lib) | n/a — no subprocess |
| `RFdiffusionRunner` | in-package `rfdiffusion` console script (or `${RFDIFFUSION_BIN}`); requires `RFDIFFUSION_HOME` → upstream clone with weights | `--output_dir DIR`, `--<dotted.hydra.key> <value>` pairs (underscores hyphenated), e.g. `--inference.num-designs N`, `--contigmap.contigs SPEC`, `--inference.input-pdb PDB`, `--inference.design-startnum N`, `--inference.cyclic True`, `--ppi.hotspot-res A51,A52` |
| `ProteinMPNNRunner` | `proteinmpnn` (or `${PROTEINMPNN_BIN}`) | `--input_path DIR`, `--output_path DIR`, `--batch_size N`, `--seed N`, `--sampling_temp F`, `--num_seq_per_target N`, `--model_name CHECKPOINT` (one of `v_48_002`, `v_48_010`, `v_48_020`, `v_48_030`), `--ca_only`, `--omit_AA`, `--fixed_positions` |
| `RosettaRunner` | Rosetta InterfaceAnalyzer | license-gated; skipped when absent |
| `GROMACSRunner` | n/a (parser-only) | fixture `.xvg` files |

The RFdiffusion adapter ships **in the package**: the installed wheel
provides the `rfdiffusion` console script
(`biolab_runners.rfdiffusion.cli`), which validates the flag pairs,
translates list-typed keys to Hydra list syntax
(`contigmap.contigs=[...]`, `ppi.hotspot_res=['A51','A52']`), quotes
string scalars only when needed (types preserved), and executes stock
`scripts/run_inference.py` with positional `key=value` overrides (no
shell). Runtime requirements: `RFDIFFUSION_HOME` pointing at the
upstream `RosettaCommons/RFdiffusion` clone root (default
`~/tools/RFdiffusion`), model weights downloaded, and a **Python with
PyTorch + CUDA** in the interpreter running the console script (the
clone is auto-prepended to `PYTHONPATH`; Hydra's own metadata is
confined under the output directory). `--help` needs none of these.
The host-level bootstrap script
`~/.local/bin/install-proteinmpnn-rfdiffusion.sh` clones the upstream
`dauparas/ProteinMPNN` and `RosettaCommons/RFdiffusion` repos shallowly
into `~/tools/` (satisfying the default `RFDIFFUSION_HOME`). After the
clone + weights are present, `biolab_runners.proteinmpnn.utils.proteinmpnn_available()`
and `biolab_runners.rfdiffusion.utils.rfdiffusion_available()` both
return True.

## Installation

### With `uv` (recommended — matches the project's lockfile)

```bash
# Clone and install all extras
git clone https://github.com/Lambda-Biolab/biolab-runners
cd biolab-runners
uv sync --all-extras
```

### With `pip`

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

### RFdiffusion Target-Conditioned Binder Design

```python
from pathlib import Path

from biolab_runners.rfdiffusion import RFdiffusionConfig, RFdiffusionRunner

# Target-conditioned peptide binder: two fixed target chains (A, B)
# followed by a generated 14-18-residue binder segment. ``contigs``
# is canonical stock syntax, forwarded byte-for-byte as
# ``contigmap.contigs``; ``target_pdb`` becomes ``inference.input_pdb``.
# ``target_pdb`` must point at a REAL structure on disk — a missing
# file fails closed, and binder contigs/hotspots without a target are
# rejected (stock upstream would silently fall back to its bundled
# example PDB).
config = RFdiffusionConfig(
    name="binder-campaign",
    target_pdb="path/to/real/target_complex.pdb",
    contigs="A1-110/0 B1-110/0 14-18",
    seed=7,
    task_count=10,
    mode="head_to_tail",          # cyclic binder (cyc_chains="a": first generated chain)
    hotspots=("A51", "A52"),      # optional binding-site residues
)

runner = RFdiffusionRunner(output_root=Path("results/backbones"))
result = runner.run(config, dry_run=True)  # validate inputs without a GPU
```

- `mode="linear"` — acyclic binder. `mode="disulfide"` — acyclic, with
  `disulfide_pairs` recorded as **downstream closure intent**:
  stock RFdiffusion cannot encode residue-pair disulfides, so no
  cyclic flags are emitted (`inference.cyclic`/`cyc_chains` express
  head-to-tail cyclization only). `mode="head_to_tail_and_disulfide"`
  — head-to-tail cyclic binder whose pairs are applied downstream
  (e.g. `biolab_runners.peptide_prep`).
- Binder contigs (chain references) and hotspots both require
  `target_pdb`, and a set-but-missing `target_pdb` is a hard error —
  all fail closed so a run can never silently design against the
  wrong structure. With an empty `target_pdb`, `contigs="14-18"` is
  generic **unconditional** generation — not a binder.
- **Generated-chain-aware output parsing (stock assignment)**: stock
  output PDBs carry the generated binder chain(s) **plus** the
  receptor chains copied from the target. The generated design's
  output chain is derived exactly as stock assigns it
  (`model_runners.py` `chain_idx`) — the lexicographically first
  ASCII letter not used by the contig-referenced receptor chains
  (receptor A+B → `C`, receptor A → `B`, unconditional → `A`) — and
  `RecordData.sequence` is parsed from that chain alone
  (`config.design_chains`, resolved from `contigs`; single source of
  truth, no override), never target+peptide. `RecordData.path` keeps
  the full target+binder complex so downstream interface filtering
  still has receptor coordinates. An output PDB missing the derived
  chain — or with no parseable residues — is a `failed` record (fail
  closed, never a fake-empty success), and the derivation itself
  fails closed on malformed/ambiguous contigs. `design_chains` is
  parse/provenance semantics, not a Hydra flag: it is bound into the
  cache identity but never forwarded upstream. `cyc_chains` lives in
  a separate **HAL space** (the internal chain-index space of
  `contigs.py`): `"a"` cyclizes the first generated chain — the
  binder — regardless of the output-PDB letter the binder gets.
- Runtime requirement: `RFDIFFUSION_HOME` must point at the upstream
  clone root (`~/tools/RFdiffusion` by default) with the model
  weights downloaded — the wheel's `rfdiffusion` console script
  resolves `scripts/run_inference.py` there.

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

`OpenMMConfig` ships with preset classmethods for common biological environments. Presets set ionic concentration (NaCl only — see caveat below), pH, and temperature; all other fields can still be passed as keyword overrides.

```python
# Saliva-like (140 mM NaCl, pH 6.2, 310 K)
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

For environments not covered by a preset, instantiate `OpenMMConfig` directly and set `nacl_mol`, `protonation_ph`, and `temperature_k` explicitly.

**Ionic strength:** the runner currently models only NaCl ionic strength (the `addSolvent(ionicStrength=…)` call takes a single value). The saliva preset's Ca²⁺ and KH₂PO₄ contributions from the original literature composition are documented in the docstring as unmodelled context — they don't reach the OpenMM call. Multi-ion modeling is future work.

Note: very low pH (e.g. gastric) affects protonation of His/Asp/Glu/N-termini. Verify that the selected protein force field handles the target regime.

### OpenMM Dry Run

```python
result = runner.run(dry_run=True)  # Validates config, no GPU needed
```

### OpenMM: canonical `MDSpec` → `OpenMMConfig` (slice 12)

For new code, prefer the canonical construction path over the direct
`OpenMMConfig(...)` keyword form. The `MDSpec` from
[`bioml_tools.md.system_spec`](https://github.com/Lambda-Biolab/bioml-tools)
is the engine-neutral source of truth for the MD protocol; pass it to
`OpenMMConfig.from_md_spec()` and add OpenMM-only overlays as needed.

```python
from biolab_runners.openmm import OpenMMRunner
from biolab_tools.md.system_spec import ACTIVIN_E_PRODUCTION_PROFILE, MDSpec

# Canonical production profile (Amber99SB-ILDN / TIP3P / cubic / 1.0 nm padding /
# 0.150 M NaCl / 310 K / PME / 50k-iter minimization cap / 100+100+200 ps
# equilibration / 200 ns production). Defined in bioml-tools 1.9.0.
spec = MDSpec.from_profile(
    ACTIVIN_E_PRODUCTION_PROFILE,
    receptor_pdb="receptor.pdb",
    peptide_pdb="peptide.pdb",
    output_dir="results/md/demo",
    target="demo",
    peptide_id="PEP001",
)

# Engine-specific overlays are passed via **engine_overrides; unknown
# keys raise TypeError at the construction boundary.
config = OpenMMConfig.from_md_spec(
    spec,
    openmm_platform="OpenCL",
    target_irmsd_threshold_a=3.5,
)

runner = OpenMMRunner(config)
result = runner.run()
```

The allowlist on `**engine_overrides` is fail-closed: only
`openmm_platform`, `water_ff_xml`, `extra_forcefields`, and
`target_irmsd_threshold_a` are accepted. Engine-neutral fields
(force field, water model, box geometry, ionic strength, temperature,
production ns) live on the `MDSpec` and must be changed there, not
via `from_md_spec`.

### gmx_MMPBSA per-residue decomposition (slice 14)

`GmxMMPBSARunner` (in `biolab_runners.mmpbsa`) drives the
[gmx_MMPBSA](https://github.com/Valdes-Tresanco-MS/gmx_MMPBSA) CLI
for per-residue MM/PBSA decomposition of a finished MD trajectory.
The runner is **opt-in** — when the binary is missing (no AmberTools
install, no container), `run()` returns `status="unsupported"` and
an empty `per_residue_records` list, not a fabricated value. This
matches the slice-14 acceptance contract: missing optional tooling
must not break the calling pipeline.

```python
from biolab_runners.mmpbsa import GmxMMPBSARunner, GmxMMPBSAStatus
from biolab_runners.openmm import OpenMMConfig

config = OpenMMConfig(
    receptor_pdb="receptor.pdb",
    peptide_pdb="peptide.pdb",
    output_dir="results/md/demo",
    target="demo",
    peptide_id="PEP001",
)

runner = GmxMMPBSARunner(
    config=config,
    prefix="demo_residue_decomposition",
    mmpbsa_binary="gmx_MMPBSA",  # or "container://docker://valdes-tre/msdocker:latest"
)
result = runner.run()

if result["status"] == GmxMMPBSAStatus.SUCCEEDED:
    for record in result["per_residue_records"]:
        print(f"{record.residue} ({record.chain}): "
              f"ΔG = {record.total_A:.2f} kcal/mol")
elif result["status"] == GmxMMPBSAStatus.UNSUPPORTED:
    # gmx_MMPBSA not on PATH; slice 14 graceful-degradation path.
    logger.info("Per-residue decomposition skipped: gmx_MMPBSA not installed")
```

The `mmpbsa_binary` accepts either a bare command name (PATH lookup)
or a `container://<engine>://<image>` URL (delegated to whichever
container runtime you've configured). `parse_residue_decomposition`
exposes the per-residue `.dat` parser for direct use without
re-running gmx_MMPBSA.

## Quality Gates (Boltz-2)

Predictions are automatically classified into `PASS` / `CONDITIONAL` / `FAIL`
based on ipTM, pLDDT, and clash counts. The thresholds are the single source
of truth in `biolab_runners.boltz2.config` (constants `IPTM_PASS`,
`IPTM_CONDITIONAL`, `PLDDT_PASS`, `PLDDT_CONDITIONAL`, `CONFIDENCE_SCORE_*`,
and the per-atom clash heuristic in `apply_quality_gate`).

## Early Abort (OpenMM)

The MD runner checks peptide stability at 5 ns and 10 ns:

- **5 ns check:** Peptide Cα RMSD vs post-equilibration reference. Abort if it exceeds `2 × config.target_irmsd_threshold_a`.
- **10 ns check:** RMSD slope between 5–10 ns. Abort if > 0.05 Å/ns (drift).

`target_irmsd_threshold_a` defaults to 3.5 Å, which is a reasonable mid-range value for peptide-protein complexes. Tighter binders (small pockets, rigid peptides) justify lower values; floppier binders justify higher values. Set it per system rather than relying on the default — binding-site geometry varies and there is no universal threshold.

## Equilibration Protocol

3-stage protocol (defined in `biolab_runners.openmm.runner._run_equilibration`):

1. **NVT 100 ps** — Strong backbone restraints (k=1000 kJ/mol/nm²)
2. **NPT 100 ps** — Reduced restraints (k=100 kJ/mol/nm²)
3. **NPT 200 ps** — Gradual ramp (100→0) + 100 ps unrestrained

Solvation: dodecahedral box with TIP3P water. Ionic conditions are configurable via `OpenMMConfig` fields or the buffer presets (`physiological`, `saliva`, `gastric`, `intestinal`). Defaults are physiological PBS-like (150 mM NaCl, pH 7.4, 310 K).

## Development

The full gate (`make validate`) runs ruff → pyright → complexipy → pytest and
is CI-safe (read-only).

```bash
# Install dev dependencies (with uv)
uv sync --all-extras

# Or with pip
pip install -e ".[all]"

# Full validation gate (lint + type + complexity + tests)
make validate

# Auto-fix lint/format issues
make lint

# Run tests only
make test

# Integration suite (gated by GPU + binary availability)
# Verifies scientific contracts on real reference inputs:
#   - ProteinMPNN FASTA parser
#   - GROMACS energy.xvg parser
#   - OpenMM minimization physics on barnase chain A
#   - OpenMM CUDA Plugin registration
#   - OpenMMRunner construction smoke
#   - Heavy runner pipeline (set BIOLAB_RUN_HEAVY_CUDA_TESTS=1 for 90-s CUDA end-to-end)
pytest -m integration
```

See `docs/testing/scientific-validation.md` for the validation plan,
threshold sources, and the failure-classification rule.

## License

[Apache License 2.0](LICENSE) — Copyright © 2026 Lambda Biolab.
