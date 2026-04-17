# biolab-runners

> Standalone, modular Python runners for Boltz-2 structure prediction and OpenMM molecular dynamics simulations.

## Documentation

- [README](${BLOB}/README.md): Project overview, architecture, usage, and dependency notes
- [CLAUDE.md](${BLOB}/CLAUDE.md): Agent instructions, architecture details, and common gotchas

## Source — biolab_runners

### Core

- [biolab_runners/__init__.py](${BLOB}/biolab_runners/__init__.py): Package root and version

### Boltz-2 structure prediction (boltz2/)

- [boltz2/runner.py](${BLOB}/biolab_runners/boltz2/runner.py): Boltz2Runner class — invokes boltz predict CLI, parses confidence scores, applies quality gate
- [boltz2/config.py](${BLOB}/biolab_runners/boltz2/config.py): Boltz2Config, ConfidenceScores, PredictionResult, QualityGate dataclasses
- [boltz2/utils.py](${BLOB}/biolab_runners/boltz2/utils.py): YAML input writer, output parser, boltz CLI availability check

### OpenMM molecular dynamics (openmm/)

- [openmm/runner.py](${BLOB}/biolab_runners/openmm/runner.py): OpenMMRunner class — system building, multi-stage equilibration, production NPT with checkpointing
- [openmm/config.py](${BLOB}/biolab_runners/openmm/config.py): OpenMMConfig, SimulationResult, EquilibrationStage dataclasses
- [openmm/utils.py](${BLOB}/biolab_runners/openmm/utils.py): Output verification, checkpoint loading, OpenMM availability check
