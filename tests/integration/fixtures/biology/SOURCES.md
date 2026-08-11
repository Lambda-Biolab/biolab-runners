# Scientific Validation — Reference Inputs

The fixtures in this directory are used by the integration tests
under `tests/integration/`. They are committed to the repo so that
any developer with the test suite gets the same reference inputs.

| File | Tool | Source | Size | License |
|---|---|---|---|---|
| `barnase_barstar_proteinmpnn.fa` | ProteinMPNN | Synthetic (matches ProteinMPNN's expected output format) | <1 KB | Project (CC0) |
| `ala2_vacuum.energy.xvg` | GROMACS | Synthetic (matches GROMACS energy.xvg format with a 5-row drift sanity-check sample) | <1 KB | Project (CC0) |
| `barnase_chainA.pdb` | OpenMM | RCSB 1BRS, chain A (barnase) only, heavy atoms | ~140 KB | Public domain |

## How to add a fixture

1. Keep the file under 500 KB. If you need more, document the
   alternative source in `SOURCES.md` and commit a one-time
   setup script in `scripts/` that re-acquires it.
2. Add a row to the table above with: tool, source, size, licence.
3. Citing the licence matters: PDB files are public domain; in-house
   structure files need an explicit licence comment in `SOURCES.md`.
