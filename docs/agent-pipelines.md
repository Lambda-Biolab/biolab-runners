# Agent pipeline composition

`biolab-runners` provides reusable runner boundaries. An AI agent composes
them by creating the tool-specific frozen config, calling the runner's public
entry point, validating the typed result, and handing explicit files or
records to the next stage. There is no universal `Runner` base class and no
implicit consumer pipeline.

## Composition rules

- Use the config and result type for the runner being called. Where exposed,
  shared `ExecutionStatus`, `ExecutionMode`, `ArtifactReference`, typed
  errors, and `ProvenanceMetadata` standardize handoffs; result fields remain
  runner-specific.
- For runners exposing shared fields, treat `status`, record statuses,
  required artifacts, and provenance as part of the handoff. OpenMM retains
  its legacy `SimulationResult` boundary: inspect `error` and its concrete
  artifact-path fields instead. Do not infer success from a directory
  existing or from an exit code alone.
- Keep consumer policy outside this package. Candidate registries, ranking,
  acceptance thresholds, finalist selection, campaign iteration, uploads, and
  cloud job orchestration belong to the consuming pipeline.

## Design: RFdiffusion → ProteinMPNN

The current design path is:

```text
RFdiffusionRunner.run(RFdiffusionConfig)
  → RFdiffusionResult.records: RecordData (full output PDB path + design sequence)
  → ProteinMPNNRunner.run(input_pdb, ProteinMPNNConfig)
  → ProteinMPNNResult.records: DesignRecord (FASTA path + designed sequence)
```

The public imports are `RFdiffusionConfig` and `RFdiffusionRunner` from
`biolab_runners.rfdiffusion`, and `ProteinMPNNConfig` and
`ProteinMPNNRunner` from `biolab_runners.proteinmpnn`. Both runners are
subprocess integrations and expose `run()` with `dry_run`, `force`, and
provenance-aware results.

RFdiffusion target-conditioned design requires `target_pdb` plus stock
chain-referencing `contigs`. `RFdiffusionResult.records` contains the full
target-plus-binder PDB in `RecordData.path`, while `RecordData.sequence` is
parsed from the generated design chain only. A consumer should pass a
successful record's PDB path to ProteinMPNN. Set
`ProteinMPNNConfig.extra["pdb_path_chains"]` to the space-separated
`RFdiffusionConfig.design_chains` so only generated binder chains are
designed and receptor chains remain fixed. Branch on ProteinMPNN's
`DesignRecordStatus` and result status before using a sequence.
RFdiffusion's head-to-tail mode is distinct from its disulfide intent:
residue-pair disulfides are downstream topology intent, not an RFdiffusion
feature.

`ProteinMPNNRunner.run_batch(inputs, config)` is available when the consumer
has already selected several backbone paths. Its result records contain the
designed FASTA sequences; ProteinMPNN does not perform downstream D-residue
conversion, cyclization, or candidate ranking.

## Rosetta scoring and the decoy artifact boundary

`RosettaRunner.run(RosettaConfig)` invokes the configured
`rosetta_scripts` executable and returns `RosettaResult`. The result contains
`RelaxRecord` entries parsed from `score.sc`, shared status/provenance, and
`ArtifactReference` values for score outputs. The config requires
`script_file`, `input_pdb`, and `license_acknowledged=True`; the library does
not ship or provide a licensed Rosetta runtime. Use `rosetta_available()` as
an availability probe, but do not treat availability as scientific
acceptance.

`RosettaDecoyArtifact` is a separate public, frozen handoff record imported
from `biolab_runners.rosetta`. `RosettaRunner` does not construct it. A
consumer that has verified a decoy can construct this boundary from:

- `PDBIdentity` values for input and output URIs with bare lowercase SHA-256
  digests;
- unique `ChainAudit` values;
- a typed `RelaxScore`; and
- candidate, parent, protocol, config, and runtime identities plus a terminal
  `ExecutionStatus`.

`RosettaDecoyArtifact.to_dict()` is the JSON-safe serialization boundary for
that validated decoy. The consumer remains responsible for deciding which
decoys pass its scientific and campaign policy.

## Peptide preparation and the full-complex boundary

`PeptidePrepRunner.run(PeptidePrepConfig, coordinate_transformer=...,`
`chirality_validator=...)` is an in-process preparation path. It consumes a
peptide backbone PDB, a designed canonical 20-letter sequence, a peptide
`chain_id`, and an optional `PeptideTopologyDescriptor`. Linear all-L
preparation needs no callbacks; D-substitution requires both callbacks and
their explicit identity strings in the config.

`PeptidePrepResult` reports `success`, `reused`, `error`, source/config
digests, and the required artifacts `prepared_pdb`, `gromacs_top`, and
`gromacs_gro`. The PDB and GROMACS exports come from the same prepared
OpenMM system; failed closure-integrity or parity checks do not produce a
successful prepared artifact.

This is peptide preparation, not a full-complex preparation service. The
package has no `ComplexPrepRunner`. An RFdiffusion output retains receptor
coordinates, so a consumer must explicitly select or produce the peptide
backbone input expected by `PeptidePrepConfig` rather than assuming that a
full target-plus-binder PDB is the peptide-prep boundary. Reusable
full-complex preparation is planned upstream. Target-specific chemistry and
campaign policy remain consumer-owned. Neither the reusable preparation
surface nor a qualified full-complex MD campaign is complete.

## GROMACS execution and resume

For a complete GROMACS protocol, import `GromacsProtocolConfig` and
`GromacsProtocolRunner` from `biolab_runners.gromacs` and call
`GromacsProtocolRunner.run_protocol(config)`. The returned
`GromacsProtocolResult` exposes per-stage `stage_statuses`, counters for
`succeeded`, `skipped`, `failed`, `interrupted`, and `validated`, plus a
normalized `status` property and artifact/provenance accessors.

The protocol runs topology, box, solvation, ions, minimization, NVT, NPT,
and production stages. Stages recorded as completed in the manifest are
skipped. An MD stage with a `.cpt` resumes with GROMACS checkpoint flags; a
stage without one starts fresh. SIGTERM leaves an interrupted stage resumable
when GROMACS has written its checkpoint. With both `prebuilt_topology` and
`prebuilt_coordinates` set on `GromacsProtocolConfig`, the caller's `.top`
and `.gro` are staged and the `pdb2gmx` topology stage is bypassed. The
remaining solvation and MD stages are unchanged.

`GromacsRunner` remains the separate one-shot wrapper for callers that
already have a `.tpr`. Neither GROMACS path submits cloud jobs; a
`binary_prefix` is an executable command boundary, not a cloud execution
implementation.

## OpenMM execution and resume

`OpenMMRunner(config).run(force=False, dry_run=False,
enable_early_abort=True)` returns `SimulationResult`. Its normal artifact
fields point to `trajectory.dcd`, `energy.csv`, the state XML, and
`topology.pdb`; the result also reports `total_ns`, `ns_per_day`, early-abort
fields, and `error`. This result does not expose the shared `status`,
`artifacts`, `provenance`, or `execution_mode` fields; use `error` plus the
concrete path fields at this legacy boundary.

The runner itself delegates run-state decisions to
`biolab_runners.openmm.run_state.decide(output_dir, config, force)`, which
returns the typed union `FreshPlan | ResumePlan | SkipPlan | FailurePlan`.
The canonical coherent checkpoint read is
`biolab_runners.openmm.checkpoint.inspect_checkpoint(output_dir, config)`;
`decide()` carries terminal fields on `SkipPlan`. Normally the agent should
call `OpenMMRunner.run()` and let the runner perform this dispatch. A normal
rerun skips a valid terminal result, an in-progress manifest resumes from its
validated state file, and `force=True` quarantines stale checkpoint files
before a fresh build. Invalid terminal data or unusable terminal artifacts
produce a failure result rather than a silent success.

`OpenMMConfig.from_md_spec(spec, **engine_overrides)` is the canonical bridge
from the engine-neutral `bioml_tools.md.system_spec.MDSpec`. The accepted
OpenMM-only overrides are `openmm_platform`, `water_ff_xml`,
`extra_forcefields`, and `target_irmsd_threshold_a`; engine-neutral protocol
fields stay on the `MDSpec`. OpenMM accepts separate receptor and peptide PDB
inputs, but that engine boundary does not provide full-complex preparation or
consumer-specific qualification.

## Optional decomposition

After a compatible MD result exists, construct `GmxMMPBSARunner` from
`biolab_runners.mmpbsa` with an `OpenMMConfig` and output `prefix` to run
optional per-residue gmx_MMPBSA decomposition. `run()` returns the legacy
JSON-stable dictionary with `status`, `per_residue_records`, and `error` plus
shared metadata. Missing optional tooling is `status="unsupported"`; a
successful invocation with absent required decomposition output is
`status="incomplete"`.

For tool-level smoke coverage and scientific-validation scope, use
[`docs/testing/scientific-validation.md`](testing/scientific-validation.md).
