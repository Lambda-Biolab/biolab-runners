# Security Policy

## Supported Versions

This project is in early development (`0.1.x`). Only the latest minor
version receives security fixes.

| Version | Supported          |
|---------|--------------------|
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

**Do not** open a public GitHub issue for security vulnerabilities.

Email **antonio@sp2tx.com** with:

1. A description of the vulnerability and its impact
2. Steps to reproduce, ideally with a minimal code snippet
3. The affected version(s)
4. Your assessment of severity (critical / high / medium / low)

You will receive an acknowledgement within 72 hours and a remediation
plan within 7 days. We follow [coordinated disclosure](https://en.wikipedia.org/wiki/Coordinated_vulnerability_disclosure):
please give us a reasonable window to fix the issue before publishing
details.

## Scope

In-scope:

- Code execution from untrusted input (PDB / trajectory / YAML parsing)
- Sandbox escapes in the OpenMM or Boltz-2 subprocess calls
- Path traversal in output directory handling
- Pickle / deserialization in checkpoint resume
- Subprocess command injection (the boltz CLI is invoked with
  user-controlled args)

Out of scope:

- Vulnerabilities in upstream dependencies (file with the upstream
  project; we will bump the version)
- Denial-of-service via pathological but valid input (e.g. a 1 GB PDB)

## Notes

This library shells out to the `boltz` CLI and loads OpenMM C extensions.
Both run in the developer's environment; there is no service-side attack
surface. The realistic threat model is a malicious PDB or YAML file
tricking the runner into executing attacker-controlled code during
parsing or subprocessing.
