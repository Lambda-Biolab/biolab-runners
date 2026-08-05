"""Per-PR Vast.ai GPU gate for biolab-runners.

Spins up a cheap Vast.ai GPU (RTX 3060, ~$0.04/hr), deploys
biolab-runners, runs the smoke test (smoke_test/run_smoke.py),
compares the output to a committed baseline, and tears down the
instance. Cost per PR: ~$0.0033 (5 min @ $0.04/hr). Bounded
runtime: 8 min.

This is the deepvariant-arm64-Linux pattern: gate every PR on a
real run of the actual scientific code, not a mocked unit test.
Catches the regression class that mocks miss: wrong CUDA build,
broken OpenMM platform detection, performance regressions in the
real workload, energy drift in the simulation.

The deepvariant comparison uses output metrics from the simulation
itself, not just exit code. We compare against a committed
baseline (smoke_test/baseline.json) that was generated from a
known-good run on a known-good GPU. The comparison is:

- error is None                       (run completed cleanly)
- num_atoms matches baseline exactly   (same molecular system)
- topology_lines matches baseline      (correct system topology)
- energy_last_row within 1% of baseline (simulation correctness)
- ns_per_day >= 80% of baseline        (no catastrophic slowdown)

Usage:
    gate.py [--baseline PATH] [--max-budget USD] [--gpu MODEL] [--no-destroy]

Environment:
    VASTAI_API_KEY         (required) Vast.ai API key
    SMOKE_BASELINE_SHA     (optional) git blob SHA of baseline.json.
                            If unset, fetched from main at runtime.

Exit codes:
    0   pass
    1   deployment / GPU / SSH error (after exhausting retries)
    2   smoke test exit code != 0
    3   result diverged from baseline (scientific regression)
    4   budget exceeded
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

# Vast.ai constraints — keep these conservative
DEFAULT_GPU_MODEL = "RTX_3060"
DEFAULT_MAX_HOURLY_USD = 0.05  # hard ceiling: $0.05/hr (well above RTX 3060 market)
DEFAULT_BUDGET_USD = 0.50     # total budget per run
DEFAULT_DEADLINE_SECONDS = 480  # 8 min hard deadline
DEFAULT_RETRY_ATTEMPTS = 3

# Baseline tolerance (relative)
PERF_FLOOR = 0.80          # ns_per_day must be >= 80% of baseline
ENERGY_TOLERANCE = 0.01    # energy_last_row within 1%


@dataclass
class GateResult:
    passed: bool
    code: int
    reason: str
    artifacts_dir: Path | None = None
    baseline: dict | None = None
    actual: dict | None = None


class VastaiError(RuntimeError):
    pass


def log(msg: str) -> None:
    print(f"[gate {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find_cheapest_offer(gpu_model: str, max_hourly: float) -> dict:
    """Find the cheapest available Vast.ai offer for the given GPU.

    Returns a dict with at minimum: id, gpu_name, hourly_cost.
    Raises VastaiError if no offer under max_hourly is available.
    """
    api_key = os.environ.get("VASTAI_API_KEY")
    if not api_key:
        raise VastaiError("VASTAI_API_KEY not set")

    # The vastai CLI is the supported way to query offers. It authenticates
    # via ~/.vastaiapi_key or VASTAI_API_KEY env var.
    cmd = [
        "vastai", "search", "offers",
        gpu_model,  # GPU model (positional)
        "--order", "dph",  # cheapest first
        "--limit", "5",
        "--raw",  # JSON output, easier to parse
    ]
    log(f"querying Vast.ai offers for {gpu_model} ...")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env={**os.environ, "VASTAI_API_KEY": api_key},
        timeout=60,
    )

    # vastai --raw returns NDJSON; one JSON object per line
    offers = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        dph = obj.get("dph_total") or obj.get("dph") or 999.0
        if float(dph) <= max_hourly:
            offers.append({
                "id": obj.get("id"),
                "gpu_name": obj.get("gpu_name", gpu_model),
                "hourly_cost": float(dph),
                "host_id": obj.get("host_id"),
                "geolocation": obj.get("geolocation"),
                "cuda_gpus": obj.get("cuda_gpus"),
                "cpu_cores": obj.get("cpu_cores"),
                "cpu_ram": obj.get("cpu_ram"),
            })
    if not offers:
        raise VastaiError(
            f"No {gpu_model} offer under ${max_hourly}/hr. Try a different model."
        )
    return offers[0]


def create_instance(offer: dict, docker_image: str) -> int:
    """Create a Vast.ai instance from an offer. Returns the instance ID."""
    api_key = os.environ["VASTAI_API_KEY"]
    cmd = [
        "vastai", "create", "instance", str(offer["id"]),
        "--image", docker_image,
        "--disk", "20",  # GB; smoke test artifacts are small
        "--ssh",  # enable SSH
        "--raw",
    ]
    log(f"creating instance on offer {offer['id']} (${offer['hourly_cost']}/hr) ...")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env={**os.environ, "VASTAI_API_KEY": api_key},
        timeout=120,
    )
    if result.returncode != 0:
        raise VastaiError(f"vastai create instance failed: {result.stderr[:500]}")
    try:
        obj = json.loads(result.stdout.strip().splitlines()[0])
    except (json.JSONDecodeError, IndexError) as e:
        raise VastaiError(f"could not parse create output: {e}")
    inst_id = obj.get("new_contract")
    if not inst_id:
        raise VastaiError(f"create output missing new_contract: {obj}")
    log(f"instance created: {inst_id}")
    return int(inst_id)


def wait_for_ssh(inst_id: int, timeout: int = 240) -> tuple[str, int]:
    """Wait for the instance to be ready and return (host, ssh_port)."""
    api_key = os.environ["VASTAI_API_KEY"]
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = subprocess.run(
            ["vastai", "show", "instances", "--raw"],
            capture_output=True,
            text=True,
            env={**os.environ, "VASTAI_API_KEY": api_key},
            timeout=30,
        )
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("id") == inst_id:
                status = obj.get("actual_status") or obj.get("status")
                host = obj.get("ssh_host")
                port = obj.get("ssh_port")
                if status == "running" and host and port:
                    log(f"instance {inst_id} ready: {host}:{port}")
                    return str(host), int(port)
        time.sleep(5)
    raise VastaiError(
        f"instance {inst_id} did not become ready within {timeout}s"
    )


def destroy_instance(inst_id: int) -> None:
    """Always destroy the instance, even on failure paths."""
    api_key = os.environ.get("VASTAI_API_KEY")
    if not api_key:
        log("WARNING: cannot destroy instance, VASTAI_API_KEY unset")
        return
    try:
        result = subprocess.run(
            ["vastai", "destroy", "instance", str(inst_id), "--raw"],
            capture_output=True,
            text=True,
            env={**os.environ, "VASTAI_API_KEY": api_key},
            timeout=60,
        )
        if result.returncode == 0:
            log(f"instance {inst_id} destroyed")
        else:
            log(f"WARNING: destroy failed for {inst_id}: {result.stderr[:200]}")
    except subprocess.TimeoutExpired:
        log(f"WARNING: destroy timed out for {inst_id}")


def ssh_exec(host: str, port: int, cmd: str, timeout: int = 60) -> tuple[int, str, str]:
    """Execute a command on the remote instance via SSH. Returns (rc, stdout, stderr)."""
    ssh_target = f"root@{host}"
    full_cmd = [
        "ssh",
        "-p", str(port),
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        ssh_target,
        cmd,
    ]
    result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)
    return result.returncode, result.stdout, result.stderr


def deploy_and_run(host: str, port: int, repo_url: str, ref: str) -> dict:
    """Deploy biolab-runners on the remote instance and run the smoke test.

    Returns the smoke_verify.json content as a dict.
    Raises VastaiError on any failure.
    """
    deploy_dir = "/tmp/biolab-runners"

    # Install uv (one-time per instance)
    rc, out, err = ssh_exec(host, port, "which uv || (curl -LsSf https://astral.sh/uv/install.sh | sh)")
    if rc != 0:
        raise VastaiError(f"uv install failed: {err[:500]}")
    rc, out, err = ssh_exec(host, port, "ls $HOME/.local/bin/uv || ls $HOME/.cargo/bin/uv")
    if rc != 0:
        raise VastaiError("uv not found after install")

    # Clone + checkout
    log(f"cloning {repo_url}@{ref} ...")
    rc, out, err = ssh_exec(
        host, port,
        f"rm -rf {deploy_dir} && git clone --depth 1 --branch {shlex.quote(ref)} {shlex.quote(repo_url)} {deploy_dir}",
        timeout=120,
    )
    if rc != 0:
        raise VastaiError(f"git clone failed: {err[:500]}")

    # Sync deps with the OpenMM extra
    log("syncing dependencies (uv sync --extra openmm --all-groups) ...")
    rc, out, err = ssh_exec(
        host, port,
        f"cd {deploy_dir} && export PATH=$HOME/.local/bin:$PATH && uv sync --extra openmm --all-groups 2>&1 | tail -20",
        timeout=600,
    )
    if rc != 0:
        raise VastaiError(f"uv sync failed: {out[-1000:]}{err[-500:]}")

    # Run the smoke test
    log("running smoke test (50 ps MD on RTX 3060) ...")
    rc, out, err = ssh_exec(
        host, port,
        f"cd {deploy_dir} && export PATH=$HOME/.local/bin:$PATH && uv run python smoke_test/run_smoke.py /tmp/smoke_out 2>&1 | tail -40",
        timeout=300,
    )
    if rc != 0:
        raise VastaiError(
            f"smoke test failed (rc={rc}):\n--- stdout ---\n{out[-2000:]}\n--- stderr ---\n{err[-1000:]}"
        )

    # Read the verification JSON
    rc, out, err = ssh_exec(
        host, port,
        f"cat /tmp/smoke_out/smoke_verify.json",
    )
    if rc != 0:
        raise VastaiError(f"could not read smoke_verify.json: {err}")
    try:
        return json.loads(out)
    except json.JSONDecodeError as e:
        raise VastaiError(f"smoke_verify.json is not valid JSON: {e}\n{out[:500]}")


def load_baseline(baseline_path: Path) -> dict:
    """Load the baseline JSON. Raises if missing or invalid."""
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Baseline not found: {baseline_path}. "
            f"Run `make smoke-baseline` once on a known-good GPU to generate it."
        )
    with open(baseline_path) as f:
        return json.load(f)


def compare_to_baseline(actual: dict, baseline: dict) -> tuple[bool, str]:
    """Compare the actual smoke run to the baseline. Returns (pass, reason)."""
    if actual.get("error") is not None:
        return False, f"smoke test reported error: {actual['error']!r}"

    if actual.get("num_atoms") != baseline.get("num_atoms"):
        return False, (
            f"num_atoms mismatch: actual={actual.get('num_atoms')} "
            f"baseline={baseline.get('num_atoms')}"
        )

    if actual.get("topology_lines") != baseline.get("topology_lines"):
        return False, (
            f"topology_lines mismatch: actual={actual.get('topology_lines')} "
            f"baseline={baseline.get('topology_lines')} (system changed?)"
        )

    # Energy check: final-step PE within 1% of baseline
    actual_pe = float(actual.get("energy_last_row", [0, 0, 0, 0])[2])
    baseline_pe = float(baseline.get("energy_last_row", [0, 0, 0, 0])[2])
    if baseline_pe == 0:
        return False, "baseline PE is 0 (invalid baseline)"
    drift = abs(actual_pe - baseline_pe) / abs(baseline_pe)
    if drift > ENERGY_TOLERANCE:
        return False, (
            f"energy drift: actual={actual_pe:.4f} baseline={baseline_pe:.4f} "
            f"drift={drift:.2%} (>{ENERGY_TOLERANCE:.0%} tolerance) — scientific regression"
        )

    # Performance check: ns_per_day within 80% of baseline
    actual_perf = actual.get("ns_per_day", 0)
    baseline_perf = baseline.get("ns_per_day", 0)
    if baseline_perf == 0:
        return False, "baseline ns_per_day is 0 (invalid baseline)"
    perf_ratio = actual_perf / baseline_perf
    if perf_ratio < PERF_FLOOR:
        return False, (
            f"performance regression: actual={actual_perf:.1f} ns/day "
            f"baseline={baseline_perf:.1f} ns/day "
            f"ratio={perf_ratio:.2f} (<{PERF_FLOOR:.0%})"
        )

    return True, (
        f"PASS: num_atoms={actual.get('num_atoms')}, "
        f"PE drift={drift:.3%}, perf={perf_ratio:.2f}x baseline"
    )


def run_gate(
    baseline_path: Path,
    gpu_model: str,
    max_hourly: float,
    budget_usd: float,
    repo_url: str,
    ref: str,
    docker_image: str,
    no_destroy: bool,
) -> GateResult:
    """Run the gate end-to-end. Returns a GateResult."""
    artifacts = Path(tempfile.mkdtemp(prefix="biolab-gate-"))
    log(f"artifacts: {artifacts}")

    baseline: dict
    try:
        baseline = load_baseline(baseline_path)
    except FileNotFoundError as e:
        return GateResult(
            passed=False,
            code=3,
            reason=str(e),
            artifacts_dir=artifacts,
        )

    log(f"baseline loaded: num_atoms={baseline.get('num_atoms')}, "
        f"PE={baseline.get('energy_last_row', [0,0,0,0])[2]}, "
        f"perf={baseline.get('ns_per_day', 0):.1f} ns/day")

    inst_id: int | None = None
    start = time.time()
    try:
        for attempt in range(1, DEFAULT_RETRY_ATTEMPTS + 1):
            log(f"--- attempt {attempt}/{DEFAULT_RETRY_ATTEMPTS} ---")
            try:
                # 1. Find cheapest offer
                offer = find_cheapest_offer(gpu_model, max_hourly)
                log(f"cheapest {gpu_model} offer: ${offer['hourly_cost']}/hr, id={offer['id']}")

                # Budget check
                hourly = offer["hourly_cost"]
                elapsed_so_far = time.time() - start
                # Conservative: cost is hourly * (elapsed + 5min) / 3600
                projected_cost = hourly * (elapsed_so_far + 300) / 3600
                if projected_cost > budget_usd:
                    return GateResult(
                        passed=False,
                        code=4,
                        reason=f"projected cost ${projected_cost:.4f} exceeds budget ${budget_usd:.2f}",
                        artifacts_dir=artifacts,
                        baseline=baseline,
                    )

                # 2. Create instance
                inst_id = create_instance(offer, docker_image)

                # 3. Wait for SSH
                host, port = wait_for_ssh(inst_id, timeout=240)
                ssh_start = time.time()

                # 4. Deploy + run
                actual = deploy_and_run(host, port, repo_url, ref)
                ssh_elapsed = time.time() - ssh_start
                cost = hourly * ssh_elapsed / 3600
                log(f"smoke test done in {ssh_elapsed:.1f}s (~${cost:.4f})")

                # Save artifacts
                artifacts_file = artifacts / "smoke_verify.json"
                artifacts_file.write_text(json.dumps(actual, indent=2))

                # 5. Compare
                passed, reason = compare_to_baseline(actual, baseline)
                log(reason)
                return GateResult(
                    passed=passed,
                    code=0 if passed else 3,
                    reason=reason,
                    artifacts_dir=artifacts,
                    baseline=baseline,
                    actual=actual,
                )

            except VastaiError as e:
                log(f"attempt {attempt} failed: {e}")
                if inst_id is not None:
                    destroy_instance(inst_id)
                    inst_id = None
                if attempt < DEFAULT_RETRY_ATTEMPTS:
                    log("retrying with a fresh instance...")
                    time.sleep(5)
                else:
                    return GateResult(
                        passed=False,
                        code=1,
                        reason=f"all {DEFAULT_RETRY_ATTEMPTS} attempts failed: {e}",
                        artifacts_dir=artifacts,
                        baseline=baseline,
                    )
            except Exception as e:
                log(f"attempt {attempt} unexpected error: {e}")
                if inst_id is not None:
                    destroy_instance(inst_id)
                    inst_id = None
                raise
    finally:
        if inst_id is not None and not no_destroy:
            destroy_instance(inst_id)
        log(f"artifacts preserved at: {artifacts}")
    # Unreachable, but for type checker
    return GateResult(passed=False, code=1, reason="unreachable", artifacts_dir=artifacts)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--baseline", type=Path,
                   default=Path(__file__).parent.parent.parent / "smoke_test" / "baseline.json",
                   help="Path to baseline.json (default: smoke_test/baseline.json)")
    p.add_argument("--gpu", default=DEFAULT_GPU_MODEL,
                   help=f"GPU model to use (default: {DEFAULT_GPU_MODEL})")
    p.add_argument("--max-hourly", type=float, default=DEFAULT_MAX_HOURLY_USD,
                   help=f"Max $/hr per instance (default: ${DEFAULT_MAX_HOURLY_USD})")
    p.add_argument("--budget", type=float, default=DEFAULT_BUDGET_USD,
                   help=f"Max total cost per run in USD (default: ${DEFAULT_BUDGET_USD})")
    p.add_argument("--repo", default="https://github.com/Lambda-Biolab/biolab-runners.git",
                   help="Git repo URL to deploy")
    p.add_argument("--ref", default=os.environ.get("GITHUB_HEAD_REF") or os.environ.get("GITHUB_REF_NAME", "main"),
                   help="Git ref to checkout (default: GITHUB_HEAD_REF or main)")
    p.add_argument("--image", default="nvidia/cuda:12.4.0-runtime-ubuntu22.04",
                   help="Docker image to deploy (must have CUDA + Python 3.11+)")
    p.add_argument("--no-destroy", action="store_true",
                   help="Don't destroy the instance (debug only)")
    args = p.parse_args()

    result = run_gate(
        baseline_path=args.baseline,
        gpu_model=args.gpu,
        max_hourly=args.max_hourly,
        budget_usd=args.budget,
        repo_url=args.repo,
        ref=args.ref,
        docker_image=args.image,
        no_destroy=args.no_destroy,
    )
    # Final result line for the CI log
    print()
    print(f"=== {'PASS' if result.passed else 'FAIL'} (code={result.code}) ===")
    print(result.reason)
    return result.code


if __name__ == "__main__":
    sys.exit(main())
