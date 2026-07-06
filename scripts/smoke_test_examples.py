#!/usr/bin/env python3
"""Smoke test runner for WanPhys examples.

Auto-discovers runnable examples under wanphys/examples/, runs each in headless
mode, and checks both exit codes and output for errors.

Usage:
    uv run python smoke_test_examples.py
    uv run python smoke_test_examples.py --pattern rigid
    uv run python smoke_test_examples.py --verbose --timeout 180
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────

EXAMPLES_ROOT = Path(__file__).resolve().parent.parent / "wanphys" / "examples"

SKIP_FILENAMES = {"__init__.py", "utils.py", "task2_examples_utils.py"}

# Patterns that indicate a real error in output
FATAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^Traceback \(most recent call last\):", re.MULTILINE),
    re.compile(r"(?:^|\s)((?:\w+\.)*\w*Error):", re.MULTILINE),
    re.compile(r"(?:^|\s)((?:\w+\.)*\w*Exception):", re.MULTILINE),
    re.compile(r"\bSIGSEGV\b", re.IGNORECASE),
    re.compile(r"\bsegfault\b", re.IGNORECASE),
    re.compile(r"\bpanic\b", re.IGNORECASE),
    re.compile(r"\bkilled\b", re.IGNORECASE),
    re.compile(r"\bFAILED\b"),
]

# Lines matching these are false positives — skip them before checking FATAL
FALSE_POSITIVE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"DeprecationWarning", re.IGNORECASE),
    re.compile(r"no error", re.IGNORECASE),
    re.compile(r"error_count\s*=\s*0", re.IGNORECASE),
    re.compile(r"raise_on_error", re.IGNORECASE),
    re.compile(r"NumPy\.error", re.IGNORECASE),
    re.compile(r"\bseterr\b", re.IGNORECASE),
    re.compile(r"\bErrorState\b"),
    re.compile(r"error.*(handling|message|code|log)", re.IGNORECASE),
    re.compile(r"FutureWarning", re.IGNORECASE),
    re.compile(r"UserWarning", re.IGNORECASE),
    re.compile(r"RuntimeWarning", re.IGNORECASE),
    # Warp module load messages that contain "error" incidentally
    re.compile(r"Module .* load on device"),
]

TORCH_DEPENDENCY_SKIP_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"No module named ['\"]torch['\"]"), "optional torch dependency is not installed"),
    (re.compile(r"requires the optional torch dependency", re.IGNORECASE), "optional torch dependency is not installed"),
]

ALWAYS_SKIP_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"requires Newton hydroelastic geometry support", re.IGNORECASE),
        "Newton hydroelastic geometry support is not available",
    ),
]


# ── Data ─────────────────────────────────────────────────────────────────────


@dataclass
class ExampleInfo:
    """Metadata about a discovered example."""

    module: str  # e.g. "wanphys.examples.rigid_pendulum"
    path: Path
    cli_args: list[str] = field(default_factory=list)


@dataclass
class RunResult:
    """Outcome of running one example."""

    example: ExampleInfo
    status: str  # "PASS", "FAIL", "SKIP", "TIMEOUT"
    exit_code: int | None = None
    duration_s: float = 0.0
    error_lines: list[str] = field(default_factory=list)
    output_tail: str = ""
    reason: str = ""


# ── Discovery ────────────────────────────────────────────────────────────────


def discover_examples() -> list[ExampleInfo]:
    """Walk wanphys/examples/ and find runnable example scripts."""
    examples: list[ExampleInfo] = []

    for py_file in sorted(EXAMPLES_ROOT.rglob("*.py")):
        if py_file.name in SKIP_FILENAMES:
            continue

        source = py_file.read_text(encoding="utf-8", errors="replace")

        # Must have an entry point to be considered runnable
        has_main_guard = 'if __name__' in source and '"__main__"' in source or "'__main__'" in source
        has_main_func = re.search(r"^def main\s*\(", source, re.MULTILINE) is not None
        if not (has_main_guard or has_main_func):
            continue

        # Convert file path to module name
        rel = py_file.relative_to(EXAMPLES_ROOT.parent.parent)
        module = str(rel.with_suffix("")).replace(os.sep, ".").replace("/", ".")

        examples.append(ExampleInfo(module=module, path=py_file))

    return examples


# ── CLI Arg Detection ────────────────────────────────────────────────────────


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    root = Path(__file__).resolve().parent
    env.setdefault("UV_CACHE_DIR", str(root / ".uv-cache"))
    env.setdefault("NEWTON_CACHE_PATH", str(root / ".newton-cache"))
    env.setdefault("WARP_CACHE_PATH", str(root / ".warp-cache-local"))
    return env


def build_refresh_env_command(
    *,
    script: Path,
    torch: str,
    python: str | None,
    clear_venv: bool,
) -> list[str]:
    """Build the command that refreshes WanPhys dependencies before smoking examples."""
    command = [sys.executable, os.fspath(script), "--torch", torch]
    if python:
        command.extend(["--python", python])
    if clear_venv:
        command.append("--clear-venv")
    return command


def refresh_environment(*, torch: str, python: str | None, clear_venv: bool) -> int:
    """Run the repo-local environment refresh script."""
    root = Path(__file__).resolve().parent
    script = root / "scripts" / "refresh_wanphys_env.py"
    command = build_refresh_env_command(script=script, torch=torch, python=python, clear_venv=clear_venv)
    print(_cyan("Refreshing WanPhys environment..."))
    proc = subprocess.run(command, cwd=root, env=_subprocess_env(), check=False)
    return proc.returncode


def detect_cli_args(example: ExampleInfo, num_frames: int, device: str | None) -> list[str]:
    """Run --help on the example and figure out what args it accepts."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", example.module, "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            env=_subprocess_env(),
        )
        help_text = result.stdout + result.stderr
    except (subprocess.TimeoutExpired, Exception):
        # If --help itself fails, try running with no args
        return []

    args: list[str] = []

    if "--viewer" in help_text:
        args.extend(["--viewer", "null"])
    if "--num-frames" in help_text or "--num_frames" in help_text:
        args.extend(["--num-frames", str(num_frames)])
    if device and "--device" in help_text:
        args.extend(["--device", device])
    if "--generate-test-mesh" in help_text:
        args.append("--generate-test-mesh")
    if "--generate-test-cloud" in help_text:
        args.append("--generate-test-cloud")

    return args


# ── Output Analysis ──────────────────────────────────────────────────────────


def find_error_lines(output: str) -> list[str]:
    """Parse output for real error lines, filtering out false positives."""
    errors: list[str] = []

    for line in output.splitlines():
        # Skip false positives first
        if any(fp.search(line) for fp in FALSE_POSITIVE_PATTERNS):
            continue

        # Check for fatal patterns
        if any(pat.search(line) for pat in FATAL_PATTERNS):
            errors.append(line.strip())

    return errors


def extract_traceback(output: str) -> str | None:
    """Extract the last traceback block from output, if any."""
    # Find all traceback blocks
    tb_pattern = re.compile(
        r"(Traceback \(most recent call last\):.*?(?:\w+Error|\w+Exception):.*?)$",
        re.MULTILINE | re.DOTALL,
    )
    matches = list(tb_pattern.finditer(output))
    if matches:
        return matches[-1].group(1).strip()
    return None


def find_skip_reason(output: str, *, allow_torch_dependency_skip: bool = True) -> str | None:
    for pattern, reason in ALWAYS_SKIP_PATTERNS:
        if pattern.search(output):
            return reason
    if not allow_torch_dependency_skip:
        return None
    for pattern, reason in TORCH_DEPENDENCY_SKIP_PATTERNS:
        if pattern.search(output):
            return reason
    return None


# ── Runner ───────────────────────────────────────────────────────────────────


def run_example(example: ExampleInfo, timeout: int, *, allow_torch_dependency_skip: bool) -> RunResult:
    """Run a single example and analyze its output."""
    cmd = [sys.executable, "-m", example.module] + example.cli_args

    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_subprocess_env(),
        )
        duration = time.monotonic() - t0
    except subprocess.TimeoutExpired as e:
        duration = time.monotonic() - t0
        output = (e.stdout or "") + (e.stderr or "")
        return RunResult(
            example=example,
            status="TIMEOUT",
            duration_s=duration,
            output_tail=_tail(output, 30),
            reason=f"timed out after {timeout}s",
        )

    output = proc.stdout + proc.stderr
    skip_reason = find_skip_reason(output, allow_torch_dependency_skip=allow_torch_dependency_skip)
    if skip_reason is not None:
        return RunResult(
            example=example,
            status="SKIP",
            exit_code=proc.returncode,
            duration_s=duration,
            output_tail=_tail(output, 10),
            reason=skip_reason,
        )

    error_lines = find_error_lines(output)
    traceback_block = extract_traceback(output)

    # Determine result
    reasons: list[str] = []
    if proc.returncode != 0:
        reasons.append(f"exit_code={proc.returncode}")
    if error_lines:
        reasons.append(f"{len(error_lines)} error(s) in output")

    if reasons:
        tail = traceback_block if traceback_block else _tail(output, 30)
        return RunResult(
            example=example,
            status="FAIL",
            exit_code=proc.returncode,
            duration_s=duration,
            error_lines=error_lines,
            output_tail=tail,
            reason="; ".join(reasons),
        )

    return RunResult(
        example=example,
        status="PASS",
        exit_code=proc.returncode,
        duration_s=duration,
        output_tail=_tail(output, 5),
    )


def _tail(text: str, n: int) -> str:
    lines = text.strip().splitlines()
    return "\n".join(lines[-n:])


# ── Reporting ────────────────────────────────────────────────────────────────

# ANSI colors (disabled if not a tty)
_use_color = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    if not _use_color:
        return text
    return f"\033[{code}m{text}\033[0m"


def _green(t: str) -> str:
    return _c("32", t)


def _red(t: str) -> str:
    return _c("31", t)


def _yellow(t: str) -> str:
    return _c("33", t)


def _cyan(t: str) -> str:
    return _c("36", t)


def _dim(t: str) -> str:
    return _c("2", t)


def print_result(result: RunResult, verbose: bool = False) -> None:
    mod = result.example.module
    dt = f"{result.duration_s:.1f}s"

    if result.status == "PASS":
        print(f"  {_green('PASS')}  {mod}  {_dim(dt)}")
        if verbose and result.output_tail:
            for line in result.output_tail.splitlines():
                print(f"        {_dim(line)}")
    elif result.status == "SKIP":
        print(f"  {_yellow('SKIP')}  {mod}  {_dim(result.reason)}")
    elif result.status == "TIMEOUT":
        print(f"  {_red('TIMEOUT')}  {mod}  {_dim(result.reason)}")
        if result.output_tail:
            print()
            for line in result.output_tail.splitlines():
                print(f"        {line}")
            print()
    else:
        print(f"  {_red('FAIL')}  {mod}  {_dim(dt)}  {_dim(result.reason)}")
        if result.output_tail:
            print()
            for line in result.output_tail.splitlines():
                print(f"        {line}")
            print()


def print_summary(results: list[RunResult]) -> None:
    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")
    timed_out = sum(1 for r in results if r.status == "TIMEOUT")
    skipped = sum(1 for r in results if r.status == "SKIP")
    total = len(results)
    total_time = sum(r.duration_s for r in results)

    print()
    print("=" * 60)
    print(f"  SMOKE TEST SUMMARY  ({total_time:.0f}s total)")
    print("=" * 60)
    print(f"  {_green('Passed')}:   {passed}/{total}")
    if failed:
        print(f"  {_red('Failed')}:   {failed}/{total}")
    if timed_out:
        print(f"  {_red('Timeout')}:  {timed_out}/{total}")
    if skipped:
        print(f"  {_yellow('Skipped')}:  {skipped}/{total}")

    failures = [r for r in results if r.status in ("FAIL", "TIMEOUT")]
    if failures:
        print()
        print("  Failed examples:")
        for r in failures:
            print(f"    - {r.example.module}: {r.reason}")

    print("=" * 60)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test WanPhys examples")
    parser.add_argument("--timeout", type=int, default=120, help="Per-example timeout in seconds (default: 120)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show output for passing examples too")
    parser.add_argument("--pattern", "-p", type=str, default=None, help="Only run examples matching this substring")
    parser.add_argument("--num-frames", type=int, default=10, help="Number of frames to run (default: 10)")
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to pass to examples that support --device. Omit by default and use the current Warp device.",
    )
    parser.add_argument(
        "--refresh-env",
        action="store_true",
        help="Run scripts/refresh_wanphys_env.py before discovering and smoking examples.",
    )
    parser.add_argument(
        "--torch",
        choices=("cu12", "cu13", "none"),
        default="cu12",
        help="Torch extra to install when --refresh-env is set (default: cu12).",
    )
    parser.add_argument(
        "--refresh-python",
        default=None,
        help="Override the Python interpreter/request passed to uv during --refresh-env.",
    )
    parser.add_argument(
        "--refresh-clear-venv",
        action="store_true",
        help="Recreate .venv during --refresh-env. Do not use while running from the repo .venv on Windows.",
    )
    args = parser.parse_args()

    if args.refresh_env:
        refresh_code = refresh_environment(
            torch=args.torch,
            python=args.refresh_python,
            clear_venv=args.refresh_clear_venv,
        )
        if refresh_code != 0:
            print(_red(f"Environment refresh failed with exit code {refresh_code}"))
            return refresh_code
        print()

    print(_cyan("Discovering examples..."))
    examples = discover_examples()

    if args.pattern:
        examples = [e for e in examples if args.pattern in e.module]

    if not examples:
        print("No examples found.")
        return 1

    print(f"Found {len(examples)} runnable example(s)")
    print()

    # Detect CLI args for each example
    print(_cyan("Detecting CLI interfaces..."))
    device = args.device or None
    for ex in examples:
        ex.cli_args = detect_cli_args(ex, args.num_frames, device)
    print()

    # Run
    print(_cyan("Running smoke tests..."))
    print()

    results: list[RunResult] = []
    allow_torch_dependency_skip = (not args.refresh_env) or args.torch == "none"
    for ex in examples:
        result = run_example(ex, timeout=args.timeout, allow_torch_dependency_skip=allow_torch_dependency_skip)
        results.append(result)
        print_result(result, verbose=args.verbose)

    print_summary(results)

    has_failures = any(r.status in ("FAIL", "TIMEOUT") for r in results)
    return 1 if has_failures else 0


if __name__ == "__main__":
    sys.exit(main())
