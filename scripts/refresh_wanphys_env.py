#!/usr/bin/env python3
"""Refresh the WanPhys project environment with uv.

This script centralizes the dependency set used by example smoke runs. Torch is
not part of the repo's ``dev`` extra, and the available torch extras are
mutually exclusive, so callers must choose one CUDA flavor explicitly.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

TORCH_EXTRAS = {
    "cu12": "torch-cu12",
    "cu13": "torch-cu13",
    "none": None,
}


def repo_root() -> Path:
    """Return the repository root containing this script."""
    return Path(__file__).resolve().parents[1]


def build_refresh_env(root: Path, base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Build an environment using repo-local caches for uv, Warp, and Newton."""
    env = dict(os.environ if base_env is None else base_env)
    env.setdefault("UV_CACHE_DIR", os.fspath(root / ".uv-cache"))
    env.setdefault("WARP_CACHE_PATH", os.fspath(root / ".warp-cache-local"))
    env.setdefault("NEWTON_CACHE_PATH", os.fspath(root / ".newton-cache"))
    return env


def _python_in_venv(venv: Path) -> Path | None:
    """Return the Python executable inside a virtual environment, if present."""
    for candidate in (venv / "Scripts" / "python.exe", venv / "bin" / "python"):
        if candidate.exists():
            return candidate
    return None


def find_repo_venv_python(root: Path, base_env: dict[str, str] | None = None) -> Path | None:
    """Find the Python executable from uv's repo-local virtual environment."""
    env = os.environ if base_env is None else base_env
    resolved_root = root.resolve()

    active_venv = env.get("VIRTUAL_ENV")
    if active_venv:
        active_venv_path = Path(active_venv).resolve()
        try:
            active_venv_path.relative_to(resolved_root)
        except ValueError:
            pass
        else:
            active_python = _python_in_venv(active_venv_path)
            if active_python is not None:
                return active_python

    return _python_in_venv(root / ".venv")


def resolve_sync_python(root: Path, python: str | None, base_env: dict[str, str] | None = None) -> str | None:
    """Resolve the interpreter uv should use while syncing dependencies."""
    if python:
        return python
    venv_python = find_repo_venv_python(root, base_env=base_env)
    return os.fspath(venv_python) if venv_python is not None else None


def build_venv_command(*, uv: str, python: str | None) -> list[str]:
    """Build the uv command that recreates the project virtual environment."""
    command = [uv, "venv"]
    if python:
        command.extend(["--python", python])
    command.append("--clear")
    return command


def build_sync_command(*, uv: str, python: str | None, torch: str) -> list[str]:
    """Build the uv command that installs WanPhys development/example dependencies."""
    if torch not in TORCH_EXTRAS:
        allowed = ", ".join(TORCH_EXTRAS)
        raise ValueError(f"Unknown torch option '{torch}'. Expected one of: {allowed}")

    command = [uv, "sync"]
    if python:
        command.extend(["--python", python])
    command.extend(["--extra", "dev"])

    torch_extra = TORCH_EXTRAS[torch]
    if torch_extra is not None:
        command.extend(["--extra", torch_extra])
    return command


def _is_running_from_repo_venv(root: Path) -> bool:
    try:
        Path(sys.executable).resolve().relative_to((root / ".venv").resolve())
    except ValueError:
        return False
    return True


def _run(command: list[str], *, cwd: Path, env: dict[str, str], dry_run: bool) -> int:
    print("+ " + " ".join(command), flush=True)
    if dry_run:
        return 0
    return subprocess.run(command, cwd=cwd, env=env, check=False).returncode


def refresh_environment(
    *,
    root: Path,
    uv: str,
    python: str | None,
    torch: str,
    clear_venv: bool,
    dry_run: bool,
) -> int:
    """Refresh the WanPhys uv environment and return the process exit code."""
    env = build_refresh_env(root)
    sync_python = resolve_sync_python(root, python, base_env=env)

    if clear_venv:
        if _is_running_from_repo_venv(root):
            print(
                "Refusing --clear-venv while this script is running from the target .venv. "
                "Run the script with a host Python, or omit --clear-venv.",
                file=sys.stderr,
            )
            return 2
        code = _run(build_venv_command(uv=uv, python=python), cwd=root, env=env, dry_run=dry_run)
        if code != 0:
            return code
        sync_python = resolve_sync_python(root, python, base_env=env)

    return _run(build_sync_command(uv=uv, python=sync_python, torch=torch), cwd=root, env=env, dry_run=dry_run)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh the WanPhys uv environment")
    parser.add_argument("--uv", default="uv", help="uv executable to run (default: uv)")
    parser.add_argument("--python", default=None, help="Python interpreter/request to pass to uv")
    parser.add_argument("--torch", choices=sorted(TORCH_EXTRAS), default="cu12", help="Torch extra to install")
    parser.add_argument("--clear-venv", action="store_true", help="Recreate .venv before syncing dependencies")
    parser.add_argument("--dry-run", action="store_true", help="Print uv commands without executing them")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return refresh_environment(
        root=repo_root(),
        uv=args.uv,
        python=args.python,
        torch=args.torch,
        clear_venv=args.clear_venv,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
