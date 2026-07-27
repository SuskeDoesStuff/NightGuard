"""Run provenance. PROJECT.md 7, v1.1 criterion 7.

*Runs that are not reproducible are not results.* Every run writes one of these before it starts, so
a number in `CHANGELOG.md` can be traced back to the exact commit, config and seed that produced it.

Torch-free: the version block is collected by importing lazily, so the manifest schema can be tested
without ``[train]`` installed.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..core.config import REPO_ROOT

RUNS_DIR = REPO_ROOT / "runs"


def _git(args: list[str], base: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=base, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_sha(root: Path | None = None) -> str:
    """The current commit, with ``-dirty`` appended if the tree has uncommitted changes.

    A dirty marker matters more than it looks: a result produced from an edited tree is not
    reproducible from the SHA alone, and silently recording the clean SHA would claim it was.
    """
    base = REPO_ROOT if root is None else root
    sha = _git(["rev-parse", "HEAD"], base)
    dirty = _git(["status", "--porcelain"], base)
    if sha is None or dirty is None:
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


def dirty_paths(root: Path | None = None) -> list[str]:
    """Which files were uncommitted when the run started.

    ``-dirty`` alone says a run is not reproducible from its SHA; it does not say *how far* from it.
    v1.1 finished with two of three runs dirty in documentation only, and no way to demonstrate that
    from the artefact. Listing the paths makes "reproducible apart from the prose" a checkable claim
    rather than an assurance.
    """
    base = REPO_ROOT if root is None else root
    status = _git(["status", "--porcelain"], base)
    if not status:
        return []
    return sorted(line[3:].strip() for line in status.splitlines() if len(line) > 3)


def package_versions() -> dict[str, str]:
    """Versions of everything that can change a result. Missing packages are recorded as absent."""
    versions = {"python": platform.python_version()}
    for name in ("numpy", "gymnasium", "torch", "stable_baselines3", "sb3_contrib"):
        module = sys.modules.get(name)
        if module is None:
            try:
                module = __import__(name)
            except ImportError:
                versions[name] = "absent"
                continue
        versions[name] = str(getattr(module, "__version__", "unknown"))
    return versions


def torch_device_report() -> dict[str, Any]:
    """What the run will actually compute on. Reported rather than assumed."""
    try:
        import torch
    except ImportError:
        return {"available": False}
    report: dict[str, Any] = {"available": True, "cuda": bool(torch.cuda.is_available())}
    if report["cuda"]:
        report["cuda_device"] = torch.cuda.get_device_name(0)
    return report


@dataclass
class RunManifest:
    """Everything needed to reproduce one run.

    Attributes:
        run_id: Directory name under ``runs/``.
        stage: Curriculum stage name.
        config_hash: :func:`~nightguard.train.config.config_hash` over the training and night
            configs together.
        seed: The run seed.
        git_sha: Commit, ``-dirty`` if the tree was modified.
        dirty_paths: Which files were uncommitted at launch, so "dirty in documentation only" is
            checkable rather than asserted.
        started_at: UTC ISO timestamp.
        wall_clock_s: Filled in on completion.
        machine: Architecture, OS and Python.
        versions: See :func:`package_versions`.
        device: See :func:`torch_device_report`.
        extra: Free-form per-run notes, such as the training config itself.
    """

    run_id: str
    stage: str
    config_hash: str
    seed: int
    git_sha: str = field(default_factory=git_sha)
    dirty_paths: list[str] = field(default_factory=dirty_paths)
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    wall_clock_s: float | None = None
    machine: str = field(
        default_factory=lambda: (
            f"{platform.machine()} {platform.system()}, Python {platform.python_version()}"
        )
    )
    versions: dict[str, str] = field(default_factory=package_versions)
    device: dict[str, Any] = field(default_factory=torch_device_report)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def directory(self) -> Path:
        """Where this run's artefacts live."""
        return RUNS_DIR / self.run_id

    def write(self) -> Path:
        """Write ``manifest.json`` into the run directory, creating it if needed."""
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / "manifest.json"
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")
        return path


def new_run_id(stage: str, tag: str = "") -> str:
    """A sortable, unique run directory name."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    suffix = f"-{tag}" if tag else ""
    return f"{stamp}-{stage}{suffix}"
