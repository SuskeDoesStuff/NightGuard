"""The curriculum. PROJECT.md 10, replacing the stale "curriculum start: night 3" row.

Three stages, chosen from measured headroom rather than from difficulty intuition:

===== ================================ ================ ==================
Stage Config                            ``rhythm``       Graduation
===== ================================ ================ ==================
1     Night 5                           0.973            survival >= 0.90
2     Night 6                           0.325            survival >= 0.40
3     ``custom_max`` (20/20/20/20)      0.185            report only
===== ================================ ================ ==================

Nights 1 to 4 are **not** stages. `rhythm` scores 1.000, 1.000, 1.000 and 0.993 on them, so an agent
that graduates from one has learned nothing a fixed script did not already do. They stay as sanity
checks that the environment does not kill spuriously.

**Stage 1 depends entirely on 6.3's dense survival term.** On night 5 `do_nothing` scores 0.000 and
`monitor_down` 0.007, so a random policy essentially never reaches dawn and the terminal reward is
never seen. What produces a gradient is `+0.01` per step survived. Under ``sparse_mode`` there is no
such term, and stage 1 is expected to produce no learning signal at all -- run it as a final
ablation, and report the flat curve rather than treating it as a bug.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..core.topology import Topology, load_topology
from .config import StageConfig, TrainConfig, as_dict, config_hash
from .evaluate import EvalResult
from .manifest import RunManifest, new_run_id
from .recurrent_ppo import (
    CurvePoint,
    EvaluationCallback,
    build_model,
    final_evaluation,
    make_vec_env,
    stage_config,
)


@dataclass
class StageResult:
    """What one stage produced.

    Attributes:
        stage: The stage name.
        timesteps: Environment steps actually spent.
        wall_clock_s: Wall clock for the stage.
        graduated: Whether the graduation threshold was met, or ``None`` for a report-only stage.
        reference: `rhythm`'s survival on this config, for the comparison.
        final: The full held-out evaluation at the end of the stage.
        curve: Periodic evaluations, in order.
        model_path: Where the finished policy was saved.
    """

    stage: str
    timesteps: int
    wall_clock_s: float
    graduated: bool | None
    reference: float
    final: EvalResult
    curve: list[CurvePoint] = field(default_factory=list)
    model_path: str = ""

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable form."""
        return {
            "stage": self.stage,
            "timesteps": self.timesteps,
            "wall_clock_s": round(self.wall_clock_s, 1),
            "graduated": self.graduated,
            "reference": self.reference,
            "beats_reference": self.final.survival > self.reference,
            "final": self.final.as_dict(),
            "curve": [asdict(point) for point in self.curve],
            "model_path": self.model_path,
        }


def run_stage(
    stage: StageConfig,
    train: TrainConfig,
    run_dir: Path,
    topology: Topology | None = None,
    initial_model_path: Path | None = None,
) -> StageResult:
    """Train one stage, evaluating periodically and once in full at the end.

    ``initial_model_path`` continues from the previous stage's policy, which is what makes this a
    curriculum rather than three independent runs.
    """
    config = stage_config(stage, train.sparse)
    shared = topology if topology is not None else load_topology(config.topology_path())
    venv = make_vec_env(config, train, shared)

    stage_dir = run_dir / stage.name
    stage_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(venv, train)
    if initial_model_path is not None:
        model.set_parameters(str(initial_model_path))

    callback = EvaluationCallback(config, train, shared, stage_dir / "curve.csv")
    started = time.perf_counter()
    model.learn(total_timesteps=stage.total_steps, callback=callback, progress_bar=False)
    wall_clock = time.perf_counter() - started

    model_path = stage_dir / "policy"
    model.save(str(model_path))
    venv.close()

    final = final_evaluation(model, config, train, shared)
    graduated = None if stage.graduation is None else final.survival >= stage.graduation
    return StageResult(
        stage=stage.name,
        timesteps=int(model.num_timesteps),
        wall_clock_s=wall_clock,
        graduated=graduated,
        reference=stage.reference,
        final=final,
        curve=list(callback.curve),
        model_path=str(model_path.with_suffix(".zip")),
    )


def run_curriculum(
    train: TrainConfig,
    tag: str = "",
    stop_on_failed_graduation: bool = False,
) -> dict[str, Any]:
    """Run every stage in order, carrying the policy forward.

    Args:
        train: The full training specification.
        tag: Appended to the run directory name, to keep ablation arms apart.
        stop_on_failed_graduation: Whether a missed threshold ends the run. Defaults to False:
            PROJECT.md 7's v1.1 criteria are about measurements being reported, and a stage that
            falls short is a result rather than an error. Never move the threshold instead.

    Returns:
        A JSON-serialisable summary, also written to ``summary.json`` in the run directory.
    """
    run_id = new_run_id(train.algorithm, tag)
    manifest = RunManifest(
        run_id=run_id,
        stage="curriculum",
        config_hash=config_hash(train),
        seed=train.seed,
        extra={"train_config": as_dict(train), "tag": tag},
    )
    manifest.write()

    started = time.perf_counter()
    results: list[StageResult] = []
    carry: Path | None = None
    for stage in train.stages:
        result = run_stage(stage, train, manifest.directory, initial_model_path=carry)
        results.append(result)
        carry = Path(result.model_path)
        _write_summary(manifest, results, time.perf_counter() - started)
        if stop_on_failed_graduation and result.graduated is False:
            break

    manifest.wall_clock_s = time.perf_counter() - started
    manifest.write()
    return _write_summary(manifest, results, manifest.wall_clock_s)


def _write_summary(
    manifest: RunManifest, results: list[StageResult], wall_clock: float
) -> dict[str, Any]:
    summary = {
        "run_id": manifest.run_id,
        "git_sha": manifest.git_sha,
        "config_hash": manifest.config_hash,
        "seed": manifest.seed,
        "machine": manifest.machine,
        "device": manifest.device,
        "wall_clock_s": round(wall_clock, 1),
        "stages": [result.as_dict() for result in results],
    }
    path = manifest.directory / "summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary
