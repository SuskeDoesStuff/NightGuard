"""JSONL trace writing. PROJECT.md 5.

One file per episode: a header record, then one record per sim tick, then a footer. Writing every
tick produces ~5350 records per episode, which is fine for inspection traces; ``stride`` subsamples
for long batch runs, where a stride of 5 gives one record per decision step.

The writer never mutates simulation state. It attaches to :attr:`NightSim.on_tick`, which core
invokes at PROJECT.md 3.13 step 9, so the dependency runs ``trace -> core`` and never the reverse.

Records are serialised with fixed key order and compact separators, so the same seed and action
script always produce a byte-identical file (v0.2 exit criterion 2).
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from types import TracebackType
from typing import IO, Any, Self

from ..core.sim import NightSim
from ..core.state import Action, SimState
from .schema import event_rank, footer_record, header_record, tick_record


class CamDuty:
    """Rolling camera duty cycle over the last N decision steps. PROJECT.md 5.

    Computed at write time so the viewer never has to derive it.
    """

    def __init__(self, window_steps: int) -> None:
        self._window: deque[bool] = deque(maxlen=window_steps)
        self._total = 0
        self._count = 0

    def observe_step(self, monitor_up: bool) -> None:
        """Record one completed decision step."""
        self._window.append(monitor_up)
        self._total += int(monitor_up)
        self._count += 1

    @property
    def rolling(self) -> float:
        """Fraction of the last N decision steps with the monitor up."""
        if not self._window:
            return 0.0
        return sum(self._window) / len(self._window)

    @property
    def mean(self) -> float:
        """Fraction over the whole episode."""
        if self._count == 0:
            return 0.0
        return self._total / self._count


class TraceWriter:
    """Writes one episode's trace to a JSONL file.

    Use as a context manager, or call :meth:`close` explicitly::

        with TraceWriter(path, sim, night=5, seed=918442) as writer:
            sim.run(script)
    """

    def __init__(
        self,
        path: Path,
        sim: NightSim,
        night: int,
        seed: int | None = None,
        stride: int | None = None,
        policy: str | None = None,
    ) -> None:
        self.path = path
        self.sim = sim
        self.stride = sim.config.trace.stride if stride is None else stride
        if self.stride < 1:
            raise ValueError(f"stride must be at least 1, found {self.stride}")
        self._cam_duty = CamDuty(sim.config.trace.cam_duty_window_steps)
        self._ticks_per_step = sim.clock.ticks_per_decision_step
        self._event_cursor = 0
        self._closed = False

        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle: IO[str] = path.open("w", encoding="utf-8", newline="\n")
        self._write(header_record(sim, night, seed, sim.topology, policy))
        sim.on_tick = self._on_tick

    # --- context manager -----------------------------------------------------------------------

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # --- writing -------------------------------------------------------------------------------

    def _write(self, record: dict[str, Any]) -> None:
        self._handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    def _pending_event(self, state: SimState) -> str | None:
        """The most significant event recorded on this tick, or ``None``.

        Core records events as ``(tick, name)`` pairs and several can land on one tick — an
        invasion always jams its door on the same tick. PROJECT.md 5's ``event`` field holds a
        single value, so co-occurring events are resolved by :data:`EVENT_PRIORITY` rather than by
        arrival order, which would silently drop the invasion in favour of the jam.
        """
        candidates: list[str] = []
        while self._event_cursor < len(state.events):
            tick, name = state.events[self._event_cursor]
            if tick > state.tick:
                break
            candidates.append(name)
            self._event_cursor += 1
        if not candidates:
            return None
        return min(candidates, key=event_rank)

    def _on_tick(self, state: SimState, action: Action | None) -> None:
        """Emit one tick record. Attached to :attr:`NightSim.on_tick`."""
        if state.tick % self._ticks_per_step == 0:
            self._cam_duty.observe_step(state.office.monitor_up)

        event = self._pending_event(state)
        terminal = state.terminated
        if state.tick % self.stride == 0 or terminal:
            self._write(tick_record(self.sim, state, action, self._cam_duty.rolling, event))

    def close(self) -> None:
        """Write the footer and close the file. Idempotent."""
        if self._closed:
            return
        self._closed = True
        if self.sim.on_tick is self._on_tick:
            self.sim.on_tick = None
        self._write(footer_record(self.sim, self.sim.state, self._cam_duty.mean))
        self._handle.close()


def write_episode(
    path: Path,
    sim: NightSim,
    night: int,
    actions: Any = (),
    seed: int | None = None,
    stride: int | None = None,
    policy: str | None = None,
) -> Path:
    """Run ``sim`` to termination while writing its trace, and return the path."""
    with TraceWriter(path, sim, night=night, seed=seed, stride=stride, policy=policy):
        sim.run(actions)
    return path
