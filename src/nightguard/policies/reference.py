"""The reference policies. PROJECT.md 8.2.

``do_nothing`` and ``rhythm`` ship permanently and act as the regression suite. ``monitor_down``
exists to answer v0.2 exit criterion 7: is there a degenerate strategy that never raises the
monitor?
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator

from ..core.state import Action, action_for_camera
from ..core.topology import Node
from .base import Percept


class DoNothing:
    """Always `NOOP`. PROJECT.md 8.2."""

    name = "do_nothing"

    def reset(self) -> None:
        """No state to clear."""

    def __call__(self, percept: Percept) -> Action:
        """Do nothing, ever."""
        return Action.NOOP


class RandomPolicy:
    """Uniform over the action space. Used as a smoke test, not a baseline."""

    name = "random"

    def __init__(self, rng: Generator | None = None, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed) if rng is None else rng

    def reset(self) -> None:
        """No state to clear; the generator carries across episodes deliberately."""

    def __call__(self, percept: Percept) -> Action:
        """Choose uniformly."""
        return Action(int(self._rng.integers(len(Action))))


class Rhythm:
    """The hand-coded reference loop. PROJECT.md 8.2.

    Flash alternating lights, close the corresponding door when the flash reveals an entity, drop
    the monitor, and raise it briefly on a fixed cadence to freeze SPRINTER.

    **Tuned once by hand and then frozen.** `peek_every_steps` was chosen by a sweep over
    {10, 12, 14, 16, 20, 28} at 400 seeds per night; 12 dominated. `door_hold_steps` made no
    measurable difference because the peek cycle resets the hold anyway. Do not retune these to
    chase a benchmark number.

    The ordering encodes the real mechanics. Both doors are shut *before* the scheduled peek,
    because raising the monitor is exactly what lets a waiting entity into the office (3.4, 3.5) —
    and the peek is not optional, because it is the only thing that freezes SPRINTER and resets its
    immunity window. Skipping it trades entity deaths for an escalating bang drain and a blackout.
    """

    name = "rhythm"

    def __init__(
        self,
        peek_every_steps: int = 12,
        door_hold_steps: int = 6,
        camera: Node = Node.COVE,
    ) -> None:
        self.peek_every_steps = peek_every_steps
        self.door_hold_steps = door_hold_steps
        self.camera = camera
        self.reset()

    def reset(self) -> None:
        """Clear the door timers and the peek state machine."""
        self._left_until = 0
        self._right_until = 0
        self._phase = "idle"

    def __call__(self, percept: Percept) -> Action:
        """One step of the loop."""
        step = percept.step

        # 1. SPRINTER is the only threat that arrives without warning; the running cue is the
        #    single reaction window there is.
        if percept.running and not percept.door_left:
            self._left_until = step + self.door_hold_steps
            return Action.TOGGLE_DOOR_LEFT

        # 2. Act on what the last light flash revealed.
        if percept.left_occupied and not percept.door_left:
            self._left_until = step + self.door_hold_steps
            return Action.TOGGLE_DOOR_LEFT
        if percept.right_occupied and not percept.door_right:
            self._right_until = step + self.door_hold_steps
            return Action.TOGGLE_DOOR_RIGHT

        # 3. The peek cycle. Raising the monitor is what lets a waiting entity into the office
        #    (3.4, 3.5), so both doors go down first and come back up afterwards. The peek itself
        #    is what freezes SPRINTER and resets its immunity window, which is the only way to stop
        #    it charging; skipping it entirely trades entity deaths for an escalating bang drain.
        if self._phase == "idle" and step % self.peek_every_steps == 0:
            self._phase = "closing"

        if self._phase == "closing":
            if not percept.door_left:
                return Action.TOGGLE_DOOR_LEFT
            if not percept.door_right:
                return Action.TOGGLE_DOOR_RIGHT
            self._phase = "peeking"
            return action_for_camera(self.camera)

        if self._phase == "peeking":
            if percept.monitor_up:
                return Action.MONITOR_DOWN
            self._phase = "idle"
            self._left_until = step
            self._right_until = step

        # 4. Reopen doors once their hold expires: power is the binding constraint.
        if percept.door_left and step >= self._left_until:
            return Action.TOGGLE_DOOR_LEFT
        if percept.door_right and step >= self._right_until:
            return Action.TOGGLE_DOOR_RIGHT

        # 5. Otherwise alternate light flashes to keep the door corners observed.
        return Action.FLASH_LIGHT_LEFT if (step // 2) % 2 == 0 else Action.FLASH_LIGHT_RIGHT


class MonitorDown:
    """Never raises the monitor. The v0.2 exit criterion 7 probe.

    Under the §3.4 and §3.5 monitor gates, WARDEN, DRIFTER and PROWLER cannot enter the office
    while the monitor stays down, so this policy only has to survive SPRINTER — which it does by
    closing the left door on the `running` cue. Its cost is the escalating bang drain, since a
    frozen-out SPRINTER re-arms and bangs again for `1.0 + 5.0n` each time.

    If this beats `rhythm`, the environment has a degenerate strategy. Report the numbers either
    way; it is a finding, not a pass/fail gate.
    """

    name = "monitor_down"

    def __init__(self, door_hold_steps: int = 4) -> None:
        self.door_hold_steps = door_hold_steps
        self._left_until = 0
        self._right_until = 0

    def reset(self) -> None:
        """Clear the door timers."""
        self._left_until = 0
        self._right_until = 0

    def __call__(self, percept: Percept) -> Action:
        """One step of the loop, with the monitor never raised."""
        step = percept.step

        if percept.running and not percept.door_left:
            self._left_until = step + self.door_hold_steps
            return Action.TOGGLE_DOOR_LEFT
        if percept.left_occupied and not percept.door_left:
            self._left_until = step + self.door_hold_steps
            return Action.TOGGLE_DOOR_LEFT
        if percept.right_occupied and not percept.door_right:
            self._right_until = step + self.door_hold_steps
            return Action.TOGGLE_DOOR_RIGHT

        if percept.door_left and step >= self._left_until:
            return Action.TOGGLE_DOOR_LEFT
        if percept.door_right and step >= self._right_until:
            return Action.TOGGLE_DOOR_RIGHT

        return Action.FLASH_LIGHT_LEFT if (step // 2) % 2 == 0 else Action.FLASH_LIGHT_RIGHT
