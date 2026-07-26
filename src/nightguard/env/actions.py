"""The Gymnasium action space. PROJECT.md 3.2 and 6.1.

This module owns the ``Discrete(17)`` space and the encode/decode boundary between raw space
samples and the semantic :class:`~nightguard.core.state.Action` the simulator consumes. PROJECT.md
1.3 forbids *action decoding* in ``core/``, and that is what lives here.

``core.Action`` keeps the integer values, because PROJECT.md 3.2's table makes them normative and
because ``trace/`` must serialise them (PROJECT.md 5's ``"action": 5``) while PROJECT.md 1's
dependency rule allows only ``trace -> core``. Moving the indices into this module would leave the
trace writer unable to produce its own format without importing ``env/``. The split is therefore
between the *indices*, which core owns, and the *space*, which is here.
"""

from __future__ import annotations

from typing import SupportsInt

import numpy as np
from gymnasium.spaces import Discrete

from ..core.state import Action

ACTION_COUNT = len(Action)


def action_space(seed: int | None = None) -> Discrete[np.int64]:
    """The ``Discrete(17)`` action space of PROJECT.md 6.1."""
    return Discrete(ACTION_COUNT, seed=seed)


def decode(index: SupportsInt) -> Action:
    """Map a raw action-space sample to a semantic :class:`Action`.

    Raises:
        ValueError: If ``index`` is outside the space.
    """
    value = int(index)
    if not 0 <= value < ACTION_COUNT:
        raise ValueError(f"action {value} is outside Discrete({ACTION_COUNT})")
    return Action(value)


def encode(action: Action) -> int:
    """Map a semantic :class:`Action` back to its action-space index."""
    return int(action)


def decode_batch(indices: np.ndarray) -> list[Action]:
    """Decode a vector of action-space samples, for the vectorised runner in v1.0."""
    return [decode(value) for value in indices.tolist()]
