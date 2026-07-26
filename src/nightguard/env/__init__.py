"""Gymnasium wrapper layer. PROJECT.md 6.

Only the action space exists in v0.2 (PROJECT.md 7's v0.2 scope excludes all of 6 except this).
Spaces, observation encoding, reward and the env itself arrive in v1.0.
"""

from .actions import ACTION_COUNT, action_space, decode, encode

__all__ = ["ACTION_COUNT", "action_space", "decode", "encode"]
