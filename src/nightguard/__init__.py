"""NightGuard: a partially-observable surveillance-and-resource-management environment.

Importing this package registers the Gymnasium IDs of PROJECT.md 2.4, so::

    import gymnasium as gym
    import nightguard  # noqa: F401 - registers NightGuard-v0

    env = gym.make("NightGuard-v0", night=5)

works. Gymnasium 1.x dropped the entry-point plugin system, so an explicit import is the only way an
external ID can resolve; this is the smallest one that makes 2.4's declaration true.

The layering of PROJECT.md 1 is unchanged -- ``core/`` imports neither gymnasium nor torch -- but note
that this line does mean importing *the package* pulls the ``env`` layer in. Gymnasium is a base
runtime dependency (1.2), so that costs an import, not a dependency.
"""

from .env.registration import register_environments

__version__ = "0.1.0"

register_environments()
