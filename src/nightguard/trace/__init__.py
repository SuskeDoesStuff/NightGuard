"""Serialisation of ground-truth state to JSONL. PROJECT.md 5.

Consumed by the viewer; never mutates state. Depends on ``core/`` only.
"""

from .schema import (
    EVENT_PRIORITY,
    TRACE_VERSION,
    config_hash,
    event_rank,
    footer_record,
    header_record,
    tick_record,
)
from .writer import CamDuty, TraceWriter, write_episode

__all__ = [
    "EVENT_PRIORITY",
    "TRACE_VERSION",
    "CamDuty",
    "TraceWriter",
    "config_hash",
    "event_rank",
    "footer_record",
    "header_record",
    "tick_record",
    "write_episode",
]
