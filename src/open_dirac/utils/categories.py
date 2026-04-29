"""Compensation category labels for scaffold event logging."""

from enum import StrEnum


class CompensationCategory(StrEnum):
    CALL_RELIABILITY = "call_reliability"  # old layers 1-3
    STATE_INVARIANTS = "state_invariants"  # old layer 4
    LOOP_CONTROL = "loop_control"  # old layers 5-6
    OUTPUT_NORMALIZATION = "output_normalization"  # old layers 7-10
