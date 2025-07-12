"""
Slot State Enumeration for LLM Inference Simulation

This module defines the possible states of a batch slot during
the simulation of LLM inference processing.
"""

from enum import Enum


class SlotState(Enum):
    """
    Enumeration of possible states for a batch slot.

    States:
        empty (0): Slot is not currently processing any request
        prefill (1): Slot is processing a request in prefill phase
        decoding (2): Slot is processing a request in decode phase
    """
    empty: int = 0
    prefill: int = 1
    decoding: int = 2
