"""
Request Models for LLM Inference Simulation

This module defines the request models used in the LLM inference simulation.
It includes both standard requests and chunked context requests.
"""

from dataclasses import dataclass
from typing import Optional
from .slot_state import SlotState


@dataclass
class Request:
    """
    Base class representing a single LLM inference request.

    This class models the behavior of a request as it moves through the inference
    pipeline, tracking its progress from queue to completion.

    Attributes:
        id (str): Unique identifier for the request
        prefill_time (float): Time taken for prefill phase when running alone
        itl (float): Inter-token latency for decoding phase when running alone
        target_output_len_tokens (int): Number of tokens to generate
        added_to_queue_at (Optional[float]): Time when request entered queue
        started_at (Optional[float]): Time when request started processing
        tokens_generated (int): Number of tokens generated so far
    """

    id: str
    # how long does prefill of this request take, if it is the only request in the engine
    prefill_time: float = 2.0
    # how long does decoding of one token of this request take, if it is the only request in the engine
    itl: float = 1.0
    # target output length, how many tokens to decode in total to complete the request
    target_output_len_tokens: int = 4

    added_to_queue_at: Optional[float] = None
    started_at: Optional[float] = None
    tokens_generated: int = 0

    def is_in_prefill(self) -> bool:
        """
        Check if request is in prefill phase (no tokens generated yet).

        Returns:
            bool: True if in prefill phase, False otherwise.
        """
        return self.tokens_generated == 0

    def get_slot_state_at(self, current_time: float) -> SlotState:
        """
        Get the state of the slot containing this request at a given time.

        Args:
            current_time (float): Time at which to check the state

        Returns:
            SlotState: Current state (empty, prefill, or decoding)
        """
        if self.started_at is None:
            return SlotState.empty
        elif current_time < self.started_at:
            return SlotState.empty
        elif self.is_in_prefill():
            return SlotState.prefill
        elif self.tokens_generated < self.target_output_len_tokens:
            return SlotState.decoding
        else:
            return SlotState.empty

    def get_current_latency_at(self, current_time: float) -> float:
        """
        Calculate total latency from queue entry to current time.

        Args:
            current_time (float): Time at which to calculate latency

        Returns:
            float: Current latency in time units

        Raises:
            AssertionError: If request hasn't been added to queue
        """
        assert self.added_to_queue_at is not None
        return current_time - self.added_to_queue_at

    def tick(self) -> None:
        """
        Progress the request by one step.
        For standard requests, generates one token.
        """
        self.tokens_generated += 1

    def get_current_duration(self) -> float:
        """
        Get duration of current processing step.

        Returns:
            float: Time for current step (prefill_time or itl)
        """
        if self.is_in_prefill():
            return self.prefill_time
        else:
            return self.itl


@dataclass
class ChunkedContextRequest(Request):
    """
    Request with chunked context processing capability.

    Extends the base Request class to support splitting the prefill phase
    into multiple chunks, allowing for more efficient processing.

    Additional Attributes:
        total_prefill_chunks (int): Number of chunks to split prefill into
        prefill_chunks_completed (int): Number of chunks processed so far
    """

    total_prefill_chunks: int = 4
    prefill_chunks_completed: int = 1

    def tick(self) -> None:
        """
        Progress the request by one step.
        For chunked requests, either completes a prefill chunk
        or generates a token if prefill is complete.
        """
        if self.prefill_chunks_completed < self.total_prefill_chunks:
            self.prefill_chunks_completed += 1
        else:
            self.tokens_generated += 1

    def get_current_duration(self) -> float:
        """
        Get duration of current processing step.

        For chunked requests, prefill time is divided by number of chunks.

        Returns:
            float: Time for current step (chunked prefill_time or itl)
        """
        if self.is_in_prefill():
            return self.prefill_time / self.total_prefill_chunks
        else:
            return self.itl
