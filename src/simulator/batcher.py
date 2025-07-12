"""
Batching Strategy Implementations for LLM Inference

This module provides different batching strategies for the LLM inference simulator.
Each strategy determines how requests are grouped together for processing.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import Engine


class Batcher:
    """
    Base class for batching strategies.

    Defines the interface for request batching implementations.
    Each concrete implementation provides its own logic for how
    requests should be grouped together for processing.

    Attributes:
        engine (Engine): Reference to the simulation engine
    """

    engine: "Engine"

    def add_requests(self) -> None:
        """
        Base implementation of static batching.

        Only batches requests when all slots are empty.
        Cannot mix prefill and decode phases in the same batch.
        """
        engine = self.engine
        if engine.get_occupied_slots():
            return  # static batcher cannot batch together new prefills with old decodings
        for slot in engine.get_all_slots():
            if not len(engine.queue):  # checking if we still have more requests to run
                break
            request = engine.queue.pop(0)
            engine.assign_request_to_slot(request, slot)

    def __str__(self) -> str:
        """
        Return the string representation of the batcher class.

        Returns:
            str: The class name of the batcher.
        """
        return f"{self.__class__.__name__}"


class StaticBatcher(Batcher):
    """
    Simple static batching implementation.

    Uses the base Batcher implementation which only batches
    requests when all slots are empty. This results in higher
    latency as requests must wait for the entire batch to complete.
    """

    pass


class IFBatcher(Batcher):
    """
    In-Flight Batching (IFB) implementation.

    Allows mixing of prefill and decode phases in the same batch.
    Fills empty slots immediately as they become available,
    leading to better resource utilization.
    """

    engine: "Engine"

    def add_requests(self) -> None:
        """
        Add requests using in-flight batching strategy.

        Fills any empty slots immediately with new requests,
        regardless of the state of other slots. This allows
        prefill and decode phases to be processed together.
        """
        engine = self.engine
        empty_slots = engine.get_all_slots() - engine.get_occupied_slots()
        for slot in empty_slots:
            if not len(engine.queue):
                break
            request = engine.queue.pop(0)
            engine.assign_request_to_slot(request, slot)


class IFBatcherWithOnePrefillOnly(IFBatcher):
    """
    Enhanced IFB implementation that limits prefill operations.

    Extends IFB to ensure only one request can be in prefill phase
    at any time. This helps balance compute-bound (prefill) and
    memory-bound (decode) operations for optimal performance.
    """

    def add_requests(self):
        """
        Add requests while maintaining single prefill constraint.

        Only adds a new request if:
        1. No request is currently in prefill phase
        2. There are empty slots available
        3. There are requests in the queue

        Once a request is added, waits for its prefill to complete
        before adding another.
        """
        engine = self.engine
        # Only one request can be in prefill simultaneously
        if len(engine.get_prefilling_requests()):
            return
        empty_slots = engine.get_all_slots() - engine.get_occupied_slots()
        for slot in empty_slots:
            if not len(engine.queue):
                break
            request = engine.queue.pop(0)
            engine.assign_request_to_slot(request, slot)
            break  # Only one new request can be taken
