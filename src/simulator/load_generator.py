"""
Load Generation Strategies for LLM Inference Simulation

This module provides different strategies for generating inference requests
in the simulation. It includes implementations for batch loading, concurrent
requests, and constant request rate scenarios.
"""

from typing import TYPE_CHECKING
import numpy as np

from .request import ChunkedContextRequest

if TYPE_CHECKING:
    from .engine import Engine


class LoadGenerator:
    """
    Base class for load generation strategies.

    Provides core functionality for creating and adding requests
    to the simulation engine's queue.

    Attributes:
        engine (Engine): Reference to the simulation engine
        prefill_time (float): Time taken for prefill phase
        itl (float): Inter-token latency
        target_output_len_tokens (int): Target number of tokens to generate
        total_prefill_chunks (int): Number of chunks for prefill phase
    """

    engine: "Engine"
    prefill_time: float = 2
    itl: float = 1
    target_output_len_tokens: int = 4
    total_prefill_chunks: int = 1

    def __init__(
        self,
        prefill_time: float = 2,
        itl: float = 1,
        target_output_len_tokens: int = 4,
        total_prefill_chunks: int = 1,
    ) -> None:
        """
        Initialize the load generator with timing parameters.

        Args:
            prefill_time: Time for prefill phase
            itl: Inter-token latency
            target_output_len_tokens: Number of tokens to generate
            total_prefill_chunks: Number of prefill chunks
        """
        self.prefill_time = prefill_time
        self.itl = itl
        self.target_output_len_tokens = target_output_len_tokens
        self.total_prefill_chunks = total_prefill_chunks

    def generate_load(self):
        """Generate a single request and add it to the queue."""
        self.engine.queue.append(self.get_request(""))

    def get_request(self, id_postfix):
        """
        Create a new request with current parameters.

        Args:
            id_postfix: Identifier suffix for the request

        Returns:
            ChunkedContextRequest: New request instance
        """
        return ChunkedContextRequest(
            id=f"{self.engine.current_time}-{id_postfix}",
            prefill_time=self.prefill_time,
            itl=self.itl,
            total_prefill_chunks=self.total_prefill_chunks,
            target_output_len_tokens=self.target_output_len_tokens,
            added_to_queue_at=self.engine.current_time,
        )

    def add_n_requests_to_queue(self, n_requests):
        """
        Add multiple requests to the queue.

        Args:
            n_requests: Number of requests to add
        """
        for i in range(n_requests):
            self.engine.queue.append(self.get_request(i))

    def __str__(self):
        res = f"{self.__class__.__name__}("
        for k, v in self.__dict__.items():
            if k not in ["engine", "last_generation_time"]:
                res += f"\n     {k}={v}"
        res += "\n)"
        return res


class LoadGeneratorNormalOutputLength(LoadGenerator):
    """
    Load generator with normally distributed output lengths.

    Extends base generator to create requests with output lengths
    drawn from a normal distribution.

    Additional Attributes:
        target_output_len_std (int): Standard deviation for output length
    """

    target_output_len_std: int = 0

    def get_request(self, id_postfix):
        """
        Create request with normally distributed output length.

        Args:
            id_postfix: Identifier suffix for the request

        Returns:
            ChunkedContextRequest: New request with random output length
        """
        target_output_len_tokens = max(
            np.random.normal(self.target_output_len_tokens, self.target_output_len_std),
            1,
        )
        return ChunkedContextRequest(
            id=f"{self.engine.current_time}-{id_postfix}",
            prefill_time=self.prefill_time,
            itl=self.itl,
            total_prefill_chunks=self.total_prefill_chunks,
            target_output_len_tokens=int(target_output_len_tokens),
            added_to_queue_at=self.engine.current_time,
        )


class BatchLoadGenerator(LoadGeneratorNormalOutputLength):
    """
    Generator that creates an initial batch of requests.

    Useful for testing system behavior with a fixed number
    of concurrent requests from the start.

    Additional Attributes:
        initial_batch (int): Number of requests to create initially
    """

    initial_batch: int = 30

    def __init__(
        self,
        prefill_time: float = 2,
        itl: float = 1,
        target_output_len_tokens: int = 4,
        total_prefill_chunks: int = 1,
        initial_batch: int = 3,
    ):
        """
        Initialize with batch size parameter.

        Args:
            prefill_time: Time for prefill phase
            itl: Inter-token latency
            target_output_len_tokens: Number of tokens to generate
            total_prefill_chunks: Number of prefill chunks
            initial_batch: Size of initial request batch
        """
        super().__init__(
            prefill_time, itl, target_output_len_tokens, total_prefill_chunks
        )
        self.initial_batch = initial_batch

    def generate_load(self):
        """Generate initial batch of requests only at start time."""
        if self.engine.current_time > 0:
            return
        self.add_n_requests_to_queue(self.initial_batch)


class ConcurrentLoadGenerator(LoadGeneratorNormalOutputLength):
    """
    Generator that maintains a target level of concurrency.

    Adds new requests as needed to maintain a specified
    number of concurrent requests in the system.

    Additional Attributes:
        target_concurrency (int): Target number of concurrent requests
    """

    target_concurrency: int = 3

    def __init__(
        self,
        prefill_time: float = 2,
        itl: float = 1,
        target_output_len_tokens: int = 4,
        total_prefill_chunks: int = 1,
        target_concurrency: int = 3,
    ):
        """
        Initialize with concurrency target.

        Args:
            prefill_time: Time for prefill phase
            itl: Inter-token latency
            target_output_len_tokens: Number of tokens to generate
            total_prefill_chunks: Number of prefill chunks
            target_concurrency: Target number of concurrent requests
        """
        super().__init__(
            prefill_time, itl, target_output_len_tokens, total_prefill_chunks
        )
        self.target_concurrency = target_concurrency

    def generate_load(self):
        """Add requests to maintain target concurrency level."""
        current_concurrency = len(self.engine.get_occupied_slots())
        already_in_queue = len(self.engine.queue)
        ## We want to reach target concurrency but not overshoot it, so limit queue buffer
        need_to_add = self.target_concurrency - current_concurrency - already_in_queue
        self.add_n_requests_to_queue(need_to_add)
        self.last_generation_time = self.engine.current_time


class RequestRateLoadGenerator(LoadGeneratorNormalOutputLength):
    """
    Generator that creates requests at a constant rate.

    Adds new requests based on a specified requests/time unit rate,
    regardless of system state.

    Additional Attributes:
        request_rate (float): Number of requests to generate per time unit
        last_generation_time (float): Time of last request generation
    """

    request_rate: float = 1.0
    last_generation_time: float = 0

    def __init__(
        self,
        prefill_time: float = 2,
        itl: float = 1,
        target_output_len_tokens: int = 4,
        total_prefill_chunks: int = 1,
        request_rate: float = 1.0,
    ):
        """
        Initialize with request rate parameter.

        Args:
            prefill_time: Time for prefill phase
            itl: Inter-token latency
            target_output_len_tokens: Number of tokens to generate
            total_prefill_chunks: Number of prefill chunks
            request_rate: Requests per time unit to generate
        """
        super().__init__(
            prefill_time, itl, target_output_len_tokens, total_prefill_chunks
        )
        self.request_rate = request_rate

    def generate_load(self):
        """Generate requests at specified constant rate."""
        generate_every = 1.0 / self.request_rate
        already_generated = self.last_generation_time // generate_every
        final_generated = self.engine.current_time // generate_every
        num_requests = final_generated - already_generated
        self.add_n_requests_to_queue(int(num_requests))
        self.last_generation_time = self.engine.current_time
