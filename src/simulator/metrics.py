"""
Performance Metrics Collection for LLM Inference Simulation

This module handles the collection and processing of performance metrics
during the simulation, including latencies, throughput, and queue statistics.
"""

from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import Engine


class Metrics:
    """
    Metrics collection and processing for LLM inference simulation.

    Tracks various performance metrics throughout the simulation,
    including latencies (E2E, TTFT, ITL), queue sizes, and output lengths.

    Attributes:
        engine (Engine): Reference to the simulation engine
        num_slots (int): Number of batch slots being tracked
        times (List[float]): Timestamps for each measurement
        queue_size (List[int]): Queue length at each measurement
        e2e_latency (List[Tuple[float, float]]): End-to-end latency measurements
        ttft (List[Tuple[float, float]]): Time to first token measurements
        itl (List[Tuple[float, float]]): Inter-token latency measurements
        osl (List[Tuple[float, float]]): Output sequence length measurements
    """

    engine: "Engine"
    num_slots: int
    times: List[float]  # time_id -> beginning time of the slot in ticks
    queue_size: List[int]
    e2e_latency: List[Tuple[float, float]]  # pairs of time, e2e_latency
    ttft: List[Tuple[float, float]]  # pairs of time, ttft
    itl: List[Tuple[float, float]]  # pairs of time, itl
    osl: List[Tuple[float, float]]  # pairs of time, output sequence length

    def __init__(self, num_slots: int, engine: "Engine") -> None:
        """
        Initialize metrics collection.

        Args:
            num_slots (int): Number of batch slots to track
            engine (Engine): Reference to simulation engine
        """
        self.num_slots = num_slots
        self.engine = engine
        self.times = [0]
        self.queue_size = []
        self.e2e_latency = []
        self.ttft = []
        self.itl = []
        self.osl = []  # target_output_len_tokens

    def track_previous_batch(self) -> None:
        """
        Record metrics for the previous batch.

        Tracks:
        - TTFT when first token is generated
        - E2E latency when request completes
        - Output sequence length for completed requests
        """
        for slot, req in self.engine.current_batch.items():
            if req.tokens_generated == 1:
                self.ttft.append(
                    (
                        self.engine.current_time,
                        req.get_current_latency_at(self.engine.current_time),
                    )
                )
            if req.tokens_generated == req.target_output_len_tokens:
                self.e2e_latency.append(
                    (
                        self.engine.current_time,
                        req.get_current_latency_at(self.engine.current_time),
                    )
                )
                self.osl.append(
                    (self.engine.current_time, req.target_output_len_tokens)
                )

    def track_current_batch(self) -> None:
        """
        Record metrics for the current batch.

        Tracks:
        - Current queue size
        - ITL for requests that have generated more than one token
        """
        self.queue_size.append(len(self.engine.queue))
        for slot, req in self.engine.current_batch.items():
            if req.tokens_generated > 1:
                self.itl.append(
                    (self.engine.current_time, self.engine.get_current_batch_duration())
                )

    def get_time_interval(self, time_id: int) -> Tuple[float, float]:
        """
        Get time interval for a specific measurement.

        Args:
            time_id (int): Index of the measurement

        Returns:
            Tuple[float, float]: Start and end time of the interval
        """
        return (self.times[time_id], self.times[time_id + 1])

    @classmethod
    def get_values(cls, latencies: List[Tuple[float, float]]) -> List[float]:
        """
        Extract just the latency values from time-value pairs.

        Args:
            latencies (List[Tuple[float, float]]): List of (time, value) tuples

        Returns:
            List[float]: List of just the values

        Example:
            >>> latencies = Metrics.get_values(metrics.itl)
        """
        return list(latency for time, latency in latencies)

    def get_e2e_latencies(self) -> List[float]:
        """
        Get list of all end-to-end latency measurements.

        Returns:
            List[float]: List of E2E latencies
        """
        return Metrics.get_values(self.e2e_latency)

    def get_ttfts(self) -> List[float]:
        """
        Get list of all time to first token measurements.

        Returns:
            List[float]: List of TTFTs
        """
        return Metrics.get_values(self.ttft)

    def get_itls(self) -> List[float]:
        """
        Get list of all inter-token latency measurements.

        Returns:
            List[float]: List of ITLs
        """
        return Metrics.get_values(self.itl)

    def get_osls(self) -> List[float]:
        """
        Get list of all output sequence length measurements.

        Returns:
            List[float]: List of output sequence lengths
        """
        return Metrics.get_values(self.osl)
