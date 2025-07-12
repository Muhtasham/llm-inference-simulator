"""
LLM Inference Engine Simulator

This module provides a simulation engine for LLM inference batching strategies.
It models how different batching approaches affect throughput and latency.
"""

from typing import List, Dict, Set, TYPE_CHECKING

from .request import Request
from .plotting import PlotData
from .load_generator import LoadGenerator
from .batcher import Batcher

if TYPE_CHECKING:
    from .request import Request
    from .plotting import PlotData
    from .load_generator import LoadGenerator
    from .batcher import Batcher


class Engine:
    """
    Core simulation engine for LLM inference batching.

    This class simulates the behavior of an LLM inference engine, handling request batching,
    token generation, and performance tracking. It supports various batching strategies
    through pluggable batcher implementations.

    Attributes:
        max_batch_size (int): Maximum number of requests that can be processed in parallel
        queue (List[Request]): Pending requests waiting to be processed
        current_batch (Dict[int, Request]): Currently processing requests mapped to their slots
        current_time (float): Current simulation time
        plot_data (PlotData): Collected metrics and visualization data
        batcher (Batcher): Strategy for batching requests
        load_generator (LoadGenerator): Generator for simulated inference requests
    """

    max_batch_size: int
    queue: List[Request]
    current_batch: Dict[int, Request]
    current_time: float
    plot_data: PlotData
    batcher: Batcher
    load_generator: LoadGenerator

    def __init__(
        self, max_batch_size: int, load_generator: LoadGenerator, batcher: Batcher
    ) -> None:
        """
        Initialize the simulation engine.

        Args:
            max_batch_size (int): Maximum number of parallel requests
            load_generator (LoadGenerator): Strategy for generating inference requests
            batcher (Batcher): Strategy for batching requests
        """
        self.max_batch_size = max_batch_size
        self.plot_data = PlotData(num_slots=max_batch_size, engine=self)
        self.load_generator = load_generator
        self.load_generator.engine = self
        self.queue = []
        self.current_batch = {}
        self.batcher = batcher
        self.batcher.engine = self
        self.current_time = 0.0

    def run(self, time_limit: float = 10.0) -> None:
        """
        Run the simulation until the time limit is reached.

        For each time step:
        1. Generate tokens for current requests
        2. Track metrics
        3. Complete finished requests
        4. Generate new load
        5. Add new requests to batch

        Args:
            time_limit (float): How long to run the simulation
        """
        while self.current_time < time_limit:
            # generate tokens
            for req in self.current_batch.values():
                req.tick()

            self.plot_data.track_previous_batch()

            # Complete previous requests. Keep only the requests that are not completed yet
            self.current_batch = {
                slot: req
                for slot, req in self.current_batch.items()
                if req.tokens_generated < req.target_output_len_tokens
            }
            # generate load, add requests to the queue
            self.load_generator.generate_load()

            # Take requests from the queue to the batch
            self.batcher.add_requests()

            self.plot_data.track_current_batch()

            duration = self.get_current_batch_duration()
            self.current_time += duration

    def get_all_slots(self) -> Set[int]:
        """
        Return all available slot indices.

        Returns:
            Set[int]: Set of all slot indices.
        """
        return set(range(self.max_batch_size))

    def get_occupied_slots(self) -> Set[int]:
        """
        Return indices of slots currently processing requests.

        Returns:
            Set[int]: Set of occupied slot indices.
        """
        return set(self.current_batch.keys())

    def assign_request_to_slot(self, req: Request, slot: int) -> None:
        """
        Assign a request to a specific batch slot.

        Args:
            req (Request): Request to be processed
            slot (int): Slot index to assign the request to
        """
        req.started_at = self.current_time
        self.current_batch[slot] = req

    def add_requests_ifb(self) -> None:
        """
        Add requests using in-flight batching strategy.
        Fills empty slots with requests from the queue.
        """
        empty_slots = self.get_all_slots() - self.get_occupied_slots()
        for slot in empty_slots:
            if not len(self.queue):
                break
            req = self.queue.pop(0)
            self.assign_request_to_slot(req, slot)

    def get_prefilling_requests(self) -> List[Request]:
        """
        Return list of requests currently in prefill phase.

        Returns:
            List[Request]: Requests in prefill phase.
        """
        return [req for req in self.current_batch.values() if req.is_in_prefill()]

    def get_decoding_requests(self) -> List[Request]:
        """
        Return list of requests currently in decode phase.

        Returns:
            List[Request]: Requests in decode phase.
        """
        return [req for req in self.current_batch.values() if not req.is_in_prefill()]

    def get_current_batch_duration(self) -> float:
        """
        Calculate duration of current simulation step.

        For chunked context, considers individual chunk durations.
        For regular requests, uses total prefill time.
        Takes maximum between prefill and decode times.

        Returns:
            float: Duration of the current step in simulation time units
        """
        decoding_requests = self.get_decoding_requests()
        prefill_time = sum(
            [req.get_current_duration() for req in self.get_prefilling_requests()]
        )
        itl_time = (
            max([req.itl for req in decoding_requests]) if decoding_requests else 0
        )
        return max(prefill_time, itl_time, 1.0)  # 1. is the minimal step duration
