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


class DisaggregatedEngine:
    """
    Simulation engine for disaggregated prefill and decode inference.
    Maintains separate worker pools/queues for prefill and decode phases.
    """

    def __init__(
        self,
        max_prefill_workers: int,
        max_decode_workers: int,
        load_generator,
        prefill_batcher,
        decode_batcher,
    ):
        self.max_prefill_workers = max_prefill_workers
        self.max_decode_workers = max_decode_workers
        self.load_generator = load_generator
        self.load_generator.engine = self
        self.prefill_batcher = prefill_batcher
        self.prefill_batcher.engine = self
        self.decode_batcher = decode_batcher
        self.decode_batcher.engine = self
        self.prefill_queue = []  # Requests waiting for prefill
        self.decode_queue = []  # Requests waiting for decode
        self.prefill_slots = {}  # slot_id: Request
        self.decode_slots = {}  # slot_id: Request
        self.current_time = 0.0
        self.plot_data = PlotData(
            num_slots=max_prefill_workers + max_decode_workers, engine=self
        )
        self.completed_requests = []

    def run(self, time_limit: float = 10.0) -> None:
        while self.current_time < time_limit:
            # Step 1: Progress prefill and decode requests
            for req in list(self.prefill_slots.values()):
                req.tick()
            for req in list(self.decode_slots.values()):
                req.tick()

            self.plot_data.track_previous_batch()

            # Step 2: Move completed prefill requests to decode queue
            finished_prefill = [
                slot
                for slot, req in self.prefill_slots.items()
                if not req.is_in_prefill()
            ]
            for slot in finished_prefill:
                req = self.prefill_slots.pop(slot)
                self.decode_queue.append(req)

            # Step 3: Remove completed decode requests
            finished_decode = [
                slot
                for slot, req in self.decode_slots.items()
                if req.tokens_generated >= req.target_output_len_tokens
            ]
            for slot in finished_decode:
                req = self.decode_slots.pop(slot)
                self.completed_requests.append(req)

            # Step 4: Generate new load
            self.load_generator.generate_load()

            # Step 5: Add new requests to prefill slots
            self.prefill_batcher.add_requests()
            # Step 6: Add new requests to decode slots
            self.decode_batcher.add_requests()

            self.plot_data.track_current_batch()

            # Debug print for request flow
            print(
                f"Time: {self.current_time:.2f} | "
                f"PrefillQ: {len(self.prefill_queue)} | "
                f"PrefillSlots: {len(self.prefill_slots)} | "
                f"DecodeQ: {len(self.decode_queue)} | "
                f"DecodeSlots: {len(self.decode_slots)} | "
                f"Completed: {len(self.completed_requests)}"
            )

            # Step 7: Advance time by max of prefill/decode durations (or 1.0 min step)
            prefill_durations = [
                req.get_current_duration() for req in self.prefill_slots.values()
            ] or [0]
            decode_durations = [
                req.get_current_duration() for req in self.decode_slots.values()
            ] or [0]
            duration = max(max(prefill_durations), max(decode_durations), 1.0)
            self.current_time += duration

    def get_all_slots(self):
        # Return all slot indices (prefill + decode)
        return set(range(self.max_prefill_workers + self.max_decode_workers))

    def get_prefill_slots(self):
        return set(range(self.max_prefill_workers))

    def get_decode_slots(self):
        return set(
            range(
                self.max_prefill_workers,
                self.max_prefill_workers + self.max_decode_workers,
            )
        )

    def get_occupied_prefill_slots(self):
        return set(self.prefill_slots.keys())

    def get_occupied_decode_slots(self):
        return set(self.decode_slots.keys())

    def get_occupied_slots(self):
        # Return all occupied slot indices (prefill + decode)
        return self.get_occupied_prefill_slots().union(self.get_occupied_decode_slots())

    def assign_request_to_prefill_slot(self, req, slot):
        req.started_at = self.current_time
        self.prefill_slots[slot] = req

    def assign_request_to_decode_slot(self, req, slot):
        # Optionally, simulate KV cache transfer time here
        req.started_at = self.current_time
        self.decode_slots[slot] = req

    def get_current_batch_duration(self):
        prefill_durations = [
            req.get_current_duration() for req in self.prefill_slots.values()
        ] or [0]
        decode_durations = [
            req.get_current_duration() for req in self.decode_slots.values()
        ] or [0]
        return max(max(prefill_durations), max(decode_durations), 1.0)

    # For compatibility with plotting/metrics
    @property
    def queue(self):
        # For metrics: total queue size (prefill + decode)
        return self.prefill_queue + self.decode_queue

    @property
    def current_batch(self):
        # For metrics: all active requests (prefill + decode)
        return {**self.prefill_slots, **self.decode_slots}

    @property
    def max_batch_size(self):
        return self.max_prefill_workers + self.max_decode_workers

    def get_prefilling_requests(self):
        return list(self.prefill_slots.values())

    def get_decoding_requests(self):
        return list(self.decode_slots.values())

    # For batchers to use
    def pop_prefill_request(self):
        if self.prefill_queue:
            return self.prefill_queue.pop(0)
        return None

    def pop_decode_request(self):
        if self.decode_queue:
            return self.decode_queue.pop(0)
        return None
