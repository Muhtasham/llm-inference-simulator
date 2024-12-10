"""
Performance Metrics Collection for LLM Inference Simulation

This module handles the collection and processing of performance metrics
during the simulation, including latencies, throughput, and queue statistics.
"""

from .request import Request
from .load_generator import LoadGenerator
from .batcher import Batcher
from .metrics import Metrics
from .engine import Engine

__all__ = [
    "Request",
    "LoadGenerator",
    "Batcher",
    "Metrics",
    "Engine",
]


class extra:
    """
    Additional utility functions for the simulator.

    Functions:
        print_experiment_metrics: Display performance metrics
        catpure_function_prints: Capture function output for testing
        check_print_metrics: Validate metrics output format
    """

    from .extra import (
        print_experiment_metrics,
        catpure_function_prints,
        check_print_metrics,
    )
