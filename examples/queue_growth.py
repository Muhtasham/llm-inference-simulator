"""
Demo: Queue Growth Analysis Example

This script analyzes queue growth under different load generators using the simulator engine.
You can select which analysis to run and set simulation parameters via command-line arguments.

Usage:
    python queue_growth.py --analysis request_rate --short 100 --long 10000
    python queue_growth.py --analysis all
"""

import argparse
from typing import Dict
from rich.console import Console
from rich.table import Table
import simulator as sim
import numpy as np
from simulator.load_generator import RequestRateLoadGenerator, ConcurrentLoadGenerator
from simulator.batcher import IFBatcherWithOnePrefillOnly

console = Console()

def run_simulation(load_generator, time_limit: int, show_plot: bool = False) -> Dict[str, float]:
    """Run simulation with given load generator and time limit."""
    engine = sim.Engine(
        max_batch_size=4,
        load_generator=load_generator,
        batcher=IFBatcherWithOnePrefillOnly(),
    )
    engine.run(time_limit=time_limit)

    if show_plot:
        engine.plot_data.show()

    metrics = engine.plot_data.metrics
    return {
        "queue_size": len(engine.queue),
        "ttft": np.mean(metrics.get_ttfts()),
        "e2e_latency": np.mean(metrics.get_e2e_latencies()),
    }

def print_comparison(title: str, short_run: Dict[str, float], long_run: Dict[str, float]) -> None:
    """Print comparison between short and long runs."""
    console.rule(f"[bold]{title}")

    table = Table(title=f"{title} Comparison")
    table.add_column("Metric", style="cyan")
    table.add_column("Short Run", style="green")
    table.add_column("Long Run", style="red")
    table.add_column("Difference", style="yellow")

    # Queue size
    table.add_row(
        "Final Queue Size",
        f"{short_run['queue_size']}",
        f"{long_run['queue_size']}",
        f"{long_run['queue_size'] - short_run['queue_size']}",
    )

    # TTFT
    table.add_row(
        "Average TTFT",
        f"{short_run['ttft']:.2f}",
        f"{long_run['ttft']:.2f}",
        f"{long_run['ttft'] - short_run['ttft']:.2f}",
    )

    # E2E Latency
    table.add_row(
        "Average E2E Latency",
        f"{short_run['e2e_latency']:.2f}",
        f"{long_run['e2e_latency']:.2f}",
        f"{long_run['e2e_latency'] - short_run['e2e_latency']:.2f}",
    )

    console.print(table)

def analyze_request_rate(short_ticks: int, long_ticks: int) -> None:
    """Analyze RequestRateLoadGenerator performance."""
    np.random.seed(42)
    load_generator = RequestRateLoadGenerator(
        request_rate=460.0 / 1000.0,  # 460 requests in 1000 ticks
        target_output_len_tokens=10,
        total_prefill_chunks=2,
    )
    load_generator.target_output_len_std = 5

    # Run short simulation
    short_run = run_simulation(load_generator, time_limit=short_ticks, show_plot=True)
    # Run long simulation
    long_run = run_simulation(load_generator, time_limit=long_ticks, show_plot=False)

    print_comparison("Request Rate Load Generator", short_run, long_run)

def analyze_concurrent_load(short_ticks: int, long_ticks: int) -> None:
    """Analyze ConcurrentLoadGenerator performance."""
    np.random.seed(42)
    load_generator = ConcurrentLoadGenerator(
        target_concurrency=6,
        target_output_len_tokens=10,
        total_prefill_chunks=2,
    )
    load_generator.target_output_len_std = 5

    # Run short simulation
    short_run = run_simulation(load_generator, time_limit=short_ticks, show_plot=True)
    # Run long simulation
    long_run = run_simulation(load_generator, time_limit=long_ticks, show_plot=False)

    print_comparison("Concurrent Load Generator", short_run, long_run)

def main():
    parser = argparse.ArgumentParser(description="Queue Growth Analysis Example")
    parser.add_argument(
        "--analysis",
        choices=["request_rate", "concurrent_load", "all"],
        default="all",
        help="Which analysis to run (default: all)",
    )
    parser.add_argument(
        "--short",
        type=int,
        default=100,
        help="Short run time limit in ticks (default: 100)",
    )
    parser.add_argument(
        "--long",
        type=int,
        default=10000,
        help="Long run time limit in ticks (default: 10000)",
    )
    args = parser.parse_args()

    if args.analysis in ("request_rate", "all"):
        analyze_request_rate(args.short, args.long)
        print("\n")
    if args.analysis in ("concurrent_load", "all"):
        analyze_concurrent_load(args.short, args.long)

if __name__ == "__main__":
    main()
