"""
Disaggregated Prefill & Decode Example

This script demonstrates disaggregated prefill and decode inference using the simulator engine.
It uses Rich for console output and Plotly for visualization. You can select which example to run
and set the time limit via command-line arguments.

Usage:
    python disaggregated.py --example basic --time-limit 10
    python disaggregated.py --example complex --time-limit 100
"""

import argparse
from rich.console import Console
from rich.table import Table
import simulator as sim
from simulator.load_generator import BatchLoadGenerator
import numpy as np
import os
import plotly.io as pio

console = Console()


def save_plot(fig, name: str) -> None:
    """Save plotly figure to plots directory as HTML only."""
    os.makedirs("plots", exist_ok=True)
    if fig is not None:
        pio.write_html(fig, f"plots/{name}.html")


def run_basic_example(time_limit: int = 10) -> None:
    """Run a basic disaggregated example with 2 prefill and 2 decode workers."""
    console.rule("[bold blue]Disaggregated Basic Example: 2 prefill, 2 decode workers")

    # Initial batch of 2 requests
    load_generator = BatchLoadGenerator(initial_batch=2)
    engine = sim.DisaggregatedEngine(
        max_prefill_workers=2,
        max_decode_workers=2,
        load_generator=load_generator,
        prefill_batcher=sim.DisaggregatedPrefillBatcher(),
        decode_batcher=sim.DisaggregatedDecodeBatcher(),
    )
    # Directly add requests to prefill queue at start
    for i in range(load_generator.initial_batch):
        req = load_generator.get_request(str(i))
        engine.prefill_queue.append(req)
    engine.run(time_limit=time_limit)

    # Table for batch info
    table = Table(title="Disaggregated Batch Information")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    duration = engine.get_current_batch_duration()
    table.add_row("Current Batch Duration", f"{duration:.2f}")
    table.add_row("Prefill Workers", str(engine.max_prefill_workers))
    table.add_row("Decode Workers", str(engine.max_decode_workers))
    table.add_row("Time Limit", str(time_limit))
    console.print(table)
    fig = engine.plot_data.show()
    save_plot(fig, "disagg_basic")
    console.print("[green]Plot saved to 'plots/disagg_basic.html'.")


def run_complex_example(time_limit: int = 100) -> None:
    """Run a complex disaggregated example with more requests and workers."""
    console.rule(
        "[bold green]Disaggregated Complex Example: 4 prefill, 4 decode workers"
    )
    np.random.seed(42)
    load_generator = BatchLoadGenerator(
        initial_batch=100,
        target_output_len_tokens=10,
    )
    load_generator.target_output_len_std = 5
    engine = sim.DisaggregatedEngine(
        max_prefill_workers=4,
        max_decode_workers=4,
        load_generator=load_generator,
        prefill_batcher=sim.DisaggregatedPrefillBatcher(),
        decode_batcher=sim.DisaggregatedDecodeBatcher(),
    )
    # Directly add requests to prefill queue at start
    for i in range(load_generator.initial_batch):
        req = load_generator.get_request(str(i))
        engine.prefill_queue.append(req)
    engine.run(time_limit=500)
    metrics = engine.plot_data.metrics
    e2e_latencies = metrics.get_e2e_latencies()
    table = Table(title="Disaggregated Performance Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    if e2e_latencies:
        table.add_row("Average E2E Latency", f"{np.mean(e2e_latencies):.2f}")
        table.add_row("Median E2E Latency", f"{np.percentile(e2e_latencies, 50):.2f}")
    else:
        table.add_row("Average E2E Latency", "N/A")
        table.add_row("Median E2E Latency", "N/A")
    table.add_row(
        "Current Batch Duration", f"{engine.get_current_batch_duration():.2f}"
    )
    console.print(table)
    fig = engine.plot_data.show()
    save_plot(fig, "disagg_complex")
    console.print("[green]Plot saved to 'plots/disagg_complex.html'.")


def main():
    parser = argparse.ArgumentParser(description="Disaggregated Inference Examples")
    parser.add_argument(
        "--example",
        choices=["basic", "complex", "all"],
        default="all",
        help="Which example to run: 'basic', 'complex', or 'all' (default: all)",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=None,
        help="Time limit for the simulation (overrides default for each example)",
    )
    args = parser.parse_args()
    if args.example in ("basic", "all"):
        run_basic_example(time_limit=args.time_limit or 10)
        print("\n")
    if args.example in ("complex", "all"):
        run_complex_example(time_limit=args.time_limit or 100)


if __name__ == "__main__":
    main()
