"""
Demo: Batch Duration and Performance Examples

This script demonstrates basic and complex batch duration scenarios using the simulator engine.
It uses Rich for console output and Plotly for visualization. You can select which example to run
and set the time limit via command-line arguments.

Usage:
    python batch_duration_demo.py --example basic --time-limit 10
    python batch_duration_demo.py --example complex --time-limit 100
"""

import argparse
from rich.console import Console
from rich.table import Table
import simulator as sim
from simulator.load_generator import BatchLoadGenerator
from simulator.batcher import StaticBatcher
import numpy as np
import os
import plotly.io as pio

console = Console()


def save_plot(fig, name: str) -> None:
    """Save plotly figure to plots directory as HTML and PNG."""
    os.makedirs("plots", exist_ok=True)
    if fig is not None:
        pio.write_html(fig, f"plots/{name}.html")
        pio.write_image(fig, f"plots/{name}.png")


def run_basic_example(time_limit: int = 10) -> None:
    """Run a basic example with 2 slots and 2 initial requests."""
    console.rule("[bold blue]Basic Example: 2 slots, 2 initial requests")

    engine = sim.Engine(
        max_batch_size=2,
        load_generator=BatchLoadGenerator(initial_batch=2),
        batcher=StaticBatcher(),
    )
    engine.run(time_limit=time_limit)

    # Create rich table for batch info
    table = Table(title="Batch Information")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    duration = engine.get_current_batch_duration()
    table.add_row("Current Batch Duration", f"{duration:.2f}")
    table.add_row("Batch Size", str(engine.max_batch_size))
    table.add_row("Time Limit", str(time_limit))

    console.print(table)
    fig = engine.plot_data.show()
    save_plot(fig, "basic_example")
    console.print(
        "[green]Plots saved to 'plots/basic_example.html' and 'plots/basic_example.png'."
    )


def run_complex_example(time_limit: int = 100) -> None:
    """Run a complex example with varying output lengths."""
    console.rule("[bold green]Complex Example: Varying Output Lengths")

    np.random.seed(42)
    load_generator = BatchLoadGenerator(
        initial_batch=100,
        target_output_len_tokens=10,
    )
    load_generator.target_output_len_std = 5

    engine = sim.Engine(
        max_batch_size=4, load_generator=load_generator, batcher=StaticBatcher()
    )
    engine.run(time_limit=time_limit)

    metrics = engine.plot_data.metrics
    e2e_latencies = metrics.get_e2e_latencies()

    # Create rich table for metrics
    table = Table(title="Performance Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Average E2E Latency", f"{np.mean(e2e_latencies):.2f}")
    table.add_row("Median E2E Latency", f"{np.percentile(e2e_latencies, 50):.2f}")
    table.add_row(
        "Current Batch Duration", f"{engine.get_current_batch_duration():.2f}"
    )

    console.print(table)
    fig = engine.plot_data.show()
    save_plot(fig, "complex_example")
    console.print(
        "[green]Plots saved to 'plots/complex_example.html' and 'plots/complex_example.png'."
    )


def main():
    parser = argparse.ArgumentParser(description="Batch Duration Demo Examples")
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
