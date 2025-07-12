"""
Demo: Metrics Visualization Example

This script demonstrates how to visualize and print experiment metrics using the simulator engine.
You can specify output file names and simulation parameters via command-line arguments.

Usage:
    python metrics_visualization.py --output metrics_demo --time-limit 100
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


def print_experiment_metrics(engine: sim.Engine) -> None:
    """Print experiment metrics with rich formatting."""
    console.rule("[bold yellow]Experiment Metrics")

    # Config Table
    config_table = Table(title="[bold]Experiment Configuration")
    config_table.add_column("Component", style="cyan")
    config_table.add_column("Settings", style="magenta")

    config_table.add_row("Load Generator", str(engine.load_generator))
    config_table.add_row("Batcher", str(engine.batcher))
    console.print(config_table)

    metrics = engine.plot_data.metrics
    e2e_latencies = metrics.get_e2e_latencies()
    ttfts = metrics.get_ttfts()
    itls = metrics.get_itls()

    # Latency Table
    latency_table = Table(title="[bold]Latency Metrics")
    latency_table.add_column("Metric", style="cyan")
    latency_table.add_column("Average", style="green")
    latency_table.add_column("Median", style="blue")

    latency_table.add_row(
        "E2E Latency",
        f"{np.mean(e2e_latencies):.2f}",
        f"{np.percentile(e2e_latencies, 50):.2f}",
    )
    latency_table.add_row(
        "TTFT", f"{np.mean(ttfts):.2f}", f"{np.percentile(ttfts, 50):.2f}"
    )
    latency_table.add_row(
        "ITL", f"{np.mean(itls):.2f}", f"{np.percentile(itls, 50):.2f}"
    )
    console.print(latency_table)

    # Throughput Table
    throughput_table = Table(title="[bold]Throughput Metrics")
    throughput_table.add_column("Metric", style="cyan")
    throughput_table.add_column("Value", style="magenta")

    num_requests = len(e2e_latencies)
    run_time = metrics.times[-1]
    requests_per_1k_ticks = 1000.0 * num_requests / run_time

    current_batch_tokens = sum(
        req.tokens_generated for req in engine.current_batch.values()
    )
    total_tokens = sum(metrics.get_osls()) + current_batch_tokens
    tokens_per_1k_ticks = 1000 * total_tokens / run_time

    throughput_table.add_row(
        "Requests/(1K ticks)/instance", f"{requests_per_1k_ticks:.2f}"
    )
    throughput_table.add_row("Tokens/(1K ticks)/instance", f"{tokens_per_1k_ticks:.2f}")
    console.print(throughput_table)


def main():
    parser = argparse.ArgumentParser(description="Metrics Visualization Example")
    parser.add_argument(
        "--output",
        type=str,
        default="metrics_visualization",
        help="Base name for output plot files (default: metrics_visualization)",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=100,
        help="Time limit for the simulation (default: 100)",
    )
    args = parser.parse_args()

    np.random.seed(42)
    load_generator = BatchLoadGenerator(
        initial_batch=100,
        target_output_len_tokens=10,
    )
    load_generator.target_output_len_std = 5

    engine = sim.Engine(
        max_batch_size=4, load_generator=load_generator, batcher=StaticBatcher()
    )
    engine.run(time_limit=args.time_limit)

    print_experiment_metrics(engine)
    fig = engine.plot_data.show()
    save_plot(fig, args.output)
    console.print(
        f"[green]Plots saved to 'plots/{args.output}.html' and 'plots/{args.output}.png'."
    )


if __name__ == "__main__":
    main()
