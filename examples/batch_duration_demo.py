from rich.console import Console
from rich.table import Table
import simulator as sim
from simulator.load_generator import BatchLoadGenerator
from simulator.batcher import StaticBatcher
import numpy as np
import os
import plotly.io as pio

console = Console()


def save_plot(fig, name):
    """Save plotly figure to plots directory"""
    os.makedirs("plots", exist_ok=True)
    if fig is not None:
        pio.write_html(fig, f"plots/{name}.html")
        pio.write_image(fig, f"plots/{name}.png")


def run_basic_example():
    """Basic example with 2 slots and 2 initial requests"""
    console.rule("[bold blue]Basic Example: 2 slots, 2 initial requests")

    engine = sim.Engine(
        max_batch_size=2,
        load_generator=BatchLoadGenerator(initial_batch=2),
        batcher=StaticBatcher(),
    )
    engine.run(time_limit=10)

    # Create rich table for batch info
    table = Table(title="Batch Information")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    duration = engine.get_current_batch_duration()
    table.add_row("Current Batch Duration", f"{duration:.2f}")
    table.add_row("Batch Size", str(engine.max_batch_size))
    table.add_row("Time Limit", "10")

    console.print(table)
    fig = engine.plot_data.show()
    save_plot(fig, "basic_example")


def run_complex_example():
    """Complex example with varying output lengths"""
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
    engine.run(time_limit=100)

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


if __name__ == "__main__":
    run_basic_example()
    print("\n")
    run_complex_example()
