"""
Demo: Batching Strategies Examples

This script demonstrates various batching strategies using the simulator engine.
You can select which strategy to run via command-line arguments.

Usage:
    python batching_strategies.py --strategy static
    python batching_strategies.py --strategy all
"""

import argparse
from typing import Optional
from rich.console import Console
import simulator as sim
import numpy as np
from simulator.load_generator import (
    BatchLoadGenerator,
    ConcurrentLoadGenerator,
    RequestRateLoadGenerator,
)
from simulator.batcher import StaticBatcher, IFBatcher, IFBatcherWithOnePrefillOnly
import os
import plotly.io as pio

console = Console()

def save_plot(fig, name: str) -> None:
    """Save plotly figure to plots directory as HTML and PNG."""
    os.makedirs("plots", exist_ok=True)
    if fig is not None:
        pio.write_html(fig, f"plots/{name}.html")
        pio.write_image(fig, f"plots/{name}.png")

def run_static_batch() -> None:
    """Example with static batching."""
    console.rule("[bold blue]Static Batching Example")

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
    fig = engine.plot_data.show()
    save_plot(fig, "static_batch")
    sim.extra.print_experiment_metrics(engine)
    console.print("[green]Plots saved to 'plots/static_batch.html' and 'plots/static_batch.png'.")

def run_ifb() -> None:
    """Example with In-Flight Batching."""
    console.rule("[bold green]In-Flight Batching Example")

    np.random.seed(42)
    load_generator = BatchLoadGenerator(
        initial_batch=100,
        target_output_len_tokens=10,
    )
    load_generator.target_output_len_std = 5

    engine = sim.Engine(
        max_batch_size=4, load_generator=load_generator, batcher=IFBatcher()
    )
    engine.run(time_limit=100)
    fig = engine.plot_data.show()
    save_plot(fig, "ifb")
    sim.extra.print_experiment_metrics(engine)
    console.print("[green]Plots saved to 'plots/ifb.html' and 'plots/ifb.png'.")

def run_chunked_context() -> None:
    """Example with Chunked Context."""
    console.rule("[bold yellow]Chunked Context Example")

    np.random.seed(42)
    load_generator = BatchLoadGenerator(
        initial_batch=100,
        target_output_len_tokens=10,
        total_prefill_chunks=2,
    )
    load_generator.target_output_len_std = 5

    engine = sim.Engine(
        max_batch_size=4, load_generator=load_generator, batcher=IFBatcher()
    )
    engine.run(time_limit=100)
    fig = engine.plot_data.show()
    save_plot(fig, "chunked_context")
    sim.extra.print_experiment_metrics(engine)
    console.print("[green]Plots saved to 'plots/chunked_context.html' and 'plots/chunked_context.png'.")

def run_one_prefill() -> None:
    """Example with One Prefill Per Batch."""
    console.rule("[bold magenta]One Prefill Per Batch Example")

    np.random.seed(42)
    load_generator = BatchLoadGenerator(
        initial_batch=100,
        target_output_len_tokens=10,
        total_prefill_chunks=2,
    )
    load_generator.target_output_len_std = 5

    engine = sim.Engine(
        max_batch_size=4,
        load_generator=load_generator,
        batcher=IFBatcherWithOnePrefillOnly(),
    )
    engine.run(time_limit=100)
    fig = engine.plot_data.show()
    save_plot(fig, "one_prefill")
    sim.extra.print_experiment_metrics(engine)
    console.print("[green]Plots saved to 'plots/one_prefill.html' and 'plots/one_prefill.png'.")

def run_concurrent_load() -> None:
    """Example with Concurrent Load Generation."""
    console.rule("[bold cyan]Concurrent Load Example")

    np.random.seed(42)
    load_generator = ConcurrentLoadGenerator(
        target_concurrency=6,
        target_output_len_tokens=10,
        total_prefill_chunks=2,
    )
    load_generator.target_output_len_std = 5

    engine = sim.Engine(
        max_batch_size=4,
        load_generator=load_generator,
        batcher=IFBatcherWithOnePrefillOnly(),
    )
    engine.run(time_limit=100)
    fig = engine.plot_data.show()
    save_plot(fig, "concurrent_load")
    sim.extra.print_experiment_metrics(engine)
    console.print("[green]Plots saved to 'plots/concurrent_load.html' and 'plots/concurrent_load.png'.")

def run_request_rate() -> None:
    """Example with Request Rate Load Generation."""
    console.rule("[bold red]Request Rate Example")

    np.random.seed(42)
    load_generator = RequestRateLoadGenerator(
        request_rate=460.0 / 1000.0,  # 460 requests in 1000 ticks
        target_output_len_tokens=10,
        total_prefill_chunks=2,
    )
    load_generator.target_output_len_std = 5

    engine = sim.Engine(
        max_batch_size=4,
        load_generator=load_generator,
        batcher=IFBatcherWithOnePrefillOnly(),
    )
    engine.run(time_limit=100)
    fig = engine.plot_data.show()
    save_plot(fig, "request_rate")
    print(f"Final Queue: {len(engine.queue)}")
    sim.extra.print_experiment_metrics(engine)
    console.print("[green]Plots saved to 'plots/request_rate.html' and 'plots/request_rate.png'.")

def main():
    parser = argparse.ArgumentParser(description="Batching Strategies Demo")
    parser.add_argument(
        "--strategy",
        choices=[
            "static",
            "ifb",
            "chunked_context",
            "one_prefill",
            "concurrent_load",
            "request_rate",
            "all",
        ],
        default="all",
        help="Which strategy to run (default: all)",
    )
    args = parser.parse_args()

    if args.strategy in ("static", "all"):
        run_static_batch()
        print("\n")
    if args.strategy in ("ifb", "all"):
        run_ifb()
        print("\n")
    if args.strategy in ("chunked_context", "all"):
        run_chunked_context()
        print("\n")
    if args.strategy in ("one_prefill", "all"):
        run_one_prefill()
        print("\n")
    if args.strategy in ("concurrent_load", "all"):
        run_concurrent_load()
        print("\n")
    if args.strategy in ("request_rate", "all"):
        run_request_rate()

if __name__ == "__main__":
    main()
