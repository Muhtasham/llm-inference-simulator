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


def save_plot(fig, name):
    """Save plotly figure to plots directory"""
    os.makedirs("plots", exist_ok=True)
    if fig is not None:
        pio.write_html(fig, f"plots/{name}.html")
        pio.write_image(fig, f"plots/{name}.png")


def run_static_batch():
    """Example with static batching"""
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


def run_ifb():
    """Example with In-Flight Batching"""
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


def run_chunked_context():
    """Example with Chunked Context"""
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


def run_one_prefill():
    """Example with One Prefill Per Batch"""
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


def run_concurrent_load():
    """Example with Concurrent Load Generation"""
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


def run_request_rate():
    """Example with Request Rate Load Generation"""
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


if __name__ == "__main__":
    run_static_batch()
    print("\n")
    run_ifb()
    print("\n")
    run_chunked_context()
    print("\n")
    run_one_prefill()
    print("\n")
    run_concurrent_load()
    print("\n")
    run_request_rate()
