from .metrics import Metrics
import numpy as np
import io
import sys
import difflib


def print_experiment_metrics(engine, show_median=False):
    print("# Experiment Config:")
    print(f"load_generator = {str(engine.load_generator)}")
    print(f"batcher = {str(engine.batcher)}")
    metrics: Metrics = engine.plot_data.metrics
    # we record the latency of every completed request
    e2e_latencies = metrics.get_e2e_latencies()
    ttfts = metrics.get_ttfts()
    itls = metrics.get_itls()

    print("\n# Latency Metrics:")
    print(f"Average E2E Latency: {np.mean(e2e_latencies):.2f}")
    print(f"Average TTFT: {np.mean(ttfts):.2f}")
    print(f"Average ITL: {np.mean(itls):.2f}")

    if show_median:
        print(f"Median E2E Latency: {np.percentile(e2e_latencies, 0.5):.2f}")
        print(f"Median TTFT: {np.percentile(ttfts, 0.5):.2f}")
        print(f"Median ITL: {np.percentile(itls, 0.5):.2f}")

    print("\n# Throughput Metrics:")
    num_requests: int = len(e2e_latencies)
    run_time: float = metrics.times[-1]

    requests_per_1k_ticks_per_instance: float = 1000.0 * num_requests / run_time

    print(f"Requests/(1K ticks)/instance = {requests_per_1k_ticks_per_instance:.2f}")

    current_batch_tokens = sum(
        req.tokens_generated for req in engine.current_batch.values()
    )
    total_tokens_generated = sum(metrics.get_osls()) + current_batch_tokens
    tokens_per_1k_ticks_per_instance = 1000 * total_tokens_generated / run_time
    print(f"Tokens/(1K ticks)/instance = {tokens_per_1k_ticks_per_instance:.2f}")


def catpure_function_prints(fn):
    try:
        capturedOutput = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = capturedOutput
        fn()
        sys.stdout = old_stdout
        return capturedOutput.getvalue()

    finally:
        if old_stdout:
            sys.stdout = old_stdout


def check_print_metrics(print_experiment_metrics_function, engine, show_median=False):
    test_print = catpure_function_prints(
        lambda: print_experiment_metrics_function(engine)
    )
    valid_print = catpure_function_prints(
        lambda: print_experiment_metrics(engine, show_median)
    )
    for line in difflib.unified_diff(
        test_print.split("\n"),
        valid_print.split("\n"),
        fromfile="test",
        tofile="valid",
        lineterm="",
    ):
        print(line)
