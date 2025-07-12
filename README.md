# LLM Inference Simulator

A simulator for exploring different batching strategies and load patterns in LLM inference.

## Installation & Setup

```bash
# Install uv package manager
pip install uv

# Install dependencies
uv sync
```

## Understanding Ticks

In this simulator:

- A **tick** is the basic unit of time
- `prefill_time=2` means the prefill phase takes 2 ticks
- `itl=1` (Inter-Token Latency) means generating each token takes 1 tick
- Metrics are often reported per 1000 ticks for easier comparison
- Example: `460./1000.` request rate means 460 requests per 1000 ticks

## Running Examples

Each example demonstrates different aspects of the simulator:

```bash
# Basic examples with simple configurations
uv run examples/batch_duration_demo.py

# Detailed metrics visualization
uv run examples/metrics_visualization.py

# Advanced batching strategies comparison
uv run examples/batching_strategies.py

# Queue growth analysis for long runs
uv run examples/queue_growth.py
```

## Features

- Multiple batching strategies (Static, In-Flight, Chunked Context)
- Various load generation patterns (Batch, Concurrent, Request Rate)
- Rich metrics visualization
- Configurable batch sizes and request parameters
- Queue growth analysis for long-running simulations

## Batching Strategies and Performance

### Static Batching

Basic batching strategy that only batches requests when all slots are empty.

```python
# Configuration
engine = sim.Engine(
    max_batch_size=4,  # Maximum 4 requests in a batch
    load_generator=BatchLoadGenerator(
        initial_batch=100,  # Send 100 requests at start
        prefill_time=2,    # Each prefill takes 2 ticks
        itl=1,             # Each token generation takes 1 tick
        target_output_len_tokens=10  # Generate 10 tokens per request
    ),
    batcher=StaticBatcher()
)
```

Performance:

```bash
Average E2E Latency: 58.16
Average TTFT: 52.80
Average ITL: 1.00
Requests/(1K ticks)/instance = 190.00
Tokens/(1K ticks)/instance = 1680.00
```

### In-Flight Batching (IFB)

Allows mixing prefill and decode phases in the same batch.

```python
# Configuration
engine = sim.Engine(
    max_batch_size=4,
    load_generator=BatchLoadGenerator(
        initial_batch=100,
        prefill_time=2,
        itl=1,
        target_output_len_tokens=10
    ),
    batcher=IFBatcher()
)
```

Performance:

```bash
Average E2E Latency: 58.44
Average TTFT: 52.90
Average ITL: 1.39
Requests/(1K ticks)/instance = 267.33  # 41% improvement over Static
Tokens/(1K ticks)/instance = 2376.24
```

### Chunked Context

Optimizes performance by separating prefill into chunks.

```python
# Configuration
load_generator = BatchLoadGenerator(
    initial_batch=100,
    prefill_time=2,
    itl=1,
    target_output_len_tokens=10,
    total_prefill_chunks=2  # Split prefill into 2 chunks
)
engine = sim.Engine(
    max_batch_size=4,
    load_generator=load_generator,
    batcher=IFBatcher()
)
```

Performance:

```bash
Average E2E Latency: 57.42
Average TTFT: 54.51
Average ITL: 1.14
Requests/(1K ticks)/instance = 310.00  # 15% improvement over basic IFB
Tokens/(1K ticks)/instance = 2730.00
```

### One Prefill Per Batch

Limits to one prefill request at a time for balanced compute/memory usage.

```python
# Configuration
engine = sim.Engine(
    max_batch_size=4,
    load_generator=load_generator,
    batcher=IFBatcherWithOnePrefillOnly()
)
```

Performance:

```bash
Average E2E Latency: 55.94
Average TTFT: 52.13
Average ITL: 1.00
Requests/(1K ticks)/instance = 360.00  # Best throughput
Tokens/(1K ticks)/instance = 3170.00
```

## Load Generation Patterns

### Concurrent Load

Maintains a target level of concurrent requests.

```python
# Configuration
load_generator = ConcurrentLoadGenerator(
    target_concurrency=6,    # Maintain 6 concurrent requests
    target_output_len_tokens=10,
    total_prefill_chunks=2,
    prefill_time=2,
    itl=1
)
```

Performance:

```bash
Average E2E Latency: 15.14
Average TTFT: 7.87
Average ITL: 1.00
Requests/(1K ticks)/instance = 360.00
Tokens/(1K ticks)/instance = 3170.00
```

### Request Rate

Generates requests at a constant rate.

```python
# Configuration
load_generator = RequestRateLoadGenerator(
    request_rate=460./1000.,  # 460 requests per 1000 ticks
    target_output_len_tokens=10,
    total_prefill_chunks=2,
    prefill_time=2,
    itl=1
)
```

Performance:

```bash
Average E2E Latency: 17.66
Average TTFT: 11.03
Average ITL: 1.00
Requests/(1K ticks)/instance = 350.00
Tokens/(1K ticks)/instance = 3060.00
```

## Queue Growth Analysis

Compare performance between short (100 ticks) and long (10000 ticks) runs:

```bash
Request Rate Load Generator (460 requests/1000 ticks)
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Metric           ┃ 100 ticks  ┃ 10000 ticks ┃ Difference ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ Final Queue Size │ 6          │ 1138        │ 1132       │
│ Average TTFT     │ 11.03      │ 1245.77     │ 1234.75    │
│ Average E2E      │ 17.66      │ 1253.78     │ 1236.12    │
└──────────────────┴────────────┴─────────────┴────────────┘

Concurrent Load Generator (6 concurrent requests)
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Metric           ┃ 100 ticks  ┃ 10000 ticks ┃ Difference ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ Final Queue Size │ 2          │ 2           │ 0          │
│ Average TTFT     │ 7.87       │ 8.61        │ 0.74       │
│ Average E2E      │ 15.14      │ 17.32       │ 2.19       │
└──────────────────┴────────────┴─────────────┴────────────┘
```

Key observations:

- Request Rate generator shows significant queue growth over time
- Concurrent Load generator maintains stable queue size and latencies
- TTFT and E2E latency increase dramatically with queue growth
- One Prefill Per Batch strategy achieves best throughput (3170 tokens/1K ticks)
- IFB improves throughput by 41% over Static Batching
- Chunked Context further improves throughput by 15% over basic IFB

## Key Metrics

- **E2E Latency**: End-to-end latency for request completion (in ticks)
- **TTFT**: Time to first token (in ticks)
- **ITL**: Inter-token latency (ticks between tokens)
- **Throughput**: Requests and tokens processed per 1K ticks per instance
- **Queue Size**: Number of requests waiting to be processed

## Future Tasks:

**Note:**
This simulator currently models the *core disaggregated inference flow* of [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) (separate prefill and decode workers, global prefill queue, and request flow through both phases). However, it does **not yet** implement advanced features such as conditional disaggregation (router logic), KV transfer simulation, prefix cache hit logic, dynamic worker scaling, or distributed infrastructure (NATS/ETCD). These are planned for future development.

- [ ] Add conditional disaggregation (router) to decide if prefill is done locally or remotely
- [ ] Simulate KV cache transfer time and bandwidth between prefill and decode phases
- [ ] Implement prefix cache hit logic to sometimes skip or reduce prefill phase
- [ ] Allow dynamic scaling (add/remove) of prefill and decode workers during simulation
- [ ] (Advanced) Simulate KV descriptor/layout management and block merging
- [ ] (Advanced) Model distributed infrastructure (NATS/ETCD) for realistic coordination and scaling