#!/usr/bin/env python3
"""
Benchmark causal_conv1d performance across different thread counts.
Logs execution time to CSV.
"""

import torch
import time
import csv
from pathlib import Path

from causal_conv1d.causal_conv1d_interface import causal_conv1d_fn


def benchmark_causal_conv1d(
    batch_size=4,
    dim=2304,
    seqlen=2048,
    width=2,
    num_threads=128,
    warmup_runs=10,
    benchmark_runs=1000,
    dtype=torch.bfloat16,
):
    """
    Benchmark causal_conv1d_fn with specified parameters.
    
    Returns:
        float: Average execution time in milliseconds
    """
    device = "cuda"
    
    # Allocate tensors
    x = torch.randn(batch_size, dim, seqlen, device=device, dtype=dtype)
    weight = torch.randn(dim, width, device=device, dtype=torch.float32)
    bias = torch.randn(dim, device=device, dtype=torch.float32)
    
    # Warmup runs
    for _ in range(warmup_runs):
        out = causal_conv1d_fn(
            x, weight, bias, 
            activation="silu",
            number_of_threads=num_threads
        )
        torch.cuda.synchronize()
    
    # Benchmark runs
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(benchmark_runs)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(benchmark_runs)]
    
    for i in range(benchmark_runs):
        start_events[i].record()
        out = causal_conv1d_fn(
            x, weight, bias,
            activation="silu", 
            number_of_threads=num_threads
        )
        end_events[i].record()
    
    torch.cuda.synchronize()
    
    # Calculate average time
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    avg_time = sum(times) / len(times)
    
    return avg_time


def main():
    print("Starting causal_conv1d benchmarking...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print()
    
    # Configuration
    batch_size = 2
    dim = 2304
    seqlen = 2048
    width = 2
    dtype = torch.bfloat16
    
    # Thread counts to benchmark (powers of 2 from 64 to 1024)
    thread_counts = [64, 128, 256, 512, 1024]
    
    # Output CSV file
    output_file = Path("causal_conv1d_benchmark.csv")
    
    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Dimensions: {dim}")
    print(f"  Sequence length: {seqlen}")
    print(f"  Kernel width: {width}")
    print(f"  Data type: {dtype}")
    print(f"  Thread counts: {thread_counts}")
    print()
    
    results = []
    
    for num_threads in thread_counts:
        print(f"Benchmarking with {num_threads} threads...", end=" ", flush=True)
        
        try:
            avg_time = benchmark_causal_conv1d(
                batch_size=batch_size,
                dim=dim,
                seqlen=seqlen,
                width=width,
                num_threads=num_threads,
                dtype=dtype,
            )
            
            results.append({
                "threads": num_threads,
                "avg_time_ms": avg_time,
                "batch_size": batch_size,
                "dim": dim,
                "seqlen": seqlen,
                "width": width,
            })
            
            print(f"✓ {avg_time:.4f} ms")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            results.append({
                "threads": num_threads,
                "avg_time_ms": None,
                "batch_size": batch_size,
                "dim": dim,
                "seqlen": seqlen,
                "width": width,
                "error": str(e),
            })
    
    # Write results to CSV
    print()
    print(f"Writing results to {output_file}...")
    
    with open(output_file, "w", newline="") as f:
        fieldnames = ["threads", "avg_time_ms", "batch_size", "dim", "seqlen", "width"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print("Done!")
    print()
    print("Results summary:")
    print("-" * 50)
    print(f"{'Threads':<10} {'Avg Time (ms)':<15} {'Speedup':<10}")
    print("-" * 50)
    
    baseline_time = None
    for result in results:
        if result["avg_time_ms"] is not None:
            if baseline_time is None:
                baseline_time = result["avg_time_ms"]
                speedup = 1.0
            else:
                speedup = baseline_time / result["avg_time_ms"]
            
            print(f"{result['threads']:<10} {result['avg_time_ms']:<15.4f} {speedup:<10.2f}x")
        else:
            print(f"{result['threads']:<10} {'Failed':<15} {'-':<10}")
    
    print("-" * 50)


if __name__ == "__main__":
    main()