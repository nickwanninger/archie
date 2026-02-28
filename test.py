#!/usr/bin/env python3
"""
Multi-threaded Roofline Benchmark

Explicitly controls threading to measure single-core vs multi-core performance.
Tests both sequential and parallel performance characteristics.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
import os
import ctypes

@dataclass
class ThreadedResults:
    num_threads: int
    bandwidth_gbps: float
    compute_gflops: float
    
def set_thread_affinity(core_id):
    """Pin thread to specific core (Linux only)"""
    try:
        os.sched_setaffinity(0, {core_id})
    except:
        pass

def control_numpy_threads(num_threads):
    """Control NumPy's internal threading"""
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['MKL_NUM_THREADS'] = str(num_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(num_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(num_threads)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(num_threads)

def get_current_numpy_threads():
    """Check how many threads NumPy is using"""
    try:
        import mkl
        return mkl.get_max_threads()
    except:
        pass
    
    try:
        import openblas
        return openblas.get_num_threads()
    except:
        pass
    
    return "Unknown"

def memory_bandwidth_worker(args):
    """Worker function for parallel memory bandwidth test"""
    core_id, size, iterations = args
    set_thread_affinity(core_id)
    
    # Each thread works on its own memory region
    a = np.random.rand(size)
    b = np.random.rand(size)
    c = np.random.rand(size)
    scalar = 3.14159
    
    # Warmup
    a[:] = b[:] + scalar * c[:]
    
    # Timed runs
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        a[:] = b[:] + scalar * c[:]
        end = time.perf_counter()
        times.append(end - start)
    
    bytes_per_op = 3 * size * 8  # 3 arrays, 8 bytes per double
    bandwidth = bytes_per_op / min(times) / 1e9
    
    return bandwidth

def measure_parallel_bandwidth(num_threads, size_per_thread_mb=50, iterations=5):
    """
    Measure aggregate memory bandwidth using multiple threads
    Each thread works on independent memory regions
    """
    size_per_thread = size_per_thread_mb * 1024 * 1024 // 8
    
    print(f"\nMeasuring parallel memory bandwidth ({num_threads} threads)...")
    print(f"  Each thread: {size_per_thread_mb}MB")
    
    # Use ProcessPoolExecutor for true parallelism (bypasses GIL)
    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        args = [(i, size_per_thread, iterations) for i in range(num_threads)]
        bandwidths = list(executor.map(memory_bandwidth_worker, args))
    
    total_bandwidth = sum(bandwidths)
    
    print(f"  Individual thread BWs: {[f'{bw:.2f}' for bw in bandwidths]} GB/s")
    print(f"  Aggregate bandwidth: {total_bandwidth:.2f} GB/s")
    
    return total_bandwidth

def measure_sequential_bandwidth(size_mb=100, iterations=10):
    """Single-threaded memory bandwidth (reference)"""
    size = size_mb * 1024 * 1024 // 8
    
    a = np.random.rand(size)
    b = np.random.rand(size)
    c = np.random.rand(size)
    scalar = 3.14159
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        a[:] = b[:] + scalar * c[:]
        end = time.perf_counter()
        times.append(end - start)
    
    bytes_transferred = 3 * size * 8
    bandwidth = bytes_transferred / min(times) / 1e9
    
    return bandwidth

def compute_worker(args):
    """Worker for parallel compute benchmark"""
    core_id, size, iterations = args
    set_thread_affinity(core_id)
    
    # Force single-threaded NumPy for this worker
    control_numpy_threads(1)
    
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    
    # Warmup
    C = A @ B
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        C = A @ B
        end = time.perf_counter()
        times.append(end - start)
    
    flops = 2 * size ** 3
    gflops = (flops / min(times)) / 1e9
    
    return gflops

def measure_parallel_compute(num_threads, matrix_size=512, iterations=20):
    """
    Measure parallel compute performance
    Each thread performs independent matrix multiplications
    """
    print(f"\nMeasuring parallel compute ({num_threads} threads)...")
    print(f"  Matrix size: {matrix_size}x{matrix_size}")
    
    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        args = [(i, matrix_size, iterations) for i in range(num_threads)]
        gflops_list = list(executor.map(compute_worker, args))
    
    total_gflops = sum(gflops_list)
    
    print(f"  Individual thread GFLOP/s: {[f'{gf:.2f}' for gf in gflops_list]}")
    print(f"  Aggregate GFLOP/s: {total_gflops:.2f}")
    
    return total_gflops

def measure_numpy_matmul_threading(size=1024, num_threads=None):
    """
    Measure NumPy's internal multi-threading for matrix multiply
    This shows how well BLAS scales
    """
    if num_threads:
        control_numpy_threads(num_threads)
    
    print(f"\nMeasuring NumPy/BLAS threading (set to {num_threads} threads)...")
    print(f"  Matrix size: {size}x{size}")
    
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    
    # Warmup
    C = A @ B
    
    times = []
    iterations = 50
    for _ in range(iterations):
        start = time.perf_counter()
        C = A @ B
        end = time.perf_counter()
        times.append(end - start)
    
    flops = 2 * size ** 3
    gflops = (flops / min(times)) / 1e9
    
    actual_threads = get_current_numpy_threads()
    print(f"  NumPy using: {actual_threads} threads")
    print(f"  Performance: {gflops:.2f} GFLOP/s")
    
    return gflops

def scaling_study(max_threads=None):
    """
    Perform a thread scaling study
    """
    if max_threads is None:
        max_threads = mp.cpu_count()
    
    print("="*60)
    print("Thread Scaling Study")
    print("="*60)
    print(f"Testing from 1 to {max_threads} threads")
    
    # Thread counts to test
    thread_counts = [1, 2, 4] + list(range(8, max_threads + 1, 4))
    thread_counts = [t for t in thread_counts if t <= max_threads]
    if max_threads not in thread_counts:
        thread_counts.append(max_threads)
    thread_counts = sorted(set(thread_counts))
    
    results = []
    
    for num_threads in thread_counts:
        print(f"\n{'='*60}")
        print(f"Testing with {num_threads} threads")
        print(f"{'='*60}")
        
        # Test parallel memory bandwidth
        bw = measure_parallel_bandwidth(num_threads, size_per_thread_mb=50, iterations=5)
        
        # Test parallel compute
        gflops = measure_parallel_compute(num_threads, matrix_size=512, iterations=10)
        
        results.append(ThreadedResults(num_threads, bw, gflops))
    
    return results

def plot_scaling(results, output_file='thread_scaling.png'):
    """Plot thread scaling results"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    threads = [r.num_threads for r in results]
    bandwidths = [r.bandwidth_gbps for r in results]
    gflops = [r.compute_gflops for r in results]
    
    # Bandwidth scaling
    ax1.plot(threads, bandwidths, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Threads', fontsize=12)
    ax1.set_ylabel('Memory Bandwidth (GB/s)', fontsize=12)
    ax1.set_title('Memory Bandwidth Scaling', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
    
    # Compute scaling
    ax2.plot(threads, gflops, 'o-', linewidth=2, markersize=8, color='orange')
    ax2.set_xlabel('Number of Threads', fontsize=12)
    ax2.set_ylabel('Compute Performance (GFLOP/s)', fontsize=12)
    ax2.set_title('Compute Performance Scaling', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)
    
    # Efficiency (speedup / num_threads)
    bw_speedup = [bw / bandwidths[0] for bw in bandwidths]
    compute_speedup = [gf / gflops[0] for gf in gflops]
    bw_efficiency = [speedup / threads[i] * 100 for i, speedup in enumerate(bw_speedup)]
    compute_efficiency = [speedup / threads[i] * 100 for i, speedup in enumerate(compute_speedup)]
    
    ax3.plot(threads, bw_efficiency, 'o-', linewidth=2, markersize=8, label='Memory')
    ax3.plot(threads, compute_efficiency, 's-', linewidth=2, markersize=8, label='Compute')
    ax3.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='Ideal (100%)')
    ax3.set_xlabel('Number of Threads', fontsize=12)
    ax3.set_ylabel('Parallel Efficiency (%)', fontsize=12)
    ax3.set_title('Parallel Efficiency', fontsize=13)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    ax3.set_xlim(left=0)
    ax3.set_ylim(0, 110)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nScaling plot saved to: {output_file}")

def plot_roofline_multithreaded(results, output_file='roofline_multithreaded.png'):
    """Generate roofline plots for different thread counts"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Select interesting thread counts to plot
    indices = [0, len(results)//3, 2*len(results)//3, -1]
    
    oi_range = np.logspace(-2, 2, 1000)
    
    for idx, ax in enumerate(axes):
        if idx >= len(results):
            break
            
        result = results[indices[idx]]
        
        # Calculate roofline
        mem_bound = result.bandwidth_gbps * oi_range
        comp_bound = np.full_like(oi_range, result.compute_gflops)
        roofline = np.minimum(mem_bound, comp_bound)
        
        ax.loglog(oi_range, roofline, 'r-', linewidth=3, label='Roofline')
        ax.loglog(oi_range, mem_bound, 'b--', alpha=0.5, linewidth=2)
        ax.axhline(y=result.compute_gflops, color='g', linestyle='--', 
                  alpha=0.5, linewidth=2)
        
        ridge = result.compute_gflops / result.bandwidth_gbps
        ax.plot(ridge, result.compute_gflops, 'ko', markersize=10)
        
        ax.set_xlabel('Operational Intensity (FLOP/Byte)', fontsize=11)
        ax.set_ylabel('Performance (GFLOP/s)', fontsize=11)
        ax.set_title(f'{result.num_threads} Thread(s)\n'
                    f'BW: {result.bandwidth_gbps:.1f} GB/s, '
                    f'Compute: {result.compute_gflops:.1f} GFLOP/s\n'
                    f'Ridge: {ridge:.2f} FLOP/Byte', fontsize=12)
        ax.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Multi-threaded roofline plots saved to: {output_file}")

def numpy_blas_study():
    """
    Study how NumPy's internal threading (BLAS) scales
    This is different from our manual parallelism
    """
    max_threads = mp.cpu_count()
    thread_counts = [1, 2, 4, 8, max_threads] if max_threads >= 8 else [1, 2, 4, max_threads]
    thread_counts = sorted(set([t for t in thread_counts if t <= max_threads]))
    
    print("\n" + "="*60)
    print("NumPy/BLAS Internal Threading Study")
    print("="*60)
    
    blas_results = []
    for nt in thread_counts:
        gflops = measure_numpy_matmul_threading(size=1024, num_threads=nt)
        blas_results.append((nt, gflops))
    
    print("\nBLAS Scaling Summary:")
    print(f"{'Threads':<10} {'GFLOP/s':<12} {'Speedup':<10} {'Efficiency':<10}")
    print("-" * 45)
    base_perf = blas_results[0][1]
    for nt, gflops in blas_results:
        speedup = gflops / base_perf
        efficiency = speedup / nt * 100
        print(f"{nt:<10} {gflops:<12.2f} {speedup:<10.2f}x {efficiency:<10.1f}%")
    
    return blas_results

def main():
    print("="*60)
    print("Multi-threaded Roofline Benchmark")
    print("="*60)
    
    num_cores = mp.cpu_count()
    print(f"\nDetected {num_cores} CPU cores")
    
    # First, study NumPy's internal threading
    blas_results = numpy_blas_study()
    
    # Then, study our explicit parallelism
    results = scaling_study(max_threads=num_cores)
    
    # Print summary
    print("\n" + "="*60)
    print("Scaling Summary")
    print("="*60)
    print(f"{'Threads':<10} {'Memory BW (GB/s)':<18} {'Compute (GFLOP/s)':<20} {'Ridge Point':<12}")
    print("-" * 65)
    for r in results:
        ridge = r.compute_gflops / r.bandwidth_gbps
        print(f"{r.num_threads:<10} {r.bandwidth_gbps:<18.2f} {r.compute_gflops:<20.2f} {ridge:<12.2f}")
    
    # Generate plots
    plot_scaling(results)
    plot_roofline_multithreaded(results)
    
    # Best overall performance
    best_result = max(results, key=lambda r: r.compute_gflops)
    print(f"\nBest Performance: {best_result.num_threads} threads")
    print(f"  Memory BW: {best_result.bandwidth_gbps:.2f} GB/s")
    print(f"  Compute: {best_result.compute_gflops:.2f} GFLOP/s")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
