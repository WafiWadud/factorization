"""
Comprehensive benchmark suite for integer factorization library.

Benchmarks:
1. Core Algorithms: Trial Division, Pollard Rho, Quadratic Sieve, ECM
2. SIMD Operations: Trial Division JIT, Batch Inverse, ECM Point Operations
3. Primality Testing: Miller-Rabin with memoization
4. Cache Performance: Repeated calls with and without caching
5. Algorithm Selection: Automatic selection based on number size
6. Memory Efficiency: Peak memory usage and cache size
"""

import time
import sys
import random
import statistics
from typing import List, Tuple, Callable, Any
import numpy as np

from factorization import (
    is_prime, trial_division, pollard_rho_brent, factor, 
    quadratic_sieve, ecm, clear_caches, is_simd_available
)
from simd_operations import (
    _trial_division_simd, _batch_inverse_simd, _batch_inverse_pure_python,
    _ecm_point_double_simd, _ecm_scalar_mult_simd
)

__all__: List[str] = [
    'BenchmarkResult', 'benchmark', 'benchmark_primality', 
    'benchmark_trial_division', 'benchmark_batch_inverse',
    'benchmark_ecm_point_ops', 'benchmark_pollard_rho',
    'benchmark_quadratic_sieve', 'benchmark_ecm_factorization',
    'benchmark_complete_factorization', 'benchmark_caching_impact',
    'benchmark_algorithm_selection', 'benchmark_stress_test',
    'run_all_benchmarks'
]


# ============================================================================
# BENCHMARK UTILITIES
# ============================================================================

class BenchmarkResult:
    """Store benchmark results with statistics."""
    
    name: str
    times: List[float]
    operations: int
    min: float
    max: float
    mean: float
    median: float
    stdev: float
    
    def __init__(self, name: str, times: List[float], operations: int = 1) -> None:
        self.name = name
        self.times = sorted(times)  # Remove outliers better with sorted
        self.operations = operations
        
        # Calculate statistics
        self.min = min(times)
        self.max = max(times)
        self.mean = statistics.mean(times)
        self.median = statistics.median(times)
        self.stdev = statistics.stdev(times) if len(times) > 1 else 0.0
        
    def __str__(self) -> str:
        return (f"{self.name:40} | "
                f"Mean: {self.mean*1000:8.3f}ms | "
                f"Median: {self.median*1000:8.3f}ms | "
                f"StdDev: {self.stdev*1000:8.3f}ms | "
                f"Min: {self.min*1000:8.3f}ms | "
                f"Max: {self.max*1000:8.3f}ms")


def benchmark(func: Callable[..., Any], *args: Any, iterations: int = 5, **kwargs: Any) -> BenchmarkResult:
    """
    Benchmark a function and return statistics.
    
    Args:
        func: Function to benchmark
        *args: Positional arguments to function
        iterations: Number of iterations to run
        **kwargs: Keyword arguments to function
        
    Returns:
        BenchmarkResult with timing statistics
    """
    times: List[float] = []
    
    # Warm up
    func(*args, **kwargs)
    
    # Run benchmark
    for _ in range(iterations):
        start: float = time.perf_counter()
        func(*args, **kwargs)
        elapsed: float = time.perf_counter() - start
        times.append(elapsed)
    
    return BenchmarkResult(func.__name__, times)


# ============================================================================
# 1. PRIMALITY TESTING BENCHMARKS
# ============================================================================

def benchmark_primality() -> None:
    """Benchmark Miller-Rabin primality testing."""
    print("\n" + "="*100)
    print("PRIMALITY TESTING BENCHMARKS")
    print("="*100)
    
    test_primes: List[Tuple[int, str]] = [
        (104729, "Small prime (5 digits)"),
        (1299709, "Medium prime (7 digits)"),
        (15485863, "Large prime (8 digits)"),
        (982451653, "Very large prime (10 digits)"),
    ]
    
    for prime, description in test_primes:
        # Fresh: no cache
        clear_caches()
        result_fresh: BenchmarkResult = benchmark(is_prime, prime, iterations=10)
        result_fresh.name = f"{description:30} (no cache)"
        print(result_fresh)
        
        # Cached: multiple calls
        times: List[float] = []
        for _ in range(100):
            start: float = time.perf_counter()
            is_prime(prime)
            times.append(time.perf_counter() - start)
        
        result_cached: BenchmarkResult = BenchmarkResult(f"{description:30} (cached)", times)
        print(result_cached)
        
        # Speedup ratio
        speedup: float = result_fresh.mean / result_cached.mean
        print(f"  → Cache speedup: {speedup:.1f}x\n")


# ============================================================================
# 2. TRIAL DIVISION BENCHMARKS
# ============================================================================

def benchmark_trial_division() -> None:
    """Benchmark trial division algorithm."""
    print("\n" + "="*100)
    print("TRIAL DIVISION BENCHMARKS")
    print("="*100)
    
    test_cases: List[Tuple[int, int, str]] = [
        (360, 10000, "Small composite (360)"),
        (30030, 10000, "Product of primes (2*3*5*7*11*13)"),
        (1234567, 50000, "Medium number with small factors"),
        (123456789, 50000, "Large number"),
    ]
    
    for n, bound, description in test_cases:
        clear_caches()
        result: BenchmarkResult = benchmark(trial_division, n, bound=bound, iterations=5)
        result.name = description
        print(result)
    
    # SIMD vs Pure Python
    if is_simd_available():
        print("\n[SIMD Comparison]")
        n = 2310  # 2 * 3 * 5 * 7 * 11
        primes: np.ndarray = np.array(list(range(2, 100)), dtype=np.int64)
        
        # Warm up JIT
        _trial_division_simd(n, primes)
        
        # SIMD
        times_simd: List[float] = []
        for _ in range(50):
            start: float = time.perf_counter()
            _trial_division_simd(n, primes)
            times_simd.append(time.perf_counter() - start)
        
        result_simd: BenchmarkResult = BenchmarkResult("Trial Division SIMD (Numba JIT)", times_simd)
        print(result_simd)


# ============================================================================
# 3. BATCH INVERSE BENCHMARKS
# ============================================================================

def benchmark_batch_inverse() -> None:
    """Benchmark batch modular inversion."""
    print("\n" + "="*100)
    print("BATCH INVERSE BENCHMARKS")
    print("="*100)
    
    p: int = 1000000007  # Large prime
    
    batch_sizes: List[int] = [5, 10, 50, 100, 500, 1000]
    
    for batch_size in batch_sizes:
        values: List[int] = list(range(1, batch_size + 1))
        
        # Pure Python
        clear_caches()
        result_py: BenchmarkResult = benchmark(_batch_inverse_pure_python, values, p, iterations=5)
        result_py.name = f"Pure Python (batch size {batch_size})"
        print(result_py)
        
        # NumPy SIMD (if large batch)
        if batch_size >= 10:
            result_np: BenchmarkResult = benchmark(_batch_inverse_simd, values, p, use_numpy=True, iterations=5)
            result_np.name = f"NumPy SIMD  (batch size {batch_size})"
            print(result_np)
            
            speedup: float = result_py.mean / result_np.mean
            print(f"  → SIMD speedup: {speedup:.2f}x\n")


# ============================================================================
# 4. ECM POINT OPERATIONS BENCHMARKS
# ============================================================================

def benchmark_ecm_point_ops() -> None:
    """Benchmark ECM point operations."""
    print("\n" + "="*100)
    print("ECM POINT OPERATIONS BENCHMARKS")
    print("="*100)
    
    n: int = 123456789
    a: int = 2
    
    num_curves_list: List[int] = [1, 4, 8, 16]
    
    for num_curves in num_curves_list:
        x_arr: np.ndarray = np.random.randint(1, n, size=num_curves, dtype=np.int64)
        z_arr: np.ndarray = np.ones(num_curves, dtype=np.int64)
        
        result: BenchmarkResult = benchmark(
            _ecm_point_double_simd, x_arr, z_arr, a, n,
            iterations=10
        )
        result.name = f"Point Doubling ({num_curves} curves)"
        print(result)
    
    # Scalar multiplication
    print("\n[Scalar Multiplication]")
    for num_curves in [1, 4, 8]:
        P: np.ndarray = np.random.randint(1, n, size=(2, num_curves), dtype=np.int64)
        k: int = 12345
        
        result: BenchmarkResult = benchmark(
            _ecm_scalar_mult_simd, k, P, a, n,
            iterations=5
        )
        result.name = f"Scalar Multiplication ({num_curves} curves)"
        print(result)


# ============================================================================
# 5. POLLARD RHO BENCHMARKS
# ============================================================================

def benchmark_pollard_rho() -> None:
    """Benchmark Pollard Rho algorithm."""
    print("\n" + "="*100)
    print("POLLARD RHO (BRENT) BENCHMARKS")
    print("="*100)
    
    test_cases: List[Tuple[int, str]] = [
        (1073, "Small semiprime (29 * 37)"),
        (10403, "Medium semiprime (101 * 103)"),
        (1000003 * 1000033, "Large semiprime (~10^12)"),
    ]
    
    for n, description in test_cases:
        clear_caches()
        result: BenchmarkResult = benchmark(pollard_rho_brent, n, use_cache=False, iterations=3)
        result.name = description
        print(result)


# ============================================================================
# 6. QUADRATIC SIEVE BENCHMARKS
# ============================================================================

def benchmark_quadratic_sieve() -> None:
    """Benchmark Quadratic Sieve algorithm."""
    print("\n" + "="*100)
    print("QUADRATIC SIEVE BENCHMARKS")
    print("="*100)
    
    # Numbers in optimal range for QS (10^8 - 10^12)
    test_cases: List[Tuple[int, str]] = [
        (10**8 + 39, "10^8 range"),
        (10**9 + 39, "10^9 range"),
        (10**10 + 39, "10^10 range"),
    ]
    
    for n, description in test_cases:
        clear_caches()
        result: BenchmarkResult = benchmark(quadratic_sieve, n, iterations=1)
        result.name = f"Quadratic Sieve {description}"
        print(result)


# ============================================================================
# 7. ECM BENCHMARKS
# ============================================================================

def benchmark_ecm_factorization() -> None:
    """Benchmark Elliptic Curve Method."""
    print("\n" + "="*100)
    print("ELLIPTIC CURVE METHOD BENCHMARKS")
    print("="*100)
    
    # Test cases in ECM's optimal range (10^10 - 10^15)
    test_cases: List[Tuple[int, int, str]] = [
        (100003 * 100019, 1000, "Small semiprime, B1=1000"),
        (1000003 * 1000033, 10000, "Medium semiprime, B1=10000"),
        (10000019 * 10000079, 100000, "Large semiprime, B1=100000"),
    ]
    
    for n, B1, description in test_cases:
        clear_caches()
        result: BenchmarkResult = benchmark(ecm, n, B1=B1, iterations=1)
        result.name = description
        print(result)


# ============================================================================
# 8. COMPLETE FACTORIZATION BENCHMARKS
# ============================================================================

def benchmark_complete_factorization() -> None:
    """Benchmark complete factorization with automatic algorithm selection."""
    print("\n" + "="*100)
    print("COMPLETE FACTORIZATION BENCHMARKS (Automatic Algorithm Selection)")
    print("="*100)
    
    test_cases: List[Tuple[int, str]] = [
        (360, "Small composite"),
        (30030, "Product of first 6 primes"),
        (1234567, "Medium number"),
        (10**8 + 39, "10^8 range (uses QS)"),
        (10**10 + 39, "10^10 range (uses QS)"),
        (1000003 * 1000033, "10^12 range semiprime (uses ECM/QS)"),
    ]
    
    for n, description in test_cases:
        clear_caches()
        result: BenchmarkResult = benchmark(factor, n, iterations=3)
        result.name = description
        print(result)


# ============================================================================
# 9. CACHING IMPACT BENCHMARKS
# ============================================================================

def benchmark_caching_impact() -> None:
    """Benchmark the impact of memoization caching."""
    print("\n" + "="*100)
    print("CACHING IMPACT BENCHMARKS")
    print("="*100)
    
    numbers: List[int] = [1234567, 9876543, 10101010, 12345678, 98765432]
    
    # Without cache
    clear_caches()
    times_no_cache: List[float] = []
    for n in numbers:
        start: float = time.perf_counter()
        factor(n)
        times_no_cache.append(time.perf_counter() - start)
    
    result_no_cache: BenchmarkResult = BenchmarkResult("Factor (cold cache)", times_no_cache)
    print(result_no_cache)
    
    # With cache (repeated calls)
    times_cached: List[float] = []
    for n in numbers:
        start: float = time.perf_counter()
        factor(n)  # Already in cache
        times_cached.append(time.perf_counter() - start)
    
    result_cached: BenchmarkResult = BenchmarkResult("Factor (warm cache)", times_cached)
    print(result_cached)
    
    speedup: float = result_no_cache.mean / result_cached.mean
    print(f"Cache speedup: {speedup:.2f}x\n")
    
    # Repeated calls to same number
    n: int = 123456789
    clear_caches()
    
    times_single: List[float] = []
    for _ in range(100):
        start: float = time.perf_counter()
        factor(n)
        times_single.append(time.perf_counter() - start)
    
    result_single: BenchmarkResult = BenchmarkResult("Repeated factor() calls (cached)", times_single)
    print(result_single)
    first_call: float = times_single[0]
    avg_cached_call: float = statistics.mean(times_single[1:])
    speedup: float = first_call / avg_cached_call if avg_cached_call > 0 else float('inf')
    print(f"Speedup vs first call: {speedup:.1f}x\n")


# ============================================================================
# 10. ALGORITHM SELECTION BENCHMARKS
# ============================================================================

def benchmark_algorithm_selection() -> None:
    """Benchmark algorithm selection across different number ranges."""
    print("\n" + "="*100)
    print("ALGORITHM SELECTION BENCHMARKS (Different Ranges)")
    print("="*100)
    
    test_suites: List[Tuple[str, List[Tuple[int, str]]]] = [
        ("Trial Division Range (< 10^8)", [
            (2 * 3 * 5 * 7 * 11 * 13 * 17, "Product of small primes"),
            (10**7 + 51, "10^7 composite"),
        ]),
        ("Quadratic Sieve Range (10^8 - 10^12)", [
            (10**8 + 39, "10^8"),
            (10**10 + 39, "10^10"),
        ]),
        ("ECM Range (10^12 - 10^18)", [
            (10**12 + 39, "10^12 range"),
            (10**15 + 37, "10^15 range (very large)"),
        ]),
    ]
    
    for suite_name, test_cases in test_suites:
        print(f"\n{suite_name}")
        print("-" * 100)
        
        for n, description in test_cases:
            clear_caches()
            result: BenchmarkResult = benchmark(factor, n, iterations=1)
            result.name = description
            print(result)


# ============================================================================
# 11. STRESS TEST
# ============================================================================

def benchmark_stress_test() -> None:
    """Stress test with diverse inputs."""
    print("\n" + "="*100)
    print("STRESS TEST (100 Random Numbers)")
    print("="*100)
    
    clear_caches()
    
    # Generate random composite numbers
    test_numbers: List[int] = []
    for _ in range(50):
        # Random products of two primes
        p: int = random.randint(1000, 100000)
        while is_prime(p):
            p = random.randint(1000, 100000)
        
        q: int = random.randint(1000, 100000)
        while is_prime(q):
            q = random.randint(1000, 100000)
        
        n: int = p * q
        if n < 10**12:  # Keep in reasonable range
            test_numbers.append(n)
    
    times: List[float] = []
    successful: int = 0
    
    start_total: float = time.perf_counter()
    
    for n in test_numbers[:20]:  # Test 20 numbers
        try:
            start: float = time.perf_counter()
            factors: List[int] = factor(n)
            elapsed: float = time.perf_counter() - start
            
            # Verify
            product: int = 1
            for f in factors:
                product *= f
            
            if product == n:
                successful += 1
                times.append(elapsed)
        except Exception as e:
            print(f"Error factoring {n}: {e}")
    
    total_time: float = time.perf_counter() - start_total
    
    if times:
        result: BenchmarkResult = BenchmarkResult("Stress test factorizations", times)
        print(result)
        print(f"Successful: {successful}/{len(test_numbers[:20])}")
        print(f"Total time: {total_time:.3f}s")


# ============================================================================
# MAIN BENCHMARK SUITE
# ============================================================================

def run_all_benchmarks() -> None:
    """Run all benchmarks."""
    print("\n")
    print("╔" + "="*98 + "╗")
    print("║" + " "*20 + "FACTORIZATION LIBRARY BENCHMARK SUITE" + " "*41 + "║")
    print("║" + f" SIMD Available: {str(is_simd_available())} {' '*75} " + "║")
    print("╚" + "="*98 + "╝")
    
    try:
        benchmark_primality()
        benchmark_trial_division()
        benchmark_batch_inverse()
        benchmark_ecm_point_ops()
        benchmark_pollard_rho()
        benchmark_quadratic_sieve()
        benchmark_ecm_factorization()
        benchmark_complete_factorization()
        benchmark_caching_impact()
        benchmark_algorithm_selection()
        benchmark_stress_test()
        
        print("\n" + "="*100)
        print("BENCHMARK COMPLETE")
        print("="*100 + "\n")
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nBenchmark error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_all_benchmarks()
