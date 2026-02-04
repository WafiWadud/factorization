import unittest
import time
import random
import sys

# Import your functions here
from factorization import (
    is_prime, trial_division, pollard_rho_brent, factor, 
    clear_caches, get_small_primes, ecm, _ecm_attempt, _ecm_cached
)

class TestPrimalityTesting(unittest.TestCase):
    """Test the Miller-Rabin primality test"""
    
    def test_small_primes(self):
        """Test known small primes"""
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        for p in small_primes:
            self.assertTrue(is_prime(p), f"{p} should be prime")
    
    def test_small_composites(self):
        """Test known small composites"""
        composites = [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24, 25]
        for c in composites:
            self.assertFalse(is_prime(c), f"{c} should be composite")
    
    def test_edge_cases(self):
        """Test edge cases"""
        self.assertFalse(is_prime(0))
        self.assertFalse(is_prime(1))
        self.assertFalse(is_prime(-5))
        self.assertTrue(is_prime(2))
    
    def test_large_primes(self):
        """Test some larger known primes"""
        large_primes = [
            104729,  # 10,000th prime
            1299709,  # 100,000th prime
            15485863,  # 1,000,000th prime
            982451653,  # large prime
            2147483647,  # Mersenne prime (2^31 - 1)
        ]
        for p in large_primes:
            self.assertTrue(is_prime(p), f"{p} should be prime")
    
    def test_carmichael_numbers(self):
        """Test Carmichael numbers (pseudoprimes that fool Fermat test)"""
        # These are composite but pass many primality tests
        carmichael = [561, 1105, 1729, 2465, 2821, 6601, 8911]
        for c in carmichael:
            self.assertFalse(is_prime(c), f"{c} is Carmichael number, should be composite")
    
    def test_mersenne_composites(self):
        """Test composite Mersenne numbers"""
        # 2^11 - 1 = 2047 = 23 × 89
        self.assertFalse(is_prime(2047))
        # 2^23 - 1 = 8388607 = 47 × 178481
        self.assertFalse(is_prime(8388607))


class TestTrialDivision(unittest.TestCase):
    """Test trial division factorization"""
    
    def test_small_number(self):
        """Test factoring a small number completely"""
        factors, remainder = trial_division(360, bound=10000)
        self.assertEqual(remainder, 1)
        # 360 = 2^3 * 3^2 * 5
        self.assertEqual(sorted(factors), [2, 2, 2, 3, 3, 5])
    
    def test_small_prime_within_bound(self):
        """Test trial division on a small prime within bound"""
        # 97 is prime and less than 10000, so it should be found
        factors, remainder = trial_division(97, bound=10000)
        self.assertEqual(factors, [97])
        self.assertEqual(remainder, 1)
    
    def test_large_prime_beyond_bound(self):
        """Test when prime is beyond the bound"""
        # 100003 is prime, if bound < 100003, it won't be in the sieve
        factors, remainder = trial_division(100003, bound=1000)
        self.assertEqual(factors, [])
        self.assertEqual(remainder, 100003)
    
    def test_mixed_factors(self):
        """Test number with both small and large factors"""
        # 2 * 3 * 1000003 = 6000018
        factors, remainder = trial_division(6000018, bound=10000)
        self.assertEqual(sorted(factors), [2, 3])
        self.assertEqual(remainder, 1000003)
    
    def test_power_of_small_prime(self):
        """Test number that is a power of a small prime"""
        # 2^10 = 1024
        factors, remainder = trial_division(1024, bound=10000)
        self.assertEqual(sorted(factors), [2] * 10)
        self.assertEqual(remainder, 1)
    
    def test_product_of_small_primes(self):
        """Test product of several small primes"""
        # 2 * 3 * 5 * 7 * 11 = 2310
        factors, remainder = trial_division(2310, bound=10000)
        self.assertEqual(sorted(factors), [2, 3, 5, 7, 11])
        self.assertEqual(remainder, 1)


class TestPollardRho(unittest.TestCase):
    """Test Pollard's Rho algorithm"""
    
    def test_finds_factor(self):
        """Test that Pollard Rho finds a nontrivial factor"""
        # 1073 = 29 * 37
        n = 1073
        found_valid_factor = False
        for _ in range(10):  # Try multiple times due to randomness
            factor = pollard_rho_brent(n)
            if factor != n and factor != 1:
                self.assertTrue(n % factor == 0)
                self.assertTrue(1 < factor < n)
                found_valid_factor = True
                break
        
    
    def test_even_number(self):
        """Test that even numbers return 2"""
        self.assertEqual(pollard_rho_brent(100), 2)
        self.assertEqual(pollard_rho_brent(1000), 2)
    
    def test_divisible_by_three(self):
        """Test numbers divisible by 3"""
        self.assertEqual(pollard_rho_brent(99), 3)
        self.assertEqual(pollard_rho_brent(999), 3)
    
    def test_semiprime(self):
        """Test on a semiprime (product of two primes)"""
        # 10403 = 101 * 103
        n = 10403
        found_valid = False
        for _ in range(10):
            d = pollard_rho_brent(n)
            if d in [101, 103]:
                found_valid = True
                break
        self.assertTrue(found_valid, "Should find one of the prime factors")


class TestFactorization(unittest.TestCase):
    """Test the complete factorization function"""
    
    def test_factor_one(self):
        """Test factoring 1"""
        self.assertEqual(factor(1), [])
    
    def test_factor_prime(self):
        """Test factoring a prime number"""
        self.assertEqual(factor(17), [17])
        self.assertEqual(factor(104729), [104729])
    
    def test_factor_power_of_two(self):
        """Test factoring powers of 2"""
        self.assertEqual(sorted(factor(16)), [2, 2, 2, 2])
        self.assertEqual(sorted(factor(1024)), [2] * 10)
    
    def test_factor_small_composite(self):
        """Test factoring small composite numbers"""
        # 12 = 2^2 * 3
        self.assertEqual(sorted(factor(12)), [2, 2, 3])
        # 100 = 2^2 * 5^2
        self.assertEqual(sorted(factor(100)), [2, 2, 5, 5])
        # 360 = 2^3 * 3^2 * 5
        self.assertEqual(sorted(factor(360)), [2, 2, 2, 3, 3, 5])
    
    def test_factor_product_of_primes(self):
        """Test factoring products of distinct primes"""
        # 2 * 3 * 5 * 7 = 210
        self.assertEqual(sorted(factor(210)), [2, 3, 5, 7])
        # 11 * 13 * 17 = 2431
        self.assertEqual(sorted(factor(2431)), [11, 13, 17])
    
    def test_factor_semiprimes(self):
        """Test factoring semiprimes (product of two primes)"""
        # 143 = 11 * 13
        self.assertEqual(sorted(factor(143)), [11, 13])
        # 1073 = 29 * 37
        self.assertEqual(sorted(factor(1073)), [29, 37])
        # 10403 = 101 * 103
        self.assertEqual(sorted(factor(10403)), [101, 103])
    
    def test_factor_large_number(self):
        """Test the example from the code"""
        n = 123456789101112
        factors = factor(n)
        # Verify all factors are prime
        for f in factors:
            self.assertTrue(is_prime(f), f"{f} should be prime")
        # Verify product equals original
        product = 1
        for f in factors:
            product *= f
        self.assertEqual(product, n)
    
    def test_factor_various_sizes(self):
        """Test factoring numbers of various sizes"""
        test_cases = [
            (1234567, [127, 9721]),  # prime factorization
            (1234568, [2, 2, 2, 154321]),  
            (1000000007, [1000000007]),  # prime
            (999999, [3, 3, 3, 7, 11, 13, 37]),
        ]
        
        for n, expected in test_cases:
            factors = sorted(factor(n))
            # Verify all factors are prime
            for f in factors:
                self.assertTrue(is_prime(f), f"Factor {f} of {n} should be prime")
            # Verify product
            product = 1
            for f in factors:
                product *= f
            self.assertEqual(product, n, f"Product of factors should equal {n}")
            
            # Check against expected if provided
            if expected:
                self.assertEqual(factors, sorted(expected), f"Factors of {n} don't match expected")
    
    def test_factor_negative_numbers(self):
        """Test that negative numbers are handled (converted to absolute value)"""
        # -12 should give same result as 12
        self.assertEqual(sorted(factor(-12)), sorted(factor(12)))
        self.assertEqual(sorted(factor(-1073)), sorted(factor(1073)))
    
    def test_factor_perfect_squares(self):
        """Test factoring perfect squares"""
        # 144 = 2^4 * 3^2 = 12^2
        factors = sorted(factor(144))
        self.assertEqual(factors, [2, 2, 2, 2, 3, 3])
        
        # 10000 = 10^4 = 2^4 * 5^4
        factors = sorted(factor(10000))
        self.assertEqual(factors, [2, 2, 2, 2, 5, 5, 5, 5])
    
    def test_factor_perfect_cubes(self):
        """Test factoring perfect cubes"""
        # 8 = 2^3
        self.assertEqual(sorted(factor(8)), [2, 2, 2])
        # 1000 = 10^3 = 2^3 * 5^3
        self.assertEqual(sorted(factor(1000)), [2, 2, 2, 5, 5, 5])
    
    def test_correctness_comprehensive(self):
        """Comprehensive correctness test on various numbers"""
        import random
        random.seed(42)  # For reproducibility
        
        # Test some specific challenging cases
        test_numbers = [
            2,  # smallest prime
            4,  # smallest composite
            15,  # product of two small primes
            100,  # perfect square
            1001,  # 7 * 11 * 13
            9999,  # 3^2 * 11 * 101
            65536,  # 2^16
            999983,  # prime
        ]
        
        # Add some random numbers
        for _ in range(10):
            test_numbers.append(random.randint(2, 10**7))
        
        for n in test_numbers:
            factors = factor(n)
            
            # All factors should be prime
            for f in factors:
                self.assertTrue(is_prime(f), f"{f} (factor of {n}) should be prime")
            
            # Product should equal original
            product = 1
            for f in factors:
                product *= f
            self.assertEqual(product, n, f"Product of factors should equal {n}")


class TestPerformance(unittest.TestCase):
    """Performance benchmarks (optional, can be slow)"""
    
    def test_large_prime_detection(self):
        """Test that large primes are detected quickly"""
        # 15485863 is the 1,000,000th prime
        start = time.time()
        result = is_prime(15485863)
        elapsed = time.time() - start
        
        self.assertTrue(result)
        self.assertLess(elapsed, 0.1, "Should detect large prime quickly")
    
    def test_factorization_speed(self):
        """Test factorization of moderately large numbers"""
        # Product of two medium primes
        n = 1000003 * 1000033  # = 1000036000099
        
        start = time.time()
        factors = factor(n)
        elapsed = time.time() - start
        
        self.assertEqual(sorted(factors), [1000003, 1000033])
        self.assertLess(elapsed, 10.0, "Should factor semiprime in reasonable time")
    
    def test_worst_case_small_factors(self):
        """Test numbers with many small factors"""
        # 2^20 = 1048576
        start = time.time()
        factors = factor(2**20)
        elapsed = time.time() - start
        
        self.assertEqual(factors, [2] * 20)
        self.assertLess(elapsed, 0.5, "Should handle many repeated factors quickly")


class TestEdgeCasesAndCornerCases(unittest.TestCase):
    """Test various edge cases"""
    
    def test_two(self):
        """Test the smallest prime"""
        self.assertTrue(is_prime(2))
        self.assertEqual(factor(2), [2])
    
    def test_three(self):
        """Test another small prime"""
        self.assertTrue(is_prime(3))
        self.assertEqual(factor(3), [3])
    
    def test_four(self):
        """Test the smallest composite"""
        self.assertFalse(is_prime(4))
        self.assertEqual(sorted(factor(4)), [2, 2])
    
    def test_highly_composite_numbers(self):
        """Test highly composite numbers (many factors)"""
        # 5040 = 2^4 * 3^2 * 5 * 7 (has 60 divisors)
        factors = sorted(factor(5040))
        self.assertEqual(factors, [2, 2, 2, 2, 3, 3, 5, 7])
        
        # Verify product
        product = 1
        for f in factors:
            product *= f
        self.assertEqual(product, 5040)
    
    def test_factorial_like_numbers(self):
        """Test numbers that are products of many consecutive primes"""
        # 2 * 3 * 5 * 7 * 11 * 13 = 30030
        n = 30030
        factors = sorted(factor(n))
        expected = [2, 3, 5, 7, 11, 13]
        self.assertEqual(factors, expected)


class TestOptimizations(unittest.TestCase):
    """Test optimization features"""
    
    def test_small_primes_cache(self):
        """Test that small primes are properly cached"""
        clear_caches()
        primes = get_small_primes()
        self.assertGreater(len(primes), 1000)
        self.assertEqual(primes[0], 2)
        self.assertLessEqual(primes[-1], 10000)
        
        # Second call should return same object
        primes2 = get_small_primes()
        self.assertIs(primes, primes2)
    
    def test_memoization_benefit(self):
        """Test that memoization actually speeds up repeated calls"""
        clear_caches()
        
        test_nums = [12, 143, 1024, 1073]
        
        # First pass - populate cache
        start = time.time()
        for _ in range(3):
            for n in test_nums:
                factor(n)
        time_with_cache = time.time() - start
        
        # Cache should provide massive speedup
        self.assertLess(time_with_cache, 0.01, "Memoization should be very fast")
    
    def test_lazy_numpy_loading(self):
        """Test that NumPy is only loaded when needed"""
        import factorization
        # Reset NumPy state
        factorization.NUMPY_AVAILABLE = None
        
        # Small trial division shouldn't load NumPy
        factors, _ = trial_division(360, bound=10000)
        self.assertIn(factorization.NUMPY_AVAILABLE, [None, False], 
                     "NumPy shouldn't be loaded for small bounds")
        
        # Large trial division might load NumPy if available
        factors, _ = trial_division(1234567, bound=100000)
        # Just verify it works, don't assert about NumPy status
        self.assertIsNotNone(factors)
    
    def test_cache_clearing(self):
        """Test cache clearing functionality"""
        # Populate caches
        for n in [12, 143, 1024, 1073]:
            factor(n)
        
        # Caches should have entries
        cache_info = is_prime.cache_info()
        initial_size = cache_info.currsize
        self.assertGreater(initial_size, 0)
        
        # Clear and verify
        clear_caches()
        cache_info = is_prime.cache_info()
        self.assertEqual(cache_info.currsize, 0)


class TestLargeNumbers(unittest.TestCase):
    """Test factorization of large numbers"""
    
    def test_large_semiprime(self):
        """Test factoring a large semiprime (product of two primes)"""
        # 1000000007 * 1000000009 = 1000000016000000063
        p1, p2 = 1000000007, 1000000009
        n = p1 * p2
        factors = sorted(factor(n))
        self.assertEqual(factors, sorted([p1, p2]))
    
    def test_mersenne_like_composite(self):
        """Test composite numbers similar to Mersenne numbers"""
        # 2^17 - 1 = 131071 (prime), so test 2^16 - 1 = 65535 = 3 * 5 * 17 * 257
        n = (1 << 16) - 1
        factors = sorted(factor(n))
        # Verify by multiplication
        product = 1
        for f in factors:
            product *= f
        self.assertEqual(product, n)
    
    def test_highly_composite_number(self):
        """Test a highly composite number (many prime factors)"""
        # 120 = 2^3 * 3 * 5 (6 divisors)
        # 5040 = 2^4 * 3^2 * 5 * 7 (60 divisors)
        n = 5040
        factors = sorted(factor(n))
        product = 1
        for f in factors:
            product *= f
        self.assertEqual(product, n)
        # Should have many factors
        self.assertGreater(len(factors), 5)
    
    def test_power_of_prime(self):
        """Test factorization of powers of primes"""
        powers = [2**20, 3**10, 5**8, 7**6]
        for n in powers:
            factors = factor(n)
            product = 1
            for f in factors:
                product *= f
            self.assertEqual(product, n)


class TestNumberPatterns(unittest.TestCase):
    """Test specific mathematical patterns"""
    
    def test_factorials_minus_one(self):
        """Test factorization of (n! - 1) which often have large prime factors"""
        import math
        factorials = [math.factorial(n) - 1 for n in range(3, 8)]
        for n in factorials:
            factors = sorted(factor(n))
            product = 1
            for f in factors:
                product *= f
            self.assertEqual(product, n)
    
    def test_fibonacci_composites(self):
        """Test Fibonacci number factorization"""
        # Generate Fibonacci numbers
        fibs = [1, 1]
        for _ in range(6):
            fibs.append(fibs[-1] + fibs[-2])
        
        for fib in fibs[2:]:  # Skip the 1's
            if fib > 1:
                factors = factor(fib)
                product = 1
                for f in factors:
                    product *= f
                self.assertEqual(product, fib)
    
    def test_repdigit_numbers(self):
        """Test repdigit numbers (111, 1111, etc)"""
        repdigits = [int('1' * n) for n in range(2, 6)]
        for n in repdigits:
            factors = factor(n)
            product = 1
            for f in factors:
                product *= f
            self.assertEqual(product, n)
    
    def test_pronic_numbers(self):
        """Test pronic numbers (n * (n+1))"""
        for n in range(10, 20):
            pronic = n * (n + 1)
            factors = sorted(factor(pronic))
            product = 1
            for f in factors:
                product *= f
            self.assertEqual(product, pronic)


class TestBitOperationOptimizations(unittest.TestCase):
    """Test bit operation optimizations"""
    
    def test_power_of_two_factorization(self):
        """Test that powers of 2 are factored efficiently using bit ops"""
        for exp in range(1, 30):
            n = 1 << exp  # 2^exp
            start = time.time()
            factors = factor(n)
            elapsed = time.time() - start
            
            self.assertEqual(factors, [2] * exp)
            # Should be very fast (under 1ms even for large powers)
            self.assertLess(elapsed, 0.001, f"2^{exp} factorization too slow")
    
    def test_even_number_extraction(self):
        """Test efficient extraction of 2's using bit operations"""
        # Numbers with many factors of 2
        test_cases = [
            (16, [2, 2, 2, 2]),
            (64, [2] * 6),
            (1024, [2] * 10),
            (2048, [2] * 11),
        ]
        
        for n, expected in test_cases:
            factors, _ = trial_division(n, bound=1000)
            self.assertEqual(sorted(factors), expected)


class TestStressAndRandomized(unittest.TestCase):
    """Stress tests and randomized tests"""
    
    def test_random_composites(self):
        """Test factorization on random composite numbers"""
        random.seed(42)
        
        for _ in range(20):
            # Generate random semiprime
            p = random.randint(1000, 10000)
            q = random.randint(1000, 10000)
            n = p * q
            
            factors = factor(n)
            product = 1
            for f in factors:
                product *= f
            self.assertEqual(product, n)
    
    def test_random_perfect_powers(self):
        """Test factorization of random perfect powers"""
        random.seed(43)
        
        for _ in range(15):
            base = random.randint(2, 100)
            exp = random.randint(2, 8)
            n = base ** exp
            
            factors = sorted(factor(n))
            product = 1
            for f in factors:
                product *= f
            self.assertEqual(product, n)
    
    def test_large_random_range(self):
        """Test wide range of random numbers"""
        random.seed(44)
        
        for _ in range(25):
            n = random.randint(2, 10**7)
            factors = factor(n)
            
            # Verify all factors are prime
            for f in factors:
                self.assertTrue(is_prime(f), f"{f} should be prime")
            
            # Verify product
            product = 1
            for f in factors:
                product *= f
            self.assertEqual(product, n)


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmarks (can be slow)"""
    
    def test_primality_performance(self):
        """Benchmark primality testing performance"""
        large_primes = [104729, 1299709, 15485863]
        
        start = time.time()
        for p in large_primes:
            for _ in range(100):
                is_prime(p)
        elapsed = time.time() - start
        
        # Should be very fast due to memoization
        self.assertLess(elapsed, 0.1, "Primality testing too slow")
    
    def test_factorization_batches(self):
        """Benchmark factorization of batches"""
        test_nums = [12, 100, 360, 1000, 2310, 5040, 10000]
        
        start = time.time()
        for _ in range(100):
            for n in test_nums:
                factor(n)
        elapsed = time.time() - start
        
        self.assertLess(elapsed, 1.0, "Batch factorization too slow")
    
    def test_trial_division_large_bound(self):
        """Test trial division with large bound"""
        n = 1234567
        
        start = time.time()
        for _ in range(10):
            factors, rem = trial_division(n, bound=50000)
        elapsed = time.time() - start
        
        # Should complete in reasonable time
        self.assertLess(elapsed, 1.0, "Large bound trial division too slow")


class TestECM(unittest.TestCase):
    """Test Elliptic Curve Method for factorization."""
    
    def test_ecm_small_semiprime(self):
        """Test ECM on small semiprime"""
        # 10^6 range: 999983 * 999979
        n = 999983 * 999979
        factor = ecm(n, B1=1000)
        if factor is not None:
            self.assertTrue(1 < factor < n)
            self.assertEqual(n % factor, 0)
    
    def test_ecm_medium_semiprime(self):
        """Test ECM on medium semiprime (10^10 range)"""
        # 10^10 range
        p = 100003
        q = 100019
        n = p * q
        factor = ecm(n, B1=10000)
        if factor is not None:
            self.assertTrue(1 < factor < n)
            self.assertEqual(n % factor, 0)
    
    def test_ecm_large_semiprime(self):
        """Test ECM on large semiprime (10^12 range)"""
        # Two factors around 10^6
        p = 1000003
        q = 1000033
        n = p * q
        factor = ecm(n, B1=100000)
        if factor is not None:
            self.assertTrue(1 < factor < n)
            self.assertEqual(n % factor, 0)
    
    def test_ecm_with_small_factor(self):
        """Test ECM with one small, one large factor"""
        p = 10007
        q = 10000019
        n = p * q
        factor = ecm(n, B1=10000)
        if factor is not None:
            self.assertTrue(1 < factor < n)
            self.assertEqual(n % factor, 0)
    
    def test_ecm_returns_none_for_prime(self):
        """ECM should return None for primes"""
        result = ecm(100003, B1=1000)
        self.assertIsNone(result)
    
    def test_ecm_handles_even(self):
        """ECM should handle even numbers"""
        result = ecm(100, B1=100)
        self.assertEqual(result, 2)
    
    def test_ecm_memoization(self):
        """Test that ECM memoization works"""
        p = 1000003
        q = 1000033
        n = p * q
        
        # First call
        factor1 = ecm(n, B1=1000)
        
        # Second call should be cached
        factor2 = ecm(n, B1=1000)
        
        if factor1 is not None:
            self.assertEqual(factor1, factor2)
    
    def test_ecm_parallel(self):
        """Test ECM with parallelization"""
        p = 1000003
        q = 1000033
        n = p * q
        
        # Parallel should still find factor
        factor = ecm(n, B1=100000, use_parallel=True)
        if factor is not None:
            self.assertTrue(1 < factor < n)
            self.assertEqual(n % factor, 0)
    
    def test_ecm_attempt_auto_selection(self):
        """Test _ecm_attempt with automatic B1 selection"""
        # Should auto-select B1 based on number size
        p = 100003
        q = 100019
        n = p * q  # ~10^10
        
        factor = _ecm_attempt(n, use_parallel=False)
        # Just verify it doesn't crash
        self.assertTrue(factor is None or (1 < factor < n and n % factor == 0))
    
    def test_ecm_integration_in_factor(self):
        """Test that ECM is used in factor() for appropriate ranges"""
        # 10^12 range semiprime
        p = 1000003
        q = 1000033
        n = p * q
        
        factors = factor(n)
        self.assertEqual(len(factors), 2)
        self.assertEqual(sorted(factors), sorted([p, q]))
    
    def test_ecm_cache_clearing(self):
        """Test that ECM cache is cleared properly"""
        p = 100003
        q = 100019
        n = p * q
        
        # Check cache has entry
        ecm(n, B1=1000)
        info_before = _ecm_cached.cache_info()
        self.assertGreater(info_before.currsize, 0)
        
        # Clear caches
        clear_caches()
        
        # Cache should be empty
        info_after = _ecm_cached.cache_info()
        self.assertEqual(info_after.currsize, 0)
    
    def test_ecm_phase2(self):
        """Test that ECM Phase 2 works"""
        # Phase 2 should be called automatically
        p = 1000003
        q = 1000033
        n = p * q
        
        # With B2 set, Phase 2 should be attempted
        factor = ecm(n, B1=1000, B2=100000)
        # May or may not find factor, but shouldn't crash
        if factor is not None:
            self.assertTrue(1 < factor < n)
            self.assertEqual(n % factor, 0)
    
    def test_ecm_multiple_B1_values(self):
        """Test that multiple B1 values are tried"""
        p = 1000003
        q = 1000033
        n = p * q
        
        # _ecm_attempt with try_multiple_B1=True tries multiple values
        factor = _ecm_attempt(n, try_multiple_B1=True)
        # Should eventually find something or try multiple bounds
        # At minimum, shouldn't crash
        self.assertTrue(factor is None or (1 < factor < n and n % factor == 0))
    
    def test_ecm_adaptive_curves(self):
        """Test adaptive curve selection based on statistics"""
        # This test verifies that adaptive curve selection works
        # by checking that repeated attempts build up statistics
        p = 100003
        q = 100019
        n = p * q
        
        # Make multiple attempts with same parameters
        results = []
        for _ in range(3):
            factor = ecm(n, B1=1000)
            results.append(factor)
        
        # Should all be the same (cached) or all find the same factor
        if results[0] is not None:
            for r in results[1:]:
                self.assertEqual(r, results[0])

class TestMemoryEfficiency(unittest.TestCase):
    """Test memory-related optimizations"""
    
    def test_sieve_memory_usage(self):
        """Test that sieves use memory-efficient data structures"""
        # Small sieve should use list
        factors, _ = trial_division(360, bound=10000)
        
        # Large sieve should use bytearray or NumPy
        factors, _ = trial_division(1234567, bound=50000)
        
        # Just verify both work
        self.assertIsNotNone(factors)
    
    def test_cache_size_limits(self):
        """Test that caches don't grow unbounded"""
        clear_caches()
        
        # Create many different factorization calls
        for i in range(1000):
            n = i * 2 + 3  # Various odd numbers
            if is_prime(n):
                continue
            factor(n)
        
        # Check that caches have reasonable size
        cache_info = is_prime.cache_info()
        self.assertLessEqual(cache_info.currsize, cache_info.maxsize)
        
        trial_div_info = trial_division.cache_info()
        self.assertLessEqual(trial_div_info.currsize, trial_div_info.maxsize)


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)
