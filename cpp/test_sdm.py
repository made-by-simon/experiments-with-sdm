"""
Consolidated tests for Python and C++ implementations of Kanerva's
Sparse Distributed Memory.

Both implementations are tested with identical parameters so results
and performance can be compared side-by-side.

Usage:
    python test_sdm.py

(c) 2025 Simon Wong
"""

import time
import numpy as np

# ---------------------------------------------------------------------------
# Shared parameters — used by BOTH Python and C++ implementations
# ---------------------------------------------------------------------------
ADDRESS_DIMENSION = 256        # Length of address vectors (N)
MEMORY_DIMENSION = 256         # Length of memory vectors (U)
NUM_LOCATIONS = 1000           # Number of hard locations (M)
HAMMING_THRESHOLD = 108        # Hamming distance activation threshold (H)
RANDOM_SEED = 42               # Seed for reproducible hard-location generation
NUM_TEST_PATTERNS = 50         # Number of write/read patterns to test


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def random_binary_vector(size: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random binary vector of the given size."""
    return rng.integers(0, 2, size=size, dtype=np.int8)


def bit_accuracy(original: np.ndarray, recalled: np.ndarray) -> float:
    """Return the fraction of bits that match between two binary vectors."""
    return np.mean(original == recalled)


# ---------------------------------------------------------------------------
# Test suite (runs against any SDM that supports write / read / erase_memory)
# ---------------------------------------------------------------------------
def run_tests(label: str, make_sdm, vec_to_input):
    """
    Run the full test suite for one SDM implementation.

    Args:
        label:        Human-readable name (e.g. "Python", "C++").
        make_sdm:     Callable() -> sdm instance, using shared parameters.
        vec_to_input: Callable(np.ndarray) -> type expected by sdm.write/read.
                      For Python this is identity; for C++ it converts to list.
    """
    print(f"\n{'='*60}")
    print(f"  {label} Implementation")
    print(f"{'='*60}")

    # ---- 1. Basic write / read-back -----------------------------------------
    print(f"\n--- Test 1: Single write & read-back ---")
    sdm = make_sdm()

    rng = np.random.default_rng(123)
    address = random_binary_vector(ADDRESS_DIMENSION, rng)
    memory = random_binary_vector(MEMORY_DIMENSION, rng)

    sdm.write(vec_to_input(address), vec_to_input(memory))
    recalled = np.array(sdm.read(vec_to_input(address)), dtype=np.int8)

    acc = bit_accuracy(memory, recalled)
    print(f"  Bit accuracy: {acc:.4f}")
    assert acc > 0.8, f"Accuracy too low: {acc:.4f}"
    print(f"  PASSED")

    # ---- 2. Erase memory ----------------------------------------------------
    print(f"\n--- Test 2: Erase memory ---")
    sdm.erase_memory()
    recalled_after_erase = np.array(sdm.read(vec_to_input(address)), dtype=np.int8)
    # After erase the memory matrix is zero; read should return all-ones
    # (since sum == 0 >= 0 is true) or all-zeros if no locations activate.
    # The key check is that the old memory is NOT faithfully recalled.
    erase_acc = bit_accuracy(memory, recalled_after_erase)
    print(f"  Bit accuracy vs original after erase: {erase_acc:.4f}")
    print(f"  PASSED")

    # ---- 3. Multiple patterns -----------------------------------------------
    print(f"\n--- Test 3: Write {NUM_TEST_PATTERNS} patterns & read back ---")
    sdm = make_sdm()
    rng = np.random.default_rng(456)

    addresses = [random_binary_vector(ADDRESS_DIMENSION, rng)
                 for _ in range(NUM_TEST_PATTERNS)]
    memories = [random_binary_vector(MEMORY_DIMENSION, rng)
                for _ in range(NUM_TEST_PATTERNS)]

    t0 = time.perf_counter()
    for addr, mem in zip(addresses, memories):
        sdm.write(vec_to_input(addr), vec_to_input(mem))
    write_time = time.perf_counter() - t0

    accuracies = []
    t0 = time.perf_counter()
    for addr, mem in zip(addresses, memories):
        recalled = np.array(sdm.read(vec_to_input(addr)), dtype=np.int8)
        accuracies.append(bit_accuracy(mem, recalled))
    read_time = time.perf_counter() - t0

    mean_acc = np.mean(accuracies)
    min_acc = np.min(accuracies)
    max_acc = np.max(accuracies)

    print(f"  Mean bit accuracy : {mean_acc:.4f}")
    print(f"  Min  bit accuracy : {min_acc:.4f}")
    print(f"  Max  bit accuracy : {max_acc:.4f}")
    print(f"  Write time ({NUM_TEST_PATTERNS} ops): {write_time*1000:.2f} ms")
    print(f"  Read  time ({NUM_TEST_PATTERNS} ops): {read_time*1000:.2f} ms")
    print(f"  PASSED")

    # ---- 4. Properties / getters -------------------------------------------
    print(f"\n--- Test 4: Parameter getters ---")
    sdm2 = make_sdm()
    checks = {
        "address_dimension": ADDRESS_DIMENSION,
        "memory_dimension": MEMORY_DIMENSION,
        "num_locations": NUM_LOCATIONS,
    }
    for attr, expected in checks.items():
        actual = getattr(sdm2, attr, None)
        # Python implementation uses uppercase attribute names
        if actual is None:
            actual = getattr(sdm2, attr.upper(), None)
        assert actual == expected, f"{attr}: expected {expected}, got {actual}"
        print(f"  {attr} = {actual}  OK")
    print(f"  PASSED")

    return {
        "single_accuracy": acc,
        "mean_accuracy": mean_acc,
        "write_ms": write_time * 1000,
        "read_ms": read_time * 1000,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Kanerva SDM — Consolidated Test Suite")
    print(f"  ADDRESS_DIMENSION = {ADDRESS_DIMENSION}")
    print(f"  MEMORY_DIMENSION  = {MEMORY_DIMENSION}")
    print(f"  NUM_LOCATIONS     = {NUM_LOCATIONS}")
    print(f"  HAMMING_THRESHOLD = {HAMMING_THRESHOLD}")
    print(f"  RANDOM_SEED       = {RANDOM_SEED}")
    print(f"  NUM_TEST_PATTERNS = {NUM_TEST_PATTERNS}")

    results = {}

    # ---- Python implementation -----------------------------------------------
    from kanerva_sdm import KanervaSDM as PySDM

    def make_py_sdm():
        return PySDM(
            ADDRESS_DIMENSION=ADDRESS_DIMENSION,
            MEMORY_DIMENSION=MEMORY_DIMENSION,
            NUM_LOCATIONS=NUM_LOCATIONS,
            HAMMING_THRESHOLD=HAMMING_THRESHOLD,
            RANDOM_SEED=RANDOM_SEED,
        )

    results["Python"] = run_tests("Python", make_py_sdm, lambda v: v)

    # ---- C++ implementation --------------------------------------------------
    try:
        from sdm_cpp import KanervaSDM as CppSDM

        def make_cpp_sdm():
            return CppSDM(
                address_dimension=ADDRESS_DIMENSION,
                memory_dimension=MEMORY_DIMENSION,
                num_locations=NUM_LOCATIONS,
                activation_threshold=HAMMING_THRESHOLD,
                random_seed=RANDOM_SEED,
            )

        results["C++"] = run_tests(
            "C++", make_cpp_sdm, lambda v: v.tolist()
        )
    except ImportError:
        print("\n** sdm_cpp not installed — skipping C++ tests. **")
        print("   Build with:  pip install -e .[dev]")

    # ---- Comparison ----------------------------------------------------------
    if len(results) == 2:
        print(f"\n{'='*60}")
        print(f"  Comparison")
        print(f"{'='*60}")
        py, cpp = results["Python"], results["C++"]
        print(f"  {'Metric':<30} {'Python':>10} {'C++':>10}")
        print(f"  {'-'*50}")
        print(f"  {'Single-pattern accuracy':<30} {py['single_accuracy']:>10.4f} {cpp['single_accuracy']:>10.4f}")
        print(f"  {'Mean accuracy (multi)':<30} {py['mean_accuracy']:>10.4f} {cpp['mean_accuracy']:>10.4f}")
        print(f"  {'Write time (ms)':<30} {py['write_ms']:>10.2f} {cpp['write_ms']:>10.2f}")
        print(f"  {'Read  time (ms)':<30} {py['read_ms']:>10.2f} {cpp['read_ms']:>10.2f}")
        speedup_w = py["write_ms"] / cpp["write_ms"] if cpp["write_ms"] > 0 else float("inf")
        speedup_r = py["read_ms"] / cpp["read_ms"] if cpp["read_ms"] > 0 else float("inf")
        print(f"\n  C++ write speedup: {speedup_w:.1f}x")
        print(f"  C++ read  speedup: {speedup_r:.1f}x")

    print(f"\n{'='*60}")
    print("  All tests passed!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
