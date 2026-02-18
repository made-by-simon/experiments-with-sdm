import itertools
import time
import csv
import numpy as np
from tqdm import tqdm
import kanerva_sdm as sdm_py
import sdm_cpp

DIMENSIONS = [10, 100, 1000]
NUM_LOCATIONS_LIST = [100, 10_000, 1_000_000]
MEMORY_COUNT = 100
OUTPUT_FILE = "grid_search_results.csv"

rng = np.random.default_rng(42)


def run_timing(sdm, addresses, memories):
    """Write then read all memories; return elapsed seconds."""
    start = time.perf_counter()
    for i in tqdm(range(MEMORY_COUNT), leave=False):
        sdm.write(addresses[i], memories[i])
    for i in tqdm(range(MEMORY_COUNT), leave=False):
        sdm.read(addresses[i])
    return time.perf_counter() - start


results = []

pbar = tqdm(total=18, desc="Grid search", unit="test")

for dimension, num_locations in itertools.product(DIMENSIONS, NUM_LOCATIONS_LIST):
    threshold = max(1, int(dimension * 0.37))

    addresses = rng.integers(0, 2, (MEMORY_COUNT, dimension), dtype=np.int8)
    memories  = rng.integers(0, 2, (MEMORY_COUNT, dimension), dtype=np.int8)

    pbar.set_postfix(dim=dimension, locs=f"{num_locations:,}", impl="python")
    py_sdm  = sdm_py.KanervaSDM(
        ADDRESS_DIMENSION=dimension,
        MEMORY_DIMENSION=dimension,
        NUM_LOCATIONS=num_locations,
        HAMMING_THRESHOLD=threshold,
    )
    py_time = run_timing(py_sdm, addresses, memories)
    pbar.update(1)

    pbar.set_postfix(dim=dimension, locs=f"{num_locations:,}", impl="cpp")
    cpp_sdm  = sdm_cpp.KanervaSDM(
        address_dimension=dimension,
        memory_dimension=dimension,
        num_locations=num_locations,
        activation_threshold=threshold,
    )
    cpp_time = run_timing(cpp_sdm, addresses, memories)
    pbar.update(1)

    tqdm.write(f"dim={dimension:>5}, locs={num_locations:>10,}  |  py={py_time:.3f}s  cpp={cpp_time:.3f}s")

    results.append({
        "dimension":    dimension,
        "num_locations": num_locations,
        "memory_count": MEMORY_COUNT,
        "python_time_s": py_time,
        "cpp_time_s":   cpp_time,
    })

pbar.close()

fieldnames = ["dimension", "num_locations", "memory_count", "python_time_s", "cpp_time_s"]
with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"\nResults saved to {OUTPUT_FILE}")
