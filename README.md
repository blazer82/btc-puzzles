# Bitcoin Puzzle Solver using Pollard's Kangaroo Algorithm

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run a simple test with CPU implementation
python src/main.py --puzzle 5 --profile verify_cpu

# Run with GPU implementation (Apple Silicon only)
python src/main.py --puzzle 5 --profile verify_gpu
```

## 1. Objective

The primary goal of this project is to find a specific Bitcoin private key that corresponds to a given public key. The search is confined to a predefined range of possible keys, as specified by the "Bitcoin Puzzle" challenges. Each puzzle number `n` defines a search space, typically from `2^(n-1)` to `2^n - 1`.

## 2. Methodology

Instead of a computationally expensive brute-force search, this project employs **Pollard's Kangaroo Algorithm** (also known as Pollard's Lambda Algorithm). This is a specialized algorithm for solving the discrete logarithm problem, which is the mathematical foundation of Bitcoin's public/private key cryptography.

### High-Level Concept

The algorithm's runtime is proportional to the square root of the size of the search range, making it vastly more efficient than checking every key one by one. It operates on the following principles:

1.  **Two Herds of "Kangaroos"**: The algorithm simulates two independent sets of "walkers" or "kangaroos" that hop across points on the secp256k1 elliptic curve.
    *   **Tame Kangaroos**: This herd begins at a known starting point within the search range (e.g., the middle of the puzzle's key space). We know the "private key" for their starting point.
    *   **Wild Kangaroos**: This herd begins at the target public key for which we want to find the private key.

2.  **Deterministic Hops**: Each kangaroo takes "hops" from one point on the curve to another. The direction and distance of each hop are determined by the kangaroo's current position, making their paths deterministic but seemingly random. The distance of each hop is a known power-of-2 multiple of the generator point `G`.

3.  **Tracking Distance**: As each kangaroo hops, it accumulates a total distance traveled.
    *   For a tame kangaroo, this distance represents an offset from its known starting private key.
    *   For a wild kangaroo, this distance represents an offset from the *unknown* target private key.

4.  **Collision and Solution**: The algorithm runs until a tame kangaroo and a wild kangaroo land on the exact same point on the elliptic curve. This event is called a **collision**. Because the paths are deterministic, this collision is bound to happen.

    When a collision occurs, we have:
    `start_tame + distance_tame = start_wild + distance_wild`

    Since we know `start_tame`, `distance_tame`, and `distance_wild`, we can solve for the unknown `start_wild` (the target private key).

### Efficiency Strategy: Distinguished Points

To make the collision detection process efficient, the algorithm does not check for collisions after every single hop. Instead, it uses a technique called **distinguished points**.

*   A point is "distinguished" if it meets a predefined, rare criterion (e.g., its x-coordinate has a certain number of trailing zero bits).
*   Kangaroos only report their location and total distance traveled when they land on one of these distinguished points.
*   These reported locations are stored in a "trap". When a new distinguished point is reported, the trap is checked for a matching point from the opposite herd. This is how collisions are detected without the overhead of constant checking.

## 3. Expected Performance

The runtime of Pollard's Kangaroo Algorithm is proportional to the square root of the search space size (`O(sqrt(N))`). This provides a significant speedup over brute-force, which is `O(N)`.

- **Scaling**: Each subsequent puzzle number doubles the size of the search space. This means that solving puzzle `n+1` is expected to take approximately `sqrt(2) ≈ 1.4` times longer than solving puzzle `n`.

- **Low-Difficulty Puzzles (e.g., #5 - #30)**: These have very small search spaces. The solver should find the key almost instantly, often in the first few iterations. This serves as a good test to ensure the algorithm is working correctly.

- **Mid-Difficulty Puzzles (e.g., #40 - #60)**: These puzzles are solvable within a reasonable timeframe on modern consumer hardware (from minutes to hours). Performance will depend on the GPU's capabilities and the number of walkers used.

- **High-Difficulty Puzzles (e.g., #65+)**: These represent a significant computational challenge. Solving them requires a highly optimized implementation and can take days, weeks, or longer, depending on the hardware available.

## 4. Implementation Options

This project provides two implementation options to accommodate different hardware capabilities:

### CPU Implementation

The CPU implementation (`KangarooRunnerCPU`) is suitable for:
- Systems without a compatible GPU
- Solving smaller puzzles (up to around #30-40)
- Testing and development

### GPU Implementation for Apple Silicon

The GPU implementation (`KangarooRunnerGPU`) is optimized specifically for Apple Silicon (M-series chips) and offers significantly higher performance:

- **Direct Metal API**: The implementation uses Apple's Metal API directly for maximum performance, with custom Metal kernels for all elliptic curve operations.
- **Unified Memory**: The algorithm takes advantage of Apple's unified memory architecture to minimize data transfers between the CPU and GPU.

Due to these specific optimizations, the GPU solver is not portable to other architectures (e.g., NVIDIA or AMD GPUs) without significant modification.

## 5. How to Use

### Installation

#### Prerequisites
- Python 3.8 or higher
- For GPU support: Apple Silicon Mac (M1/M2/M3 series)

#### Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/bitcoin-puzzle-solver.git
   cd bitcoin-puzzle-solver
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

The solver is configured through two types of files:

-   **`puzzles.json`**: This file contains the definitions for the Bitcoin puzzles. Each puzzle entry includes its number, the target public key, and the key search range. You can add more puzzles to this file as needed.
-   **`profiles/`**: This directory contains `.ini` files that define solver profiles. Each profile specifies parameters like `num_walkers` (the number of kangaroos), `distinguished_point_threshold`, and `start_point_strategy`. The `start_point_strategy` can be `midpoint` (default, deterministic) or `random` (useful for running multiple independent solver instances).
    -   `verify_cpu.ini`: A profile for quick checks on easy puzzles using the CPU.
    -   `verify_gpu.ini`: A profile for quick checks on easy puzzles using the GPU.
    -   `solve_gpu.ini`: A profile for serious attempts on hard puzzles using the GPU.

### Running the Solver

To run the solver, execute the `src/main.py` script from the command line, specifying the puzzle number and the desired profile.

**Command:**

```bash
python src/main.py --puzzle <number> --profile <name>
```

**Examples:**

To solve puzzle #5 using the CPU implementation:

```bash
python src/main.py --puzzle 5 --profile verify_cpu
```

To solve a harder puzzle using the GPU implementation:

```bash
python src/main.py --puzzle 30 --profile solve_gpu
```

The solver will then initialize and start searching for the private key, printing progress updates to the console.
