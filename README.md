# Bitcoin Puzzle Solver using Pollard's Kangaroo Algorithm

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

## 4. Target Platform: Apple Silicon

This project is exclusively designed and optimized for Apple Silicon (M-series chips). It heavily relies on the unique features of this platform to achieve high performance:

-   **GPU Acceleration**: All cryptographic computations are offloaded to the GPU using PyTorch with the Metal Performance Shaders (MPS) backend.
-   **Unified Memory**: The algorithm takes advantage of Apple's unified memory architecture to eliminate costly data transfers between the CPU and GPU, which is critical for an iterative, high-throughput algorithm like Pollard's Kangaroo.

Due to these specific optimizations, this solver is not intended to be portable to other architectures (e.g., NVIDIA or AMD GPUs) without significant modification.
