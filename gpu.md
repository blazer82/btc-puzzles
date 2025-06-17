# GPU Acceleration Specification for Apple Silicon

## 1. Objective

To achieve a significant performance increase, the core computational work of the Pollard's Kangaroo algorithm will be offloaded from the CPU to the Apple Silicon GPU. This involves porting all per-kangaroo operations—elliptic curve point addition, distance tracking, and distinguished point checking—into a Metal compute shader. The Python application will be refactored to act as an orchestrator, managing GPU resources and processing the rare results returned by the GPU.

## 2. Technology Stack

- **Orchestration Language**: Python 3.
- **GPU Interface**: A Python library that provides bindings to the Metal framework, such as `metal-python`. This will be used for device management, buffer allocation, and kernel dispatch.
- **Compute Shaders**: Metal Shading Language (MSL) will be used to write the high-performance compute kernels.

## 3. GPU Data Structures & Memory Management

The state of individual `Kangaroo` objects will be replaced by large, contiguous memory buffers on the GPU, managed via the Metal interface from Python. This leverages the Unified Memory Architecture of Apple Silicon to minimize explicit data copies.

- **Kangaroo State Buffers**: Two sets of buffers (one for the tame herd, one for the wild herd) will hold the state for all walkers.

  - `points_buffer`: A buffer of `num_walkers * 2` 32-bit unsigned integers, storing the (x, y) coordinates for each kangaroo's current point on the curve.
  - `distances_buffer`: A buffer of `num_walkers` 64-bit unsigned integers, storing the accumulated hop distance for each kangaroo.

- **Pre-computed Hops Buffer**:

  - `hop_points_buffer`: A read-only buffer containing the (x, y) coordinates of the 16 pre-computed hop points (`2^i * G`). This is generated once by the CPU and made available to the GPU for the entire run.

- **Distinguished Point (DP) Output Buffers**:
  - `dp_output_buffer`: A buffer large enough to hold potential distinguished points found in a single step. It will store a struct containing `(x, y, distance)`.
  - `dp_counter`: An atomic integer on the GPU, used as a thread-safe counter. When a walker finds a DP, it will atomically increment this counter to get a unique index into the `dp_output_buffer` to write its result.

## 4. Metal Compute Kernel (`kangaroo_hop.metal`)

A single, highly optimized Metal compute kernel will perform one parallel hop for all kangaroos.

- **Execution Model**: The kernel will be dispatched with a 1D grid of `num_walkers` threads. Each thread `i` is responsible for updating the state of kangaroo `i`.

- **Kernel Signature (Conceptual)**:

  ```c
  kernel void kangaroo_step(
      // Input/Output: Kangaroo state
      device packed_uint2 *current_points [[buffer(0)]],
      device ulong *distances [[buffer(1)]],

      // Input: Read-only data
      constant packed_uint2 *hop_points [[buffer(2)]],
      constant uint dp_threshold [[buffer(3)]],

      // Output: For distinguished points
      device atomic_uint *dp_counter [[buffer(4)]],
      device DPResult *dp_output_buffer [[buffer(5)]],

      // Thread identifier
      uint gid [[thread_position_in_grid]]
  );
  ```

- **Logic within a Single GPU Thread**:
  1.  **Load State**: Read the current point `(x, y)` and `distance` for the kangaroo at index `gid`.
  2.  **Select Hop**: Implement the `select_hop_index` logic (`current_point.x % 16`) directly in MSL.
  3.  **Perform Hop**:
      - Fetch the corresponding `hop_point` from the `hop_points_buffer`.
      - Calculate the new point by performing secp256k1 point addition: `new_point = current_point + hop_point`. **This requires a full implementation of 256-bit modular arithmetic and the elliptic curve addition formula in MSL.**
      - Update the distance: `new_distance = current_distance + (2^hop_index)`.
  4.  **Check for Distinguished Point**: Implement the `is_distinguished` logic (`new_point.x & ((1 << dp_threshold) - 1) == 0`) in MSL.
  5.  **Write Back State**: Write the `new_point` and `new_distance` back to the main state buffers at index `gid`.
  6.  **Report Distinguished Point**: If the `new_point` is distinguished:
      a. Atomically increment `dp_counter` to reserve an output slot.
      b. Write the `(new_point.x, new_point.y, new_distance)` to the `dp_output_buffer` at the reserved index.

## 5. Refactoring of Python Codebase

- **`src/gpu_executor.py` (New File)**: This new module will encapsulate all interaction with the Metal API.

  - Responsibilities: Device initialization, compiling the `.metal` file, creating and managing all GPU buffers, and wrapping the kernel dispatch in a simple Python method.

- **`src/kangaroo_runner.py` (Major Refactor)**:

  - **`__init__`**: Will no longer create `Kangaroo` objects. Instead, it will instantiate `GpuExecutor`. It will prepare the initial kangaroo states (points and distances) in NumPy arrays and use the executor to create and populate the corresponding GPU buffers. The warm-up phase will be handled by dispatching the GPU kernel repeatedly.
  - **`step()`**: The loop over kangaroos will be removed. The method will now:
    1.  Call the `gpu_executor.execute_step()` method, which dispatches the kernel.
    2.  Read the `dp_counter` value from the GPU.
    3.  If the counter is greater than zero, copy the `dp_output_buffer` from the GPU to a NumPy array on the CPU.
    4.  Iterate over the DPs on the CPU, checking for collisions in the existing `PointTrap` instances.
    5.  Reset the `dp_counter` on the GPU to zero for the next step.

- **`src/kangaroo.py` (To Be Removed)**: This class becomes obsolete, as its state and logic are moved to the GPU buffers and kernel, respectively.

- **`src/cryptography_utils.py` (Reduced Role)**: The performance-critical functions (`point_add`, `scalar_multiply`) will no longer be used in the main loop. The module will only be used for initial setup tasks on the CPU, such as parsing the public key and calculating the initial tame start point.

- **`src/hop_strategy.py` & `src/distinguished_points.py` (Logic Ported)**: The core logic from `select_hop_index` and `is_distinguished` will be re-implemented in the MSL kernel. The Python versions will no longer be used in the main loop. The `PointTrap` class will remain unchanged and will be used on the CPU.

## 6. Summary of Logic to be Ported to MSL

To be explicit, the following pieces of logic, currently implemented in Python or its C-based dependencies, must be re-implemented from scratch in the Metal Shading Language (`.metal` file) to run on the GPU:

1.  **`secp256k1` Elliptic Curve Point Addition**:

    - **Source**: This logic is currently handled by the `coincurve` library via `cryptography_utils.point_add`.
    - **Requirement**: This is the most complex part. It requires a complete, from-scratch implementation of the underlying 256-bit modular arithmetic (addition, subtraction, multiplication, inversion) over the finite field used by `secp256k1`, followed by an implementation of the point addition formula itself.

2.  **Hop Selection**:

    - **Source**: The `select_hop_index` function in `src/hop_strategy.py`.
    - **Requirement**: Re-implement the `point.x % 16` logic in MSL to deterministically select a hop from the pre-computed hops buffer.

3.  **Distinguished Point Identification**:
    - **Source**: The `is_distinguished` function in `src/distinguished_points.py`.
    - **Requirement**: Re-implement the bitwise check (`(point.x & mask) == 0`) in MSL to efficiently check every new point on the GPU without CPU intervention.

## 7. Implementation Plan

To manage complexity, the Metal shader development will be broken down into four distinct, verifiable stages:

### Stage 1: Implement 256-bit Big-Integer Arithmetic

- **Goal**: Create a foundational Metal shader file for basic 256-bit integer operations.
- **Details**:
  - Define a data structure to hold 256-bit numbers, mirroring `secp256k1_fe`'s 5x52-bit limb representation.
  - Implement core functions for addition, subtraction, and multiplication, porting the logic from `_libsecp256k1/src/field_5x52_int128_impl.h`.

### Stage 2: Implement Modular (Field) Arithmetic

- **Goal**: Build on Stage 1 to implement arithmetic specific to the `secp256k1` finite field.
- **Details**:
  - Port the modular reduction function (`secp256k1_fe_impl_normalize`).
  - Port the modular inverse function (`secp256k1_fe_impl_inv`), which is the most complex part of the field arithmetic and relies on the logic in `_libsecp256k1/src/modinv64_impl.h`.

### Stage 3: Implement Elliptic Curve (Group) Operations

- **Goal**: Use the field arithmetic from Stage 2 to implement the elliptic curve point operations.
- **Details**:
  - Port the point addition (`secp256k1_gej_add_ge`) and point doubling (`secp256k1_gej_double_var`) functions from `_libsecp256k1/src/group_impl.h`.

### Stage 4: Implement the Final Kangaroo Kernel

- **Goal**: Assemble the cryptographic primitives into the final `kangaroo_step` kernel.
- **Details**:
  - The kernel will use the group operations from Stage 3 to perform the kangaroo hops.
  - It will also include the simpler logic for hop selection and distinguished point checking, ported from the Python implementation.

### Implement the Final Kangaroo Kernel

1.  **Create `src/gpu_executor.py`:** This new file, specified in the plan, is the critical missing piece. It should encapsulate all direct interaction with the Metal API, including:

    - Initializing the Metal device.
    - Compiling the `kangaroo_hop.metal` kernel.
    - Creating, populating, and managing the required GPU buffers (`current_points`, `distances`, `hop_points`, `dp_output_buffer`, `dp_counter`).
    - Providing a Python method to dispatch the `kangaroo_step` kernel.

2.  **Refactor `src/kangaroo_runner.py`:** This class must be heavily modified to act as a GPU orchestrator instead of running the logic on the CPU.

    - **Remove `Kangaroo` objects:** The lists of `tame_kangaroos` and `wild_kangaroos` must be removed.
    - **Use `GpuExecutor`:** The `__init__` method should instantiate the new `GpuExecutor`. It should prepare the initial kangaroo states in NumPy arrays and use the executor to create the corresponding GPU buffers.
    - **Rewrite `step()` method:** The `step()` method needs to be completely rewritten. Instead of looping over `Kangaroo` objects, it will now make a single call to the `GpuExecutor` to run the `kangaroo_step` kernel on the GPU. It will then read the distinguished point results back from the GPU for collision checking.

3.  **Remove `src/kangaroo.py`:** As specified in the plan, this file will become obsolete once `KangarooRunner` is refactored to use the GPU. It can then be deleted.
