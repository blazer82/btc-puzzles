#!/usr/bin/env python3
"""
Apple Silicon GPU Pollard's Rho Implementation

HOW-TO GUIDE: Apple Silicon Bitcoin Puzzle Solver
-------------------------------------------------
This script uses Pollard's Rho algorithm, accelerated on Apple Silicon GPUs via PyTorch MPS,
to search for private keys within the ranges defined by the Bitcoin Puzzles.

--- Step 1: Installation ---
Ensure you have PyTorch for Apple Silicon installed. If not, run this in your terminal:
$ pip install torch torchvision

--- Step 2: Configuration ---
All settings are at the VERY BOTTOM of this file, inside the main() function.
1.  PUZZLE_NUMBER: Set this to the difficulty level you want to attempt (e.g., 40).
2.  NUM_WALKERS: The number of parallel searches. A good start is 20,000 for M-Pro,
    50,000+ for M-Max/Ultra. More walkers use more GPU memory.
3.  DISTINGUISHED_BITS: Controls the rarity of points checked for collisions. 18-22 is a
    well-balanced range.

--- Step 3: Adding New Puzzles ---
To solve a puzzle, its public key must be known to the script.
1.  Find the `AppleSiliconBitcoinPuzzleSolver` class in this file.
2.  Inside its `__init__` method, find the `puzzle_data` dictionary.
3.  Add a new entry for your target puzzle in the format:
    { puzzle_number: (public_key_x, public_key_y), ... }

--- Step 4: Execution ---
Navigate to the script's directory in your terminal and run:
$ python solve_puzzle.py

The script will show its progress. To stop it, press Ctrl+C.

--- Realistic Expectations ---
- This script is excellent for testing on already-solved puzzles in the 40-60 range.
- Each puzzle number DOUBLES the difficulty.
- Solving high-difficulty puzzles (70+) is computationally infeasible for a single machine.
"""

import torch
import time
from typing import Tuple, Optional, Dict
import platform

# --- Core secp256k1 Parameters ---
P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
GX = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
GY = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8

# --- Helper Class for Pure Python Verification (CPU-based) ---
class BitcoinVerifier:
    P, N, Gx, Gy = P, N, GX, GY
    
    @classmethod
    def point_add(cls, p1, p2):
        if p1 is None: return p2
        if p2 is None: return p1
        if p1[0] == p2[0] and p1[1] != p2[1]: return None
        if p1 == p2:
            if p1[1] == 0: return None
            lam = (3 * p1[0] * p1[0] * pow(2 * p1[1], -1, cls.P)) % cls.P
        else:
            if p2[0] == p1[0]: return None
            lam = ((p2[1] - p1[1]) * pow(p2[0] - p1[0], -1, cls.P)) % cls.P
        x3 = (lam * lam - p1[0] - p2[0]) % cls.P
        y3 = (lam * (p1[0] - x3) - p1[1]) % cls.P
        return (x3, y3)

    @classmethod
    def point_multiply(cls, k, p):
        res = None
        add = p
        while k > 0:
            if k & 1: res = cls.point_add(res, add)
            add = cls.point_add(add, add)
            k >>= 1
        return res

    @classmethod
    def get_public_key(cls, private_key: int):
        return cls.point_multiply(private_key, (cls.Gx, cls.Gy))

class AppleSiliconECCOps:
    P, N = P, N
    G = (torch.tensor([[GX]]), torch.tensor([[GY]]))

    def __init__(self, device='mps'):
        self.device = device
        self.Gx_cpu = GX
        self.Gy_cpu = GY

    def point_add_batch(self, p1x, p1y, p2x, p2y):
        den = p2x - p1x
        if torch.any(den == 0): return None, None # Collision on same x, should be rare
        den_inv = pow(den, self.P - 2, self.P)
        s = ((p1y - p2y) * den_inv) % self.P
        x3 = (s*s - p1x - p2x) % self.P
        y3 = (s*(p1x - x3) - p1y) % self.P
        return x3, y3

    def point_double_batch(self, p1x, p1y):
        den = 2 * p1y
        if torch.any(den == 0): return None, None # Point at infinity
        den_inv = pow(den, self.P - 2, self.P)
        s = (3 * p1x * p1x * den_inv) % self.P
        x3 = (s*s - 2*p1x) % self.P
        y3 = (s*(p1x - x3) - p1y) % self.P
        return x3, y3
    
    def advance_walkers(self, x, y, a, b, partition, target_point):
        px, py = target_point
        x_cpu, y_cpu, a_cpu, b_cpu = x.cpu(), y.cpu(), a.cpu(), b.cpu()
        partition_cpu = partition.cpu()

        mask0 = partition_cpu == 0
        mask1 = partition_cpu == 1
        mask2 = partition_cpu == 2

        if torch.any(mask0):
            x_cpu[mask0], y_cpu[mask0] = self.point_double_batch(x_cpu[mask0], y_cpu[mask0])
            a_cpu[mask0] = (a_cpu[mask0] * 2) % self.N
            b_cpu[mask0] = (b_cpu[mask0] * 2) % self.N

        if torch.any(mask1):
            x_cpu[mask1], y_cpu[mask1] = self.point_add_batch(x_cpu[mask1], y_cpu[mask1], self.Gx_cpu, self.Gy_cpu)
            a_cpu[mask1] = (a_cpu[mask1] + 1) % self.N

        if torch.any(mask2):
            x_cpu[mask2], y_cpu[mask2] = self.point_add_batch(x_cpu[mask2], y_cpu[mask2], px, py)
            b_cpu[mask2] = (b_cpu[mask2] + 1) % self.N
            
        return x_cpu.to(self.device), y_cpu.to(self.device), a_cpu.to(self.device), b_cpu.to(self.device)


class AppleSiliconRhoWalker:
    def __init__(self, num_walkers: int, target_point: Tuple[int, int], distinguished_bits: int, device='mps'):
        self.num_walkers = num_walkers
        self.device = device
        self.target_x, self.target_y = target_point
        self.distinguished_bits = distinguished_bits
        self.distinguished_mask = (1 << distinguished_bits) - 1
        
        self.ecc_ops = AppleSiliconECCOps(device)
        self.distinguished_points: Dict[int, Tuple[int, int]] = {}
        
        self.a = torch.tensor([torch.randint(1, N, (1,)).item() for _ in range(num_walkers)], device=device)
        self.b = torch.tensor([torch.randint(1, N, (1,)).item() for _ in range(num_walkers)], device=device)
        
        print("Initializing walker starting points (one-time CPU operation)...")
        start_points_x, start_points_y = [], []
        for i in range(num_walkers):
            ag = BitcoinVerifier.point_multiply(self.a[i].item(), (GX, GY))
            bp = BitcoinVerifier.point_multiply(self.b[i].item(), (self.target_x, self.target_y))
            start_point = BitcoinVerifier.point_add(ag, bp)
            start_points_x.append(start_point[0])
            start_points_y.append(start_point[1])
        
        self.x = torch.tensor(start_points_x, device=device)
        self.y = torch.tensor(start_points_y, device=device)

        print(f"🚀 Initialized {num_walkers} GPU walkers on Apple Silicon")

    def iterate_batch(self) -> Optional[int]:
        partition = self.x % 3
        self.x, self.y, self.a, self.b = self.ecc_ops.advance_walkers(
            self.x, self.y, self.a, self.b, partition, (self.target_x, self.target_y))
        
        is_distinguished = (self.x & self.distinguished_mask) == 0
        if torch.any(is_distinguished):
            dist_indices = torch.where(is_distinguished)[0]
            for idx in dist_indices.cpu().tolist():
                x_val = self.x[idx].item()
                if x_val in self.distinguished_points:
                    a1_coll, b1_coll = self.distinguished_points[x_val]
                    a2_coll, b2_coll = self.a[idx].item(), self.b[idx].item()
                    
                    if a1_coll == a2_coll or b1_coll == b2_coll: continue
                    
                    da = (a1_coll - a2_coll) % N
                    db_inv = pow((b2_coll - b1_coll) % N, -1, N)
                    private_key = (da * db_inv) % N
                    
                    pk_x, pk_y = BitcoinVerifier.get_public_key(private_key)
                    if pk_x == self.target_x and pk_y == self.target_y:
                        return private_key
                else:
                    self.distinguished_points[x_val] = (self.a[idx].item(), self.b[idx].item())
        return None

class AppleSiliconBitcoinPuzzleSolver:
    def __init__(self, puzzle_number: int):
        self.puzzle_number = puzzle_number
        self.device = 'mps'
        
        # --- DATABASE OF PUZZLE PUBLIC KEYS ---
        # Add the public keys for the puzzles you want to solve here.
        # The key is the puzzle number, the value is a tuple of (PublicKeyX, PublicKeyY)
        puzzle_data = {
            40: (0xa2efa402fd5268400c77c20e574ba86409ededee7c4020e4b9f0edbee53de0d4,
                 0x7ba1a987013e78aef5295bf842749bdf97e25336a82458bbaba8c00d16a79ea7),
            # Add more puzzles here...
        }
        
        if puzzle_number not in puzzle_data:
            raise ValueError(f"Public key for puzzle #{puzzle_number} is not defined in the script.")

        # Set the target point from our database
        self.target_point = puzzle_data[puzzle_number]

        print(f"\n🧩 Loaded Bitcoin Puzzle #{puzzle_number}")
        print(f"🎯 Target PubKey X: {hex(self.target_point[0])}")

    def solve(self, num_walkers: int, distinguished_bits: int, report_interval: int = 100):
        print(f"\n🚀 Starting Apple Silicon GPU solver..."); print(f"👥 Walkers: {num_walkers:,}"); print(f"✨ Distinguished Point Bits: {distinguished_bits} (Points stored if x % {2**distinguished_bits} == 0)")
        walker = AppleSiliconRhoWalker(num_walkers=num_walkers, target_point=self.target_point, distinguished_bits=distinguished_bits, device=self.device)
        start_time = time.time(); iterations = 0; total_steps = 0
        try:
            while True:
                solution = walker.iterate_batch(); iterations += 1; total_steps += num_walkers
                if solution:
                    elapsed = time.time() - start_time; print("\n" + "="*50); print("🎉🎉🎉 SOLUTION FOUND! 🎉🎉🎉"); print(f"🔑 Private Key (hex): {hex(solution)}"); print(f"⏱️ Time Taken: {elapsed:.2f} seconds"); rate = total_steps / elapsed; print(f"🏃‍♀️ Total Steps: {total_steps:,}"); print(f"⚡ Average Rate: {rate:,.0f} steps/second"); print("="*50); return solution
                if iterations % report_interval == 0:
                    elapsed = time.time() - start_time; rate = total_steps / elapsed if elapsed > 0 else 0; dp_count = len(walker.distinguished_points); print(f"🔄 Iter: {iterations:,} | Steps: {total_steps/1e6:.2f}M | Rate: {rate:,.0f} steps/s | DPs Found: {dp_count:,}")
        except KeyboardInterrupt: print("\n⏹️ Interrupted by user.")
        except Exception as e: print(f"\n❌ An error occurred: {e}"); import traceback; traceback.print_exc()
        elapsed = time.time() - start_time; print("\n📊 Solver Stopped. Final Statistics:");
        if elapsed > 0: rate = total_steps / elapsed; print(f"⏱️ Total time: {elapsed:.2f} seconds"); print(f"👣 Total steps: {total_steps:,}"); print(f"🏃 Average rate: {rate:,.0f} steps/second")
        return None

def main():
    print("🍎 Apple Silicon GPU Pollard’s Rho Solver (Corrected Version 2)"); print("=" * 60)
    if not (platform.system() == "Darwin" and torch.backends.mps.is_available()): print("❌ This script requires an Apple Silicon Mac with PyTorch MPS support."); return
    print("✅ Apple Silicon (MPS) detected.")
    
    PUZZLE_NUMBER = 40; NUM_WALKERS = 20000; DISTINGUISHED_BITS = 18
    
    try:
        solver = AppleSiliconBitcoinPuzzleSolver(PUZZLE_NUMBER); solver.solve(num_walkers=NUM_WALKERS, distinguished_bits=DISTINGUISHED_BITS)
    except Exception as e: print(f"\n❌ A critical error occurred in main: {e}")

if __name__ == "__main__":
    main()