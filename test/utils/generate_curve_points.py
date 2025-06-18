#!/usr/bin/env python3
"""
Utility script to generate valid secp256k1 curve points and test vectors.

This script generates known answer test vectors for secp256k1 curve operations
that can be used to validate the correctness of Metal shader implementations.
"""
import random
import json
from typing import Tuple, Dict, List, Any

from coincurve.keys import PrivateKey, PublicKey

import cryptography_utils as crypto


def generate_random_point() -> Tuple[PublicKey, int]:
    """
    Generates a random point on the secp256k1 curve by creating a random scalar
    and multiplying it with the generator point G.
    
    Returns:
        Tuple[PublicKey, int]: The generated point and the scalar used
    """
    # Generate a random scalar between 1 and curve order - 1
    curve_order = crypto.get_curve_order_n()
    scalar = random.randint(1, curve_order - 1)
    
    # Multiply the generator point by this scalar
    point = crypto.scalar_multiply(scalar)
    
    return point, scalar


def print_point_info(point: PublicKey, scalar: int = None):
    """
    Prints information about a curve point in various formats.
    
    Args:
        point: The curve point to display
        scalar: The scalar used to generate the point (if known)
    """
    x, y = point.point()
    
    print(f"Point coordinates:")
    print(f"  x: {hex(x)}")
    print(f"  y: {hex(y)}")
    print(f"Compressed format: {point.format(compressed=True).hex()}")
    print(f"Uncompressed format: {point.format(compressed=False).hex()}")
    
    if scalar is not None:
        print(f"Generated using scalar: {scalar}")
        print(f"Scalar (hex): {hex(scalar)}")


def point_to_dict(point: PublicKey, scalar: int = None) -> Dict[str, Any]:
    """
    Converts a curve point to a dictionary representation.
    
    Args:
        point: The curve point to convert
        scalar: The scalar used to generate the point (if known)
        
    Returns:
        Dict containing the point information
    """
    x, y = point.point()
    
    result = {
        "x": hex(x),
        "y": hex(y),
        "x_limbs": [
            x & 0xFFFFFFFFFFFFF,
            (x >> 52) & 0xFFFFFFFFFFFFF,
            (x >> 104) & 0xFFFFFFFFFFFFF,
            (x >> 156) & 0xFFFFFFFFFFFFF,
            (x >> 208)
        ],
        "y_limbs": [
            y & 0xFFFFFFFFFFFFF,
            (y >> 52) & 0xFFFFFFFFFFFFF,
            (y >> 104) & 0xFFFFFFFFFFFFF,
            (y >> 156) & 0xFFFFFFFFFFFFF,
            (y >> 208)
        ],
        "compressed": point.format(compressed=True).hex(),
        "uncompressed": point.format(compressed=False).hex()
    }
    
    if scalar is not None:
        result["scalar"] = scalar
        result["scalar_hex"] = hex(scalar)
    
    return result


def generate_test_vectors() -> Dict[str, Any]:
    """
    Generates a comprehensive set of test vectors for secp256k1 curve operations.
    
    Returns:
        Dict containing test vectors for various curve operations
    """
    test_vectors = {
        "generator_point": {},
        "point_addition": [],
        "point_doubling": [],
        "scalar_multiplication": []
    }
    
    # Generator point
    g_point = crypto.get_generator_point()
    test_vectors["generator_point"] = point_to_dict(g_point)
    
    # Point addition test vectors (P + Q = R)
    for i in range(3):
        # Generate two random points
        p_point, p_scalar = generate_random_point()
        q_point, q_scalar = generate_random_point()
        
        # Calculate P + Q
        r_point = crypto.point_add(p_point, q_point)
        
        test_vectors["point_addition"].append({
            "P": point_to_dict(p_point, p_scalar),
            "Q": point_to_dict(q_point, q_scalar),
            "R": point_to_dict(r_point)
        })
    
    # Point doubling test vectors (2*P = R)
    for i in range(3):
        # Generate a random point
        p_point, p_scalar = generate_random_point()
        
        # Calculate 2*P
        r_point = crypto.point_add(p_point, p_point)
        
        test_vectors["point_doubling"].append({
            "P": point_to_dict(p_point, p_scalar),
            "R": point_to_dict(r_point)
        })
    
    # Scalar multiplication test vectors (k*P = R)
    # Include specific test for 4*G
    g_point = crypto.get_generator_point()
    g2_point = crypto.point_add(g_point, g_point)  # 2*G
    g4_point = crypto.point_add(g2_point, g2_point)  # 4*G
    
    test_vectors["scalar_multiplication"].append({
        "P": point_to_dict(g_point),
        "k": 4,
        "R": point_to_dict(g4_point)
    })
    
    # Add a few more random scalar multiplication tests
    for i in range(2):
        p_point, p_scalar = generate_random_point()
        k = random.randint(2, 10)  # Small scalar for easy verification
        r_point = crypto.scalar_multiply(k, p_point)
        
        test_vectors["scalar_multiplication"].append({
            "P": point_to_dict(p_point, p_scalar),
            "k": k,
            "R": point_to_dict(r_point)
        })
    
    # Add test with non-trivial Z coordinate
    # Create a point with Z=2 by taking a point and manually setting Z
    p_point, p_scalar = generate_random_point()
    x, y = p_point.point()
    
    # For a point (x,y) with Z=2, the Jacobian coordinates would be (x*Z^2, y*Z^3, Z)
    # So the equivalent affine coordinates would be (x/Z^2, y/Z^3) = (x,y)
    z_value = 2
    test_vectors["jacobian_conversion"] = {
        "affine": point_to_dict(p_point),
        "jacobian": {
            "x": hex(x * z_value * z_value),
            "y": hex(y * z_value * z_value * z_value),
            "z": z_value,
            "x_limbs": [
                (x * z_value * z_value) & 0xFFFFFFFFFFFFF,
                ((x * z_value * z_value) >> 52) & 0xFFFFFFFFFFFFF,
                ((x * z_value * z_value) >> 104) & 0xFFFFFFFFFFFFF,
                ((x * z_value * z_value) >> 156) & 0xFFFFFFFFFFFFF,
                ((x * z_value * z_value) >> 208)
            ],
            "y_limbs": [
                (y * z_value * z_value * z_value) & 0xFFFFFFFFFFFFF,
                ((y * z_value * z_value * z_value) >> 52) & 0xFFFFFFFFFFFFF,
                ((y * z_value * z_value * z_value) >> 104) & 0xFFFFFFFFFFFFF,
                ((y * z_value * z_value * z_value) >> 156) & 0xFFFFFFFFFFFFF,
                ((y * z_value * z_value * z_value) >> 208)
            ],
            "z_limbs": [z_value, 0, 0, 0, 0]
        }
    }
    
    return test_vectors


def main():
    """
    Generates and displays information about secp256k1 curve points.
    Also generates test vectors and saves them to a JSON file.
    """
    print("=== Generator Point G ===")
    g_point = crypto.get_generator_point()
    print_point_info(g_point)
    
    print("\n=== Random Point 1 ===")
    random_point1, scalar1 = generate_random_point()
    print_point_info(random_point1, scalar1)
    
    print("\n=== Random Point 2 ===")
    random_point2, scalar2 = generate_random_point()
    print_point_info(random_point2, scalar2)
    
    print("\n=== Point Addition Example ===")
    sum_point = crypto.point_add(random_point1, random_point2)
    print_point_info(sum_point)
    
    print("\n=== Scalar Multiplication Example ===")
    scalar = random.randint(1, 1000)
    mult_point = crypto.scalar_multiply(scalar, random_point1)
    print(f"Multiplying Random Point 1 by scalar: {scalar}")
    print_point_info(mult_point)
    
    print("\n=== Generating Test Vectors ===")
    test_vectors = generate_test_vectors()
    
    # Save test vectors to a JSON file
    with open('secp256k1_test_vectors.json', 'w') as f:
        json.dump(test_vectors, f, indent=2)
    
    print(f"Test vectors saved to secp256k1_test_vectors.json")
    print("\nTest vector summary:")
    print(f"- Generator point")
    print(f"- {len(test_vectors['point_addition'])} point addition tests")
    print(f"- {len(test_vectors['point_doubling'])} point doubling tests")
    print(f"- {len(test_vectors['scalar_multiplication'])} scalar multiplication tests")
    print(f"- 1 Jacobian-to-affine conversion test")


if __name__ == "__main__":
    main()
