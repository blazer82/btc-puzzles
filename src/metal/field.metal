/*
 * Bitcoin Puzzle Solver - Metal port of secp256k1 field arithmetic.
 *
 * This is the main include file that brings together all the field arithmetic
 * and elliptic curve operations. It includes the individual stage files to
 * provide a complete implementation.
 *
 * Original source:
 * - _libsecp256k1/src/field_5x52.h
 * - _libsecp256k1/src/field_5x52_int128_impl.h
 * - _libsecp256k1/src/int128_native_impl.h
 * - _libsecp256k1/src/field_5x52_impl.h
 * - _libsecp256k1/src/modinv64_impl.h
 * - _libsecp256k1/src/group_impl.h
 */

#include <metal_stdlib>
#include "int128.metal"
#include "field_basic.metal"
#include "field_modular.metal"
#include "group.metal"

using namespace metal;
