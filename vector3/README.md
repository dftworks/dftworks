# Vector3 Module

The `vector3` module provides a generic 3D vector type with optimized implementations for common numeric types (`f64`, `i32`, `c64`). It is designed for high-performance scientific computing with SIMD-friendly operations and efficient memory layout.

## Features

- Generic `Vector3<T>` type supporting multiple numeric types
- Type aliases: `Vector3f64`, `Vector3i32`, `Vector3c64`
- Vector operations: dot product, cross product, norm calculations
- Operator overloading: addition, scalar multiplication, division
- Zero-copy slice views for efficient data access
- `#[repr(C)]` layout for guaranteed memory layout and SIMD optimization
- Inline functions for optimal performance

## Type Aliases

```rust
pub type Vector3f64 = Vector3<f64>;  // 64-bit floating point vectors
pub type Vector3i32 = Vector3<i32>;  // 32-bit integer vectors
pub type Vector3c64 = Vector3<c64>;  // Complex 64-bit vectors
```

## Usage

### Basic Operations

```rust
use vector3::Vector3f64;

// Create a vector
let v = Vector3f64::new(1.0, 2.0, 3.0);

// Create zero vector
let zero = Vector3f64::zeros();

// Access components
println!("x: {}, y: {}, z: {}", v.x, v.y, v.z);
```

### Vector Operations (Vector3f64)

```rust
use vector3::Vector3f64;

let v1 = Vector3f64::new(1.0, 2.0, 3.0);
let v2 = Vector3f64::new(4.0, 5.0, 6.0);

// Dot product
let dot = v1.dot_product(&v2);  // 32.0

// Cross product
let cross = v1.cross_product(&v2);

// Norm (magnitude)
let norm = v1.norm2();  // sqrt(14.0)

// Squared norm (faster, no sqrt)
let norm_sq = v1.norm_squared();  // 14.0
```

### Arithmetic Operations

```rust
use vector3::Vector3f64;

let v1 = Vector3f64::new(1.0, 2.0, 3.0);
let v2 = Vector3f64::new(4.0, 5.0, 6.0);

// Addition
let sum = v1 + v2;  // Vector3f64 { x: 5.0, y: 7.0, z: 9.0 }

// Scalar multiplication (right)
let scaled = v1 * 2.0;  // Vector3f64 { x: 2.0, y: 4.0, z: 6.0 }

// Scalar multiplication (left)
let scaled = 2.0 * v1;  // Same result

// Scalar division
let divided = v1 / 2.0;  // Vector3f64 { x: 0.5, y: 1.0, z: 1.5 }

// Dot product via multiplication operator
let dot = v1 * v2;  // 32.0 (f64)
```

### Slice Views

```rust
use vector3::{Vector3f64, as_slice_of_element, as_mut_slice_of_element};

// Get slice view of single vector
let mut v = Vector3f64::new(1.0, 2.0, 3.0);
let slice = v.as_mut_slice();
slice[0] = 10.0;  // Modifies v.x

// Get slice view of vector array
let vectors = vec![
    Vector3f64::new(1.0, 2.0, 3.0),
    Vector3f64::new(4.0, 5.0, 6.0),
];
let flat_slice = as_slice_of_element(&vectors);
// flat_slice is &[f64] with 6 elements: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
```

### Integer Vectors (Vector3i32)

```rust
use vector3::Vector3i32;

let v1 = Vector3i32::new(1, 2, 3);
let v2 = Vector3i32::new(4, 5, 6);

// Addition
let sum = v1 + v2;  // Vector3i32 { x: 5, y: 7, z: 9 }

// Zero vector
let zero = Vector3i32::zeros();
```

## API Reference

### Generic Vector3<T> Methods

- `new(x, y, z)` - Create a new vector
- `zeros()` - Create a zero vector
- `to_vec()` - Convert to `Vec<T>`
- `as_slice()` - Get immutable slice view of components
- `as_mut_slice()` - Get mutable slice view of components
- `set_zeros()` - Set all components to zero

### Vector3f64 Specific Methods

- `dot_product(&other)` - Compute dot product: a · b
- `cross_product(&other)` - Compute cross product: a × b
- `norm_squared()` - Compute squared norm (|v|²) - **faster for comparisons**
- `norm2()` - Compute norm (|v|) = sqrt(x² + y² + z²)

### Operator Traits (Vector3f64)

- `Add<Vector3f64>` - Vector addition: `v1 + v2`
- `Mul<Vector3f64>` - Dot product: `v1 * v2` → f64
- `Mul<f64>` - Scalar multiplication (right): `v * s`
- `Mul<Vector3f64>` for `f64` - Scalar multiplication (left): `s * v`
- `Div<f64>` - Scalar division: `v / s`
- `Display` - Formatted output: `"x y z"`

### Vector3i32 Operations

- `Add<Vector3i32>` - Vector addition: `v1 + v2`
- `Display` - Formatted output: `"x y z"`

### Utility Functions

- `as_slice_of_element(v: &[Vector3<T>])` - Convert array of vectors to flat slice
- `as_mut_slice_of_element(v: &mut [Vector3<T>])` - Convert mutable array to flat slice

## Performance Considerations

### Use `norm_squared()` for Comparisons

When comparing vector magnitudes, use `norm_squared()` instead of `norm2()` to avoid the expensive `sqrt()` operation:

```rust
// Fast: compare squared norms
if v1.norm_squared() < v2.norm_squared() {
    // ...
}

// Slower: unnecessary sqrt
if v1.norm2() < v2.norm2() {  // Avoid this!
    // ...
}
```

### Memory Layout

The `Vector3<T>` struct uses `#[repr(C)]` to guarantee:
- Predictable memory layout (x, y, z in order)
- Safe use of `as_slice()` and `as_mut_slice()`
- Better SIMD vectorization opportunities

### Inlining

All hot-path operations are marked with `#[inline]` for optimal performance in release builds.

### Slice Conversions

The `as_slice_of_element()` functions enable zero-copy conversion between arrays of vectors and flat arrays, useful for:
- Interfacing with C libraries
- Efficient data processing
- BLAS/LAPACK operations

## Testing

Run tests using the justfile:

```bash
just test              # Run all tests
just test-verbose      # Run tests with output
just test-serial       # Run tests sequentially
just test-specific <test_name>  # Run a specific test
just test-doc          # Run documentation tests
```

Or directly with cargo:

```bash
cargo test --package vector3
```

## Dependencies

- `types` - Complex number types (c64)
- `num-traits` - Numeric trait definitions

## Examples

### Computing Distance

```rust
use vector3::Vector3f64;

let p1 = Vector3f64::new(0.0, 0.0, 0.0);
let p2 = Vector3f64::new(3.0, 4.0, 0.0);
let diff = Vector3f64::new(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z);
let distance = diff.norm2();  // 5.0
```

### Normalizing a Vector

```rust
use vector3::Vector3f64;

let mut v = Vector3f64::new(3.0, 4.0, 0.0);
let norm = v.norm2();
v = v / norm;  // Normalized: Vector3f64 { x: 0.6, y: 0.8, z: 0.0 }
```

### Batch Processing

```rust
use vector3::{Vector3f64, as_slice_of_element};

let vectors = vec![
    Vector3f64::new(1.0, 2.0, 3.0),
    Vector3f64::new(4.0, 5.0, 6.0),
    Vector3f64::new(7.0, 8.0, 9.0),
];

// Convert to flat slice for efficient processing
let flat = as_slice_of_element(&vectors);
// Process flat array (e.g., pass to BLAS routine)
```

## Notes

- All operations are designed to be SIMD-friendly
- The struct is `Copy` and `Clone`, so it can be passed by value efficiently
- Use `norm_squared()` instead of `norm2()` when comparing magnitudes
- Slice views rely on `#[repr(C)]` layout guarantee
- Integer vectors (`Vector3i32`) have limited operations compared to `Vector3f64`
