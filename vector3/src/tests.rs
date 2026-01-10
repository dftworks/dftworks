use super::*;

#[test]
fn test_vector3_slice_conversion() {
    let mut v = vec![
        Vector3f64 {
            x: 1.0,
            y: 2.0,
            z: 3.0
        },
        Vector3f64 {
            x: 4.0,
            y: 5.0,
            z: 6.0
        },
        Vector3f64 {
            x: 7.0,
            y: 8.0,
            z: 9.0
        },
    ];

    let v_f64 = as_mut_slice_of_element(&mut v);

    let mut v2 = vec![0.0; 9];
    for i in 0..9 {
        v2[i] = v_f64[i];
    }

    assert_eq!(v2, v_f64);
    assert_eq!(v2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
}

#[test]
fn test_vector3f64_basic() {
    let v = Vector3f64::new(1.0, 2.0, 3.0);
    assert_eq!(v.x, 1.0);
    assert_eq!(v.y, 2.0);
    assert_eq!(v.z, 3.0);
}

#[test]
fn test_vector3f64_zeros() {
    let v = Vector3f64::zeros();
    assert_eq!(v.x, 0.0);
    assert_eq!(v.y, 0.0);
    assert_eq!(v.z, 0.0);
}

#[test]
fn test_vector3f64_set_zeros() {
    let mut v = Vector3f64::new(1.0, 2.0, 3.0);
    v.set_zeros();
    assert_eq!(v.x, 0.0);
    assert_eq!(v.y, 0.0);
    assert_eq!(v.z, 0.0);
}

#[test]
fn test_vector3f64_to_vec() {
    let v = Vector3f64::new(1.0, 2.0, 3.0);
    let vec = v.to_vec();
    assert_eq!(vec, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_vector3f64_as_slice() {
    let v = Vector3f64::new(1.0, 2.0, 3.0);
    let slice = v.as_slice();
    assert_eq!(slice, &[1.0, 2.0, 3.0]);
}

#[test]
fn test_vector3f64_as_mut_slice() {
    let mut v = Vector3f64::new(1.0, 2.0, 3.0);
    let slice = v.as_mut_slice();
    slice[0] = 10.0;
    assert_eq!(v.x, 10.0);
}

#[test]
fn test_vector3f64_dot_product() {
    let v1 = Vector3f64::new(1.0, 2.0, 3.0);
    let v2 = Vector3f64::new(4.0, 5.0, 6.0);
    let dot = v1.dot_product(&v2);
    assert_eq!(dot, 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0);
    assert_eq!(dot, 32.0);
}

#[test]
fn test_vector3f64_cross_product() {
    let v1 = Vector3f64::new(1.0, 0.0, 0.0);
    let v2 = Vector3f64::new(0.0, 1.0, 0.0);
    let cross = v1.cross_product(&v2);
    assert_eq!(cross.x, 0.0);
    assert_eq!(cross.y, 0.0);
    assert_eq!(cross.z, 1.0);
}

#[test]
fn test_vector3f64_norm_squared() {
    let v = Vector3f64::new(3.0, 4.0, 0.0);
    let norm_sq = v.norm_squared();
    assert_eq!(norm_sq, 25.0);
}

#[test]
fn test_vector3f64_norm2() {
    let v = Vector3f64::new(3.0, 4.0, 0.0);
    let norm = v.norm2();
    assert_eq!(norm, 5.0);
}

#[test]
fn test_vector3f64_add() {
    let v1 = Vector3f64::new(1.0, 2.0, 3.0);
    let v2 = Vector3f64::new(4.0, 5.0, 6.0);
    let sum = v1 + v2;
    assert_eq!(sum.x, 5.0);
    assert_eq!(sum.y, 7.0);
    assert_eq!(sum.z, 9.0);
}

#[test]
fn test_vector3f64_mul_scalar() {
    let v = Vector3f64::new(1.0, 2.0, 3.0);
    let scaled = v * 2.0;
    assert_eq!(scaled.x, 2.0);
    assert_eq!(scaled.y, 4.0);
    assert_eq!(scaled.z, 6.0);
}

#[test]
fn test_vector3f64_mul_scalar_left() {
    let v = Vector3f64::new(1.0, 2.0, 3.0);
    let scaled = 2.0 * v;
    assert_eq!(scaled.x, 2.0);
    assert_eq!(scaled.y, 4.0);
    assert_eq!(scaled.z, 6.0);
}

#[test]
fn test_vector3f64_mul_dot_product() {
    let v1 = Vector3f64::new(1.0, 2.0, 3.0);
    let v2 = Vector3f64::new(4.0, 5.0, 6.0);
    let dot = v1 * v2;
    assert_eq!(dot, 32.0);
}

#[test]
fn test_vector3f64_div() {
    let v = Vector3f64::new(2.0, 4.0, 6.0);
    let divided = v / 2.0;
    assert_eq!(divided.x, 1.0);
    assert_eq!(divided.y, 2.0);
    assert_eq!(divided.z, 3.0);
}

#[test]
fn test_vector3i32_basic() {
    let v = Vector3i32::new(1, 2, 3);
    assert_eq!(v.x, 1);
    assert_eq!(v.y, 2);
    assert_eq!(v.z, 3);
}

#[test]
fn test_vector3i32_zeros() {
    let v = Vector3i32::zeros();
    assert_eq!(v.x, 0);
    assert_eq!(v.y, 0);
    assert_eq!(v.z, 0);
}

#[test]
fn test_vector3i32_add() {
    let v1 = Vector3i32::new(1, 2, 3);
    let v2 = Vector3i32::new(4, 5, 6);
    let sum = v1 + v2;
    assert_eq!(sum.x, 5);
    assert_eq!(sum.y, 7);
    assert_eq!(sum.z, 9);
}

#[test]
fn test_vector3_copy_clone() {
    let v1 = Vector3f64::new(1.0, 2.0, 3.0);
    let v2 = v1; // Copy
    let v3 = v1.clone(); // Clone
    assert_eq!(v1.x, v2.x);
    assert_eq!(v1.x, v3.x);
}

#[test]
fn test_vector3_default() {
    let v: Vector3f64 = Default::default();
    assert_eq!(v.x, 0.0);
    assert_eq!(v.y, 0.0);
    assert_eq!(v.z, 0.0);
}

#[test]
fn test_as_slice_of_element() {
    let v = vec![
        Vector3f64::new(1.0, 2.0, 3.0),
        Vector3f64::new(4.0, 5.0, 6.0),
    ];
    let slice = as_slice_of_element(&v);
    assert_eq!(slice.len(), 6);
    assert_eq!(slice[0], 1.0);
    assert_eq!(slice[5], 6.0);
}

#[test]
fn test_cross_product_orthogonality() {
    let v1 = Vector3f64::new(1.0, 0.0, 0.0);
    let v2 = Vector3f64::new(0.0, 1.0, 0.0);
    let cross = v1.cross_product(&v2);
    
    // Cross product should be orthogonal to both vectors
    let dot1 = v1.dot_product(&cross);
    let dot2 = v2.dot_product(&cross);
    assert!(dot1.abs() < 1e-10);
    assert!(dot2.abs() < 1e-10);
}

#[test]
fn test_norm_squared_vs_norm2() {
    let v = Vector3f64::new(3.0, 4.0, 0.0);
    let norm_sq = v.norm_squared();
    let norm = v.norm2();
    assert!((norm_sq - norm * norm).abs() < 1e-10);
}
